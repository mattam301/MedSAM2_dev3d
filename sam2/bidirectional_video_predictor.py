"""
Bidirectional Video Predictor for 3D Medical Image Segmentation.

All new architecture code lives in this single file.  Nothing in the original
MedSAM2 / SAM2 source tree is modified.

Design (matches Architecture_Innovation.md):

  Pass 1  –  Standard SAM2 causal forward sweep.
             Produces  M_i^(0) = SAM2_forward(I_i, {M_{i-k}^(0)})  for all i.

  Pass 2  –  Bidirectional refinement.
             For every non-conditioning slice i the MemoryAttention module is
             re-run with an augmented memory bank that contains *both* past and
             future memories from Pass 1:

               Mem_i = Attn(I_i, {M_{i-1}^(0), M_{i+1}^(0), …})

             The existing MemoryAttention transformer is used as-is; only the
             set of key/value memory tokens is extended.

  Consistency Loss (Phase 2, training)  –  identity-warp L1 regulariser:
             L_cons = || M_i - M_{i-1} ||_1  +  || M_i - M_{i+1} ||_1

Temporal positional encoding for future frames
  The model's learned maskmem_tpos_enc[k] (shape [num_maskmem,1,1,mem_dim])
  is reused symmetrically:
    - past frame k steps back  →  tpos_enc[k-1]
    - future frame k steps ahead  →  tpos_enc[k-1]          (same index)
  This requires zero new parameters and leaves existing checkpoints intact.
"""

import logging
import torch
import torch.nn.functional as F
from tqdm import tqdm

from sam2.sam2_video_predictor_npz import SAM2VideoPredictorNPZ
from sam2.modeling.sam2_utils import get_1d_sine_pe, select_closest_cond_frames
from sam2.build_sam import get_best_available_device, _load_checkpoint


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Consistency loss (Phase 2 training helper, usable independently)
# ---------------------------------------------------------------------------

def slice_consistency_loss(
    pred_masks: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    """Identity-warp slice consistency regularisation.

    For each interior slice i (i=1 … N-2):
        L_cons = mean_i( |M_i - M_{i-1}| + |M_i - M_{i+1}| )

    Args:
        pred_masks: Raw mask logits, shape [N, 1, H, W].
                    N is the number of slices in the volume.
        weight:     Scalar loss coefficient (default 1.0).

    Returns:
        Scalar tensor.  Returns 0 if N < 3.
    """
    if pred_masks.shape[0] < 3:
        return pred_masks.new_zeros(()).requires_grad_(pred_masks.requires_grad)

    probs = torch.sigmoid(pred_masks)   # [N, 1, H, W]
    interior  = probs[1:-1]             # [N-2, 1, H, W]
    prev_slc  = probs[:-2]              # [N-2, 1, H, W]
    next_slc  = probs[2:]               # [N-2, 1, H, W]

    loss = (interior - prev_slc).abs() + (interior - next_slc).abs()
    return weight * loss.mean()


# ---------------------------------------------------------------------------
# Bidirectional predictor
# ---------------------------------------------------------------------------

class BidirectionalSAM2VideoPredictorNPZ(SAM2VideoPredictorNPZ):
    """SAM2VideoPredictorNPZ extended with bidirectional attention for 3D volumes.

    Adds two public entry-points on top of the parent class:

      * propagate_in_video_bidirectional()  –  two-pass inference
      * (slice_consistency_loss is a module-level helper, not a method)

    All existing public methods of SAM2VideoPredictorNPZ remain unchanged.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # State flags set only during Pass 2; reset on completion / error.
        self._use_bidirectional_memory: bool = False
        self._pass1_output_dict: dict | None = None

    # ------------------------------------------------------------------
    # Override: memory-conditioned features (honours bidirectional flag)
    # ------------------------------------------------------------------

    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,
    ):
        """Route to bidirectional version when Pass 2 is active."""
        if self._use_bidirectional_memory and self._pass1_output_dict is not None:
            return self._prepare_bidirectional_memory_conditioned_features(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                output_dict=output_dict,
                num_frames=num_frames,
            )
        return super()._prepare_memory_conditioned_features(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            output_dict,
            num_frames,
            track_in_reverse,
        )

    # ------------------------------------------------------------------
    # Bidirectional memory assembly
    # ------------------------------------------------------------------
    def _prepare_memory_conditioned_features(self, frame_idx, is_init_cond_frame,
            current_vision_feats, current_vision_pos_embeds, feat_sizes,
            output_dict, num_frames, track_in_reverse=False):
        
        print(f'[PMCF] frame={frame_idx} bidir={self._use_bidirectional_memory} '
            f'pass1={self._pass1_output_dict is not None}')
        
        if self._use_bidirectional_memory and self._pass1_output_dict is not None:
            return self._prepare_bidirectional_memory_conditioned_features(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                output_dict=output_dict,
                num_frames=num_frames,
            )
        return super()._prepare_memory_conditioned_features(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            output_dict,
            num_frames,
            track_in_reverse,
        )
    def _prepare_bidirectional_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
    ):
        """Fuse current frame features with both past AND future memories.

        Past memories are drawn from output_dict (same as original).
        Future memories are drawn from self._pass1_output_dict (stable Pass 1
        results).  Both are fed as key/value tokens into the existing
        MemoryAttention module – no architectural change is needed.

        Temporal positional encoding:
          Past  frame at distance k  →  maskmem_tpos_enc[k-1]
          Future frame at distance k →  maskmem_tpos_enc[k-1]  (symmetric reuse)
        """
        B = current_vision_feats[-1].size(1)
        C = self.hidden_dim
        H, W = feat_sizes[-1]
        device = current_vision_feats[-1].device

        if self.num_maskmem == 0:
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            return pix_feat

        to_cat_memory: list[torch.Tensor] = []
        to_cat_memory_pos_embed: list[torch.Tensor] = []
        num_obj_ptr_tokens = 0

        pass1 = self._pass1_output_dict  # stable Pass 1 outputs

        # quick check
        future_found = sum(
            1 for k in range(1, self.num_maskmem)
            if self._pass1_output_dict["non_cond_frame_outputs"].get(frame_idx + k) is not None
        )
        print(f'[BIDIR] frame={frame_idx} '
            f'pass1_non_cond_keys={sorted(self._pass1_output_dict["non_cond_frame_outputs"].keys())} '
            f'future_found={future_found}')

        if not is_init_cond_frame:
            # ---- conditioning frames (t_pos = 0) -------------------------
            cond_outputs = output_dict["cond_frame_outputs"]
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                frame_idx, cond_outputs, self.max_cond_frames_in_attn
            )
            t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]

            # ---- past non-conditioning memories (t_rel = 1 … num_maskmem-1) ----
            stride = self.memory_temporal_stride_for_eval
            for t_pos in range(1, self.num_maskmem):
                t_rel = self.num_maskmem - t_pos  # frames before current
                if t_rel == 1:
                    prev_frame_idx = frame_idx - 1
                else:
                    prev_frame_idx = ((frame_idx - 2) // stride) * stride
                    prev_frame_idx = prev_frame_idx - (t_rel - 2) * stride

                out = pass1["non_cond_frame_outputs"].get(prev_frame_idx, None)
                if out is None:
                    out = unselected_cond_outputs.get(prev_frame_idx, None)
                t_pos_and_prevs.append((t_pos, out))

            for t_pos, prev in t_pos_and_prevs:
                if prev is None:
                    continue
                feats = prev["maskmem_features"].to(device, non_blocking=True)
                to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))
                maskmem_enc = prev["maskmem_pos_enc"][-1].to(device)
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
                # tpos_enc index for past: distance k → index k-1
                # (for cond frames t_pos=0 → index num_maskmem-1, same as original)
                maskmem_enc = (
                    maskmem_enc
                    + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
                )
                to_cat_memory_pos_embed.append(maskmem_enc)

            # ---- future non-conditioning memories (k = 1 … num_maskmem-1) ----
            for k in range(1, self.num_maskmem):
                future_frame_idx = frame_idx + k
                if future_frame_idx >= num_frames:
                    break  # reached the end of the volume

                fut = pass1["non_cond_frame_outputs"].get(future_frame_idx, None)
                if fut is None:
                    # conditioning frames are also valid future context
                    fut = output_dict["cond_frame_outputs"].get(future_frame_idx, None)
                if fut is None:
                    continue

                feats = fut["maskmem_features"].to(device, non_blocking=True)
                to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))
                maskmem_enc = fut["maskmem_pos_enc"][-1].to(device)
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
                # Symmetric reuse: future distance k → same index as past distance k
                maskmem_enc = maskmem_enc + self.maskmem_tpos_enc[k - 1]
                to_cat_memory_pos_embed.append(maskmem_enc)

            # ---- object pointers (past only, consistent with eval default) ----
            if self.use_obj_ptrs_in_encoder:
                max_ptrs = min(num_frames, self.max_obj_ptrs_in_encoder)
                only_past_eval = getattr(
                    self, "only_obj_ptrs_in_the_past_for_eval", True
                )
                if not self.training and only_past_eval:
                    ptr_cond_outputs = {
                        t: out
                        for t, out in selected_cond_outputs.items()
                        if t <= frame_idx
                    }
                else:
                    ptr_cond_outputs = selected_cond_outputs

                pos_and_ptrs = [
                    (abs(frame_idx - t), out["obj_ptr"])
                    for t, out in ptr_cond_outputs.items()
                ]
                for t_diff in range(1, max_ptrs):
                    t = frame_idx - t_diff
                    if t < 0:
                        break
                    out = pass1["non_cond_frame_outputs"].get(
                        t, unselected_cond_outputs.get(t, None)
                    )
                    if out is not None:
                        pos_and_ptrs.append((t_diff, out["obj_ptr"]))

                if len(pos_and_ptrs) > 0:
                    pos_list, ptrs_list = zip(*pos_and_ptrs)
                    obj_ptrs = torch.stack(ptrs_list, dim=0)
                    if self.add_tpos_enc_to_obj_ptrs:
                        t_diff_max = max_ptrs - 1
                        proj_tpos = getattr(self, "proj_tpos_enc_in_obj_ptrs", False)
                        tpos_dim = C if proj_tpos else self.mem_dim
                        obj_pos = torch.tensor(pos_list, device=device)
                        obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                        obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                        obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
                    else:
                        obj_pos = obj_ptrs.new_zeros(len(pos_list), B, self.mem_dim)

                    if self.mem_dim < C:
                        obj_ptrs = obj_ptrs.reshape(
                            -1, B, C // self.mem_dim, self.mem_dim
                        )
                        obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                        obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)

                    to_cat_memory.append(obj_ptrs)
                    to_cat_memory_pos_embed.append(obj_pos)
                    num_obj_ptr_tokens = obj_ptrs.shape[0]

        else:
            # Initial conditioning frame: no memory available yet
            if self.directly_add_no_mem_embed:
                pix_feat_with_mem = (
                    current_vision_feats[-1] + self.no_mem_embed
                )
                pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(
                    B, C, H, W
                )
                return pix_feat_with_mem
            to_cat_memory = [self.no_mem_embed.expand(1, B, self.mem_dim)]
            to_cat_memory_pos_embed = [
                self.no_mem_pos_enc.expand(1, B, self.mem_dim)
            ]

        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)

        pix_feat_with_mem = self.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        return pix_feat_with_mem

    # ------------------------------------------------------------------
    # Two-pass bidirectional propagation
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def propagate_in_video_bidirectional(
        self,
        inference_state,
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
    ):
        print(f"[BIDIR ENTRY] propagate_in_video_bidirectional called, num_frames={inference_state['num_frames']}")
        self.propagate_in_video_preflight(inference_state)

        output_dict = inference_state["output_dict"]
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        obj_ids = inference_state["obj_ids"]
        num_frames = inference_state["num_frames"]
        batch_size = self._get_obj_num(inference_state)

        if len(output_dict["cond_frame_outputs"]) == 0:
            raise RuntimeError("No prompts found. Please add points or a mask first.")

        if start_frame_idx is None:
            start_frame_idx = min(output_dict["cond_frame_outputs"])
        if max_frame_num_to_track is None:
            max_frame_num_to_track = num_frames

        end_frame_idx = min(start_frame_idx + max_frame_num_to_track, num_frames - 1)
        processing_order = range(start_frame_idx, end_frame_idx + 1)

        # ----------------------------------------------------------------
        # PASS 1 – standard causal forward sweep
        # ----------------------------------------------------------------
        pass1_non_cond: dict = {}

        for frame_idx in tqdm(processing_order, desc="Bidir Pass 1/2 (forward)"):
            if frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
                storage_key = "cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
            elif frame_idx in consolidated_frame_inds["non_cond_frame_outputs"]:
                storage_key = "non_cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
            else:
                storage_key = "non_cond_frame_outputs"
                current_out, pred_masks = self._run_single_frame_inference(
                    inference_state=inference_state,
                    output_dict=output_dict,
                    frame_idx=frame_idx,
                    batch_size=batch_size,
                    is_init_cond_frame=False,
                    point_inputs=None,
                    mask_inputs=None,
                    reverse=False,
                    run_mem_encoder=True,
                )
                output_dict[storage_key][frame_idx] = current_out

            if storage_key == "non_cond_frame_outputs":
                # Store reference — safe because Pass 2 replaces the dict entry
                # rather than mutating the existing dict in-place
                pass1_non_cond[frame_idx] = current_out

            self._add_output_per_object(
                inference_state, frame_idx, current_out, storage_key
            )
            inference_state["frames_already_tracked"][frame_idx] = {"reverse": False}

        # Stable Pass 1 reference for Pass 2 future-memory lookups
        pass1_output_dict = {
            "cond_frame_outputs": output_dict["cond_frame_outputs"],
            "non_cond_frame_outputs": pass1_non_cond,
        }



        # ----------------------------------------------------------------
        # PASS 2 – bidirectional refinement
        # ----------------------------------------------------------------
        self._use_bidirectional_memory = True
        self._pass1_output_dict = pass1_output_dict

        try:
            for frame_idx in tqdm(processing_order, desc="Bidir Pass 2/2 (refinement)"):
                if frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
                    # Conditioning frames: keep Pass 1 result
                    current_out = output_dict["cond_frame_outputs"][frame_idx]
                    pred_masks = current_out["pred_masks"]
                    storage_key = "cond_frame_outputs"
                else:
                    storage_key = "non_cond_frame_outputs"

                    # *** FIX 1: Clear the cache so _run_single_frame_inference
                    #     actually recomputes instead of returning cached Pass 1 ***
                    inference_state["frames_already_tracked"].pop(frame_idx, None)

                    # *** FIX 2: Remove Pass 1 output so parent doesn't short-circuit ***
                    output_dict["non_cond_frame_outputs"].pop(frame_idx, None)

                    current_out, pred_masks = self._run_single_frame_inference(
                        inference_state=inference_state,
                        output_dict=output_dict,
                        frame_idx=frame_idx,
                        batch_size=batch_size,
                        is_init_cond_frame=False,
                        point_inputs=None,
                        mask_inputs=None,
                        reverse=False,
                        run_mem_encoder=True,
                    )
                    output_dict[storage_key][frame_idx] = current_out

                self._add_output_per_object(
                    inference_state, frame_idx, current_out, storage_key
                )
                inference_state["frames_already_tracked"][frame_idx] = {"reverse": False}

                _, video_res_masks = self._get_orig_video_res_output(
                    inference_state, pred_masks
                )
                yield frame_idx, obj_ids, video_res_masks

        finally:
            self._use_bidirectional_memory = False
            self._pass1_output_dict = None


# ---------------------------------------------------------------------------
# Convenience builder (mirrors build_sam2_video_predictor_npz)
# ---------------------------------------------------------------------------

def build_bidir_sam2_video_predictor_npz(
    config_file: str,
    ckpt_path: str | None = None,
    device: str | None = None,
    mode: str = "eval",
    hydra_overrides_extra: list[str] | None = None,
    apply_postprocessing: bool = True,
    **kwargs,
) -> BidirectionalSAM2VideoPredictorNPZ:
    """Build a BidirectionalSAM2VideoPredictorNPZ from a config + checkpoint.

    Drop-in replacement for build_sam2_video_predictor_npz.  Loads the same
    checkpoint weights but returns a BidirectionalSAM2VideoPredictorNPZ instance
    that exposes propagate_in_video_bidirectional().

    Args:
        config_file:            Hydra config name (e.g. "configs/sam2.1_hiera_t512.yaml").
        ckpt_path:              Path to a MedSAM2 / SAM2 checkpoint (.pt file).
        device:                 Target device string.  Auto-detected if None.
        mode:                   "eval" or "train".
        hydra_overrides_extra:  Additional Hydra overrides.
        apply_postprocessing:   Apply SAM2 mask post-processing (recommended).

    Returns:
        BidirectionalSAM2VideoPredictorNPZ ready for inference.
    """
    from hydra import compose
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    if hydra_overrides_extra is None:
        hydra_overrides_extra = []

    device = device or get_best_available_device()
    logger.info(f"build_bidir_sam2_video_predictor_npz: device={device}")

    # Point Hydra at the bidirectional subclass instead of the base class
    hydra_overrides = [
        "++model._target_="
        "sam2.bidirectional_video_predictor.BidirectionalSAM2VideoPredictorNPZ",
    ]

    if apply_postprocessing:
        hydra_overrides_extra = list(hydra_overrides_extra) + [
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            "++model.fill_hole_area=8",
        ]

    hydra_overrides.extend(hydra_overrides_extra)

    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model
