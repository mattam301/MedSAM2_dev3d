#!/usr/bin/env bash
set -euo pipefail

# ===========================================================================
# 3-Model Comparison: Base vs Stacked Causal vs Stacked Bidirectional
#
#   1) base             → original pretrained, single slices
#   2) stacked_causal   → stacked-finetuned, 8-frame clips, causal inference
#   3) stacked_bidir    → stacked-finetuned, 8-frame clips, bidirectional inference
# ===========================================================================

PYTHON="${PYTHON:-python}"
EXPERIMENT_NAME="${1:-stack_comparison_e10_b20}"
WINDOW_SIZE="${WINDOW_SIZE:-8}"
WINDOW_STRIDE="${WINDOW_STRIDE:-8}"

SPLITS_ROOT="data/new_datasets/_splits/${EXPERIMENT_NAME}"
META_JSON="${SPLITS_ROOT}/experiment.json"
EXP_LOG="exp_log/${EXPERIMENT_NAME}"
EVAL_ROOT="${EXP_LOG}/eval_final"
COMPARISON_DIR="${EXP_LOG}/comparison"
TMP_STACKED_TEST_ROOT="${EXP_LOG}/tmp_stacked_test_w${WINDOW_SIZE}_s${WINDOW_STRIDE}"

if [ ! -f "$META_JSON" ]; then
  echo "ERROR: Experiment metadata not found: $META_JSON"
  exit 1
fi

echo "=========================================================="
echo " 3-Model Comparison"
echo " Experiment : ${EXPERIMENT_NAME}"
echo " Window     : ${WINDOW_SIZE} frames, stride ${WINDOW_STRIDE}"
echo "=========================================================="

# --------------------------------------------------------------------------
# Load metadata
# --------------------------------------------------------------------------
BASE_CHECKPOINT="${BASE_CHECKPOINT:-$("$PYTHON" -c "
import json
m = json.load(open('$META_JSON'))
print(m.get('base_checkpoint', 'checkpoints/sam2.1_hiera_tiny.pt'))
")}"
STACK_CKPT="$("$PYTHON" -c "import json; print(json.load(open('$META_JSON'))['stack_output'])")/checkpoints/checkpoint.pt"
INFER_CONFIG="$("$PYTHON" -c "import json; print(json.load(open('$META_JSON'))['infer_config'])")"
IMAGE_CHANNEL_INDEX="$("$PYTHON" -c "import json; print(json.load(open('$META_JSON'))['image_channel_index'])")"

echo ""
echo "Checkpoints:"
echo "  base    : $BASE_CHECKPOINT"
echo "  stacked : $STACK_CKPT"
echo "Config    : $INFER_CONFIG"
echo "Channel   : $IMAGE_CHANNEL_INDEX"

for CKPT in "$BASE_CHECKPOINT" "$STACK_CKPT"; do
  [ -f "$CKPT" ] || { echo "ERROR: Checkpoint not found: $CKPT"; exit 1; }
done
echo "✓ All checkpoints found"

mkdir -p "$TMP_STACKED_TEST_ROOT" "$COMPARISON_DIR"

# --------------------------------------------------------------------------
# Find slice test manifests
# --------------------------------------------------------------------------
mapfile -t SLICE_TEST_MANIFESTS < <(
  find "${SPLITS_ROOT}/manifests" -maxdepth 1 -name "*_slice_test.txt" | sort
)
if [ ${#SLICE_TEST_MANIFESTS[@]} -eq 0 ]; then
  echo "ERROR: No *_slice_test.txt manifests found"
  exit 1
fi

echo ""
echo "Found ${#SLICE_TEST_MANIFESTS[@]} dataset(s):"
printf '  %s\n' "${SLICE_TEST_MANIFESTS[@]}"

# --------------------------------------------------------------------------
# Process each dataset
# --------------------------------------------------------------------------
for MANIFEST in "${SLICE_TEST_MANIFESTS[@]}"; do
  MANIFEST_BASENAME="$(basename "$MANIFEST" .txt)"
  DATASET_SLUG="${MANIFEST_BASENAME%_slice_test}"

  DATASET_DIR="$("$PYTHON" - "$META_JSON" "$DATASET_SLUG" <<'PYEOF'
import json, re, sys
def slugify(v): return re.sub(r"[^a-zA-Z0-9._-]+", "_", v.strip()).strip("_").lower()
meta_json, ds_slug = sys.argv[1], sys.argv[2]
meta = json.load(open(meta_json))
for name, info in meta["splits"].items():
    if slugify(name) == ds_slug:
        print(info["dataset_dir"], end="")
        sys.exit(0)
sys.exit(1)
PYEOF
)" || true
  [ -z "$DATASET_DIR" ] && { echo "WARNING: Cannot resolve $DATASET_SLUG"; continue; }

  TMP_DS_ROOT="${TMP_STACKED_TEST_ROOT}/${DATASET_SLUG}"
  TMP_DS_NPZ_DIR="${TMP_DS_ROOT}/npz"
  TMP_DS_MANIFEST="${TMP_DS_ROOT}/${DATASET_SLUG}_stacked_test.txt"
  mkdir -p "$TMP_DS_NPZ_DIR"

  echo ""
  echo "=========================================================="
  echo " Dataset: ${DATASET_SLUG}"
  echo "=========================================================="

  # ── 1/3: Base on single slices ──────────────────────────────────────────
  echo "  [1/3] BASE (pretrained) on single slices..."
  "$PYTHON" scripts/eval_npz_dataset.py \
    --checkpoint "$BASE_CHECKPOINT" \
    --cfg "$INFER_CONFIG" \
    --dataset-dir "$DATASET_DIR" \
    --file-list "$MANIFEST" \
    --image-channel-index "$IMAGE_CHANNEL_INDEX" \
    --output-dir "${EVAL_ROOT}/base/${DATASET_SLUG}"

  # ── Build 8-frame clips if needed ──────────────────────────────────────
  if [ -f "$TMP_DS_MANIFEST" ]; then
    CLIP_COUNT=$(wc -l < "$TMP_DS_MANIFEST" | tr -d ' ')
    echo "  Reusing ${CLIP_COUNT} existing ${WINDOW_SIZE}-frame clips"
  else
    echo "  Building ${WINDOW_SIZE}-frame stacked test clips..."
    "$PYTHON" - "$DATASET_DIR" "$MANIFEST" "$TMP_DS_NPZ_DIR" "$TMP_DS_MANIFEST" "$WINDOW_SIZE" "$WINDOW_STRIDE" <<'PYEOF'
import re, sys, numpy as np
from pathlib import Path
from collections import defaultdict

dataset_dir = Path(sys.argv[1]).resolve()
manifest_path = Path(sys.argv[2]).resolve()
out_npz_dir = Path(sys.argv[3]).resolve()
out_manifest = Path(sys.argv[4]).resolve()
window_size = int(sys.argv[5])
stride = int(sys.argv[6])
out_npz_dir.mkdir(parents=True, exist_ok=True)

lines = [x.strip() for x in manifest_path.read_text().splitlines() if x.strip()]
if not lines: raise SystemExit(f"Empty manifest: {manifest_path}")

def extract_case_and_index(rel):
    p = Path(rel); stem = p.name; parent = str(p.parent) if str(p.parent) != "." else ""
    m = re.match(r"^(.*?)(\d+)$", stem)
    prefix = m.group(1).rstrip("_-") if m else stem
    idx = int(m.group(2)) if m else 0
    case_key = f"{parent}/{prefix}" if parent and prefix else (parent or prefix or stem)
    return case_key, idx

groups = defaultdict(list)
for rel in lines:
    key, idx = extract_case_and_index(rel)
    npz = dataset_dir / (rel if rel.endswith(".npz") else rel+".npz")
    groups[key].append((idx, npz))

num_clips = 0; manifest_entries = []
for key, items in sorted(groups.items()):
    items.sort(key=lambda x: x[0])
    for clip_id, start in enumerate(range(0, len(items), stride)):
        clip_items = items[start:start+window_size]
        if not clip_items: continue
        while len(clip_items) < window_size:
            clip_items.append(clip_items[-1])
        imgs_list, gts_list = [], []
        for _, npz_path in clip_items:
            data = np.load(npz_path, allow_pickle=True)
            imgs_list.append(data["imgs"]); gts_list.append(data["gts"])
        safe_key = re.sub(r"[^a-zA-Z0-9._/-]+", "_", key).strip("/")
        clip_rel = f"{safe_key}/clip_{clip_id:04d}"
        clip_out = out_npz_dir / f"{clip_rel}.npz"
        clip_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(clip_out, imgs=np.stack(imgs_list, axis=0), gts=np.stack(gts_list, axis=0))
        manifest_entries.append(clip_rel); num_clips += 1

out_manifest.write_text("\n".join(manifest_entries)+"\n")
d = np.load(out_npz_dir / (manifest_entries[0]+".npz"), allow_pickle=True)
print(f"  Built {num_clips} clips | Sample: imgs={d['imgs'].shape}, gts={d['gts'].shape}")
PYEOF
  fi

  # ── 2/3: Stacked Causal on 8-frame clips ───────────────────────────────
  echo "  [2/3] STACKED CAUSAL on ${WINDOW_SIZE}-frame clips..."
  "$PYTHON" scripts/eval_npz_dataset.py \
    --checkpoint "$STACK_CKPT" \
    --cfg "$INFER_CONFIG" \
    --dataset-dir "$TMP_DS_NPZ_DIR" \
    --file-list "$TMP_DS_MANIFEST" \
    --image-channel-index "$IMAGE_CHANNEL_INDEX" \
    --output-dir "${EVAL_ROOT}/stacked_causal/${DATASET_SLUG}"

  # ── 3/3: Stacked Bidirectional on 8-frame clips ────────────────────────
  echo "  [3/3] STACKED BIDIRECTIONAL on ${WINDOW_SIZE}-frame clips..."
  "$PYTHON" scripts/eval_npz_dataset_bidirectional.py \
    --checkpoint "$STACK_CKPT" \
    --cfg "$INFER_CONFIG" \
    --dataset-dir "$TMP_DS_NPZ_DIR" \
    --file-list "$TMP_DS_MANIFEST" \
    --image-channel-index "$IMAGE_CHANNEL_INDEX" \
    --output-dir "${EVAL_ROOT}/stacked_bidir/${DATASET_SLUG}"
done

# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------
echo ""
echo "=========================================================="
echo " RESULTS"
echo "=========================================================="

"$PYTHON" - "$EVAL_ROOT" "$COMPARISON_DIR" <<'PYEOF'
import json, sys, math
from pathlib import Path
from collections import defaultdict

eval_root = Path(sys.argv[1])
compare_dir = Path(sys.argv[2])
compare_dir.mkdir(parents=True, exist_ok=True)

MODEL_INFO = {
    "base":            ("Base (pretrained)",  "1-slice"),
    "stacked_causal":  ("Stacked+Causal",     "8-frame"),
    "stacked_bidir":   ("Stacked+Bidir",      "8-frame"),
}
MODEL_ORDER = list(MODEL_INFO.keys())

def find_metric_files(d):
    return sorted(d.rglob("summary.json")) + sorted(d.rglob("metrics.json"))

def get_metric(m, *keys):
    for k in keys:
        if k in m: return m[k]
    return float("nan")

def fmt(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) and not math.isnan(v) else "N/A"

rows = []
for tag in MODEL_ORDER:
    d = eval_root / tag
    if not d.exists():
        print(f"WARNING: {d} not found"); continue
    for mf in find_metric_files(d):
        ds = mf.parent.name
        try: metrics = json.loads(mf.read_text())
        except: continue
        rows.append({
            "dataset": ds, "model": tag,
            "dice": get_metric(metrics, "dice", "dsc", "mean_dice", "meanDice"),
            "iou":  get_metric(metrics, "iou", "jaccard", "mean_iou", "meanIoU"),
        })

if not rows:
    print("No metrics found."); sys.exit(1)

by_ds = defaultdict(dict)
for r in rows:
    by_ds[r["dataset"]][r["model"]] = r

W = 24; M = 20
sep = "=" * 82
thin = "-" * 82

# ── TABLE 1: Main Results ─────────────────────────────────────────────────
print(f"\n{sep}")
print("  TABLE 1: MAIN RESULTS")
print(sep)
print(f"{'Dataset':<{W}} {'Model':<{M}} {'Input':<8} {'Dice':>8} {'IoU':>8}")
print(thin)
prev = None
for ds in sorted(by_ds.keys()):
    if prev: print(thin)
    for tag in MODEL_ORDER:
        if tag not in by_ds[ds]: continue
        r = by_ds[ds][tag]
        label, inp = MODEL_INFO[tag]
        print(f"{ds:<{W}} {label:<{M}} {inp:<8} {fmt(r['dice']):>8} {fmt(r['iou']):>8}")
    prev = ds
print(sep)

# ── TABLE 2: Improvement Over Base ────────────────────────────────────────
print(f"\n{sep}")
print("  TABLE 2: IMPROVEMENT OVER BASE")
print(sep)
print(f"{'Dataset':<{W}} {'Model':<{M}} {'Δ Dice':>10} {'Δ IoU':>10}")
print(thin)
for ds in sorted(by_ds.keys()):
    if "base" not in by_ds[ds]: continue
    base = by_ds[ds]["base"]
    for tag in ("stacked_causal", "stacked_bidir"):
        if tag not in by_ds[ds]: continue
        r = by_ds[ds][tag]
        dd = r["dice"] - base["dice"]
        di = r["iou"] - base["iou"]
        label = MODEL_INFO[tag][0]
        print(f"{ds:<{W}} {label:<{M}} {dd:+10.4f} {di:+10.4f}")
    print(thin)
print(sep)

# ── TABLE 3: Bidirectional vs Causal ──────────────────────────────────────
print(f"\n{sep}")
print("  🔥 TABLE 3: BIDIRECTIONAL vs CAUSAL (same checkpoint, same input)")
print(sep)
print(f"{'Dataset':<{W}} {'Causal':>10} {'Bidir':>10} {'Δ Dice':>10} {'Δ IoU':>10} {'Winner':<12}")
print(thin)
bidir_wins = causal_wins = ties = 0
for ds in sorted(by_ds.keys()):
    if "stacked_causal" not in by_ds[ds] or "stacked_bidir" not in by_ds[ds]: continue
    c = by_ds[ds]["stacked_causal"]; b = by_ds[ds]["stacked_bidir"]
    dd = b["dice"] - c["dice"]; di = b["iou"] - c["iou"]
    if dd > 1e-4: winner = "BIDIR ✓"; bidir_wins += 1
    elif dd < -1e-4: winner = "CAUSAL ✓"; causal_wins += 1
    else: winner = "tie"; ties += 1
    print(f"{ds:<{W}} {fmt(c['dice']):>10} {fmt(b['dice']):>10} {dd:+10.4f} {di:+10.4f} {winner:<12}")
print(thin)
print(f"  Bidir wins: {bidir_wins} | Causal wins: {causal_wins} | Ties: {ties}")
print(sep)

# ── TABLE 4: Best Model Per Dataset ──────────────────────────────────────
print(f"\n{sep}")
print("  TABLE 4: BEST MODEL PER DATASET")
print(sep)
print(f"{'Dataset':<{W}} {'Best Model':<{M}} {'Input':<8} {'Dice':>8} {'IoU':>8}")
print(thin)
for ds in sorted(by_ds.keys()):
    models = by_ds[ds]
    best_tag = max(models, key=lambda t: models[t]["dice"] if isinstance(models[t]["dice"], (int,float)) else -1)
    best = models[best_tag]
    label, inp = MODEL_INFO[best_tag]
    print(f"{ds:<{W}} {label:<{M}} {inp:<8} {fmt(best['dice']):>8} {fmt(best['iou']):>8}")
print(sep)

# ── TABLE 5: Cross-Dataset Averages ──────────────────────────────────────
print(f"\n{sep}")
print("  TABLE 5: CROSS-DATASET AVERAGES")
print(sep)
print(f"{'Model':<{M}} {'Input':<8} {'Mean Dice':>12} {'Mean IoU':>12}")
print(thin)
for tag in MODEL_ORDER:
    dice_vals = [by_ds[ds][tag]["dice"] for ds in by_ds if tag in by_ds[ds] and isinstance(by_ds[ds][tag]["dice"], (int,float))]
    iou_vals  = [by_ds[ds][tag]["iou"]  for ds in by_ds if tag in by_ds[ds] and isinstance(by_ds[ds][tag]["iou"], (int,float))]
    mean_d = sum(dice_vals)/len(dice_vals) if dice_vals else float('nan')
    mean_i = sum(iou_vals)/len(iou_vals)   if iou_vals  else float('nan')
    label, inp = MODEL_INFO[tag]
    print(f"{label:<{M}} {inp:<8} {fmt(mean_d):>12} {fmt(mean_i):>12}")
print(sep)

# ── Save JSON ─────────────────────────────────────────────────────────────
out = compare_dir / "3model_comparison.json"
result = {
    "description": "Base on 1-slice, Stacked models on 8-frame clips",
    "model_info": {tag: {"label": label, "input": inp} for tag, (label, inp) in MODEL_INFO.items()},
    "per_model": rows,
    "bidir_wins": bidir_wins,
    "causal_wins": causal_wins,
    "ties": ties,
    "by_dataset": {
        ds: {tag: {"dice": models[tag]["dice"], "iou": models[tag]["iou"]}
             for tag in MODEL_ORDER if tag in models}
        for ds, models in by_ds.items()
    },
}
out.write_text(json.dumps(result, indent=2))
print(f"\n✓ Saved: {out}")
PYEOF

echo ""
echo "=========================================================="
echo " Complete!"
echo "=========================================================="
echo " Eval results:"
echo "   base (1-slice)          : ${EVAL_ROOT}/base"
echo "   stacked_causal (8-frame): ${EVAL_ROOT}/stacked_causal"
echo "   stacked_bidir (8-frame) : ${EVAL_ROOT}/stacked_bidir"
echo " Comparison                : ${COMPARISON_DIR}/3model_comparison.json"
echo "=========================================================="