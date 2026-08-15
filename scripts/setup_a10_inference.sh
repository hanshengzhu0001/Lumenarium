#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${IMAGINARIUM_ROOT:-$HOME/Lumenarium}"
MM="${MICROMAMBA:-$HOME/.local/bin/micromamba}"
ENV_PREFIX="${IMAGINARIUM_ENV:-$HOME/.venvs/lumenarium-py311}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
PHASE="${1:-all}"

export HF_HOME
export PIP_DISABLE_PIP_VERSION_CHECK=1
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"

run_python() {
  MAMBA_ROOT_PREFIX="$HOME/.mamba-root" "$MM" run -p "$ENV_PREFIX" "$@"
}

install_python() {
  test -x "$MM"
  test -x "$ENV_PREFIX/bin/python"
  cd "$ROOT"
  run_python python -m pip install --upgrade pip setuptools wheel
  run_python python -m pip install -r requirements.txt
  # SAM3 support landed after older Transformers releases; upgrade explicitly.
  run_python python -m pip install --upgrade transformers huggingface_hub
  MAMBA_ROOT_PREFIX="$HOME/.mamba-root" "$MM" install -y -p "$ENV_PREFIX" \
    -c conda-forge xorg-libsm xorg-libxext xorg-libxrender xorg-libxi \
    xorg-libxfixes xorg-libx11 libglvnd assimp
  if [[ ! -e config/config.yaml ]]; then
    cp config/config_sam3_gemini.yaml config/config.yaml
  fi
}

download_data() {
  cd "$ROOT"
  mkdir -p asset_data weights third_party "$HF_HOME"

  local hf=(run_python hf download)
  local common=(--repo-type dataset)
  if [[ -n "${HF_TOKEN:-}" ]]; then
    common+=(--token "$HF_TOKEN")
  fi

  "${hf[@]}" HiHiAllen/Imaginarium-Dataset \
    imaginarium_assets.tar.gz \
    imaginarium_assets_internal_placement_space.tar.gz \
    imaginarium_asset_info.csv \
    background_texture_dataset.tar.gz \
    "${common[@]}" --local-dir asset_data

  "${hf[@]}" binicey/Imaginarium-3D-Derived-Dataset \
    imaginarium_assets_render_results_part1.tar.gz \
    imaginarium_assets_render_results_part2.tar.gz \
    imaginarium_assets_render_results_part3.tar.gz \
    imaginarium_assets_render_results_part4.tar.gz \
    imaginarium_assets_patch_embedding.tar.gz \
    imaginarium_assets_voxels.tar.gz \
    "${common[@]}" --local-dir asset_data

  "${hf[@]}" binicey/Imaginarium-3D-Derived-Dataset \
    dinov2_vitl14.pth \
    ae_net_pretrained_weights.pth \
    depth_anything_v2_metric_hypersim_vitl.pth \
    "${common[@]}" --local-dir weights

  "${hf[@]}" binicey/Imaginarium-3D-Derived-Dataset \
    blender-4.3.2-linux-x64.tar.gz \
    "${common[@]}" --local-dir third_party
}

extract_data() {
  cd "$ROOT"
  # These two archives already include an asset_data/ prefix.
  for archive in \
    asset_data/imaginarium_assets.tar.gz \
    asset_data/imaginarium_assets_voxels.tar.gz; do
    test -s "$archive"
    tar -xzf "$archive" -C "$ROOT"
  done
  for archive in \
    asset_data/imaginarium_assets_internal_placement_space.tar.gz \
    asset_data/background_texture_dataset.tar.gz \
    asset_data/imaginarium_assets_render_results_part*.tar.gz \
    asset_data/imaginarium_assets_patch_embedding.tar.gz; do
    test -s "$archive"
    tar -xzf "$archive" -C asset_data
  done
  test -s third_party/blender-4.3.2-linux-x64.tar.gz
  tar -xzf third_party/blender-4.3.2-linux-x64.tar.gz -C third_party
}

download_hf_models() {
  cd "$ROOT"
  local auth=()
  if [[ -n "${HF_TOKEN:-}" ]]; then
    auth=(--token "$HF_TOKEN")
  fi
  # DINO processor files and SAM3 are cached once for offline test runs.
  run_python hf download facebook/dinov2-large "${auth[@]}"
  run_python hf download facebook/sam3 "${auth[@]}"
}

verify() {
  cd "$ROOT"
  run_python python -m py_compile \
    run_imaginarium_I2Layout.py \
    run_imaginarium_I2Layout_v3.py \
    run_imaginarium_I2Layout_v4_deepsearch.py \
    modules/retrieval.py

  for path in \
    asset_data/imaginarium_asset_info.csv \
    asset_data/imaginarium_assets \
    asset_data/imaginarium_assets_internal_placement_space \
    asset_data/background_texture_dataset \
    asset_data/imaginarium_assets_render_results \
    asset_data/imaginarium_assets_patch_embedding \
    asset_data/imaginarium_assets_voxels \
    weights/dinov2_vitl14.pth \
    weights/ae_net_pretrained_weights.pth \
    weights/depth_anything_v2_metric_hypersim_vitl.pth \
    third_party/blender-4.3.2-linux-x64/blender; do
    test -e "$path" || { echo "MISSING: $path" >&2; return 1; }
  done

  run_python python - <<'PY'
import torch
from transformers import Sam3Model, Sam3Processor
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
print("SAM3 imports OK")
PY

  "$ROOT/third_party/blender-4.3.2-linux-x64/blender" --version | head -1
  echo "A10 inference prerequisites verified."
}

case "$PHASE" in
  python) install_python ;;
  download) download_data ;;
  extract) extract_data ;;
  models) download_hf_models ;;
  verify) verify ;;
  all)
    install_python
    download_data
    extract_data
    download_hf_models
    verify
    ;;
  *)
    echo "Usage: $0 [all|python|download|extract|models|verify]" >&2
    exit 2
    ;;
esac
