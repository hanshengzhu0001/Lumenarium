#!/usr/bin/env bash
set -Eeuo pipefail

root="${IMAGINARIUM_ROOT:-$HOME/Lumenarium}"
env_prefix="${IMAGINARIUM_ENV:-$HOME/.venvs/lumenarium-py311}"
micromamba="${MICROMAMBA:-$HOME/.local/bin/micromamba}"
mode="${1:-all}"

die() { echo "LUMENARIUM_SETUP_ERROR=$*" >&2; exit 2; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }

check_host() {
  test "$(uname -s)" = Linux || die "Linux x86_64 is required"
  test "$(uname -m)" = x86_64 || die "x86_64 is required"
  need git
  need tar
  need nvidia-smi
  test -d "$root" || die "repository not found: $root"

  local gpu_count min_vram free_kb
  gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
  min_vram="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1 | tr -d ' ')"
  test "$gpu_count" -ge 1 || die "at least one NVIDIA GPU is required"
  test "$min_vram" -ge 22000 || die "at least 22 GB VRAM is required per active worker; found ${min_vram} MiB"
  free_kb="$(df -Pk "$root" | awk 'NR==2 {print $4}')"
  test "$free_kb" -ge 262144000 || die "at least 250 GiB free storage is required"
  echo "HOST_OK gpu_count=$gpu_count minimum_vram_mib=$min_vram free_disk_gib=$((free_kb/1024/1024))"
}

install_micromamba() {
  if test -x "$micromamba"; then return; fi
  mkdir -p "$(dirname "$micromamba")"
  local tmp
  tmp="$(mktemp -d /tmp/lumenarium_micromamba_XXXXXX)"
  if command -v curl >/dev/null 2>&1; then
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
      | tar -xj -C "$tmp" bin/micromamba
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://micro.mamba.pm/api/micromamba/linux-64/latest \
      | tar -xj -C "$tmp" bin/micromamba
  else
    die "curl or wget is required to install micromamba"
  fi
  cp "$tmp/bin/micromamba" "$micromamba"
  chmod +x "$micromamba"
  rm -rf -- "$tmp"
}

install_runtime() {
  install_micromamba
  if ! test -x "$env_prefix/bin/python"; then
    MAMBA_ROOT_PREFIX="$HOME/.mamba-root" "$micromamba" create -y \
      -p "$env_prefix" -c conda-forge python=3.11 pip
  fi
  IMAGINARIUM_ROOT="$root" MICROMAMBA="$micromamba" \
    IMAGINARIUM_ENV="$env_prefix" bash "$root/scripts/setup_a10_inference.sh" all
  if ! test -e "$root/.env.lumenarium"; then
    cp "$root/.env.lumenarium.example" "$root/.env.lumenarium"
    chmod 600 "$root/.env.lumenarium"
  fi
  echo "CONFIGURE_NEXT=$root/.env.lumenarium"
}

load_config() {
  test -s "$root/.env.lumenarium" || die "copy .env.lumenarium.example to .env.lumenarium first"
  set -a
  # shellcheck disable=SC1091
  source "$root/.env.lumenarium"
  set +a
  [[ "${GPT_API_KEY:-}" != replace-* ]] || die "GPT_API_KEY is not configured"
  [[ "${GPT_ENDPOINT:-}" == http://* || "${GPT_ENDPOINT:-}" == https://* ]] || die "GPT_ENDPOINT must be HTTP(S)"
  [[ "${OMNIVERSE_DEEPSEARCH_URL:-}" == http://* || "${OMNIVERSE_DEEPSEARCH_URL:-}" == https://* ]] || die "OMNIVERSE_DEEPSEARCH_URL must be HTTP(S)"
  [[ "${SCENEPROOF_WORKER_TOKEN:-}" != replace-* && -n "${SCENEPROOF_WORKER_TOKEN:-}" ]] || die "SCENEPROOF_WORKER_TOKEN is not configured"
  LC_ALL=C grep -q '^[[:print:]]*$' <<<"$SCENEPROOF_WORKER_TOKEN" || die "SCENEPROOF_WORKER_TOKEN must be ASCII"
}

verify_runtime() {
  check_host
  test -x "$env_prefix/bin/python" || die "Python environment missing: $env_prefix"
  IMAGINARIUM_ROOT="$root" MICROMAMBA="$micromamba" \
    IMAGINARIUM_ENV="$env_prefix" bash "$root/scripts/setup_a10_inference.sh" verify
  load_config
  "$env_prefix/bin/python" - <<'PY'
import os, requests
for name in ("GPT_ENDPOINT", "OMNIVERSE_DEEPSEARCH_URL"):
    value = os.environ[name]
    print(f"CONFIGURED_{name}={value}")
print("VISUAL_API_CONFIG_OK=1")
PY
}

start_service() {
  verify_runtime
  cd "$root"
  if test -z "${SCENEPROOF_API_GPU_IDS:-}"; then
    gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
    if test "$gpu_count" -ge 2; then
      export SCENEPROOF_API_GPU_IDS=0,1
    else
      export SCENEPROOF_API_GPU_IDS=0
    fi
  fi
  bash scripts/restart_sceneproof_api_fix140.sh
}

case "$mode" in
  all) check_host; install_runtime ;;
  verify) verify_runtime ;;
  start) start_service ;;
  *) echo "usage: $0 [all|verify|start]" >&2; exit 2 ;;
esac
