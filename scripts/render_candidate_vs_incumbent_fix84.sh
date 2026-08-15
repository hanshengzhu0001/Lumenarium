#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
root="a10_reusable_results/paper30"
scene="${SCENEPROOF_SCENE:-bedroom_01}"
source_version="v4_deepsearch"
incumbent_version="${SCENEPROOF_INCUMBENT_VERSION:-v5_sceneproof_pose_serialization_smoke1_fix76}"
candidate_version="${SCENEPROOF_CANDIDATE_VERSION:-v5_sceneproof_local_settle_candidate_smoke1_fix82}"

blender="${IMAGINARIUM_BLENDER:-$HOME/Lumenarium/third_party/blender-4.3.2-linux-x64/blender}"

source_json="$(find "$root/${scene}_${source_version}_result/S3_pose_inference" -maxdepth 1 -type f -name '*_placement_info.json' -print -quit)"

incumbent_placement="$root/${scene}_${incumbent_version}_result/S4_layout_refinement/${scene}_${incumbent_version}_placement_info_s4.json"
candidate_placement="$root/${scene}_${candidate_version}_result/S4_layout_refinement/${scene}_${candidate_version}_placement_info_s4.json"

incumbent_render="$root/${scene}_${incumbent_version}_result/S4_layout_refinement/${scene}_${incumbent_version}_render_beauty_256s.png"
candidate_render="$root/${scene}_${candidate_version}_result/S4_layout_refinement/${scene}_${candidate_version}_render_beauty_256s.png"

test -s "$source_json" || { echo "Missing source: $source_json" >&2; exit 2; }
test -s "$incumbent_placement" || { echo "Missing Fix76: $incumbent_placement" >&2; exit 2; }
test -s "$candidate_placement" || { echo "Missing Fix82 candidate: $candidate_placement" >&2; exit 2; }
test -x "$blender" || { echo "Missing Blender: $blender" >&2; exit 2; }

render_one() {
    local label="$1"
    local placement="$2"
    local output="$3"
    echo ""
    echo "=== Render: $label ==="
    echo "Placement: $placement"
    echo "Output:    $output"
    echo ""

    env CUDA_VISIBLE_DEVICES=0 \
        IMAGINARIUM_S4_RENDER_ONLY_PLACEMENT="$placement" \
        IMAGINARIUM_S4_RENDER_ONLY_OUTPUT="$output" \
        IMAGINARIUM_S4_RENDER_ONLY_SAMPLES=256 \
        LD_LIBRARY_PATH="$HOME/.venvs/lumenarium-py311/lib:${LD_LIBRARY_PATH:-}" \
        "$blender" --background \
            --python modules/S4_blender_layout_and_corr.py -- \
            --obj_placement_info_json_path "$source_json" \
            --output_folder /tmp/render_candidate_vs_incumbent_fix84 \
            > /dev/null 2>&1 < /dev/null

    if [ -s "$output" ]; then
        echo "OK: $(stat --format=%s "$output") bytes"
    else
        echo "FAILED: no output" >&2
        return 1
    fi
}

render_one "Fix76 (incumbent)" "$incumbent_placement" "$incumbent_render"
render_one "Fix82 candidate (settled chair)" "$candidate_placement" "$candidate_render"

echo ""
echo "================================================"
echo "RENDER COMPARISON"
echo "================================================"
echo "FIX76_RENDER=$incumbent_render"
echo "FIX82_RENDER=$candidate_render"
echo ""
echo "To compare side-by-side on the A10:"
echo "  python -c \"from PIL import Image; a=Image.open('$incumbent_render'); b=Image.open('$candidate_render'); print(a.size, b.size)\""
