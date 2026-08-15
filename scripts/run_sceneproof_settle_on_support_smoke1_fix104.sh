#!/usr/bin/env bash
# Fix104Smoke1 — 支撑面贴合（沉降）的 A/B 验证，bedroom_01单场景。
#
# 改的是什么：process_other_objects 里悬空与穿模两个方向不对称，穿模把间隙归零，悬空
# 只在超过 0.2m 时才修且修完仍留 0.2m，0 到 0.2m 的悬空完全不管。现在两个方向统一为
# "子物体底面对齐父物体顶面"。tree_sons 只收 SpatialRel=="on" 且父不是墙的关系，所以
# 挂墙物体不受影响。间隙超过 IMAGINARIUM_SETTLE_MAX_GAP（默认 0.5m）时视为支撑关系本
# 身可疑，保留改动前的行为。
#
# 为什么跑两次而不是跟已有的 fix61 产物比：只有同一次代码、同一个输入、只切一个开关，
# 差异才能归因到这一处改动。
#
# 第一版脚本的两个错误已修：函数用 $( ) 捕获返回值把 S4 的诊断信息一起吞掉了；输入默认
# 用了 v4_deepsearch 那份 S3 json，而 S4 会从输入 json 的上两级目录去找配套的
# *_placement_info_s3.json，产出当前基线的输入其实在 fix61 的 result 目录下。
set -euo pipefail

cd "$HOME/Lumenarium"
PY="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
root="${SCENEPROOF_RESULTS_ROOT:-a10_reusable_results/paper30}"
scene="${SCENEPROOF_SCENE:-bedroom_01}"
BASELINE="${SCENEPROOF_BASELINE:-v5_sceneproof_collision_partial_commit_certified_paper30_fix61}"
blender="${IMAGINARIUM_BLENDER:-$HOME/Lumenarium/third_party/blender-4.3.2-linux-x64/blender}"
PRECHECK_ONLY="${PRECHECK_ONLY:-0}"

OUT_ROOT="$root/sceneba_audit/v5_sceneproof_settle_on_support_smoke1_fix104/$scene"
mkdir -p "$OUT_ROOT/off" "$OUT_ROOT/on"

echo "=== FIX104 SETTLE ON SUPPORT (A/B, SMOKE1) ==="
echo "Scene:    $scene"
echo "Baseline: $BASELINE"
echo "Start:    $(date)"

# ---- 前置检查：宁可在这里失败，也不要浪费一次完整 S4 ----
echo ""
echo "--- precheck ---"
ok=1
note() { printf '  %-6s %s\n' "$1" "$2"; }

test -s config/config.yaml && note OK "config/config.yaml" \
    || { note MISS "config/config.yaml"; ok=0; }
test -x "$blender" && note OK "blender=$blender" \
    || { note MISS "blender=$blender"; ok=0; }

# S4 用 输入json 的上两级目录 去找配套的 s3 中间产物，所以候选按"配套是否齐全"排序。
source_json=""
for candidate_version in "$BASELINE" "${SCENEPROOF_SOURCE_VERSION:-v4_deepsearch}"; do
    result_root="$root/${scene}_${candidate_version}_result"
    json="$(find "$result_root/S3_pose_inference" -maxdepth 1 -type f \
        -name '*_placement_info.json' -print -quit 2>/dev/null || true)"
    s3geo="$(find "$result_root/S4_layout_refinement" -maxdepth 1 -type f \
        -name '*_placement_info_s3.json' -print -quit 2>/dev/null || true)"
    note INFO "version=$candidate_version"
    note "" "  S3 json:${json:-<none>}"
    note "" "  s3 geometry:    ${s3geo:-<none>}"
    if [ -s "$json" ] && [ -z "$source_json" ]; then
        source_json="$json"
        chosen_version="$candidate_version"
        chosen_s3geo="$s3geo"
    fi
done

if [ -z "$source_json" ]; then
    note MISS "no S3 placement_info.json found for any version"
    ok=0
else
    note PICK "$source_json  (version=$chosen_version)"
    [ -s "$chosen_s3geo" ] || note WARN \
        "该版本目录下没有 *_placement_info_s3.json；若 S4 需要它会在早期失败"
fi

read_cfg='import yaml,sys; print(yaml.safe_load(open("config/config.yaml"))["S4_blender_layout_and_corr"].get("placeable_area_info_folder",""))'
placeable="$("$PY" -c "$read_cfg" 2>/dev/null || true)"
if [ -n "$placeable" ]; then
    note INFO "placeable_area_info_folder=$placeable"
    test -d "$placeable" && note OK "该目录存在" || note WARN "该目录不存在"
else
    note WARN "无法从 config.yaml 读出 placeable_area_info_folder"
fi

[ "$ok" = "1" ] || { echo ""; echo "PRECHECK FAILED，先解决上面标 MISS 的项"; exit 2; }
if [ "$PRECHECK_ONLY" = "1" ]; then
    echo ""
    echo "PRECHECK ONLY，未运行 S4。确认无误后去掉 PRECHECK_ONLY=1 再跑。"
    exit 0
fi

# ---- 跑两次 S4，只切一个开关 ----
PLACEMENT_PATH=""
run_s4() {
    local label="$1" settle="$2" folder="$3"
    echo ""
    echo "--- S4: $label (IMAGINARIUM_SETTLE_ON_SUPPORT=$settle) ---"
    local rc=0
    env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
        IMAGINARIUM_SETTLE_ON_SUPPORT="$settle" \
        IMAGINARIUM_SETTLE_MAX_GAP="${IMAGINARIUM_SETTLE_MAX_GAP:-0.5}" \
        LD_LIBRARY_PATH="$HOME/.venvs/lumenarium-py311/lib:${LD_LIBRARY_PATH:-}" \
        "$blender" --background \
            --python modules/S4_blender_layout_and_corr.py -- \
            --obj_placement_info_json_path "$source_json" \
            --output_folder "$folder" \
            --use_layoutvlm \
            > "$folder/s4.log" 2>&1 < /dev/null || rc=$?
    echo "  exit=$rc  log=$folder/s4.log"
    echo "  [SETTLE] lines: $(grep -c '^\[SETTLE\]' "$folder/s4.log" 2>/dev/null || true)"

    PLACEMENT_PATH="$(find "$folder" -maxdepth 2 -type f \
        -name '*_placement_info_s4.json' -print -quit 2>/dev/null || true)"
    if [ -z "$PLACEMENT_PATH" ]; then
        echo "  没有产出 placement，S4 日志最后 60 行："
        tail -60 "$folder/s4.log" | sed 's/^/    | /'
        echo "  在其他可能位置搜索本次产物："
        find "$root/${scene}_${chosen_version}_result" -newer "$folder/s4.log" \
            -name '*_placement_info_s4.json' 2>/dev/null | sed 's/^/    ? /' || true
        return 1
    fi
    echo "  placement: $PLACEMENT_PATH"
}

run_s4 "settle OFF (等价于改动前)" 0 "$OUT_ROOT/off"
placement_off="$PLACEMENT_PATH"
run_s4 "settle ON" 1 "$OUT_ROOT/on"
placement_on="$PLACEMENT_PATH"

echo ""
echo "--- S4 记录的沉降决策 (settle ON, 前 40 条) ---"
grep '^\[SETTLE\]' "$OUT_ROOT/on/s4.log" | head -40 || true

echo ""
echo "--- 写盘 placement 的 A/B ---"
"$PY" sceneproof_settle_ab_compare.py \
    --scene "$scene" \
    --settle-off "$placement_off" \
    --settle-on "$placement_on" \
    --settle-log "$OUT_ROOT/on/s4.log" \
    --out-report "$OUT_ROOT/settle_ab.json"

render_one() {
    local placement="$1" output="$2"
    env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
        IMAGINARIUM_S4_RENDER_ONLY_PLACEMENT="$placement" \
        IMAGINARIUM_S4_RENDER_ONLY_OUTPUT="$output" \
        IMAGINARIUM_S4_RENDER_ONLY_SAMPLES="${SCENEPROOF_RENDER_SAMPLES:-256}" \
        LD_LIBRARY_PATH="$HOME/.venvs/lumenarium-py311/lib:${LD_LIBRARY_PATH:-}" \
        "$blender" --background \
            --python modules/S4_blender_layout_and_corr.py -- \
            --obj_placement_info_json_path "$source_json" \
            --output_folder /tmp/fix104_render \
            > /tmp/fix104_render.log 2>&1 < /dev/null || true
    if [ -s "$output" ]; then
        echo "  OK $(stat --format=%s "$output") bytes  $output"
    else
        echo "  渲染失败，日志最后 30 行："
        tail -30 /tmp/fix104_render.log | sed 's/^/    | /'
    fi
}

echo ""
echo "--- renders ---"
render_off="$OUT_ROOT/${scene}_settle_off_beauty_256s.png"
render_on="$OUT_ROOT/${scene}_settle_on_beauty_256s.png"
render_one "$placement_off" "$render_off"
render_one "$placement_on"  "$render_on"

echo ""
echo "========================================"
echo "FIX104 COMPLETE  $(date)"
echo "SETTLE_OFF_RENDER=$(readlink -f "$render_off" 2>/dev/null || echo MISSING)"
echo "SETTLE_ON_RENDER=$(readlink -f "$render_on" 2>/dev/null || echo MISSING)"
echo "AB_REPORT=$(readlink -f "$OUT_ROOT/settle_ab.json" 2>/dev/null || echo MISSING)"
echo ""
echo "Reading guide:"
echo "  1. moved 与 largest_drop：位移应当全是向下的小量。出现一个很大的下沉说明某个"
echo "     物体的支撑关系判错了，此时调小 IMAGINARIUM_SETTLE_MAX_GAP 重跑，而不是接受。"
echo "  2. raised 只应来自穿模修正，那条分支未改动；raised 变多说明改动有副作用。"
echo "  3. 日志里的决策数与 placement 反算出的 moved 数应当接近。差得多说明位移在后续"
echo "     步骤（靠墙平移、组内 scale、刚体仿真、位姿序列化）里被覆盖了。"
echo "  4. 两张图下载对比，判据只有一个：杯子、书、摆件是否还浮在支撑面上方。物体身份"
echo "     和尺寸的错误不在本次范围内。"
