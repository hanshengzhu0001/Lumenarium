#!/usr/bin/env bash
set -euo pipefail

# Iteration-budget frontier for the second-order SceneLM solver.
#
# Why this exists: the frozen Fix61 baseline and the live API both run
# IMAGINARIUM_LAYOUTVLM_ITERATIONS=2.  Two is the smallest value the
# stable-linearization gate accepts -- _s4_layoutvlm_ops.py:3707 raises when
# iterations < SCENEPROOF_STABLE_LINEARIZATIONS -- so the record shows it was
# chosen to clear a gate, not to reach convergence.  Residual floating and
# interpenetration may therefore be an unconverged solve rather than an
# upstream identity or size error.  This measures which one it is.
#
# The budget is an upper bound, not a step count: the solver stops early on a
# small gradient, on lm_patience consecutive small reductions, or as soon as an
# in-loop guarded Schur trial is accepted (_s4_layoutvlm_ops.py:6701-6705).
# The unguarded branch cannot reach the patience criterion within two
# iterations, so it is under-converged by construction; the guarded branch may
# already stop at its first accepted trial.  Read executed_iterations and
# converged from the emitted scenelm_solver record before reading the metrics.
#
# Reading guide:
#   1. every branch differs only in the iteration budget.  Solver, both Schur
#      gates, warm start, plane anchoring and all tolerances are copied
#      verbatim from the production branch of
#      run_sceneproof_fix43_inloop_fullstack_smoke5_fix56.sh, so a difference
#      in the output can only come from the budget
#   2. each budget writes its own S4 target version, so no run overwrites
#      another and Fix61 itself is never touched
#   3. wall-clock per budget is printed and appended to the audit directory,
#      because the cost of a larger budget is half of the question
#   4. one evaluation call covers all budgets, so collision and support come
#      from identical tolerances
#   5. the ladder stops at 32: the patience criterion needs one substantive
#      step plus lm_patience small ones, and Gauss-Newton on a few hundred
#      residuals typically reaches the gradient tolerance well inside that
#   6. rejected alternative: sweeping Paper30 first.  One scene shows the shape
#      of the curve at 1/30 of the cost; only a visible knee justifies the full
#      sweep

cd "$HOME/Lumenarium"

python="${IMAGINARIUM_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
manifest="${SCENELM_BUDGET_MANIFEST:-/tmp/scenelm_iteration_budget_smoke1.txt}"
results_root="${SCENELM_BUDGET_RESULTS_ROOT:-a10_reusable_results/paper30}"
log_root="${SCENELM_BUDGET_LOG_ROOT:-logs/scenelm_iteration_budget_frontier}"
audit="${SCENELM_BUDGET_AUDIT_ROOT:-$results_root/sceneba_audit/scenelm_iteration_budget}"
source_version="${SCENELM_BUDGET_SOURCE_VERSION:-v4_deepsearch}"
budgets="${SCENELM_BUDGET_LIST:-2 8 16 32}"
runner="scripts/run_paper30_v4_s4_only_dual_gpu.sh"

test -f "$manifest" || printf '%s\n' "${SCENELM_BUDGET_SCENE:-livingroom_10}" > "$manifest"
mkdir -p "$log_root" "$audit"
timing="$audit/wall_clock.tsv"
test -f "$timing" || printf 'budget\tversion\tseconds\tstatus\n' > "$timing"

versions=""
for budget in $budgets; do
  version="v5_lm_budget_${budget}"
  versions="${versions:+$versions,}$version"
  echo "===== START budget=$budget version=$version $(date) ====="
  start="$(date +%s)"
  set +e
  env \
    IMAGINARIUM_PAPER30_MANIFEST="$manifest" \
    IMAGINARIUM_PAPER30_RESULTS_ROOT="$results_root" \
    IMAGINARIUM_S4_SOURCE_VERSION="$source_version" \
    IMAGINARIUM_S4_SOURCE_STAGE=S3_pose_inference \
    IMAGINARIUM_S4_SOURCE_PATTERN='*_placement_info.json' \
    IMAGINARIUM_S4_TARGET_VERSION="$version" \
    IMAGINARIUM_S4_ENGINE=layoutvlm \
    IMAGINARIUM_LAYOUTVLM_STAGE=full \
    IMAGINARIUM_LAYOUTVLM_SOLVER=v5_scenelm \
    IMAGINARIUM_LAYOUTVLM_ITERATIONS="$budget" \
    IMAGINARIUM_LAYOUTVLM_ACTIVE_SET_ROUTER=0 \
    IMAGINARIUM_SCENEPROOF_PROGRAM_IR=1 \
    IMAGINARIUM_SCENEPROOF_REQUIRE_FACTOR_PARITY=1 \
    IMAGINARIUM_SCENEPROOF_REQUIRE_BINDING_AUDIT=1 \
    IMAGINARIUM_SCENEPROOF_SHADOW_JACOBIAN_OWNERSHIP=1 \
    IMAGINARIUM_SCENEPROOF_STABLE_LINEARIZATIONS=2 \
    IMAGINARIUM_SCENEPROOF_FULL_SO3_GUARDED_SCHUR=1 \
    IMAGINARIUM_SCENEPROOF_IN_LOOP_GUARDED_SCHUR=1 \
    IMAGINARIUM_SCENEPROOF_WARM_START_ANCHORED_PLANE_TRANSLATION=1 \
    IMAGINARIUM_SCENEPROOF_PLANE_ANCHOR_NORMAL_LIMIT_M=0.02 \
    IMAGINARIUM_SCENEPROOF_PLANE_PROXY_ABSTAIN_GAP_M=0.02 \
    IMAGINARIUM_SCENEPROOF_PLANE_ATTACH_REQUIRES_WITNESS=1 \
    IMAGINARIUM_SCENEPROOF_MATERIALIZED_WARM_START=1 \
    IMAGINARIUM_SCENEPROOF_PLANE_SIBLING_TANGENT_PROJECTION=1 \
    IMAGINARIUM_SCENEPROOF_PLANE_SIBLING_MAX_SHIFT_M=0.35 \
    IMAGINARIUM_SCENEPROOF_PLANE_COMPONENT_IMAGE_GAUGE=0 \
    IMAGINARIUM_SCENEPROOF_MESH_VISIBILITY_AUDIT=0 \
    IMAGINARIUM_SCENELM_KINEMATIC_BACKSUB=0 \
    IMAGINARIUM_GPU_FREE_FLOOR_MB="${IMAGINARIUM_GPU_FREE_FLOOR_MB:-16000}" \
    IMAGINARIUM_S4_SCENE_TIMEOUT="${IMAGINARIUM_S4_SCENE_TIMEOUT:-7200}" \
    IMAGINARIUM_S4_WORKER_LOG_ROOT="$log_root/$version" \
    bash "$runner"
  status=$?
  set -e
  elapsed="$(( $(date +%s) - start ))"
  printf '%s\t%s\t%s\t%s\n' "$budget" "$version" "$elapsed" "$status" >> "$timing"
  echo "===== FINISH budget=$budget version=$version seconds=$elapsed rc=$status $(date) ====="
done

echo "===== EVALUATE $versions $(date) ====="
runtime_args=()
for budget in $budgets; do
  runtime_args+=(--runtime-log "v5_lm_budget_${budget}=$log_root/v5_lm_budget_${budget}")
done

"$python" eval_physical_realizability.py \
  --saved-results "$results_root" \
  --scenes "$manifest" \
  --versions "$versions" \
  --geometry-version "$source_version" \
  --baseline-version "v5_lm_budget_$(echo "$budgets" | awk '{print $1}')" \
  "${runtime_args[@]}" \
  --metrics-out "$audit/physical.json" \
  --scene-csv "$audit/physical_scenes.csv" \
  --object-csv "$audit/physical_objects.csv" \
  --report-out "$audit/physical.ascii"

echo "SCENELM_ITERATION_BUDGET_FRONTIER_COMPLETE $(date)"
echo "wall clock: $timing"
echo "metrics:    $audit/physical.json"
echo "report:     $audit/physical.ascii"
