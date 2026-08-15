#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
PY="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"

"$PY" -m py_compile \
  modules/_sceneproof_program_ir.py \
  modules/_sceneproof_compile.py \
  modules/_sceneproof_execute.py \
  modules/_sceneproof_certificate.py \
  modules/_sceneproof_block_system.py \
  modules/_sceneproof_residual_bridge.py \
  modules/_sceneproof_factor_binding.py \
  modules/_s4_layoutvlm_ops.py \
  modules/_s4_scenelm_relational.py \
  modules/S4_blender_layout_and_corr.py \
  sceneproof_compile_audit.py \
  sceneproof_factor_binding_audit.py \
  sceneproof_full_so3_schur_audit.py \
  sceneproof_jacobian_ownership_audit.py \
  sceneproof_residual_switch_audit.py

env \
  LD_LIBRARY_PATH="$HOME/.venvs/lumenarium-py311/lib:${LD_LIBRARY_PATH:-}" \
  "$PY" -m unittest \
    tests.test_sceneproof_program_ir \
    tests.test_sceneproof_factor_binding \
    tests.test_sceneproof_solver_core \
    tests.test_s4_layoutvlm_ops \
    tests.test_s4_layoutvlm_wiring \
    -v

echo "SCENEPROOF_MILESTONE_A_PREFLIGHT=PASS"
