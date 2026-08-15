#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Lumenarium"
current="$HOME/Lumenarium/a10_reusable_results/fix116_s1_s4_smoke1"
historical="$HOME/Lumenarium/a10_reusable_results/paper30"
spec="$current/sceneba_audit/cold_start_selector_high_smoke2_fix120_candidates.json"
out="$current/sceneba_audit/cold_start_selector_high_smoke2_fix120.json"
python="$HOME/.venvs/lumenarium-py311/bin/python"
"$python" - "$current" "$historical" "$spec" <<'PY'
import json,sys
from pathlib import Path
current,historical,out=map(Path,sys.argv[1:])
def one(root,scene,geometry,placement,candidate_id):
 g=next((root/f'{scene}_{geometry}_result'/'S4_layout_refinement').glob('*_placement_info_s3.json'))
 p=root/f'{scene}_{placement}_result'/'S4_layout_refinement'/f'{scene}_{placement}_placement_info_s4.json'
 return {'candidate_id':candidate_id,'geometry_path':str(g.resolve()),'placement_path':str(p.resolve())}
data={'candidates':[
 one(current,'bedroom_01','v5_sceneproof_fix43_smooth_fix116_s1_s4_smoke1','v5_sceneproof_collision_partial_commit_certified_fix116_s1_s4_smoke1','current_cold_fix116'),
 one(historical,'bedroom_01','v4_deepsearch','v5_sceneproof_collision_partial_commit_certified_paper30_fix61','historical_frozen_paper30'),
]}
out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(data,indent=2)+'\n')
PY
"$python" sceneproof_cold_start_selector.py --candidates "$spec" --mode high --out "$out"
