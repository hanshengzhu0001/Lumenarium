#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Lumenarium"
root="$HOME/Lumenarium/a10_reusable_results/fix116_s1_s4_smoke1"
version=v5_sceneproof_vertical_support_com_projection_fix117_1
scene=bedroom_01
audit="$root/sceneba_audit/$version"
freeze="$audit/FROZEN_VISUAL_FIX119.json"
archive="$HOME/sceneproof_fix117_visual_incumbent_fix119.tar.gz"
"$HOME/.venvs/lumenarium-py311/bin/python" - "$root" "$scene" "$version" "$freeze" <<'PY'
import hashlib,json,sys
from pathlib import Path
root,scene,version,out=Path(sys.argv[1]),sys.argv[2],sys.argv[3],Path(sys.argv[4])
files=[]
for path in sorted((root/f'{scene}_{version}_result').rglob('*')):
    if path.is_file():
        files.append({'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size})
transaction=root/'sceneba_audit'/version/'transactions'/f'{scene}.json'
if transaction.is_file(): files.append({'path':str(transaction.resolve()),'sha256':hashlib.sha256(transaction.read_bytes()).hexdigest(),'bytes':transaction.stat().st_size})
record={'schema_version':'sceneproof_frozen_visual_incumbent_v1','version':version,'scene':scene,'immutable_by_manifest':True,'files':files}
out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(record,indent=2)+'\n')
print(f'FROZEN_FILES={len(files)} MANIFEST={out.resolve()}')
PY
tar -czf "$archive" -C "$root" \
  "${scene}_${version}_result" "sceneba_audit/$version"
chmod a-w "$archive"
echo "FIX117_FROZEN_MANIFEST=$freeze"
echo "FIX117_FROZEN_ARCHIVE=$archive"
