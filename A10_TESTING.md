# A10 testing through MNET2

This workflow connects from Windows PowerShell to the USD Search A10 VM without
storing the MNET2 Pin+Token.

## Connection command

Git for Windows OpenSSH is used because the built-in Windows OpenSSH client and
the MNET2 gateway do not provide a reusable `ControlMaster` session. Public-key
authentication is disabled so that loaded SSH-agent keys do not exhaust the
gateway's authentication attempts. `SSH_ASKPASS` displays the Pin+Token prompt
without storing the credential.

```powershell
$env:SSH_ASKPASS = "C:\Program Files\Git\mingw64\bin\git-askpass.exe"
$env:SSH_ASKPASS_REQUIRE = "force"
$env:DISPLAY = "required"

& "C:\Program Files\Git\usr\bin\ssh.exe" `
  -tt `
  -o IdentityAgent=none `
  -o PubkeyAuthentication=no `
  -o PreferredAuthentications=keyboard-interactive `
  -o NumberOfPasswordPrompts=1 `
  -p 36000 `
  hansenzhu@ieg.mnet2.com `
  "bf ssh 172.16.0.9" 2>&1 |
  Tee-Object -FilePath "$env:TEMP\lumenarium_a10_probe.log"
```

Enter the current MNET2 Pin+Token only in the Git for Windows prompt. Never add
it to this repository, a command, or a log file. Press Enter at the interactive
`bf` selector to accept the highlighted `user00 / 172.16.0.9` target.

Do not pipe `$commands` into `ssh`; that replaces keyboard input with the pipe.
After the A10 shell prompt appears, paste the desired commands interactively.

## File transfer

Use a WeTERM session (or another terminal with ZMODEM support) to connect to the
A10 VM as `dev`. PowerShell/OpenSSH alone does not implement ZMODEM.

On the A10 VM:

```bash
cd /data/home/dev
rz -be
```

Choose the local archive in the file picker. Verify the transfer before
extracting it:

```bash
sha256sum lumenarium_v4_deepsearch_source.tar.gz
```

For the current v4-deepsearch source bundle, the expected SHA-256 is:

```text
d6420ccc624da9d3b97b2ea1155daa553f05f41b2d38c51c8f736f2467bedfe4
```

Do not use `bf scp`, `scp_out`, or iFt on this VM unless the corresponding
account/tool has been provisioned. They were unavailable in the validated
environment.

## Pipeline checks

Once on the A10 VM, use the existing v1/v3 environment and configuration. For
v4-deepsearch, verify connectivity first, then run the pipeline:

```bash
curl --fail --show-error \
  "https://miller-unshapeable-melany.ngrok-free.dev/search?description=house&limit=2"

python run_imaginarium_I2Layout_v4_deepsearch.py \
  demo/custom_scene3.png --clean --debug
```

The S2 log must contain `Running retrieval with Omniverse DeepSearch...`. Check
that `retrieval_results.json` and `retrieval_results_final.json` are identical,
and that each non-background object's top candidate satisfies
`data[obj_name][0][0]` being an asset-name string.

## v4 LayoutVLM incremental validation

The v4 implementation is advanced in gated stages.  Stage `reproject` replaces
the SA call with a differentiable yaw/translation representation initialized
from the deterministic S4-S2 matrices.  It intentionally applies zero pose
delta: this validates coordinate conventions, gradient flow, Blender wiring,
and output compatibility before any loss is allowed to move objects.

Never extract a v4 bundle directly over the repository.  Preserve the previous
source and unpack into a staging directory first:

```bash
cd "$HOME"
STAGE="/tmp/lumenarium_v4_stage"
BACKUP="$HOME/Lumenarium/a10_reusable_results/source_backups/v4_layoutvlm"
mkdir -p "$STAGE" "$BACKUP"
tar -xzf lumenarium_v4_stage1_reproject.tar.gz -C "$STAGE"
cp "$HOME/Lumenarium/modules/S4_blender_layout_and_corr.py" "$BACKUP/"
cp "$HOME/Lumenarium/modules/layout.py" "$BACKUP/"
diff -u "$HOME/Lumenarium/modules/layout.py" "$STAGE/modules/layout.py" || true
```

Only after reviewing the diff, install the two gated shared files and add the
new v4-only files:

```bash
cd "$HOME/Lumenarium"
cp "$STAGE/modules/S4_blender_layout_and_corr.py" modules/
cp "$STAGE/modules/layout.py" modules/
cp "$STAGE/modules/_s4_layoutvlm_ops.py" modules/
cp "$STAGE/run_imaginarium_I2Layout_v4.py" .
cp "$STAGE/tests/test_s4_layoutvlm_ops.py" tests/
cp "$STAGE/tests/test_s4_layoutvlm_wiring.py" tests/
```

No result directory is overwritten. v1/v3/v4-deepsearch do not set the
LayoutVLM gate and therefore continue through the legacy SA branch.

Then run:

```bash
cd "$HOME/Lumenarium"
PY="$HOME/.venvs/lumenarium-py311/bin/python"
"$PY" -m unittest -v \
  tests.test_s4_layoutvlm_ops \
  tests.test_s4_layoutvlm_wiring
```

Then reuse an existing v4-deepsearch S3 result so the check makes no Gemini or
DeepSearch request:

```bash
cd "$HOME/Lumenarium"
SRC="saved_results_a10_v4_deepsearch/demo_0_result/S3_pose_inference/demo_0_placement_info.json"
OUT="a10_reusable_results/v4_stage1_reproject/demo_0"
test -f "$SRC" && mkdir -p "$OUT"
env CUDA_VISIBLE_DEVICES=1 \
  LD_LIBRARY_PATH="$HOME/.venvs/lumenarium-py311/lib:${LD_LIBRARY_PATH:-}" \
  "$HOME/Lumenarium/third_party/blender-4.3.2-linux-x64/blender" \
  --background \
  --python modules/S4_blender_layout_and_corr.py \
  -- \
  --obj_placement_info_json_path "$SRC" \
  --output_folder "$OUT" \
  --use_layoutvlm \
  --layoutvlm_stage reproject \
  2>&1 | tee logs/test_v4_stage1_reproject.log
```

The test passes only if all autograd tests are `ok`, the Blender log contains
`[LayoutVLM] Warm-start reprojection passed` with `max_abs_error=0`, and the
new output contains both `*_placement_info_s4.json` and `*_render_simu.png`.
The legacy v1/v3/v4-deepsearch paths remain on the SA branch because they do
not set `IMAGINARIUM_USE_LAYOUTVLM=1`.

For the next incremental stage, replace `reproject` with `collision` and set
`IMAGINARIUM_LAYOUTVLM_ITERATIONS=100`. The log must show a non-increasing
collision loss and must keep all Z translations at their S4-S2 warm starts.

For the fixed-plane stage, reuse the same frozen S3 input and run:

```bash
cd "$HOME/Lumenarium"
SRC="saved_results_a10_v4_deepsearch/demo_0_result/S3_pose_inference/demo_0_placement_info.json"
OUT="a10_reusable_results/v4_stage4_wall/demo_0"
mkdir -p "$OUT" logs
nohup env CUDA_VISIBLE_DEVICES=1 \
  IMAGINARIUM_LAYOUTVLM_ITERATIONS=200 \
  IMAGINARIUM_LAYOUTVLM_MAX_CONTACT_GAP=0.5 \
  LD_LIBRARY_PATH="$HOME/.venvs/lumenarium-py311/lib:${LD_LIBRARY_PATH:-}" \
  "$HOME/Lumenarium/third_party/blender-4.3.2-linux-x64/blender" \
  --background \
  --python modules/S4_blender_layout_and_corr.py \
  -- \
  --obj_placement_info_json_path "$SRC" \
  --output_folder "$OUT" \
  --use_layoutvlm \
  --layoutvlm_stage wall \
  > logs/test_v4_stage4_wall.log 2>&1 < /dev/null &
echo "WALL_PID=$!"
```

The test passes only when the log reports non-zero wall/ceiling constraint
counts as applicable, `projected_max_contact_gap` and
`projected_max_plane_gap` are near zero, the orientation loss is finite, and
Blender writes both the S4 JSON and final render without a traceback.
