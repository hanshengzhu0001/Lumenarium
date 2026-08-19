# 二阶与一阶求解器对比实验计划

面向执行 agent 的实施文档。目标读者是没有参与前期调查的人，因此第 2 节把已确认的事实全部列出，**这些不需要重新发现，也不应被重新推翻，除非拿出与之矛盾的实测数据**。

日期：2026-08-19。前置调查记录见 `.codebuddy/skills/lumenarium-progress/references/progress.md` 的 2026-08-19 两条 changelog。

---

## 1. 要证明什么

Lumenarium 的 S4 布局优化阶段（下称 S4）有三个可选求解器。项目对外报告的 3.513× 加速来自其中的二阶路径，但**从未做过同初值、同目标函数、同场景集下的一阶与二阶对照**。本实验补上这个对照。

需要产出的命题有三条，按重要性排序：

1. **二阶相对一阶的收益是多少，代价是多少。** 收益用物理可实现性指标（下称 macro，定义见 4.3）衡量，代价用单场景墙钟秒数衡量。
2. **关系流形与守卫 Schur 消元的增量贡献是多少。** 即 `scenelm` 到 `v5_scenelm` 的差值（两者都是二阶，区别见 3.1）。
3. **迭代预算在两类求解器上的行为是否不同。** 已知二阶在 2 步后即失效（2.3），一阶是否同样早饱和未知。

**不需要证明**「二阶更快」这个笼统说法。已报告的 3.513× 是二阶路径对 SA-5000（模拟退火 5000 步，旧 S4 主路径）的对比，不是对一阶 Adam 的对比。本实验的对照对象是 Adam，不是 SA。

---

## 2. 已确认的事实

### 2.1 生产路径跑的是二阶，不是代码默认值

`IMAGINARIUM_LAYOUTVLM_SOLVER` 的代码默认值是 `adam`（`modules/S4_blender_layout_and_corr.py:11128`），但**报告基线与线上 API 服务共用的跑法脚本显式覆盖为 `v5_scenelm`**。调用链：

```
sceneproof_api/worker.py:141
  -> scripts/run_sceneproof_frozen_single_job_fix115.sh:193
    -> scripts/run_sceneproof_fix43_inloop_fullstack_smoke5_fix56.sh
```

`fix56` 的 `run_branch` 函数设 `IMAGINARIUM_LAYOUTVLM_SOLVER=v5_scenelm`（:54），并跑两个分支：`run_branch "$control" 0`（:81，不开守卫 Schur）与 `run_branch "$candidate" 1`（:83，开）。两个分支的输出随后被逐物体认证混合成 certified 版本。

**推论**：任何以「默认值是 adam，所以生产在跑一阶」为前提的推理都是错的。

### 2.2 迭代预算是上限，不是步数

三个早停条件任一满足即 `break`（`modules/_s4_layoutvlm_ops.py:6701-6705`）：

1. `in_loop_guarded_accepted` —— 循环内守卫 Schur 试步被接受；
2. `gradient_inf < lm_gradient_tolerance`（默认 1e-5）；
3. `lm_small_reduction_count >= lm_patience`（默认 3）。

两个由此而来的结构性结论，均为 by construction，不需实验验证：

- 守卫 Schur 试步要求先积累 `SCENEPROOF_STABLE_LINEARIZATIONS`（默认 2）次线性化审计（`:6538-6542`），因此**第 1 步不做，第 2 步才做第一次**。预算为 2 时，守卫 Schur 只有一次试机会。
- 条件 3 要求 3 次连续「已接受但改善很小」的步，因此**预算小于 4 时该条件不可能触发**。

### 2.3 二阶在 2 步后即失效（已实测）

场景 `a2c4adeed3cbca9444ba280033bbca2b`（API 提交的客厅图），`v5_scenelm` + 两个守卫 Schur 门全开，只改迭代预算：

| 预算 | executed | converged | accepted / rejected | LM damping λ | 残差能量 |
|---|---|---|---|---|---|
| 2 | 2 | False | 2 / 0 | 0.0025 | 1.12818e-3 |
| 8 | 8 | False | 3 / 5 | 1.28 | 1.08406e-3 |
| 16 | 16 | False | 5 / 11 | 1311 | 1.07891e-3 |

读法：

- **能量饱和**。预算翻 8 倍换来 4.37% 的残差下降；8→16 再翻倍只多 0.48%。
- **λ 上升 5.7 个数量级**。LM 的步是 `(JᵀJ + λ·diag(JᵀJ))δ = −Jᵀr`；λ=1311 时 `δ ≈ −Jᵀr/1311`，求解器已退化为步长极小的一阶梯度下降。
- **拒绝数超过接受数**。16 步档 11 次迭代完全浪费（组装雅可比、跑 PCG、做守卫试步，然后回滚）。
- **`conv=False` 配 energy 不降**：梯度未达容差却无下降方向被接受，这是约束集不可行或停在非光滑驻点的签名，不是欠收敛的签名。

### 2.4 加预算在评测口径下是净亏（已实测）

同一场景，几何快照冻结自同一 S3 源：

| 指标 | 2 步 | 8 步 | 差 |
|---|---|---|---|
| coll | 0.6913 | 0.7050 | +0.0137 |
| support | 0.5350 | 0.5322 | −0.0028 |
| plane | 0.7745 | 0.7745 | 0 |
| semantic | 0.5345 | 0.5101 | **−0.0244** |
| macro | 0.6338 | 0.6304 | −0.0034 |
| critical | 0.5345 | 0.5101 | −0.0244 |

两个已核对的恒等式，是理解这张表的前提：

- `macro = (coll + support + plane + semantic) / 4`，**四项等权**；
- `critical = min(四项)`，而三档预算下最小项都是 semantic。

**机制**：目标函数的权重是 `collision=1.0`、`contact=2.0`、`semantic=0.5`（`modules/_s4_layoutvlm_ops.py:3614-3619` 的函数默认参数，全仓库无环境变量覆盖，S4 调用时不传）。语义在目标函数里只值碰撞的一半，在评测里与碰撞等权。因此多跑迭代等于沿「语义只值半分」的方向多走：`semantic 损失 0.0244 ÷ coll 收益 0.0137 ≈ 1.8`，与权重比 `1.0/0.5 = 2` 一致。

**这是一处口径错配，不是求解器缺陷。两类求解器共用同一目标函数，因此该偏差对双方相同，不影响本实验的可比性——但报告时必须声明它，否则 macro 的绝对值会被误读为求解器质量。**

### 2.5 支撑（悬空）与迭代预算无关（已实测）

`support` 阈值通过率在 2 / 8 / 16 三档预算下完全相同：`37.5% / 37.5% / 37.5%`。

结构性原因：最终竖直位置不由优化器决定。刚体仿真在 S4 优化之后运行，仿真结束后 `process_z` 按支撑树重新赋值 z（`modules/S4_blender_layout_and_corr.py:12087`），优化器的 z 输出被覆盖。

**推论**：本实验不应把 support 当作区分求解器的指标，也不应期望任何求解器改动改善它。它应作为「未被改动」的对照项报告。

---

## 3. 三个求解器的准确语义

### 3.1 定义

| solver 取值 | 是什么 | 关系流形（自由度裁剪） | 守卫 Schur | 诊断落盘 |
|---|---|---|---|---|
| `adam` | 一阶。`torch.optim.Adam` 作用于 yaw 与平移（`_s4_layoutvlm_ops.py:3874-3875`），配合支撑接触与平面的硬投影 | 无 | 不可用 | **无** |
| `scenelm` | 二阶。matrix-free Levenberg–Marquardt，legacy yaw-only 坐标图 | 无 | 不可用 | 有，`schema_version = scenelm_matrix_free_lm_v1` |
| `v5_scenelm` | 二阶。LM + 关系流形 + 可选守卫 Schur | 有（`:4026-4125` 仅在该分支初始化） | 可开 | 有，`schema_version = scenelm_relation_manifold_v1` |

「关系流形」指按物体与场景的关系决定其自由度个数：被支撑或贴平面的物体 2 个平移自由度，自由物体 3 个，另加 1 个 yaw。

### 3.2 守卫 Schur 是 fail-closed 依赖链

三层，每层默认关闭，且依赖不满足时在函数入口 `raise ValueError`，**不静默降级**：

```
IMAGINARIUM_SCENEPROOF_IN_LOOP_GUARDED_SCHUR=1
  要求 IMAGINARIUM_SCENEPROOF_FULL_SO3_GUARDED_SCHUR=1   (_s4_layoutvlm_ops.py:3717-3723)
    要求 IMAGINARIUM_SCENEPROOF_SHADOW_JACOBIAN_OWNERSHIP=1 (:3702-3706)
      要求 solver == "v5_scenelm"                          (:3690-3694)
```

因此 `adam + Schur` 与 `scenelm + Schur` 都是非法组合，会直接报错退出。这是刻意设计，不要试图绕过。

### 3.3 诊断字段

`v5_scenelm` 与 `scenelm` 会把求解器诊断写进 `<scene>_<version>_placement_info_s4.json` 的 `scenelm_solver` 字段（`S4_blender_layout_and_corr.py:12488-12535`）：`maximum_iterations`、`executed_iterations`、`accepted_steps`、`rejected_steps`、`final_damping`、`final_residual_energy`、`converged`、`pcg_iterations`、`gradient_tolerance`。

**`adam` 不写这个字段**（`:12489` 的集合只含 `scenelm` 与 `v5_scenelm`）。因此一阶侧**拿不到残差能量与收敛状态**，只能用 macro 与墙钟。这是本实验最重要的一处不对称，见 4.4 的处理方式。

---

## 4. 实验设计

### 4.1 场景集

第一轮用单场景 `a2c4adeed3cbca9444ba280033bbca2b`，源产物已在 `api_jobs/da0b8807e1c748a5805b9894a9436fc7/results` 下，且 2 / 8 / 16 三档二阶结果已存在，可直接复用作为对照。

第二轮扩到 Smoke5（`bedroom_01`、`livingroom_10`、`casino_01`、`official_01`、`streelitter_01`），源在 `a10_reusable_results/paper30`。

**只有第一轮出现可解释的差异，才做第二轮。** 单场景的作用是确认差异的方向与量级，不是给最终数字。

### 4.2 对照臂

| 臂 | solver | 迭代预算 | Schur | 目的 |
|---|---|---|---|---|
| A1 | `adam` | 100 | — | 一阶，代码默认预算 |
| A2 | `adam` | 400 | — | 一阶，Paper30 runner 默认预算（`run_paper30_v4_s4_only_dual_gpu.sh:13`） |
| B1 | `scenelm` | 2 | 关 | 二阶，无关系流形，与生产同预算 |
| B2 | `scenelm` | 16 | 关 | 二阶，无关系流形，放开预算 |
| C1 | `v5_scenelm` | 2 | 开 | **生产配置**，已有数据可复用 |
| C2 | `v5_scenelm` | 16 | 开 | 已有数据可复用 |

C1 与 C2 已存在（版本名 `v5_sceneproof_collision_partial_commit_api` 与 `v5_lm_budget_16`），不必重跑。**需要新跑的只有 A1、A2、B1、B2 四臂。**

### 4.3 指标

主指标取自 `eval_physical_realizability.py`：

- `macro` = coll、support、plane、semantic 四项等权平均，为总览；
- `critical` = 四项最小值，即最弱一环；
- 各项阈值通过率，用于把分数变化翻译成「几个物体」；
- 单场景墙钟秒数，在每臂外层用 `date +%s` 测。

辅助指标（仅二阶臂可得）：`executed_iterations`、`converged`、`accepted_steps` / `rejected_steps`、`final_damping`、`final_residual_energy`。

### 4.4 公平性口径：三种，都要报

「同步数」不是公平对比，因为一阶的一步与二阶的一步不是同一件事：Adam 的一步是沿梯度走一小步；LM 的一步是组装雅可比（几百条残差 × 约 120 个自由度）、用 12 次预条件共轭梯度解一次正规方程、再全场同时更新一次位姿。

因此对比结果必须同时按三种口径报告，**不得只报其中一种**：

1. **同预算**：A1(100) 对 B2(16) 无意义，改为报「各自默认配置」的绝对表现，即 A2(400) 对 C1(2)。这是「两条工程路径各自最优实践」的对比。
2. **同墙钟**：找出与 C1 墙钟最接近的一阶预算，报该点的 macro 差。若一阶在同等墙钟下达不到二阶的 macro，这是二阶收益的直接证据。
3. **同收敛程度**：仅在二阶臂之间可用（B 与 C 有 `converged` 字段）。一阶臂无此字段，需声明「一阶的收敛状态不可观测」，不得用「一阶跑满预算即视为收敛」代替。

---

## 5. 命令模板

以下命令在 A10 上执行（`ssh -p36000 hansenzhu@ieg.mnet2.com` 后 `bf ssh 172.16.0.9`）。该机的 git remote 是内网仓库，**GitHub 上的新脚本拉不到这台机器，不要尝试 `git pull`**；所有配置通过环境变量传入现有 runner。

### 5.1 一阶臂（A1、A2）

```bash
cd "$HOME/Lumenarium"
PY="$HOME/.venvs/lumenarium-py311/bin/python"
ROOT="api_jobs/da0b8807e1c748a5805b9894a9436fc7/results"
SCENE=a2c4adeed3cbca9444ba280033bbca2b
MAN=/tmp/solver_ab_manifest.txt
printf '%s\n' "$SCENE" > "$MAN"

for B in 100 400; do
  V="v5_adam_budget_$B"
  echo "===== START $V $(date) ====="
  S=$(date +%s)
  env \
    IMAGINARIUM_PAPER30_MANIFEST="$MAN" \
    IMAGINARIUM_PAPER30_RESULTS_ROOT="$ROOT" \
    IMAGINARIUM_S4_SOURCE_VERSION=v4_deepsearch \
    IMAGINARIUM_S4_SOURCE_STAGE=S3_pose_inference \
    IMAGINARIUM_S4_SOURCE_PATTERN='*_placement_info.json' \
    IMAGINARIUM_S4_TARGET_VERSION="$V" \
    IMAGINARIUM_S4_ENGINE=layoutvlm \
    IMAGINARIUM_LAYOUTVLM_STAGE=full \
    IMAGINARIUM_LAYOUTVLM_SOLVER=adam \
    IMAGINARIUM_LAYOUTVLM_ITERATIONS="$B" \
    IMAGINARIUM_LAYOUTVLM_ACTIVE_SET_ROUTER=0 \
    IMAGINARIUM_GPU_FREE_FLOOR_MB="${IMAGINARIUM_GPU_FREE_FLOOR_MB:-16000}" \
    IMAGINARIUM_S4_SCENE_TIMEOUT=7200 \
    IMAGINARIUM_S4_WORKER_LOG_ROOT="logs/$V" \
    bash scripts/run_paper30_v4_s4_only_dual_gpu.sh
  echo "===== FINISH $V seconds=$(( $(date +%s) - S )) $(date) ====="
done
```

一阶臂**不要**设任何 `IMAGINARIUM_SCENEPROOF_*_GUARDED_SCHUR` 或 `SHADOW_JACOBIAN_OWNERSHIP`，否则按 3.2 会直接报错。

### 5.2 二阶无关系流形臂（B1、B2）

与 5.1 相同，只改两处：`IMAGINARIUM_LAYOUTVLM_SOLVER=scenelm`，预算循环改为 `for B in 2 16`，版本名改为 `v5_scenelm_legacy_budget_$B`。同样不要设 Schur 相关变量。

### 5.3 评测

```bash
mkdir -p /tmp/solver_ab
"$PY" eval_physical_realizability.py \
  --saved-results "$ROOT" \
  --scenes "$MAN" \
  --versions "v5_adam_budget_100,v5_adam_budget_400,v5_scenelm_legacy_budget_2,v5_scenelm_legacy_budget_16,v5_sceneproof_collision_partial_commit_api,v5_lm_budget_16" \
  --geometry-version v5_sceneproof_fix43_smooth_api \
  --metrics-out /tmp/solver_ab/physical.json \
  --scene-csv /tmp/solver_ab/physical_scenes.csv \
  --object-csv /tmp/solver_ab/physical_objects.csv \
  --report-out /tmp/solver_ab/physical.ascii
cat /tmp/solver_ab/physical.ascii
```

### 5.4 读二阶诊断

```bash
"$PY" - <<'PY'
import json, glob, os
root = os.path.expanduser('~/Lumenarium/api_jobs/da0b8807e1c748a5805b9894a9436fc7/results')
for p in sorted(glob.glob(root + '/*/S4_layout_refinement/*_placement_info_s4.json')):
    version = p.split('/results/')[1].split('/')[0]
    d = json.load(open(p, encoding='utf-8-sig')).get('scenelm_solver')
    if not d:
        print('%-58s (first-order: no solver record)' % version[-58:])
        continue
    print('%-58s max=%-4s exec=%-4s conv=%-5s acc=%-3s rej=%-3s damp=%.4g energy=%.6g' % (
        version[-58:], d['maximum_iterations'], d['executed_iterations'],
        d['converged'], d['accepted_steps'], d['rejected_steps'],
        d['final_damping'], d['final_residual_energy']))
PY
```

---

## 6. 已知陷阱

以下每一条都在前期调查中实际发生过，或从代码确认会发生。

1. **`--geometry-version` 必须指向跑过 S4 的版本。** 评测要的几何快照是 `<scene>_<version>_result/S4_layout_refinement/*_placement_info_s3.json`（`eval_physical_realizability.py:299-311`），而 `v4_deepsearch` 只跑到 S3，没有该目录。指错的表现是 `Failures: 1` 且所有指标为 `n/a`、`scenes=0`——**不是报错退出，是静默给出空表**。用 `v5_sceneproof_fix43_smooth_api`。
2. **不要用 certified 或 best 版本做对照。** `v5_sceneproof_collision_partial_commit_certified_api` 是 control 与 guarded 两分支逐物体混合后的产物，`v5_sceneproof_best_exhaustive_support_api` 还叠了在线支撑修复。与单分支比较会把认证与修复的效果算进求解器。**生产配置的单分支代表是 `v5_sceneproof_collision_partial_commit_api`。**
3. **一阶侧没有残差能量。** 见 3.3。不要用 macro 反推能量，也不要为了对齐而修改 `:12489` 的集合把 `adam` 加进去——`lm_*` 字段在一阶路径下不存在，会 `KeyError`。正确做法是声明这处不可观测。
4. **目标函数权重与评测口径不一致。** 见 2.4。对双方相同，不影响可比性，但报告必须声明，否则 macro 的绝对值会被误读。
5. **迭代预算不是步数。** 见 2.2。任何以「设了 N 步就跑了 N 步」为前提的分析都是错的；必须读 `executed_iterations`。
6. **support 不会变。** 见 2.5。若某臂的 support 出现明显变化，**先怀疑是几何快照选错或版本串错，而不是求解器改善了支撑**。
7. **磁盘。** 每臂每场景约几十 MB。四臂跑完后 `du -sh "$ROOT"` 检查一次。
8. **中断过的运行留下半成品。** 若某臂被 Ctrl-C，`placement_info_s4.json` 可能已写而渲染缺失，评测会报该版本失败。删掉该 `*_result` 目录重跑，不要把半成品带进评测。

---

## 7. 交付物

1. 一张主表：六臂 × {macro, critical, coll, support, plane, semantic, 各项通过率, 墙钟秒}。
2. 一张诊断表：四个二阶臂的 `executed_iterations` / `converged` / `accepted` / `rejected` / `damping` / `energy`；两个一阶臂标注为不可观测。
3. 按 4.4 三种口径各写一段结论，每段一句话说清「在这个口径下，二阶相对一阶是赚还是亏，赚多少」。
4. 关系流形与守卫 Schur 的增量：`v5_scenelm(2)` 减 `scenelm(2)` 的各项差值，这是论文里 ablation 的一行。
5. 若出现与第 2 节任一条矛盾的结果，**单独写一节说明矛盾点与证据**，不要默默按新结果覆盖旧结论。

---

## 8. 明确不要做的事

- 不要为了「让一阶看起来公平」而把一阶的预算加到收敛为止再比 macro，却不报墙钟。算力代价是结论的一半。
- 不要改 `modules/` 下任何 pipeline 代码。本实验全部通过环境变量完成。唯一例外是若要扫目标函数权重（`semantic_weight` 等），那属于另一个实验，需要单独立项，并且应先把权重做成环境变量而不是就地改默认值。
- 不要在这个实验里试图改善悬空。见 2.5，它与求解器无关。
- 不要用 Paper30 全量开跑。先单场景，出现可解释差异再扩 Smoke5，最后才是 Paper30。
- 不要改动 `api_cache/` 下任何内容。那是演示用的冻结缓存，与本实验无关。
