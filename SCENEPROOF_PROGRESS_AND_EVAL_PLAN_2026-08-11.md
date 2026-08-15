# SceneProof 进展与当前评测方案（2026-08-11）

## 结论先行

8 月 10 日至 8 月 11 日没有形成新的、经过 Paper30 验证的质量提升。两天里产生了不少诊断代码和修补包，但研究线一度同时混入了 support metric 缺陷、asset/scaling 归因、沉降规则、Bullet 参数、gate 豁免和相机渲染问题，导致“fix 编号增加很多，可信结论没有同步增加”。

当前应停止扩展通用 optimizer、全场 COM 和 bulk settle。唯一在执行的实验是：以 Fix61 为冻结 baseline，只对四个直观刚体做 process-isolated full-SO(3) 局部掉落，并在 strict/relaxed Fix84 component gates 后做 scoped commit。其余 26 个场景直接复用 Fix61。

## 保留的可信节点

| 节点 | 作用 | 当前判断 |
|---|---|---|
| Fix27 | source-S3 锁定相机基础设施 | 历史基础设施节点；正式比较仍必须禁止 target-framed camera |
| Fix61 | Paper30 collision partial-commit aggregate baseline | 当前唯一冻结的 Paper30 aggregate baseline |
| Fix76–Fix81 | pose serialization、true-mesh COM responsibility、process-isolated local settle oracle | 证明了局部 full-SO(3) 掉落、恢复 incumbent 和 true-mesh 复测链条可用 |
| Fix84 | single-object 13-gate component evaluation + witnessed proxy exemption | 可作为局部候选 gate；strict 和 relaxed 必须同时报告 |
| Fix84e | rigid-only adaptive re-evaluation | 当前运行中，尚无结果，不得提前声称提升 |

## 2026-08-10：实际完成了什么

### 1. 做了较多责任归因和测量诊断

- Fix92–93：检查 collision measurement 和 normalization。
- Fix94 系列：检查 overhang、COM density 和 consistency。
- Fix96：scene defect screen。
- Fix97–100：追踪 scaling chain attribution。
- Fix101–103：检查 asset-library join 和资产替换链。

这些工作主要解释“为什么指标或几何看起来异常”，没有产生经过完整 gate 的新 pose operator，也没有带来新的 Paper30 headline improvement。它们是诊断资产，不应在周报中写成算法收益。

### 2. 修改了沉降路径，但没有得到干净消融

四个主要文件发生变化：

1. `modules/_s4_settle.py`
   - 把 `process_other_objects` 中的 z 沉降决策抽成可单测纯函数。
   - 统一悬空/穿模方向，并为大于 0.5 m 的可疑 support gap 保留旧行为。
   - 这是代码结构和确定性改善，不等于质量已经提高。

2. `modules/S4_blender_layout_and_corr.py`
   - 初始化 `optimization_history = None`：合理的独立 bug fix。
   - 为 local settle 建 passive BVH cache：合理的性能优化，尤其减少大场景重复 BVH 构建。
   - simulation 后再次运行 support z 对齐：属于算法变化，尚未被独立消融。
   - render-only target framing 后更新 camera lock baseline：后来确认会让 target-framed image 被错误标记为 source-S3 locked comparison；这些近景/空白图不能作为正式视觉比较。

3. `sceneproof_local_settle_component_gate_fix84.py`
   - E4 允许 before COM margin 缺失、after 新认证。
   - E5 允许 contact gap 在容差内最多退化 2.5 cm。
   - 增加 semantic z-only artefact 豁免。
   - 这些是 relaxed diagnostic policy，不是严格 gate 的自然延伸。必须同时保留 strict 结果，不能只报告 relaxed PASS。

4. `scripts/drop_sim_script.py`
   - linear/angular damping 从 0.5/0.5 改成 0.8/0.8。
   - 动机是减少 pillow 反弹，但没有证明它适合会倾倒的刚体；高 angular damping 可能压制应发生的旋转。

### 3. Fix104/105 风格 A/B 没有形成可信结果

- 只切换 `IMAGINARIUM_SETTLE_ON_SUPPORT` 时，post-simulation settle 仍可能默认开启。
- 两个 arm 都使用 damping 0.8，因此没有复现真正旧行为。
- pre-sim settle、post-sim settle、damping 和 gate policy 没被完全解耦。

诚实结论：8/10 增加了诊断覆盖和一些工程改进，但没有产生可报告的新算法提升，而且实验变量混得过多。

## 2026-08-11：实际完成了什么

### 1. 暴露并确认视觉评测问题

- 对多个 Fix84 candidate/commit 做了渲染。
- bedroom/livingroom/casino/official 的若干图片出现空白、贴脸或场景外构图。
- 原因不是掉落质量本身，而是 target camera framing 与 locked-camera audit 混用。
- 这些图保留为 diagnostic reference，但不能进入正式 before/after 比较。

### 2. 收窄为四个直观刚体

固定对象：

- `bedroom_01 / single_sofa_chair_1`
- `livingroom_10 / single_sofa_chair_0`
- `casino_02 / casino_chair_0`
- `official_01 / office_chair_5`

明确排除 `bedroom_18 / pillow_2`。当前目标不是解决软体、柔性资产或所有重心问题，而是验证少量明显刚体的掉落是否能在较低成本下产生可信改进。

### 3. 建立 Fix84e adaptive rigid-only eval

每个对象使用独立 Blender 进程，其他物体固定为 passive collider：

1. 主 trial：1.0 s，CONVEX_HULL active、MESH passive、10/10 substeps/iterations、damping 0.8/0.8、friction 100。
2. 只有 true-mesh support 仍认证为 unstable 时，补跑 damping 0.5/0.5。
3. 仍 unstable 时才补跑 active/passive friction 0.5。
4. 新碰撞、恢复失败、support 不可认证均 fail closed，不继续参数搜索。
5. 同时输出 strict Fix84 gate 和 relaxed Fix84 gate。
6. relaxed gate 通过的对象才进入 scoped commit；最终仍输出 aggregate collision/support/plane/semantic、rotation/translation 和 retained objects。
7. 正式渲染只使用 source-S3 camera；四个受影响场景渲染，其他 26 场复用 Fix61。

第一版 runner 在进入仿真前因 Bash `local` 变量同语句初始化顺序失败；没有产生或污染 probe。`runnerfix1` 已修复，并增加 worker TSV 缺失时立即停止的 fail-closed 检查。

诚实结论：8/11 的主要进步是把混乱路线重新缩成一个可评测的刚体实验，而不是已经取得质量提升。Fix84e 的结果尚未返回。

## 当前唯一执行方案

### Baseline

`v5_sceneproof_collision_partial_commit_certified_paper30_fix61`

### Candidate

`v5_sceneproof_rigid_only_adaptive_paper30_fix84e`

### 必须读取的结果

- `/data/home/dev/Lumenarium/a10_reusable_results/paper30/sceneba_audit/v5_sceneproof_rigid_only_adaptive_paper30_fix84e/per_object_trials.tsv`
- `/data/home/dev/Lumenarium/a10_reusable_results/paper30/sceneba_audit/v5_sceneproof_rigid_only_adaptive_paper30_fix84e/final_eval.json`
- `/data/home/dev/Lumenarium/a10_reusable_results/paper30/sceneba_audit/v5_sceneproof_rigid_only_adaptive_paper30_fix84e/final_eval.txt`
- `/data/home/dev/sceneproof_rigid_only_adaptive_comparison_fix84e.tar.gz`

### 成功标准

- 四个目标全部有明确 outcome，失败不能从分母消失。
- 至少一个 nonzero rigid pose 被保留，否则结论是 no improvement。
- 无新 exact-mesh collision、无 boundary regression、incumbent restoration 通过。
- strict 与 relaxed gate 均呈现；若只有 relaxed PASS，必须明确写 proxy-policy-dependent。
- aggregate physical family、rotation 和 translation 不退化。
- source-S3 locked-camera before/after 图可复现且构图一致。
- 运行时间只报告新增局部 probe 开销；不能把缓存复用伪装成完整端到端加速。

## 暂停项

- Fix87 bulk settle。
- 全 Paper30 true-mesh COM 重跑。
- pillow/curtain 等软体或柔性资产掉落。
- 新的通用 optimizer 修改。
- 继续增加 fix 编号而没有固定 baseline、manifest、gate 和相机协议。

## 文件保留策略

保留：当前 Fix84e 完整包及 runnerfix1、Fix84d gate 历史包、全部源代码与测试、周报 PPT/PDF、对照与诊断 PNG（整理到 `eval_reference_images`）。

删除：已被后续源文件和 Fix84e 包覆盖的旧 handoff tar.gz。删除这些传输副本不会删除源代码或 VM 结果，但本地不可直接恢复；需要时可从保留源文件重新打包。
