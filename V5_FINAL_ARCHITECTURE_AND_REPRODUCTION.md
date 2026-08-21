# Lumenarium V5 最终架构与复现指南

**状态**：最终版（2026-08-20）。S4 优化线已收口，Fix61 为冻结基线。
**适用对象**：复现本项目结果的人，以及在此基础上继续改 pipeline 的 agent。

本文档记录三个交付档位（V5-fast / V5-medium / V5-best）的完整设计思路、实现流程、
每个相关文件的意义，以及我们真实踩过的坑与解决方案。**读者不必重走我们的弯路。**

相关链接：
[项目主页](https://hanshengzhu0001.github.io/Lumenarium/) ·
[GitHub](https://github.com/hanshengzhu0001/Lumenarium) ·
[工蜂 BowerPhys](https://git.woa.com/USD/BowerPhys) ·
[在线系统](https://embedding.lightart.qq.com/) ·
[Bilibili demo](https://www.bilibili.com/video/BV1tpbD6hERB/) ·
[Imaginarium 基础工作](https://github.com/HiHiAllen/Imaginarium)（[arXiv:2510.15564](https://arxiv.org/abs/2510.15564)）

---

## 0. 术语与阶段约定（先读这节，后文全部依赖）

| 记号 | 含义 |
|---|---|
| S0 | 提示词扩写与图像生成 |
| S1 | 视觉解析：实例分割、可见掩码、相机、关系图（**全链路最大瓶颈，占 S0–S3 的 69.56%**） |
| S2 | 资产检索（DeepSearch 路径在此） |
| S3 | 位姿推断，产出 `*_placement_info.json` |
| S4 | 布局精修（SceneLM + SceneProof），产出 `*_placement_info_s4.json` |
| Fix61 | S4 的冻结配置：二阶 `v5_scenelm` + 2 步迭代 + semantic_weight 0.5 + warm_start_weight 0.01 |
| Fix114 | 保守的 true-mesh 支撑/可见性修复层（V5-medium 才启用） |
| Paper30 | 固定 30 场景评测集 |
| Primary | 只统计 S1 可见掩码 ≥ 8,000 px 的对象（headline 位姿协议） |
| macro | `(collision + support + plane + semantic) / 4`，四族等权 |

**关键区分（贯穿全文）**：
- `*_placement_info.json` = S3 输出（位姿 + 观测尺寸，**无** solver 诊断）
- `*_placement_info_s3.json` = **S4 优化前**写的几何快照（含资产 bbox/length，**无** solver 诊断）
- `*_placement_info_s4.json` = **S4 优化后**输出（**唯一**含 `scenelm_solver` 诊断的文件）

---

## 1. 三个交付档位

同一套底座，三种口径。**同一张输入图，三档复用同一份冻结的 S0–S3 缓存**，只在 S4 及之后分叉。

### 1.1 V5-fast — 论文定量主版本

**思路**：单次冷启动 + 冻结 Fix61，不做任何"为了好看"的修改。所有论文数字来自这一档，
保证可比性与可复现性。

**流程**：
```
S0 → S1 → S2(DeepSearch) → S3 → S4(SceneLM/Fix61) → 证书 → 输出
```

**实测数字**（Paper30）：

| 指标 | 值 |
|---|---:|
| Primary recovery | 88.22% |
| Primary parent | 80.14% |
| Rotation AUC@60 | 31.38% |
| Translation AUC@0.5 m | 12.14% |
| Physical macro | 62.10% |
| S4 均值 | 192.930 s/scene（1.608 GPU-h，**3.513x** vs legacy SA5000 677.770 s） |
| S0–S4 均值 | 829.879 s/scene（13.83 min，6.916 GPU-h） |

**为什么是 2 步迭代**：不是欠收敛，是刻意的隐式正则。见 §4 完整证据链。

### 1.2 V5-medium — 演示交付版本

**思路**：在 V5-fast 之上叠加 Fix114 —— 用真实网格（而非包围盒）做支撑与可见性审计，
事务化地落地或隐藏少量叶子重复物。**presentation-only，不混入论文主表。**

**流程**：
```
复用 V5-fast 的冻结 S0–S3 → S4(Fix61) → Fix114 true-mesh 修复 → 证书 → 输出
```

**实测数字**：

| 指标 | 值 |
|---|---:|
| Fix114 增量 | +166.333 s/scene（1.386 GPU-h） |
| S4 总计 | 359.263 s/scene（2.994 GPU-h，**1.887x** vs legacy） |
| S0–S4 总计 | 996.212 s/scene（8.302 GPU-h） |

**铁律**：Fix114 只在局部见证与聚合族**同时**通过时才接受修复，否则回滚到 Fix61 现任解，
并把该场景标记为"物理或视觉未解决"。**不允许用单场景好看的编辑悄悄拉低 benchmark。**
它的质量行必须与 Fix61 分开记录，**不得复用 Fix61 的质量分**。

### 1.3 V5-best — 最高质量档

**思路**：三次独立冷启动 + 无 GT selector。用可观测证据（而非评测真值）挑最好的那次。

**流程**：
```
双 A10 并行三次冷启（seed 不同） → 各自 S4 + 证书
  → selector 排序：证书覆盖率 → unresolved 数 → 碰撞 → physical → coverage
  → 选出唯一交付
```

**selector 铁律**：**GT 永远不是 selector 的特征。** 否则就是评测泄漏。
代价是约 3 倍上游算力，必须显式报告。

---

## 2. 核心数学方法（S4 内部）

### 2.1 Typed Relation Program

每个场景编译成确定性的关系程序 bundle。每个程序 `p` 声明四件事：
(i) 参与的对象/部件见证，(ii) 拥有的变量块，(iii) 给求解器的鲁棒残差通道，
(iv) 用于认证的**独立**测量与阈值。

程序种类：`Support / Stack / PlaneAttach / CeilingAttach / Hang / Inside /
CollisionExclusion / PointTowards / Align / Distance`

**类型规则**：任何求解器变量的 owner 必须是该 factor 的参与者之一。
这条简单规则使"Jacobian 所有权数值审计"成为可能——未声明的 Jacobian 块必须在容差内为零。

**求解器残差与证书测量刻意分离**，防止"优化 loss 低"冒充"物理见证通过"。

### 2.2 关系条件化的位姿坐标

每个非刚体对象拥有独立的世界系旋转切空间 `ω_i ∈ R³`：

```
R_i(ω_i) = exp([ω_i]×) · R̄_i
```

平移用关系决定的 chart：world `xyz` / support `uvh` / plane `uvn`。
若 `π(i)` 是支撑父物体，中心解码为：

```
t_i = t_π(i) + ΔR_π(i)·(t̄_i − t̄_π(i)) + ΔR_π(i)·B_i·q_i
```

**父物体运动会输运子物体中心，但子物体朝向保持独立 SO(3) 块**——避免错误继承父物体旋转。

### 2.3 SceneLM：守卫 Levenberg–Marquardt

```
(JᵀJ + λD) Δx = −Jᵀr,    D = diag(max(|diag(JᵀJ)|, 1))
```

- 支持 matrix-free `Jv / Jᵀv` + 预条件共轭梯度
- 独立组装的块正规方程用于 parity 校验与守卫求解
- 冻结对象在微分与更新中都被 mask 掉
- 全局 trust scale 只截断步长幅度，**不改变步方向**

接受准则：步长有限 **且** 非线性目标下降 **且** gain ratio `γ_k` 超阈值。
高比例步减半阻尼；弱接受步增加阻尼；拒绝步阻尼乘四。

**安全 Schur 消元**：只有稳定叶子支撑块的**平移**坐标可被消去。
```
S = H_rr − H_re H_ee⁻¹ H_er,    b̃_r = b_r − H_re H_ee⁻¹ b_e
```
**旋转永不被 Schur 消元。** 叶子只有在其稳定入射 factor 仅涉及自己与声明的父物体时才合格。

### 2.4 SceneProof：可执行证书与作用域回滚

对每个分量 `c`，令 `m_c(x)` 为越高越好的分数。分量通过的条件：

```
m_c(x′) − m_c(x) ≥ −ε_c
```

外加：现任已通过的 hard program 不得变为不通过；不得出现新的 hard failure。

失败程序的见证**只**标识其变量 owner 与声明的分隔符，**只回滚这些对象**，
不丢弃无关的已接受分量。证据缺失时给 `Abstain`；未解决的可见支撑/附着失败必须暴露出来，
不得静默标记为成功。

### 2.5 目标函数权重（当前冻结值）与设计理由

| 项 | 权重 | 源码位置 | 设计理由 |
|---|---:|---|---|
| collision | 1.0 | `_s4_layoutvlm_ops.py:3614` | 唯一 pairwise 项，残差数 $O(N^2)$；调高会让密集场景淹没所有 per-object 项 |
| contact（竖直间隙） | 2.0 | `:3615` | 硬物理违背，悬空无歧义且视觉刺眼 |
| plane | 2.0 | `:3616` | 同上，脱墙是硬违背 |
| orientation | 0.25 | `:3617` | 只对贴墙物体有意义，朝向误差比穿模温和 |
| **containment（水平足迹包含）** | **1.0** | `:3618` | 保证整个足迹在父物体内，对齐评测口径而非只测中心 |
| semantic | 0.5 | `:3619` | 关系来自语言模型场景图，单条不可靠，过度信任会转错朝向 |
| boundary | 1.0 | `:3620` | 房间边界；当前基准覆盖退化，仅作诊断 |
| warm_start | 0.01 | `:3629` | 图像证据已编码在初始化里，只需极弱锚定防漂移 |

**两条贯穿原则**：① 硬物理违背优先于软偏好（contact/plane 2.0 vs semantic 0.5）；
② 噪声证据权重低于可靠证据（LM 关系 0.5、初始化锚 0.01）。

**warm_start 不能调高**（实测反证）：调到 0.3 时所有族都变差（semantic 0.4563、
collision 0.4130 双双垫底）。机制是它与物理残差争夺同一个 Gauss–Newton 步。

**注意**：`containment` 项就是"水平足迹包含"，由 `support_planar_containment_loss`
（`:2460-2553`）实现，旋转感知、每对取最差角点、可微。
**它早就存在**——我们曾误判它缺失并浪费了一轮分析，见 §5.3。

### 2.6 各物理分量对优化的响应（实测）

四个评测族**不是可互换的代理**，在"位移能否帮到它"上性质完全不同：

| 族 | 响应 | 实测变化 | 机制 |
|---|---|---|---|
| collision | **唯一正响应** | 0.4301 → 0.4436 | 唯一最优解远离初始化的族 |
| support | 负响应 | 0.5684 → 0.5480 | 三分量（竖直 gap + 水平 containment + 足迹重叠），只有竖直有强权重残差守住 |
| plane | **完全无响应** | 恒 0.7081 | 求解器可表达的运动不改变墙/天花板的法向距离与朝向角；纯 S3 属性 |
| semantic | **最敏感** | 0.7634 → 0.5209 | 双重不利：warm-start 门控筛过 + 评测 per-object max vs 残差 mean |

**结论**：报单一 macro 平均分会掩盖「四分之三的分数在测初始化保真度」这个事实。

---

## 3. 文件清单与意义

### 3.1 S4 核心实现

| 文件 | 意义 |
|---|---|
| `modules/S4_blender_layout_and_corr.py` | S4 主编排。关键行：`:10478` collision pair 构建（**已排除支撑对**）、`:10711` `gate_support_containment_pairs` 筛选、`:11303` `optimize_semantic_stage` 唯一调用点、`:11327` semantic_weight env 出口、`:11333` warm_start_weight env 出口、`:12493-12545` `scenelm_solver` 诊断落盘、`:12850` s4 JSON 写出 |
| `modules/_s4_layoutvlm_ops.py` | 求解器与所有损失项。`:1291` `support_contact_loss`（竖直）、`:2460` `support_planar_containment_loss`（水平）、`:3614-3629` 权重默认值、`:5205` 全局步长截断、`:6092` 主循环、`:6419-6430` total 目标组装、`:6701-6705` 三个早停条件、`:6711-6746` 硬投影（**仅非 v5_scenelm 执行**）|
| `modules/_s4_layoutvlm_relations.py` | `build_semantic_relation_specs`。**关键**：`:255-264` 与 `:311-323` 会把"warm start 已违反"的关系直接 skip，故被评分的语义关系恰是初始化本来做对的那批 |
| `modules/_sceneproof_residual_bridge.py` | 影子残差桥，做 factor/block owner 归属与 parity 校验 |

### 3.2 评测

| 文件 | 意义 |
|---|---|
| `eval_physical_realizability.py` | 物理/语义可实现性评测。`:287` `find_s3`、`:299` `find_geometry_snapshot`（要 `*_placement_info_s3.json`）、`:949-981` support 族三分量（竖直 gap + 水平 containment + footprint overlap）、`:1094` semantic 取 **per-object max**、`:1534` 语义角容差 20° |
| `eval_gt_metrics.py` | GT 位姿指标（recovery / parent / rot AUC / trans AUC） |
| `eval_dashboard.py`、`EVAL_DASHBOARD.ascii` | 汇总看板 |
| `eval_matching_diagnostics.py` | 匹配诊断 |

### 3.3 运行脚本

| 文件 | 意义 |
|---|---|
| `scripts/run_paper30_v4_s4_only_dual_gpu.sh` | **底层 S4 runner**。所有 env 变量的真实入口（`:7-56` 全部 `IMAGINARIUM_*` 读取）。要跑自定义 arm 就用这个 |
| `scripts/run_sceneproof_fix43_inloop_fullstack_smoke5_fix56.sh` | 全栈 Smoke5（smooth → inloop guarded → 证书 → 渲染 → GT → gates）。`:45-77` 内部**硬编码**转发 env，外层变量名不同 |
| `scripts/setup_a10_inference.sh` | A10 环境与权重/数据集下载 |
| `monitor_paper30.sh`、`monitor_light.sh` | 进度监控 |

### 3.4 文档

| 文件 | 意义 |
|---|---|
| `V5_FINAL_ARCHITECTURE_AND_REPRODUCTION.md` | 本文件 |
| `V5_FAST_FINAL_QUALITY_SPEED_REPORT_2026-08-13.md` | fast/medium 质量与速度权威数字 |
| `paper_draft/sceneproof_paper_draft.tex` | 论文初稿 |
| `paper_draft/README_DATA_PROVENANCE.md` | 每个数字"测量/推断/占位"的出处 |
| `docs/SCENEBA_MATHEMATICAL_FORMULATION.md` | 数学形式化 |
| `docs/index.html` | 项目主页 |
| `.codebuddy/skills/lumenarium-progress/` | 进度追踪 SOP + `references/progress.md` 唯一真源 |

### 3.5 数据

| 路径 | 意义 |
|---|---|
| `asset_data/imaginarium_asset_info.csv` | 2043 资产授权真实尺寸（米）。**必须 utf-8-sig 读**。字段 `name_en` / `bbx` / `class_en` / `retrieval_class_en` / `scaling_strategy` |
| `a10_reusable_results/paper30/` | 结果根目录，结构见 §5.1 |
| `visual_results/` | livingroom_10 与 official_02 的 input/V1/V3/V5 四图对比 |

---

## 4. S4 优化预算：最终结论与完整证据链

**这一节是本项目最重要的负面结果，也是最容易被后人重复浪费时间的地方。**

### 4.1 结论

**Fix61（二阶 v5_scenelm + 2 步）是当前评测定义下的经验上界。加迭代预算净亏。**

Smoke5（bedroom_01 / livingroom_10 / casino_01 / official_01 / streelitter_01）实测：

| 配置 | macro | coll | support | semantic |
|---|---:|---:|---:|---:|
| **Fix61（2 步，默认 cap）** | **0.6183** | 0.4301 | **0.5684** | **0.7634** |
| 4 步 | 0.5782 | 0.4394 | 0.5459 | 0.5209 |
| 8 步 | 0.5828 | **0.4436** | 0.5480 | 0.5210 |
| 8 步 + yaw cap 3.75° | 0.6004 | 0.4380 | 0.5391 | 0.6741 |
| 8 步 + yaw 3.75° + trans 0.05 m | 0.5799 | 0.4304 | 0.5495 | 0.5943 |
| 8 步 + warm_start 0.3 | 0.5672 | 0.4130 | 0.5510 | 0.4563 |
| 16 步 | — | — | — | 见 §4.4 |

### 4.2 二阶铁证（Fix61 一直是二阶，不是一阶）

读 Fix61 的 S4 **优化后**输出：
```
solver = v5_scenelm
schema_version = scenelm_relation_manifold_v1   # 二阶关系流形专属，一阶不写此键
maximum_iterations = 2
executed_iterations = 2
accepted_steps = 1, rejected_steps = 1
converged = False
```
`converged=False` 说明 2 步是**人工截断**而非自然收敛。
`rejected_steps=1` 说明第 2 步试步已经过不了接受准则。

一阶 `adam` 根本不产生 `scenelm_solver` 键，故此为落盘自证。

### 4.3 机制（可进论文）

**早停本身就是对 S3 初始化的隐式 trust-region 正则。**

评测 macro 的四族里：
- `semantic`：被 warm-start 自一致性门控筛过（`_s4_layoutvlm_relations.py:255-264, 311-323`），
  被评分的恰是初始化本来做对的那批关系 → **任何位移只有下行**
- `support`：锚定 S3 支撑树
- `plane`：对位移完全不响应（所有臂恒为 0.7081）
- `collision`：**唯一**能被位移改进的族

所以 `macro ≈ 3 份保真度 + 1 份改进量`。等权下，"优化得更狠"必然负期望。

**semantic 崩坏是 2→4 步的一次性跳变，不是逐步累积**（0.7634 → 0.5209 → 0.5210 饱和）。
峰形是悬崖不是缓坡。

### 4.4 已关闭的四条路径（不要重开）

| 路径 | 结果 |
|---|---|
| 加迭代步数（4/8/16） | 全部 < Fix61。16 步时 `acc=5 rej=11 damp=1311`，λ 涨 5.7 数量级、能量只降 4.37% |
| 掐 yaw / trans 步长 cap | yaw cap 收回 63% semantic 但 support 更差；同时掐两者更差（trust region 在两维间零和） |
| 补水平包含残差 | **该项早已存在**（`containment_weight=1.0`），非缺口 |
| warm_start 显式正则（0.01→0.3） | 反效果，semantic 0.4563 垫底 |
| 提 semantic 权重（0.5→1.0） | Paper30 上 macro 0.5986 < Fix61 0.6287，已回退 |

### 4.4.1 为什么「更小步长 + 更多步数」失效（反直觉，必读）

经典优化直觉：减半信赖域半径、加倍迭代数，最差中性、通常更好——对平移和旋转都成立。
**我们做了同总额度的对照实验，直觉失效**：

- 8 步 × 3.75° = 2 步 × 15°（总角度均 30°）→ macro 0.6004 < 0.6183
- 再加 8 步 × 0.05 m = 2 步 × 0.20 m（总平移均 0.40 m）→ macro 0.5799 < 0.6183

三个原因，**没有一个是求解器缺陷**：

1. **直觉是关于 $E$ 的，不是关于评测 $M$ 的。** 细步长让迭代点更准确地到达
   $\arg\min E$，而那个点在 $M$ 下更差。优化保真度与目标错配**相乘**：
   错配固定时，优化越准损失越大。**细化离散化 = 把错的问题解得更精确。**
2. **总路径长度相同 ≠ 净位移相同。** 二阶 LM 每步重新线性化，8 小步沿弯曲下降路径
   持续朝一致方向累积；2 步轨迹被截断且**第 2 步就被 gain-ratio 拒绝**
   （落盘 `accepted=1, rejected=1`）。所以 budget-8 的有效位移 $\|T-\bar T\|$
   **反而更大**，与预期的控制相反。
3. **两个 cap 不是两个独立一维线搜索。** LM 联合求解平移与旋转，全局 trust scale
   只等比缩放整个步、不改方向，所以掐一维会把预算重分配给另一维。实测只掐 yaw 时
   support 掉到全部臂最差的 0.5391（释放的预算把子物体推出台面）；同时掐平移则
   拿掉了 `PointTowards` / `Distance` 需要的平移自由度。
   **该二维 cap 空间无内部甜点**，两个 cap 通过同一信赖域耦合、不可分离。

### 4.5 一阶 vs 二阶

- 单场景 macro：`adam@100` 0.6822 > `v5_scenelm@2` 0.6338
- **但 adam 在 official_01（112 物体、14 个足迹误差 >0.5 m）硬投影无可行解，
  直接 `raise ValueError`，Smoke5 跑不完**
- 二阶 5/5 完成 → **结构性鲁棒性差异，二阶胜出**
- 二阶 collision 相对 legacy SA-5000：0.3719 → 0.4301（**+15.6%**）

### 4.6 尚存的已知错配（未修，风险已知）

优化器的 semantic 是 **mean** 残差（`_s4_layoutvlm_ops.py:6401`），
评测的 semantic 是 **per-object max**（`eval_physical_realizability.py:1094`）。
优化器可能"平均改善但最差关系恶化"。这是代码事实；它是否为 semantic 崩坏的主因**未做隔离验证**。

---

## 5. 复现坑与解决方案

> 每一条都是我们真实踩过的。按类别组织。

### 5.1 路径与文件结构

**坑 1：找不到结果文件（`ls` 返回空）**
结果目录结构是 `<scene>_<version>_result/`，**版本名前面有场景名前缀**：
```
a10_reusable_results/paper30/bedroom_01_v5_smoke5_budget4_result/S4_layout_refinement/
  bedroom_01_v5_smoke5_budget4_placement_info_s4.json
```
正确 glob：
```bash
ls a10_reusable_results/paper30/*_${V}_result/S4_layout_refinement/*_placement_info_s4.json
```
❌ 错误写法：`paper30/${V}/*/S4_layout_refinement/*`（少了场景前缀、多了一层）

**坑 2：查 solver 诊断查错文件**
`scenelm_solver` 只写在 `*_placement_info_s4.json`（优化**后**）。
`*_placement_info_s3.json` 是优化**前**的几何快照，查它永远得到 `ABSENT → first-order adam`，
会让你误判成一阶。**记住：s3 后缀 = 优化前，s4 后缀 = 优化后。**

### 5.2 脚本 env 变量名

**坑 3：env 变量名落到脚本不看的名字上（最坑，会静默产生假结果）**

两个脚本的变量名**完全不同**：

| 你想设的 | 底层 runner（`run_paper30_v4_s4_only_dual_gpu.sh`） | 全栈脚本（`..._fullstack_smoke5_fix56.sh`） |
|---|---|---|
| 版本名 | `IMAGINARIUM_S4_TARGET_VERSION` | 内部硬编码，外层不可覆盖 |
| 迭代数 | `IMAGINARIUM_LAYOUTVLM_ITERATIONS` | 同名，默认 2 |
| 场景清单 | `IMAGINARIUM_PAPER30_MANIFEST` | `manifest` 位置参数 |

我们曾用 `TARGET_VERSION=$V ITERATIONS=8` 跑全栈脚本，结果它用自己的默认值
（`ITERATIONS=2` + 自己的版本名）跑了一遍 Fix61，**121 秒完成、数字与 Fix61 逐位相同**，
差点被当成实验结果。

**解决**：跑完必须核 banner 两行：
```
TARGET_VERSION=<你要的版本名>
ITERATIONS=<你要的步数>
```
不对就立刻 Ctrl-C。

**坑 4：worker 幂等跳过导致静默复用旧结果**
如果目标版本目录已存在，worker 会跳过该场景。表现为耗时异常短（一两分钟）。

**解决**：三重验证
```bash
# 1. 耗时是否合理（Smoke5 8 步约 400–850 s，纯求解约 20 s，全栈含渲染更久）
# 2. mtime 是否是刚才
# 3. executed_iterations 是否等于你设的步数
for V in <version>; do
  f=$(ls a10_reusable_results/paper30/*_${V}_result/S4_layout_refinement/*_placement_info_s4.json | head -1)
  python -c "
import json,sys,os,datetime
d=json.load(open(sys.argv[1])); s=d.get('scenelm_solver')
print('mtime=', datetime.datetime.fromtimestamp(os.path.getmtime(sys.argv[1])))
print({k:s[k] for k in ('solver','maximum_iterations','executed_iterations','accepted_steps','rejected_steps','converged')} if s else 'ABSENT')
" "$f"
done
```

### 5.3 分析纪律

**坑 5：没查证就断言"某项缺失"**
我们曾断言"目标函数缺水平足迹包含项、这是唯一值得走的路"，浪费一轮分析。
真相是 `support_planar_containment_loss`（`:2460`）早已存在且权重 1.0。

**解决**：断言任何"缺失"之前，必须 grep 到函数定义 + 确认它进了 total 目标 + 确认权重非零。

**坑 6：从单场景结果外推**
我们曾看 2 步档 `rej=0` 就断言"加预算纯收益、给足 32 步"，
8/16 步实测 `rej > acc`、λ=1311，被推翻。也曾在单场景上定 `semw=1.0` 为默认，
Paper30 全量证明降了 0.0301，被迫回退。

**解决**：定默认值前必须过 Smoke5，改论文数字前必须过 Paper30。
执行顺序永远 **Smoke1 → Smoke5 → Paper30**。

**坑 7：拿错基线比**
曾拿 legacy SA-5000 当基线，得出"prod_v2 更好"的错误结论。正确基线是 **Fix61**。

### 5.4 评测调用

**坑 8：`--geometry-version` 指向没有 S4 目录的版本**
指向 `v4_deepsearch` 之类只有 S3 的版本时，评测会静默给空表（`Failures:1`、`scenes=0`）。
**解决**：确认该版本下存在 `S4_layout_refinement/*_placement_info_s3.json`。

**坑 9：physical.json 顶层字段全是 None**
真实数字在 `--report-out` 产出的 `physical.txt` 里，不在 JSON 顶层。
**解决**：看 `head -16 physical.txt` 的表格。

### 5.5 Shell 与文件传输

**坑 10：源码块混入 bash 命令**
把 Python 源码直接粘进 shell 会报 `-bash: syntax error`，还可能误触发包管理器安装。
**解决**：源码一律走 heredoc 或独立 `.py` 文件。

**坑 11：heredoc 内变量不展开**
`<<'PY'`（带单引号）内 `$T` **不**展开，会得到字面量路径导致 `FileNotFoundError`。
**解决**：要展开就用 `<<PY`（无引号）；或在 heredoc 内用 `sys.argv` 接参数。

**坑 12：`rz` 传 tar 包失败**
**解决**：改为单文件 heredoc patch。

**坑 13：patch 锚点不匹配（`anchor count=0`）**
A10 上的文件与本地可能格式不同（例如本地多行、A10 单行）。
**解决**：patch 前先 `grep -n` 看 A10 上的真实内容，按实际写锚点。
**patch 脚本必须带 `assert n == 1`**，锚点不唯一或为 0 就 abort，绝不写坏文件。

**坑 14：基于字符串索引的整体改写**
曾用 `index('"""', 3)` 误命中 shebang 后的开引号而写坏文件，靠上一轮 tar.gz 恢复。
**解决**：定点 `replace` + assert，不做整体重写。

### 5.6 数据读取

**坑 15：资产 CSV 编码**
`asset_data/imaginarium_asset_info.csv` **必须** `utf-8-sig` 读，否则首列名带 BOM。

**坑 16：`class_en` 不是资产身份**
它是策展粗桶。例如 `wardrobe_0 → a_SM_Wardrobe_01`（真实授权 0.998/0.544/2.000）
却因桶名 `Storage_locker` 被误判为替换。
**解决**：资产标识符作第二独立证人，且**不对称**——标识符可开释，永不可定罪。

**坑 17：`scale=[1,1,1]` 不是中性默认值**
它是在断言"此物恰好就是其资产授权的那么大"。库会把它翻译成具体米数
（例如 1.91 m 书架）。观测框只剩 2–17 cm 碎片时会落入小物体分支并返回 `[1,1,1]`，
于是"凭空长出 1.9 m 书架"。**这个结论没有授权尺寸表说不出来。**

---

## 6. 最小复现路径

### 6.1 环境
```bash
bash scripts/setup_a10_inference.sh   # 下载权重与数据集
```
数据来源（HuggingFace）：
- FBX 资产库与元数据：[`HiHiAllen/Imaginarium-Dataset`](https://huggingface.co/datasets/HiHiAllen/Imaginarium-Dataset)
- 渲染视图/嵌入/体素/AE 权重：[`binicey/Imaginarium-3D-Derived-Dataset`](https://huggingface.co/datasets/binicey/Imaginarium-3D-Derived-Dataset)
- DINOv2：[`facebook/dinov2-large`](https://huggingface.co/facebook/dinov2-large)
- Depth Anything V2：[`depth-anything/Depth-Anything-V2-Metric-Hypersim-Large`](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large)
- SAM3：[`facebook/sam3`](https://huggingface.co/facebook/sam3)

### 6.2 跑 V5-fast 的 S4（复现 Fix61）
```bash
cd "$HOME/Lumenarium"
V=v5_repro_fix61
env IMAGINARIUM_PAPER30_MANIFEST=a10_reusable_results/paper30/manifest.txt \
  IMAGINARIUM_PAPER30_RESULTS_ROOT=a10_reusable_results/paper30 \
  IMAGINARIUM_S4_SOURCE_VERSION=v4_deepsearch \
  IMAGINARIUM_S4_SOURCE_STAGE=S3_pose_inference \
  IMAGINARIUM_S4_SOURCE_PATTERN='*_placement_info.json' \
  IMAGINARIUM_S4_TARGET_VERSION="$V" \
  IMAGINARIUM_S4_ENGINE=layoutvlm \
  IMAGINARIUM_LAYOUTVLM_STAGE=full \
  IMAGINARIUM_LAYOUTVLM_SOLVER=v5_scenelm \
  IMAGINARIUM_LAYOUTVLM_ITERATIONS=2 \
  IMAGINARIUM_LAYOUTVLM_SEMANTIC_WEIGHT=0.5 \
  IMAGINARIUM_SCENELM_WARM_START_WEIGHT=0.01 \
  IMAGINARIUM_LAYOUTVLM_ACTIVE_SET_ROUTER=0 \
  IMAGINARIUM_SCENEPROOF_PROGRAM_IR=1 \
  IMAGINARIUM_SCENEPROOF_REQUIRE_FACTOR_PARITY=1 \
  IMAGINARIUM_SCENEPROOF_REQUIRE_BINDING_AUDIT=1 \
  IMAGINARIUM_SCENEPROOF_SHADOW_JACOBIAN_OWNERSHIP=1 \
  IMAGINARIUM_SCENEPROOF_STABLE_LINEARIZATIONS=2 \
  IMAGINARIUM_SCENEPROOF_MATERIALIZED_WARM_START=1 \
  IMAGINARIUM_GPU_FREE_FLOOR_MB=16000 \
  IMAGINARIUM_S4_SCENE_TIMEOUT=3600 \
  IMAGINARIUM_S4_WORKER_LOG_ROOT="logs/$V" \
  bash scripts/run_paper30_v4_s4_only_dual_gpu.sh
```
**核 banner**：`TARGET_VERSION=v5_repro_fix61`、`ITERATIONS=2`。

### 6.3 评测
```bash
python eval_physical_realizability.py \
  --saved-results a10_reusable_results/paper30 \
  --scenes a10_reusable_results/paper30/manifest.txt \
  --versions "v5_repro_fix61" \
  --geometry-version v4_deepsearch \
  --metrics-out /tmp/repro/physical.json \
  --report-out /tmp/repro/physical.txt
head -16 /tmp/repro/physical.txt
```
期望 Smoke5 子集 macro ≈ 0.6183，Paper30 certified ≈ 0.6287。

### 6.4 验证真的跑了
见 §5.2 坑 4 的三重验证脚本。`executed_iterations` 必须为 2。

---

## 7. 给后续 agent 的硬约束

1. **不要再调 S4 迭代预算/步长 cap/warm_start/semantic 权重**。四条路径已实测关闭（§4.4）。
2. **不要断言某损失项缺失**，先 grep 定义 + 确认进 total + 确认权重非零。
3. **改默认值前过 Smoke5，改论文数字前过 Paper30。** 顺序 Smoke1 → Smoke5 → Paper30。
4. **所有 probe 先 audit-only**，确认无副作用再提交改动。
5. **patch 必须带 assert 锚点唯一**，禁止基于字符串索引的整体重写。
6. **区分 by-construction 与经验保证**。例如 `length/scale` 是网格自身 bbox（by construction），
   而"库授权值与网格一致"是 91.4% 的经验率。
7. **交付 tar.gz 必须在回复中给完整绝对路径**，禁止只给文件名。
8. **沉降/重心投影路线保持关闭**。它修竖直接触关系，前提是尺寸对；
   5 场景中 3 个的沉降目标身份与尺寸都错，对错误尺寸做沉降只是把错误锚死。
   尺寸前提修好后才可重开。

---

## 8. 尚存 limitation

1. **V3 physical 分数有 evaluator 出处冲突**（41.20% vs 52.14%），已从 headline 表剔除。
2. **V3 runtime 是重建值**（27/30 成功行 + 均值补齐），只支持数量级比较，不支持百分比声明。
3. **exact-mesh 见证会在开放或薄网格上 abstain**，保守回滚可能保留明显视觉缺陷。
4. **DeepSearch 与冷启动/资产库混杂无法分离**。需要 seed-locked、冻结 S1 的 S2-only ablation。
5. **优化器 mean 与评测 per-object max 的错配未修**（§4.6）。
6. **`critical` 指标恒为 0**（四族最小值），因为总有场景某族为 0，该指标目前无区分度。
7. **V5-best 的 selector 尚未完整实现**，三冷启 + 无 GT 排序仍是计划。

---

*最后更新：2026-08-20。S4 优化线收口于 Fix61。*
