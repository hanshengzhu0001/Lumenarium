# SceneBA: Uncertainty-Aware Discrete–Continuous Scene Bundle Adjustment

Technical formulation and implementation specification, version 0.1
Date: 2026-07-30

## 1. Scope and status

SceneBA upgrades the current staged reconstruction pipeline

\[
\text{S1 parsing}\rightarrow\text{S2 retrieval}\rightarrow
\text{S3 pose}\rightarrow\text{S4 continuous layout}
\]

into hybrid inference that can revise upstream discrete commitments using
downstream visual and physical evidence. The formulation below is complete
enough to implement a Top-K oracle and a bounded-beam SceneBA prototype.
Factor weights, proposal thresholds, and non-inferiority margins remain
empirical quantities and must be frozen on a development split.

## 2. Variables

For object \(i\), define the discrete state

\[
z_i=(A_i,G_i,Q_i)
\]

and continuous state

\[
x_i=(T_i,R_i,\ell_i),\qquad \ell_i=\log S_i.
\]

The symbols mean:

- \(A_i\): asset identity, selected from local/DeepSearch Top-K candidates.
- \(G_i=(P_i,\rho_i)\): scene-graph state: parent/support object \(P_i\) and
  relation type \(\rho_i\), such as `on`, `inside`, `wall`, or `hang`.
- \(Q_i\): discrete pose or symmetry mode.
- \(T_i\in\mathbb R^3\): world-space translation of the object origin/center.
- \(R_i\in SO(3)\): world-space 3-D rotation.
- \(S_i\in\mathbb R_{>0}^3\): positive anisotropic scale; isotropic scale is
  the special case \(S_i=s_i\mathbf1\).
- \(\ell_i=\log S_i\in\mathbb R^3\): unconstrained log-scale optimized in
  place of positive scale.

The homogeneous object-to-world transform is

\[
M_i(T_i,R_i,S_i)=
\begin{bmatrix}
R_i\,\mathrm{Diag}(S_i) & T_i\\
0^\top&1
\end{bmatrix}.
\]

Optional global variables are camera parameters \(C\), metric-depth scale
\(\alpha_d\), floor plane \(\pi_f\), and wall planes \(\Pi\). Collect them as
\(\gamma=(C,\alpha_d,\pi_f,\Pi)\).

## 3. Symmetry-aware rotation

Let \(H_{A_i}\subset SO(3)\) be the finite symmetry group of asset \(A_i\).
The physical orientation is an equivalence class

\[
[R_i]=\{R_i h:h\in H_{A_i}\}\in SO(3)/H_{A_i}.
\]

For reference rotation \(\widehat R_i\), use

\[
d_{\mathrm{sym}}(R_i,\widehat R_i;A_i)
=\min_{h\in H_{A_i}}
\left\|\log\!\left(\widehat R_i^\top R_i h\right)\right\|_2.
\]

The discrete mode \(Q_i\) indexes retained symmetry representatives or
coarse pose modes. Continuous optimization then uses a local Lie-algebra
increment:

\[
R_i(\delta\omega_i)=R_i^{(0)}
\exp([\delta\omega_i]_\times).
\]

The current LayoutVLM implementation optimizes yaw only. SceneBA can begin
with the same one-degree rotational parameterization and later generalize to
full \(SO(3)\) without changing the discrete formulation.

## 4. Posterior and energy

Let \(y=(I,D,\mathcal M,\mathcal F)\) contain the input image, metric depth,
instance masks, and cached foundation-model features. With
\(z=(z_1,\ldots,z_N)\), \(x=(x_1,\ldots,x_N)\),

\[
p(z,x,\gamma\mid y)
\propto
p(z)\,p(x,\gamma\mid z)
\prod_{f\in\mathcal F_{\mathrm{obs}}}\psi_f(z_f,x_f,\gamma;y).
\]

Equivalently,

\[
E(z,x,\gamma;y)
=-\log p(z)-\log p(x,\gamma\mid z)
+\sum_f E_f(z_f,x_f,\gamma;y),
\]

where \(E_f=-\log\psi_f\). The MAP problem is

\[
(z^\star,x^\star,\gamma^\star)
=\arg\min_{z,x,\gamma}E(z,x,\gamma;y).
\]

## 5. Factors

### 5.1 Unary visual factors

For rendered mask \(\widehat M_i\), observed mask \(M_i\), robust penalty
\(\rho\), and cached dense features \(\Phi\):

\[
E_{\mathrm{sil},i}=1-\mathrm{IoU}(\widehat M_i,M_i),
\]

\[
E_{\mathrm{feat},i}
=\frac{1}{|\Omega_i|}
\sum_{u\in\Omega_i}
\rho\!\left(1-
\cos(\widehat\Phi_i(u),\Phi_I(u))\right).
\]

A robust depth factor is

\[
E_{\mathrm{depth},i}
=\frac{1}{|\Omega_i|}
\sum_{u\in\Omega_i}
w_i(u)\rho\!\left(
\log \widehat D_i(u)-\log(\alpha_d D(u))
\right).
\]

Occlusion order can be penalized only where two rendered masks overlap:

\[
E_{\mathrm{occ},ij}
=\frac1{|\Omega_{ij}|}
\sum_{u\in\Omega_{ij}}
\rho\!\left(
\operatorname{sign}_{ij}(u)
[\widehat D_i(u)-\widehat D_j(u)]
\right).
\]

### 5.2 Retrieval and initialization priors

For calibrated retrieval probability \(p_{\mathrm{ret}}\),

\[
E_{\mathrm{ret},i}(A_i)=-\log
p_{\mathrm{ret}}(A_i\mid I,M_i,\text{label}_i).
\]

Warm-start regularization is uncertainty weighted:

\[
E_{\mathrm{init},i}
=\frac12
(x_i-\mu_i)^\top\Sigma_i^{-1}(x_i-\mu_i).
\]

Large uncertainty weakens the prior and permits revision; confident S3
estimates remain stable.

### 5.3 Graph and physical factors

For parent \(P_i\) and relation \(\rho_i\):

\[
E_{\mathrm{graph},i}
=-\log p(P_i,\rho_i\mid\text{S1 evidence}).
\]

The support graph is constrained to a rooted directed forest: every ordinary
object has at most one structural parent and directed cycles are forbidden.

The physical energy reuses the validated LayoutVLM terms:

\[
E_{\mathrm{phys}}
=\lambda_cE_{\mathrm{collision}}
+\lambda_sE_{\mathrm{contact}}
+\lambda_pE_{\mathrm{plane}}
+\lambda_bE_{\mathrm{boundary}}
+\lambda_mE_{\mathrm{semantic}}.
\]

The semantic term includes gated containment, `align_with`,
`point_towards`, and directed distance constraints. Hard contact, plane,
containment, and room-boundary projections remain feasible-set projections
after gradient updates.

### 5.4 Total energy

\[
\begin{aligned}
E={}&
\sum_i[
\lambda_{\mathrm{sil}}E_{\mathrm{sil},i}
+\lambda_{\mathrm{feat}}E_{\mathrm{feat},i}
+\lambda_dE_{\mathrm{depth},i}\\
&+\lambda_{\mathrm{ret}}E_{\mathrm{ret},i}
+\lambda_{\mathrm{init}}E_{\mathrm{init},i}
+\lambda_gE_{\mathrm{graph},i}]
+\sum_{i<j}\lambda_oE_{\mathrm{occ},ij}
+E_{\mathrm{phys}}.
\end{aligned}
\]

Every reported experiment must expose each unweighted factor separately;
the internal total energy is not itself the evaluation metric.

## 6. Why minimum optimized loss is insufficient

For a fixed discrete hypothesis \(z\), define

\[
\widehat u_z=(x_z^\star,\gamma_z^\star)
=\arg\min_u E(z,u;y).
\]

Ranking hypotheses only by \(E(z,\widehat u_z;y)\) favors brittle hypotheses
that achieve a sharp minimum in a tiny continuous region. Instead compare the
discrete marginal

\[
p(z\mid y)=\int p(z,u\mid y)\,du.
\]

Second-order expansion at the local optimum gives

\[
E(z,u;y)\approx
E(z,\widehat u_z;y)
+\frac12(u-\widehat u_z)^\top
H_z(u-\widehat u_z),
\]

where

\[
H_z=\nabla_u^2E(z,u;y)|_{u=\widehat u_z}.
\]

The Laplace approximation yields

\[
\log p(z\mid y)
\approx
-E(z,\widehat u_z;y)
-\frac12\log\det(H_z+\lambda I)
+\log p(z)
+\frac{d_z}{2}\log(2\pi)
+C.
\]

For equal-dimensional hypotheses, the last two constants cancel. Define the
ranking cost

\[
\mathcal C(z)=
E(z,\widehat u_z;y)
+\frac12\log\det(H_z+\lambda I)
-\log p(z).
\]

Lower is better. The damping \(\lambda>0\) handles nearly flat modes. In the
first implementation, approximate log-determinants can use a diagonal
Gauss–Newton/Fisher matrix or stochastic Lanczos quadrature. Exact dense
Hessians are unnecessary.

Important limitation: hypotheses using different asset parameter dimensions
or different numbers of active objects require the dimension term and a
consistent prior measure; otherwise their evidence values are not directly
comparable.

## 7. Bounded hybrid inference

The exact discrete search is combinatorial. Use uncertainty-gated local beam
search:

1. Initialize with the current Top-1 pipeline result.
2. Compute uncertainty from retrieval margin, parent entropy, symmetry
   ambiguity, visual residual, and physical residual.
3. Select a small active subgraph around the highest-risk object.
4. Propose Top-3 assets, Top-2 parents, and symmetry representatives.
5. Run 30–50 continuous projected-gradient iterations per proposal.
6. Rank with Laplace evidence and retain beam size 4–8.
7. Freeze converged low-risk subgraphs; repeat until the evidence gain or
   compute budget is exhausted.
8. Run the uniform LayoutVLM400 backend only for the winning state, or use the
   later `{30,100,400}` router.

This is coordinate search over discrete variables combined with conditional
continuous bundle adjustment. Monotonic improvement is guaranteed only for
the evaluated surrogate/evidence when the incumbent is retained; it is not a
guarantee of GT improvement.

## 8. Current pipeline correspondence

| SceneBA quantity | Current source |
|---|---|
| \(A_i\) candidates | S2 local/DeepSearch Top-10 retrieval JSON |
| \(G_i\) initial parent/relation | S1 parsing and S3 placement JSON |
| \(Q_i\) initial modes | S3 view IDs, yaw hypotheses, asset symmetry metadata |
| \(T_i,R_i,S_i\) warm start | S3 pose matrices and dimensions |
| image/mask evidence | input image and S1 instance masks |
| depth evidence | S0 Depth Anything output |
| dense feature evidence | cached DINO/AENet features |
| physical/semantic factors | validated LayoutVLM400 S4 objective |
| independent evaluation | 8000px GT evaluator and physical-realizability evaluator |

The present S4 result shows why the hybrid extension is needed: LayoutVLM400
improves physical macro score and runtime but cannot change matched-object
coverage or parent accuracy because \(A_i,G_i,Q_i\) are fixed.

## 9. Identifiability and calibration

Single-view reconstruction is not fully identifiable. Scale and depth can
trade off under perspective projection; occluded geometry is weakly observed;
symmetry creates multiple equivalent rotations. SceneBA therefore needs:

- metric-depth or support/floor priors to constrain scale-depth ambiguity;
- explicit symmetry quotienting;
- calibrated retrieval and parent probabilities;
- uncertainty-aware warm-start covariance;
- held-out calibration of factor weights;
- reporting of posterior ambiguity rather than only a point estimate.

Factor weights may be learned by likelihood calibration, ranking loss, or
black-box tuning on a development set. Paper30 must not be used both for
weight selection and final reporting.

## 10. Falsifiable oracle gates

Before implementing the full solver:

1. Measure asset Recall@1/3/5/10 and MRR.
2. Measure parent Recall@1/2.
3. Compute symmetry-aware pose oracle.
4. Evaluate asset-only, parent-only, symmetry-only, and joint oracle gains.
5. Measure how much of the oracle gap a GT-free verifier recovers.

Suggested continuation gates:

- Recall@5 minus Recall@1 at least 10 percentage points; or
- oracle rotation AUC gain at least 0.05; or
- oracle translation AUC gain at least 0.03; or
- oracle parent accuracy gain at least 0.05.

The practical solver should recover at least 40% of the available oracle gap,
expand no more than roughly 25–30% of objects, and remain within twice the
LayoutVLM400 runtime.

## 11. Claims supported now versus later

Supported now:

- a complete hybrid probabilistic formulation;
- a symmetry-aware continuous state;
- a principled Laplace-evidence ranking rule;
- bounded local beam inference compatible with current cached artifacts;
- a direct implementation path from the existing pipeline.

Not yet supported:

- that Top-K contains enough correct alternatives;
- that Laplace evidence ranks them better than raw loss;
- that SceneBA improves Paper30 GT metrics;
- final factor weights or compute thresholds;
- a formal novelty claim against every 2025–2026 system.

These are experimental questions, beginning with the Top-K oracle audit.
