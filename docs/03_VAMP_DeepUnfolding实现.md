# VAMPNet｜执行文档 03｜第一版 VAMP 信道均衡 Deep Unfolding

> 项目名称：**VAMPNet**。
> 文档版本：1.0。数学模型版本：`vamp_du_math_v1`。
> 主实现：`algorithm_name=vamp_du`，`algorithm_version=v1`，`variant=fixed_csi_core`。
> 显式增强变体：`algorithm_version=v1_refine`，`variant=structured_refinement`。不能把两个变体的结果都记作 `v1`。
> 输入来自 [文档 01](01_仿真数据生成.md)；baseline 见 [文档 02](02_Baseline算法实现.md)；训练后单独评估与复用规则见 [文档 04](04_统一评估与增量重测.md)。

## 0. 数学来源与不可变结构

本实现以已有的 `VAMP_DeepUnfolding_Mathematical_Design.md` 为数学依据。[M0] 该源文件 SHA-256：

```text
0915e67018e6ef8fb037f6626ecc42bf4e87bf6153eb63f36f5c137e650f64f0
```

本文件自包含实现所需公式，不要求代码运行时读取原 Markdown。原文中的两个版本现在明确落地：先实现第 15.1 节固定 CSI 核心版作为 `v1`；第 11 节、第 15.2 节的信道细化以独立变体保留完整数学定义。原文允许的条件控制器和可学习 $C_0$ **不在 v1 中启用**，不得由实现者临时加入。

v1 保留的结构是：

```text
上一层完整符号二阶矩
→ 结构化 CSI 不确定性协方差
→ 完整 ICI 加权线性估计
→ 条件散度 + 外信息
→ 分频组温度 QPSK 去噪
→ 与温度一致的条件散度 + 外信息
→ 自然参数阻尼
→ 下一层
```

允许的优化限于数值等价的矩阵重排、分块、线性方程求解和确定性复用。不允许为了内存或速度：删除非对角 ICI、把 $S$ 换成 $V$、取消外信息、用未经推导的方差代替散度、截断跨层梯度、把精确求解替换为固定少量 CG，或把信道细化改成任意网络输出矩阵。这些都属于新数学版本。

本文固定的是有限层条件代理模型，不宣称继承原始 VAMP 在特定随机矩阵条件下的全部状态演化结论，也不预先保证性能超过 baseline。[M1]

## 1. 维度、概率和数据接口

| 记号 | 维度/含义 |
|---|---|
| $N$ | 接收 FFT 维度；主数据 1024 |
| $D$ | 未知数据符号数；主数据 576 |
| $P$ | 导频数；主数据 192 |
| $Q$ | 接收侧残差基数；随估计路径数变化，主数据至多 24 |
| $G$ | 数据频率分组数，固定 8 |
| $T$ | 展开层数，固定 12 |
| $\mathbf y$ | $\mathbb C^N$，完整接收观测 |
| $\widehat H$ | $\mathbb C^{N\times N}$，实际估计 CSI |
| $\widehat R_n$ | 接收噪声协方差估计 |
| $B_q$ | $\mathbb C^{N\times N}$，固定的接收侧基 |
| $C_0$ | $\mathbb C^{Q\times Q}$，接收侧正定先验 |
| $E_d,E_p$ | 数据、导频选择矩阵 |
| $g(i)$ | 第 $i$ 个数据符号所属频率组 |

固定单位能量 QPSK

$$a(b_1,b_2)=\frac{(1-2b_1)+j(1-2b_2)}{\sqrt2},\qquad c=2b_1+b_2.$$

整个发射向量为 $s=E_dx+E_pp$。接收模型为 $y=H_{true}s+n$。空子载波为 0。

复高斯约定：

$$\mathcal{CN}(z;\mu,C)\propto\exp[-(z-\mu)^HC^{-1}(z-\mu)].$$

标量精度 $\gamma$ 对应复误差方差 $1/\gamma$，每个实部/虚部方差为 $1/(2\gamma)$。使用实双通道时必须保留这个因子 2。

网络只接收 `EqualizerInputV1`，不接收真实信道、真实残余尺度、真实逐块 CSI NMSE、名义 SNR 标签或真实符号。标签只交给损失函数和独立指标计算器。

## 2. 可训练量、固定量与在线状态

### 2.1 v1 固定参数

`T=12`、`G=8`、星座、频率分组、$\widehat H$、$\widehat R_n$、$B_q$、$C_0$、初始化、投影规则和数值精度固定。$B_q,C_0$ 是随样本变化的接收输入，不是可训练权重。

v1 对所有层令

$$H_t=\widehat H,\quad c_t=0,\quad C_{c,t}=C_0.$$

### 2.2 v1 可训练参数

每层学习：不确定性校准 $\kappa_t$、8 组温度 $\vartheta_{t,g}$；前 $T-1$ 层学习消息阻尼 $\beta_t$。

使用无约束实参数 $u$ 经过 Sigmoid 映射：

$$\kappa_t=\kappa_{min}+(\kappa_{max}-\kappa_{min})\sigma(u_{\kappa,t}),$$

$$\vartheta_{t,g}=\vartheta_{min}+(\vartheta_{max}-\vartheta_{min})\sigma(u_{\vartheta,t,g}),$$

$$\beta_t=\beta_{min}+(\beta_{max}-\beta_{min})\sigma(u_{\beta,t}).$$

固定边界和初始化：

| 参数 | 边界 | 初值 |
|---|---|---|
| $\kappa$ | $[10^{-3},10]$ | 1 |
| $\vartheta$ | $[0.5,4]$ | 1 |
| $\beta$ | $[0.05,1]$ | 0.5 |

端点在有限无约束值下不一定精确可达。初始化按

$$u_0=\operatorname{logit}\left(\frac{p_0-p_{min}}{p_{max}-p_{min}}\right)$$

计算，不把“raw 参数设为 0”误认为物理参数等于 1。

v1 参数总数：

$$P_{DU}=T(G+1)+(T-1)=12\times9+11=119.$$

最后一层没有需要继续使用的消息，不创建未参与损失的 $\beta_T$。这等价于原数学设计允许的最后一层提前结束，不是改变有效前向。

### 2.3 在线状态不是网络权重

每条样本独立维护

$$\mathfrak S_t=(r_{2,t},\gamma_{2,t},m_{t-1},v_{t-1},c_t,C_{c,t}).$$

$v1$ 中后两项不更新；增强变体中在线更新，但不存入全局模型权重，不跨样本传递。不能用前一条测试样本的 $c$ 初始化下一条。

## 3. 接收侧误差基与完整二阶矩

### 3.1 基底实现来源

文档 01 已规定 $B_q$ 来源于接收估计路径的复增益、时延和尺度局部变化。其基础算子为

$$
(\mathcal S_qu)(\tau)=b_q(\tau)
 u(\tau-\tau_q+\psi_q(\tau))e^{-j2\pi f_c[\tau_q-\psi_q(\tau)]},
\quad \psi_q(\tau)=\int_0^\tau a_q(u)du.
$$

频域离散表达为 $B_q=F W_{rx}T_q^{ch}T_{CP}F^H$，接收窗和 FFT 与数据生成完全一致。首版不重新学习任意稠密 $B_q$。

归一化 $B_q=B_q^{raw}/b_q^{norm}$ 时，必须同时令 $C_0=D_bC_0^{raw}D_b^H$，加上文档 01 已定义的正定项。网络不能再进行未记载的归一化。

### 3.2 符号矩

数据软均值 $m_{t-1}\in\mathbb C^D$，方差 $v_{t-1}\in\mathbb R^D$。构造

$$\overline m_{t-1}=E_dm_{t-1}+E_pp,$$

$$V_{t-1}=E_d\operatorname{diag}(v_{t-1})E_d^H,$$

$$\boxed{S_{t-1}=\overline m_{t-1}\overline m_{t-1}^H+V_{t-1}.}\tag{DU-01}$$

导频与空子载波方差均为 0，数据均值与方差满足 $|m_i|^2+v_i=1$，因此

$$\operatorname{tr}S_{t-1}=D+\|p\|^2.$$

不能把 $S$ 当成对角协方差。尤其在 $v=0$ 时，CSI 误差作用在确定符号上仍产生有效误差。

### 3.3 CSI 不确定性协方差

采用条件代理 $\delta c_t\sim\mathcal{CN}(0,C_{c,t})$，并与当前符号代理作因子化近似：

$$\boxed{U_t=\sum_{q,r}[C_{c,t}]_{qr}B_qS_{t-1}B_r^H.}\tag{DU-02}$$

v1 固定 $C_{c,t}=C_0$，但 $S_{t-1}$ 随层变化，所以 $U_t$ 仍然随层变化。

主模型不额外加入未经定义的基底外残差协方差，也不使用真实误差推导的逐测试样本修正。

### 3.4 保证半正定的等价实现

设 $C_{c,t}=L_{c,t}L_{c,t}^H$，定义

$$D_{\ell,t}=\sum_q[L_{c,t}]_{q\ell}B_q,\quad
\nu_{\ell,t}=D_{\ell,t}\overline m_{t-1}.$$

则

$$\boxed{U_t=\sum_{\ell=1}^{Q}\left[
\nu_{\ell,t}\nu_{\ell,t}^H+
D_{\ell,t}[:,\mathcal D]\operatorname{diag}(v_{t-1})D_{\ell,t}[:,\mathcal D]^H
\right].}\tag{DU-03}$$

允许用 `X = D_l[:, data_idx] * sqrt(v)[None, :]` 后累加 `X @ X.conj().T`。这是与完整 $S$ 等价的实现，不是近似对角化 $U$。$\nu$ 必须包含导频均值贡献。

输入的空子载波列可以省略，因为其均值和方差严格为零；接收行不能因而删掉。v1 中 $L_c,D_\ell$ 可在一个样本内预计算一次，相关时间计入该算法准备时间。

## 4. 初始化与数值约束

### 4.1 初始化

$$p_{0,i}(a)=1/4,\quad m_0=0,\quad v_0=1,$$

$$r_{2,1}=0,\quad\gamma_{2,1}=1,\quad c_1=0,\quad C_{c,1}=C_0.$$

初始完整二阶矩

$$S_0=E_ppp^HE_p^H+E_dE_d^H.$$

### 4.2 固定投影

$$\Pi_\alpha(a)=\operatorname{clip}(a;10^{-6},1-10^{-6}),$$

$$\Pi_\gamma(g)=\operatorname{clip}(g;10^{-8},10^8),\quad\epsilon_R=10^{-10}.$$

每次外信息必须执行：原始散度 → 投影散度 → 外信息均值 → 原始精度 → 仅投影精度。投影精度后自然参数为 $h=\gamma r$，$r$ 不重新缩放。

### 4.3 条件散度定义

对复映射 $g:\mathbb C^D\to\mathbb C^D$，

$$\alpha(g)=\frac1{2D}\operatorname{tr}\frac{\partial[\Re g;\Im g]}{\partial[\Re r;\Im r]}.$$

计算该模块的散度时，进入模块的 $H,R,\gamma,\vartheta$ 和跨层状态是条件常数；训练反向传播仍沿整个展开图传播，不能因此 `.detach()`。

## 5. 一层前向：与传统 VAMP 逐步对应

| 层内步骤 | 对应传统 VAMP | v1 的变化 |
|---|---|---|
| DU-01～04 | 建立线性似然 | 将 CSI 误差二阶统计加入有效协方差 |
| DU-05～07 | 线性 MMSE 模块与散度 | 结构保留，使用新的 $R_t$ |
| DU-08 | 线性到去噪的外信息 | 保留公式与投影顺序 |
| DU-09～11 | QPSK 先验去噪 | 增加分组温度，散度同步变化 |
| DU-12 | 去噪到线性的外信息 | 保留消息精度与散度关系 |
| DU-13 | 有限维迭代阻尼 | 学习自然参数阻尼系数 |
| RF-01～06 | 无对应必需步骤 | 仅增强变体加入结构化信道细化 |

### 5.1 有效协方差与导频均值消除

$$\boxed{R_t=\widehat R_n+\kappa_tU_t+\epsilon_RI_N.}\tag{DU-04}$$

$$A_t=H_tE_d,\qquad y_{d,t}=y-H_tE_pp.$$

v1 中 $H_t=\widehat H$。导频均值虽然已经减去，其通过信道误差产生的残差仍由完整 $S$ 进入 $U_t$。

### 5.2 精确线性估计

$$\boxed{K_t=A_t^HR_t^{-1}A_t+\gamma_{2,t}I_D,\qquad Q_t=K_t^{-1}.}\tag{DU-05}$$

$$\boxed{z_t=K_t^{-1}\left(A_t^HR_t^{-1}y_{d,t}+\gamma_{2,t}r_{2,t}\right).}\tag{DU-06}$$

代码先 Cholesky 白化 $R_t$，然后用 $K_t$ 的 Cholesky 解线性方程。$Q_t$ 不一定显式形成，但后续的迹必须精确对应同一个 $K_t$。

### 5.3 线性散度与外信息

$$\boxed{\alpha_{L,t}^{raw}=\frac{\gamma_{2,t}}D\operatorname{tr}(Q_t),\quad
\bar\alpha_{L,t}=\Pi_\alpha(\alpha_{L,t}^{raw}).}\tag{DU-07}$$

若 $K_t=L_{K,t}L_{K,t}^H$，精确迹为

$$\operatorname{tr}(Q_t)=\|L_{K,t}^{-1}\|_F^2.$$

$$\boxed{
 r_{1,t}=\frac{z_t-\bar\alpha_{L,t}r_{2,t}}{1-\bar\alpha_{L,t}},\quad
\gamma_{1,t}=\Pi_\gamma\left(\gamma_{2,t}\frac{1-\bar\alpha_{L,t}}{\bar\alpha_{L,t}}\right).
}\tag{DU-08}$$

不要先把 $\gamma_{1,t}$ 截断，再拿未截断的自然参数分子除以它。

### 5.4 分组温度与四类概率

$$\widetilde\gamma_{t,i}=\gamma_{1,t}/\vartheta_{t,g(i)}.$$

$$\boxed{
\ell_{t,i}(a)=-\widetilde\gamma_{t,i}|r_{1,t,i}-a|^2,\quad
\log p_{t,i}(a)=\ell_{t,i}(a)-\operatorname{LSE}_{a'}\ell_{t,i}(a').
}\tag{DU-09}$$

注意：$\widetilde\gamma$ 只用于去噪概率及其散度，模块的输入消息精度仍是 $\gamma_{1,t}$。

### 5.5 软均值、方差及闭式

$$\boxed{m_{t,i}=\sum_a a p_{t,i}(a),\quad v_{t,i}=1-|m_{t,i}|^2.}\tag{DU-10}$$

等价闭式为

$$m_{t,i}=\frac{\tanh(\sqrt2\widetilde\gamma_{t,i}\Re r_{1,t,i})
+j\tanh(\sqrt2\widetilde\gamma_{t,i}\Im r_{1,t,i})}{\sqrt2},$$

$$v_{t,i}=\frac12\left[\operatorname{sech}^2(\sqrt2\widetilde\gamma_{t,i}\Re r_{1,t,i})+
\operatorname{sech}^2(\sqrt2\widetilde\gamma_{t,i}\Im r_{1,t,i})\right].$$

首版概率求和与闭式必须交叉验证。中间层禁止硬判决。

### 5.6 温度一致的散度

$$\boxed{\alpha_{N,t}^{raw}=\frac1D\sum_i\widetilde\gamma_{t,i}v_{t,i}
=\frac{\gamma_{1,t}}D\sum_i\frac{v_{t,i}}{\vartheta_{t,g(i)}},\quad
\bar\alpha_{N,t}=\Pi_\alpha(\alpha_{N,t}^{raw}).}\tag{DU-11}$$

原始散度不预设小于 1。不能在温度改变后仍用 $\gamma_1\overline v$，也不能将消息输出精度直接设为 $1/\overline v$。v1 温度是层参数，不依赖当前 $r_{1,t}$，所以本式不缺少温度的链式导数。

### 5.7 去噪到线性模块的外信息

$$\boxed{
r_{2,t}^{*}=\frac{m_t-\bar\alpha_{N,t}r_{1,t}}{1-\bar\alpha_{N,t}},\quad
\gamma_{2,t}^{*}=\Pi_\gamma\left(\gamma_{1,t}\frac{1-\bar\alpha_{N,t}}{\bar\alpha_{N,t}}\right).
}\tag{DU-12}$$

### 5.8 自然参数阻尼

$$\boxed{\begin{aligned}
\gamma_{2,t+1}&=(1-\beta_t)\gamma_{2,t}+\beta_t\gamma_{2,t}^{*},\\
h_{2,t+1}&=(1-\beta_t)\gamma_{2,t}r_{2,t}+\beta_t\gamma_{2,t}^{*}r_{2,t}^{*},\\
r_{2,t+1}&=h_{2,t+1}/\gamma_{2,t+1}.
\end{aligned}}\tag{DU-13}$$

不能只平均两个 $r$；不能将两份精度不加权地相加。当 $t=T$ 时，不执行 DU-12/13，也不创建无用的状态更新；最后层散度可仅作为可选诊断计算。

## 6. 公式到函数的严格映射

| 公式 | 函数 | 必须返回 |
|---|---|---|
| DU-01 | `build_symbol_moments` | 完整均值、数据方差表示 |
| DU-02/03 | `csi_covariance_exact` | Hermitian $U_t$ |
| DU-04 | `effective_covariance` | 实际使用的 $R_t$ |
| DU-05/06/07 | `linear_module_exact` | $z_t$、精确迹、原始散度 |
| DU-08/12 | `extrinsic_with_projection` | 均值不变的投影消息 |
| DU-09/10/11 | `qpsk_temperature_denoiser` | logp、均值、方差、原始散度 |
| DU-13 | `damp_natural_parameters` | 新 $r_2,\gamma_2$ |
| RF-01～06 | `refine_channel_exact` | 增强版信道自然参数 |

共享外信息函数必须直接实现

```python
def extrinsic_with_projection(mean_out, mean_in, gamma_in, alpha_raw):
    alpha = clip(alpha_raw, alpha_eps, 1.0 - alpha_eps)
    mean_ext = (mean_out - alpha * mean_in) / (1.0 - alpha)
    gamma_raw = gamma_in * (1.0 - alpha) / alpha
    gamma_ext = clip(gamma_raw, gamma_min, gamma_max)
    return mean_ext, gamma_ext
```

该函数不能在内部归一化符号能量、截断均值或使用真实误差。

精确线性模块参考计算图：

```python
Lr = cholesky(hermitian(R))
Aw = solve_triangular(Lr, A, upper=False)
yw = solve_triangular(Lr, y_data[..., None], upper=False)[..., 0]
K = hermitian(Aw.mH @ Aw) + gamma * I_D
Lk = cholesky(K)
b = Aw.mH @ yw + gamma * r2
z = cholesky_solve(b[..., None], Lk)[..., 0]
Lk_inv = solve_triangular(Lk, I_D, upper=False)
trace_Q = (abs(Lk_inv) ** 2).sum()
alpha_raw = gamma * trace_Q / D
```

批处理时迹只沿最后两个矩阵轴求和，不得把 batch 轴也求和。`mH` 指共轭转置，不是普通转置。代码中的 `gamma` 必须按 batch 明确广播。

## 7. 完整 v1 前向调度

```python
def forward_v1(inp, params, return_layers=False):
    # inp 无标签；每个样本独立初始化
    r2, gamma2 = zeros(D, complex128), scalar(1.0)
    mean, var = zeros(D, complex128), ones(D, float64)
    H = inp.H_hat
    C = inp.C0
    layer_outputs = []
    static_factors = prepare_covariance_factors(inp.B, C)

    for t in range(1, T + 1):
        full_mean = scatter_data(mean) + scatter_pilots(inp.pilot_symbols)
        U = csi_covariance_exact(static_factors, full_mean, var, inp.data_idx)
        kappa, theta = constrained_parameters_for_layer(t)
        R = inp.R_n_hat + kappa * U + epsilon_R * I_N
        A = H[:, inp.data_idx]
        y_data = inp.y - H[:, inp.pilot_idx] @ inp.pilot_symbols

        z, trace_Q, alphaL = linear_module_exact(A, R, y_data, r2, gamma2)
        r1, gamma1 = extrinsic_with_projection(z, r2, gamma2, alphaL)
        logp, mean, var, alphaN = qpsk_temperature_denoiser(
            r1, gamma1, theta[inp.data_group]
        )
        llr = qpsk_llr_from_log_probs(logp)
        layer_outputs.append((logp, llr))  # 训练时保持计算图
        if t == T:
            break
        r2_new, gamma2_new = extrinsic_with_projection(mean, r1, gamma1, alphaN)
        beta = constrained_beta(t)
        r2, gamma2 = damp_natural_parameters(r2, gamma2, r2_new, gamma2_new, beta)

    return standardized_output(logp, mean, llr, diagnostics, layer_outputs)
```

这里的 `scatter_data`/`scatter_pilots` 使用输入的固定索引。生产实现可以减少重复构造 $A,y_d$，因为 v1 的 $H$ 不变；这是代数等价优化。$R_t$ 不固定，不能因此复用一次完整线性模块的 SVD。

## 8. 结构化信道细化：完整保留，独立版本启用

### 8.1 调度与可训练参数

增强版 `v1_refine` 使用 $t_0=4$、间隔 $J=2$，只在层 $t\in\{4,6,8,10\}$ 的去噪之后细化；最后一层不细化。固定

$$\lambda_t=0.01+(100-0.01)\sigma(u_{\lambda,t}),$$

$$\rho_t=s_t\,0.5\,\sigma(u_{\rho,t}),\quad s_t\in\{0,1\}.$$

初始化 $\lambda_t=1$、激活层 $\rho_t=0.1$。只为四个激活位置创建参数，共增加 8 个实自由度，总计 127。$C_0$、$B_q$ 仍不学习。

### 8.2 固定参考信道与目标

当前修正始终相对于输入 $\widehat H$：

$$H(c)=\widehat H+\sum_qc_qB_q,\quad W_n=(\widehat R_n+\epsilon_RI)^{-1}.$$

使用**当前层**符号矩 $\overline m_t,V_t,S_t$，最小化

$$\boxed{
J_t(c)=\|y-H(c)\overline m_t\|_{W_n}^2+
\operatorname{tr}(W_nH(c)V_tH(c)^H)+\lambda_t c^HC_0^{-1}c.
}\tag{RF-01}$$

此处不能把 $W_n$ 换成 $R_t^{-1}$，因为同一组系数已被显式估计，不能再把其不确定性重复当成独立噪声。不能仅保留第一项，把软符号均值当真实符号。

### 8.3 精确二次型

$$\boxed{[G_t]_{qr}=\operatorname{tr}(B_q^HW_nB_rS_t).}\tag{RF-02}$$

$$\boxed{[d_t]_q=(B_q\overline m_t)^HW_ny-
\operatorname{tr}(B_q^HW_n\widehat H S_t).}\tag{RF-03}$$

等价高效实现：

$$[G_t]_{qr}=(B_q\overline m_t)^HW_n(B_r\overline m_t)
+\sum_{i\in\mathcal D}v_{t,i}\,B_q[:,k_i]^HW_nB_r[:,k_i],$$

$$[d_t]_q=(B_q\overline m_t)^HW_n(y-\widehat H\overline m_t)
-\sum_{i\in\mathcal D}v_{t,i}\,B_q[:,k_i]^HW_n\widehat H[:,k_i].$$

形成

$$P_t=\operatorname{Herm}(G_t)+\lambda_tC_0^{-1}.$$

$$\boxed{c_t^*=P_t^{-1}d_t,\qquad C_t^*=P_t^{-1}.}\tag{RF-04}$$

复高斯约定下不再额外乘 2。求解使用小规模 Cholesky；目标和正规方程必须满足 $\partial J/\partial c^*=P_tc-d_t$。

### 8.4 信道自然参数阻尼

令 $\Lambda_{c,t}=C_{c,t}^{-1}$、$\xi_{c,t}=\Lambda_{c,t}c_t$：

$$\boxed{\begin{aligned}
\Lambda_{c,t+1}&=(1-\rho_t)\Lambda_{c,t}+\rho_tP_t,\\
\xi_{c,t+1}&=(1-\rho_t)\xi_{c,t}+\rho_td_t,\\
C_{c,t+1}&=\Lambda_{c,t+1}^{-1},\\
c_{t+1}&=\Lambda_{c,t+1}^{-1}\xi_{c,t+1}.
\end{aligned}}\tag{RF-05}$$

$$\boxed{H_{t+1}=\widehat H+\sum_qc_{q,t+1}B_q.}\tag{RF-06}$$

不得改为 $\Lambda_{c,t+1}=\Lambda_{c,t}+G_t$：各层仍用同一接收块，不是新增独立观测。$c_t$ 是总修正，不能在已修正 $H_t$ 上再次累加同一个总修正。

调度未激活时，$c,C_c$ 原样传递。下一层用新的 $C_c$ 和当前符号 $S_t$ 重建 $U_{t+1}$。该模块对固定当前分布的二次目标有解析最优解，不代表整个网络的训练损失逐层单调。

## 9. 最终输出与损失函数

### 9.1 推理输出

最终保存 `log_probs=log p_T`、`x_soft=m_T`、`llr_raw=L_T`。LLR 满足

$$L_{T,i,j}=\log\frac{\sum_{a:b_j(a)=0}p_{T,i}(a)}{\sum_{a:b_j(a)=1}p_{T,i}(a)}.$$

对本模型还可核对

$$L_{t,i,1}=2\sqrt2\widetilde\gamma_{t,i}\Re r_{1,t,i},\quad
L_{t,i,2}=2\sqrt2\widetilde\gamma_{t,i}\Im r_{1,t,i}.$$

若译码器需要有限幅度，输出适配层单独生成 $L_{out}=\operatorname{clip}(L_{raw};-L_{max},L_{max})$；首版保存未截断的 logp/LLR，后续改变译码截断不需要重跑均衡器。截断不能写回层内概率、方差或散度。

增强版可额外保存最后实际使用的 $c_T,C_{c,T}$ 与 $H_T$ 的紧凑表示，用于信道细化诊断。v1 的输出信道就是输入估计，不宣称信道 NMSE 被改善。

### 9.2 逐层比特损失

固定层权重

$$w_t=\frac{1.2^{t-1}}{\sum_{s=1}^{T}1.2^{s-1}}.$$

对训练标签 $b_{i,j}\in\{0,1\}$，

$$\boxed{\mathcal L_{DU}=-\sum_{t=1}^{T}\frac{w_t}{2D}
\sum_{i=1}^{D}\sum_{j=1}^{2}\log P_{t,i,j}(b_{i,j}).}\tag{LOSS-01}$$

批量损失按真实样本数平均。等价稳定实现：

$$\boxed{\mathcal L_{DU}=\sum_t\frac{w_t}{2D}\sum_{i,j}
\operatorname{softplus}((2b_{i,j}-1)L_{t,i,j}).}\tag{LOSS-02}$$

采用 `BCEWithLogitsLoss` 时，输入应为 `-L`、标签为比特 $b$，因为本项目正 LLR 表示比特 0。直接把 `L` 与 $b$ 传入会反转标签意义。

首版不加符号 MSE 辅助损失、不加真实信道监督、不加不明正则项。增加任何损失需发布训练配置版本；若最终权重变化，则产生新的预测身份。

## 10. 反向传播与训练过程

### 10.1 梯度路径

训练梯度穿过：参数约束、$U_t$、$R_t$、Cholesky/求解、外信息、QPSK 概率、方差、下一层状态。增强版还穿过 RF-01～06。

对于 $Kz=b$，

$$dz=K^{-1}(db-(dK)z).$$

对于 $C=\Lambda^{-1}$，

$$dC=-C(d\Lambda)C.$$

自动微分必须实现对应求解梯度；不能为了避免二阶路径就把 `trace_Q`、`var`、`U`、`R` 或信道细化输出 detach。条件散度是局部映射中的数学量，与训练总梯度是不同对象。

硬投影按分段导数计算，饱和区导数为 0；不能暗中使用 straight-through 梯度。统计日志可以复制脱离图的数值，但不能让脱离图的值回到递推中。

### 10.2 固定训练配置

```yaml
algorithm_name: vamp_du
algorithm_version: v1
math_model_version: vamp_du_math_v1
variant: fixed_csi_core
layers: 12
groups: 8
parameterization: per_layer_bounded_sigmoid
conditional_controller: false
learn_C0: false
learn_B: false
channel_refinement: false
linear_solver: cholesky_exact
trace_method: inverse_cholesky_frobenius_exact
compute_dtype: complex128
real_parameter_dtype: float64
alpha_epsilon: 1.0e-6
gamma_min: 1.0e-8
gamma_max: 1.0e8
noise_floor: 1.0e-10
loss:
  name: layer_weighted_bit_ce
  layer_weight_ratio: 1.2
  use_unclipped_llr: true
train:
  seed: 41001
  optimizer: Adam
  learning_rate: 0.0003
  betas: [0.9, 0.999]
  eps: 1.0e-8
  weight_decay: 0.0
  micro_batch_samples: 1
  gradient_accumulation_steps: 8
  max_epochs: 150
  gradient_clip_norm: 5.0
  early_stop_patience_epochs: 20
  selection_metric: validation_bit_cross_entropy
  mixed_precision: false
  stochastic_trace: false
  test_access: forbidden
```

基础验证使用 complex128/float64。内存不足时先用独立 debug 尺寸完成公式验证；正式训练可以梯度累积和确定性 activation checkpointing，但不得改用近似矩阵、减少实际层数或截断梯度后仍声称相同配置。训练反向的确定性重计算不是正式测试重测，不受测试结果复用规则禁止。

### 10.3 训练循环

```text
验证封存 train/val manifest → 固定 seed 与环境
→ 初始化 119 个实参数 → 固定采样顺序生成器
→ 完整前向，计算所有层 LOSS-02 → 反向传播
→ 按有效累积样本数归一化梯度 → 全局范数裁剪 → Adam 更新
→ 每个 epoch 在固定验证集上评价，不读取正式 test
→ 保存 best/last、优化器和 RNG 状态
→ 按预先规定的验证 bit CE 选择一个不可变 checkpoint
→ 冻结 inference 配置与权重摘要 → 注册算法版本
```

最后不足一个累积批的样本，应按实际累积数量归一化，不固定除以 8。验证损失相同则选择更早 epoch，不能用测试集打破并列。

训练记录 $\kappa,\vartheta,\beta$ 的范围、alpha/gamma 投影频率、梯度范数、各层概率有限性和验证指标。Cholesky 失败不得静默加入越来越大的 jitter；应定位到具体样本和实际矩阵。若要增加稳定项，必须明确修改数学/配置并重新进行一致性测试。

## 11. 推理过程及结果身份

推理使用冻结 checkpoint、`eval()` 模式和无梯度上下文；每个测试样本状态从第 4 节重新初始化。主版本无 dropout、随机迹估计或测试时优化权重。增强版的在线 $c$ 更新是既定 forward，不改变训练权重。

checkpoint 至少绑定：算法/数学版本、variant、层数、分组、参数边界、精度投影、训练/验证内容摘要、源码依赖、权重摘要和环境。正式 `model_checkpoint` 必须解析为不可变权重，不使用可变路径作为唯一身份。

修改 DU 的任一有效层公式、温度/阻尼边界、损失导致的权重、分组规则或输入解释器时，DU 预测身份需更新。仅修改本说明文档排版不影响任何预测身份。未修改的 LMMSE、传统 VAMP 与 UDNet 不应因此重新评估。

## 12. 数学一致性与梯度验收

### 12.1 必须逐项验证的恒等式

| ID | 目标 | 方法与通过条件 |
|---|---|---|
| DU-T01 | QPSK 能量 | $|m_i|^2+v_i=1$，误差 ≤ 1e-12 |
| DU-T02 | 完整矩 | $\operatorname{tr}S=D+\|p\|^2$，含导频 |
| DU-T03 | 协方差等价 | DU-02 双重和 = DU-03，随机小矩阵相对误差 ≤ 1e-10 |
| DU-T04 | PSD | Hermitian 误差 ≤ 1e-10，特征值仅允许舍入级负值 |
| DU-T05 | $v=0$ | 非零均值/非零 CSI 误差仍可得到非零 $U$ |
| DU-T06 | 线性均值 | Cholesky 与小矩阵直接逆一致 |
| DU-T07 | 精确迹 | $\|L_K^{-1}\|_F^2=\operatorname{tr}(K^{-1})$ |
| DU-T08 | 线性散度 | 固定条件后实雅可比迹与 DU-07 一致 |
| DU-T09 | QPSK 三表达 | Softmax 均值、tanh、LLR 闭式一致 |
| DU-T10 | 温度散度 | 多个温度下有限差分与 DU-11 一致 |
| DU-T11 | 投影顺序 | gamma 截断前后外信息均值不变 |
| DU-T12 | 消息阻尼 | 手动设 beta=0/1 时满足端点等式 |
| DU-T13 | RF 驻点 | $P_tc^*=d_t$，Wirtinger 梯度为零 |
| DU-T14 | RF 目标展开 | $J(c)-J(0)=c^HPc-2\Re(c^Hd)$ |
| DU-T15 | RF 阻尼 | rho=0/1 的均值、协方差端点正确 |
| DU-T16 | 反向梯度 | 小维度非饱和随机样本 gradcheck/中心差分，相对误差约 ≤ 1e-4 |
| DU-T17 | 无学习退化 | 关闭 U/细化、温度=1、同 beta/T/数值规则，与传统 VAMP 逐层对齐 |
| DU-T18 | last layer | 不做未使用更新与做完后丢弃更新，输出完全相同 |
| DU-T19 | 防泄漏 | 移除全部 truth 后 inference 正常 |
| DU-T20 | 缓存隔离 | 只改 DU 版本，其他三个 evaluator 调用次数为 0 |

端点测试通过直接向数学函数传 beta/rho，不能要求有界 Sigmoid 参数在有限 raw 值下精确达到端点。梯度检查避免投影折点；折点的分段规则另做单元测试。

### 12.2 逐层数值基准

为 $N=16,D=8,Q=2,T=3$ 的独立开发样本保存黄金中间状态：$U,R,z,\alpha_L,r_1,\gamma_1,p,m,v,\alpha_N,r_2,\gamma_2$；增强版再保存 $G,d,P,c,C_c$。黄金生成器使用最直观的双重求和和直接小矩阵运算，生产实现采用等价优化。两者不能调用同一个高层函数，否则可能共同掩盖同一错误。

黄金样本与正式测试数据分开存放。开发单测的反复执行不是正式实验重测。

## 13. 复杂度、内存与允许的精确优化

DU 参数少，但前向不一定便宜。按稠密实现，每层 $U$ 的构造约 $O(QN^2D+QN^2)$，$R$ 分解 $O(N^3)$，白化 $O(N^2D)$，形成 Gram $O(ND^2)$，$K$ 分解及精确迹 $O(D^3)$。固定 CSI 不意味着固定有效协方差，不能假定只分解一次。

允许的首版优化：忽略严格为零的空发射列；按数据列分块累加 DU-03；样本内复用固定 $D_\ell$；通过同一 Cholesky 同时求均值与迹；按 $Q$ 分桶；训练时梯度累积/确定性 activation checkpointing。

禁止的首版优化：带宽截断、对角 $U$、随机迹、提前结束迭代、只回传末几层、用单一全局温度代替 8 组、训练时准确而测试时粗略求解。实现这些优化时另设数学变体、进行对应散度推导和缓存版本管理。

## 14. 模块目录与执行命令

```text
src/vampnet/algorithms/vamp_du/
├── model.py                   # v1 / v1_refine 的显式注册
├── state.py                   # 每样本状态，无持久学习状态
├── parameters.py              # 有界参数化与初始化
├── covariance.py              # DU-01～04
├── linear.py                  # DU-05～08，精确迹
├── denoiser.py                # DU-09～11
├── extrinsic.py               # DU-08/12 与投影顺序
├── damping.py                 # DU-13
├── refinement.py              # RF-01～06，v1 关闭
├── losses.py                  # LOSS-01/02
└── manifest.py                # 推理依赖与模型结构签名
```

```bash
# 数学验收与训练
python -m vampnet.cli.verify_math --algorithm vamp_du --variant fixed_csi_core
python -m vampnet.cli.train --algorithm vamp_du --config configs/training/vamp_du_v1.yaml

# 只测试 DU；已有有效缓存则不执行 forward
python -m vampnet.cli.evaluate --algorithm vamp_du --version v1 --suite configs/evaluation/accuracy_v1.yaml

# 修改后注册 v2，并只评价 v2
python -m vampnet.cli.evaluate --algorithm vamp_du --version v2 --suite configs/evaluation/accuracy_v1.yaml

# 比较从文档 04 的独立只读模块执行，不调用本模块
python -m vampnet.cli.compare --spec configs/comparison/v2_vs_baselines.yaml
```

## 15. 实施完成条件与依据

只有公式编号与函数映射、黄金中间状态、独立散度检查、梯度检查、防泄漏测试、不可变 checkpoint 和结果缓存隔离全部通过，才允许将实现标记为 `v1` 正式算法。训练 loss 下降或某条 BER 曲线较好不能代替这些检查。

- [M0] 本项目原数学文件 `VAMP_DeepUnfolding_Mathematical_Design.md`，SHA-256 见第 0 节。对应来源章节：§4 二阶矩/不确定性、§7 投影、§8–10 VAMP 主干、§11 信道细化、§12–13 输出与训练、§15 两种实现、§16–17 数值与一致性。
- [M1] Rangan, Schniter, Fletcher, *Vector Approximate Message Passing*：<https://arxiv.org/abs/1610.03082>。作为传统双模块结构依据；本文的水声不确定性、温度和细化递推属于本项目规定的设计。
