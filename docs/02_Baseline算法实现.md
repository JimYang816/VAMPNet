# VAMPNet｜执行文档 02｜LMMSE、传统 VAMP 与 UDNet baseline

> 项目名称：**VAMPNet**。
> 文档版本：1.0。本文给出待实现的数学、接口、配置与验收顺序。
> 统一输入来自 [文档 01](01_仿真数据生成.md)，单算法测试与缓存遵守 [文档 04](04_统一评估与增量重测.md)。
> 禁止在 baseline 内生成信道、添加新噪声、重新估计共同 CSI 或读取真实信道。三个 baseline 各自是独立可测试、独立版本化的模块。

## 0. 复现范围与固定原则

LMMSE 使用完整频域矩阵，不用单抽头方法替代主 baseline。传统 VAMP 使用完整矩阵、QPSK 先验及正确外信息更新，不退化为普通线性迭代。UDNet 按原论文式 (13)–(19) 的分类展开结构实现，不用 DetNet、OAMPNet、任意 MLP 或通用残差网络冒名替代。[B1][B2]

**UDNet 的公开论文给出了层主公式、分类目标、滑动结构及跨层连接，但没有唯一确定隐藏宽度、残差系数、概率残差归一化和所有边界处理。**本文件将这些位置标为“本项目复现约定”，冻结成 `paper_structure_project_spec_v1`。这是可执行的论文结构复现，不声称逐位复现作者未提供的实现细节。若以后获得作者实现，建立新版本并记录差异，不覆盖现有结果。

所有超参数只在训练/验证集确定。正式测试开始前发布 `algorithm_manifest.json` 和 `selection_manifest.json`；不得看过测试 BER 后修改 baseline 参数却仍使用旧版本名。

## 1. 共同输入、输出与数学约定

### 1.1 观测与导频消除

设 $N$ 为 FFT 点数，$D$ 为数据子载波数，

$$
\mathbf y=\mathbf H_{true}(\mathbf E_d\mathbf x+\mathbf E_p\mathbf p)+\mathbf n.
$$

baseline 只获得 $\widehat{\mathbf H}$ 和 $\widehat{\mathbf R}_n$。构造

$$
\mathbf A=\widehat{\mathbf H}\mathbf E_d\in\mathbb C^{N\times D},\qquad
\mathbf y_d=\mathbf y-\widehat{\mathbf H}\mathbf E_p\mathbf p.
$$

LMMSE 与 VAMP 保留所有 $N$ 个接收行，包括导频和空子载波所在的接收位置。UDNet 的滑动截取是其内部结构，不改变存盘输入。

固定数值噪声矩阵

$$\mathbf R=\widehat{\mathbf R}_n+\epsilon_R\mathbf I_N,\qquad\epsilon_R=10^{-10}.$$

主数据为白噪声，故 $R$ 为标量乘单位阵。初始符号能量为 1。复高斯精度 $\gamma$ 满足 $E|e|^2=1/\gamma$，不是实部方差的倒数。

### 1.2 输出契约

```python
@dataclass(frozen=True)
class EqualizerOutputV1:
    sample_id: str
    log_probs: Array        # float64 [D, 4]，规范类别顺序，logsumexp=0
    x_soft: Array           # complex128 [D]，sum(exp(log_probs) * constellation)
    llr_raw: Array          # float64 [D, 2]，log P(bit=0)/P(bit=1)
    x_native: Array | None  # 算法原生连续输出；LMMSE 为线性均值
    diagnostics: Mapping   # 迭代数、投影次数、有限性、状态码等
```

主统计统一从 `log_probs` 生成，避免每种算法各自使用不同解调器。`x_soft` 统一为星座概率均值，主 `NMSE_x_soft` 的含义因此相同；LMMSE 原始线性均值另外保存为 `x_native`，允许单独报告 `NMSE_x_native`，不可把二者混在同一列。

令 $\mathcal A=[a_{00},a_{01},a_{10},a_{11}]$，则

$$
L_{i,j}=\operatorname{LSE}_{a:b_j(a)=0}\ell_i(a)
-\operatorname{LSE}_{a:b_j(a)=1}\ell_i(a).
$$

SER 默认使用最大符号概率；BER 默认使用比特边缘最大后验，即 $\widehat b_{i,j}=\mathbf1\{L_{i,j}<0\}$。UDNet 的符号 MAP 与逐比特 MAP 不必一致，评估器必须按文档 04 区分。并列取最低类别编号；LLR 等于 0 时取比特 0。

### 1.3 软件边界

```text
algorithms/lmmse.py         仅含 LMMSE forward
algorithms/vamp.py          仅含传统 VAMP forward
algorithms/udnet.py         仅含 UDNet 和窗口适配器
algorithms/common/qpsk.py   概率、LLR、类别映射
algorithms/common/complex_linalg.py 精确求解、白化、散度
training/                  训练与 checkpoint 管理
evaluation/               由文档 04 单独调用某个算法
comparison/               不导入上述算法和训练模块
```

以上代码片段与命令均为待实施契约，不表示本交付包含可运行工程。

## 2. LMMSE：完整矩阵基准

### 2.1 线性估计公式

采用单位方差高斯代理先验 $\mathbf x\sim\mathcal{CN}(0,I_D)$：

$$
\mathbf K=\mathbf A^H\mathbf R^{-1}\mathbf A+\mathbf I_D,
\qquad \mathbf Q=\mathbf K^{-1},
$$

$$
\boxed{\mathbf z_L=\mathbf K^{-1}\mathbf A^H\mathbf R^{-1}\mathbf y_d.}
$$

等价地，$z_L$ 最小化 $\|y_d-Az\|_{R^{-1}}^2+\|z\|^2$。不使用真实 CSI 误差协方差，不学习正则化参数。矩阵 $R$ 来自接收端噪声估计。

### 2.2 求解实现

用 $R=L_RL_R^H$ 白化：

$$\widetilde A=L_R^{-1}A,\quad\widetilde y=L_R^{-1}y_d.$$

然后形成 $K=\widetilde A^H\widetilde A+I$，Cholesky 分解并求解 $Kz_L=\widetilde A^H\widetilde y$。需要软输出时，通过同一分解求取 $q_i=Q_{ii}$。不要求调用显式 `inv`，但求解和对角方差必须来自同一个 $K$。

### 2.3 统一软解调：去偏后的高斯代理

LMMSE 线性均值包含高斯先验收缩。令

$$
d_i=1-q_i,\qquad r_i=\frac{z_{L,i}}{d_i},\qquad
\tau_i=\frac{q_i}{d_i}.
$$

在该模型下 $d_i$ 是该分量的等效线性增益，$\tau_i$ 是去偏后高斯代理误差方差。以

$$
\ell_i(a)=-\frac{|r_i-a|^2}{\tau_i},\quad
\log p_i(a)=\ell_i(a)-\operatorname{LSE}_{a'}\ell_i(a')
$$

生成四类概率。它是 LMMSE 后的固定星座软解调，不是额外训练的均衡网络，也不代表真实非高斯残差的精确后验。

数值约定：仅在 $d_i>10^{-10}$ 时计算去偏式；$d_i\le10^{-10}$ 视为无有效观测，设 $p_i(a)=1/4$。仅因舍入越界时把 $q_i$ 投影到 $[10^{-12},1]$，并记录次数。不能把 $q_i$ 随意调小来制造更大的 LLR。

输出 `x_native=z_L`，`x_soft=sum_a p_i(a)a`，`llr_raw` 按公共公式生成。

### 2.4 配置

```yaml
algorithm_name: lmmse
algorithm_version: v1
implementation_flavor: full_ici_lmmse_with_fixed_qpsk_demapper
input_schema: EqualizerInputV1
output_schema: EqualizerOutputV1
compute_dtype: complex128
linear_solver: cholesky_exact
noise_floor: 1.0e-10
symbol_prior_variance: 1.0
demapper: debiased_gaussian_qpsk
no_information_gain_threshold: 1.0e-10
model_checkpoint: null
trainable_parameters: 0
```

### 2.5 验收

单位信道下核对线性均值与软解调解析值；随机小矩阵下核对发射域与接收域两种 LMMSE 公式；零矩阵下输出均匀概率；使用完整非对角矩阵验证不是单抽头实现；原生输出与统一软输出分别核验。

## 3. 传统 VAMP：无训练参数的 QPSK 双模块递推

以下沿用原始 VAMP 的线性估计、可分离去噪、平均散度和外信息交换结构，并明确复数尺度、有限精度投影及阻尼。[B1]

### 3.1 初始化与投影

$$r_{2,1}=0,\quad\gamma_{2,1}=1,\quad m_0=0,\quad v_0=1.$$

定义

$$
\Pi_\alpha(a)=\operatorname{clip}(a;10^{-6},1-10^{-6}),\quad
\Pi_\gamma(g)=\operatorname{clip}(g;10^{-8},10^8).
$$

重要顺序：先用投影后的散度计算外信息均值，再投影精度，**保持已计算的均值不变**。不能把未投影精度的自然参数分子直接除以投影后的精度。

### 3.2 线性模块

第 $t$ 次迭代：

$$
K_t=A^HR^{-1}A+\gamma_{2,t}I_D,\qquad Q_t=K_t^{-1},
$$

$$z_t=K_t^{-1}(A^HR^{-1}y_d+\gamma_{2,t}r_{2,t}).$$

条件散度为

$$\alpha_{L,t}^{raw}=\frac{\gamma_{2,t}}D\operatorname{tr}(Q_t),\quad
\bar\alpha_{L,t}=\Pi_\alpha(\alpha_{L,t}^{raw}).$$

### 3.3 线性到 QPSK 的外信息

$$
r_{1,t}=\frac{z_t-\bar\alpha_{L,t}r_{2,t}}{1-\bar\alpha_{L,t}},\quad
\gamma_{1,t}=\Pi_\gamma\left(\gamma_{2,t}\frac{1-\bar\alpha_{L,t}}{\bar\alpha_{L,t}}\right).
$$

### 3.4 QPSK 去噪及散度

$$
\ell_{t,i}(a)=-\gamma_{1,t}|r_{1,t,i}-a|^2,\quad
\log p_{t,i}(a)=\ell_{t,i}(a)-\operatorname{LSE}_{a'}\ell_{t,i}(a'),
$$

$$m_{t,i}=\sum_a a p_{t,i}(a),\quad v_{t,i}=1-|m_{t,i}|^2.$$

使用复精度约定时：

$$
m_{t,i}=\frac{\tanh(\sqrt2\gamma_{1,t}\Re r_{1,t,i})
+j\tanh(\sqrt2\gamma_{1,t}\Im r_{1,t,i})}{\sqrt2},
$$

$$\alpha_{N,t}^{raw}=\frac{\gamma_{1,t}}D\sum_i v_{t,i},\quad
\bar\alpha_{N,t}=\Pi_\alpha(\alpha_{N,t}^{raw}).$$

只对舍入误差导致的方差越界进行 $[0,1]$ 截断；不能以另一个方差估计替代散度中的真实去噪器导数。

### 3.5 QPSK 到线性模块的外信息

$$r_{2,t}^{*}=\frac{m_t-\bar\alpha_{N,t}r_{1,t}}{1-\bar\alpha_{N,t}},$$

$$\gamma_{2,t}^{*}=\Pi_\gamma\left(\gamma_{1,t}\frac{1-\bar\alpha_{N,t}}{\bar\alpha_{N,t}}\right).$$

### 3.6 自然参数阻尼

固定非训练参数 $\beta\in(0,1]$：

$$\gamma_{2,t+1}=(1-\beta)\gamma_{2,t}+\beta\gamma_{2,t}^{*},$$

$$h_{2,t+1}=(1-\beta)\gamma_{2,t}r_{2,t}+\beta\gamma_{2,t}^{*}r_{2,t}^{*},$$

$$r_{2,t+1}=h_{2,t+1}/\gamma_{2,t+1}.$$

最后一次去噪输出后直接结束，不计算不会使用的下一轮消息。每个样本初始化独立，禁止让上一个测试样本的状态影响下一个样本。

### 3.7 固定矩阵下的精确加速

这里 $A,R$ 在迭代间固定。可预先计算

$$A^HR^{-1}A=V\operatorname{diag}(d_1,\ldots,d_D)V^H,\quad b=A^HR^{-1}y_d.$$

于是

$$z_t=V\left[\frac{V^Hb+\gamma_{2,t}V^Hr_{2,t}}{d+\gamma_{2,t}}\right],\quad
\operatorname{tr}(Q_t)=\sum_i\frac1{d_i+\gamma_{2,t}}.$$

这是精确代数重写，允许作为正式实现。小矩阵必须逐层与 Cholesky 参考实现对齐。特征分解属于该样本的算法准备时间，不可在运行时间统计时偷偷排除。

禁止用少量 CG 迭代替代精确求解却继续声称散度严格相同；近似求解器应另设版本并重新定义散度。

### 3.8 配置及验证集选择

```yaml
algorithm_name: vamp
algorithm_version: v1
implementation_flavor: complex_qpsk_vamp_exact
compute_dtype: complex128
iterations: 40
linear_solver: fixed_gram_eigh_exact
beta: 0.5
gamma_init: 1.0
gamma_min: 1.0e-8
gamma_max: 1.0e8
alpha_epsilon: 1.0e-6
noise_floor: 1.0e-10
early_stop: false
trainable_parameters: 0
model_checkpoint: null
```

默认值是开发起点。正式锁定前允许在验证集搜索 `iterations ∈ {20,40,80}`、`beta ∈ {0.3,0.5,0.8,1.0}`，记录所有候选及选择规则：验证集平均比特交叉熵优先，指标相同则选更少迭代，再选固定字典序。不得在各个测试 SNR 上事后选择不同最佳参数。

建议另报告 12 次迭代的对照作为等层数补充，但它不替代经过合理验证选择的主 VAMP baseline。等层数也不代表等计算量。

## 4. UDNet：分类展开与滑动结构

### 4.1 论文给出的核心与本项目补全

原论文式 (13)、(14) 使用当前符号估计、$H^TY$、$H^THS$ 三类输入，经仿射、Tanh、仿射和逐符号 Softmax 得到星座类别概率；初始符号来自 ZF，层间重新映射成符号，训练使用分类损失，另有跨层连接和滑动处理。[B2]

下列必须实现：三个输入分支、两个层级映射、四类概率、可微软星座回映射、ZF 初值、跨层状态和固定窗口重组。以下隐藏宽度、残差权重、掩码与窗口边界是本项目明确约定，不冒充原论文唯一公式。

### 4.2 窗口与已知符号适配

取窗口宽度 $W$，主实现无重叠，$W\in\{32,64,128\}$ 且整除 $N$。按 FFT 自然顺序分块

$$\mathcal J_j=\{jW,\ldots,(j+1)W-1\},\quad j=0,\ldots,N/W-1.$$

先全局消除已知导频 $y_d=y-\widehat H E_pp$，再取窗口。令 $m_j\in\{0,1\}^W$ 表示哪些发射位置为未知数据，定义

$$H_j=\widehat H[\mathcal J_j,\mathcal J_j]\operatorname{diag}(m_j),\quad
Y_j=y_d[\mathcal J_j].$$

主数据为白噪声，可将二者同时除以 $\sqrt{\widehat\sigma_n^2+\epsilon_R}$。非数据位置不参与损失，软符号回映射后强制为 0；因为导频贡献已经消除，此处不能再次回填导频。

窗口外未知数据的 ICI 没有从真实观测中消失，UDNet 按其局部结构将其视为未显式建模干扰。禁止用真实发送符号把窗口外干扰先减掉。记录窗口外矩阵能量作为解释指标，而不是按测试表现选择窗口。

若未来采用重叠窗口、保留中心区、窗口外软消除或彩色噪声局部白化，应新建 `implementation_flavor` 与算法版本。主结果必须明确是“完整观测输入、局部窗口结构”的 UDNet，而不是完整矩阵网络。

### 4.3 复数转实数

$$
\mathcal R(H_j)=\begin{bmatrix}\Re H_j&-\Im H_j\\\Im H_j&\Re H_j\end{bmatrix}
\equiv\widetilde H_j\in\mathbb R^{2W\times2W},
$$

$$\widetilde Y_j=[\Re Y_j;\Im Y_j]\in\mathbb R^{2W}.$$

预先计算

$$b_j=\widetilde H_j^T\widetilde Y_j,\qquad G_j=\widetilde H_j^T\widetilde H_j.$$

每个复符号能量仍为 1，不改成未归一化的 $\{\pm1\pm j\}$。一旦更换幅度约定，必须对信号、矩阵、标签和先验统一变换。

### 4.4 ZF 初始化

仅对窗口中的有效数据列 $H_{j,D}$ 求 Moore–Penrose 伪逆：

$$z_{j,0}=H_{j,D}^{\dagger}Y_j.$$

SVD 奇异值截断固定为 `rcond=1e-10`；再把有效结果填入长度 $W$ 的复向量，其余为零，转成 $s_{j,0}\in\mathbb R^{2W}$。这是数值定义明确的 ZF，不允许默默替换为 LMMSE。没有数据的窗口直接跳过，并在计时中采用相同规则。

### 4.5 第 $m$ 层完整公式

设总层数 $M=22$，隐藏维度 $h=8W$。每层、每个窗口使用同一组权重；不同层权重不共享。

$$
f_{j,m}=\begin{bmatrix}s_{j,m-1}\\\lambda_{m,1}b_j\\
\lambda_{m,2}G_js_{j,m-1}\end{bmatrix}\in\mathbb R^{6W},
$$

$$\widetilde v_{j,m}=\tanh(W_{m,1}f_{j,m}+b_{m,1}).$$

本项目固定 Tanh 输出的残差为

$$
v_{j,m}=\begin{cases}
\widetilde v_{j,1},&m=1,\\
(1-r_v)\widetilde v_{j,m}+r_vv_{j,m-1},&m>1,
\end{cases}\qquad r_v=0.5.
$$

计算 $4W$ 个 logits，按 `[W,4]` reshape：

$$\widetilde q_{j,m,i,:}=\operatorname{softmax}_4
\big(\operatorname{reshape}(W_{m,2}v_{j,m}+b_{m,2})_{i,:}\big).$$

本项目固定概率残差为

$$q_{j,m}=\begin{cases}
\widetilde q_{j,1},&m=1,\\
(1-r_q)\widetilde q_{j,m}+r_qq_{j,m-1},&m>1,
\end{cases}\qquad r_q=0.5.$$

这个凸组合保证每个符号的四类概率和为 1。不能直接把两组概率相加而不归一化，也不能对整个 $4W$ 维向量只做一次 Softmax。

层间软回映射：

$$z_{j,m,i}=m_{j,i}\sum_{c=0}^{3}a_cq_{j,m,i,c},\qquad
s_{j,m}=[\Re z_{j,m};\Im z_{j,m}].$$

中间层不使用 argmax、硬 QPSK 或 `.detach()`，否则训练图被改变。所有窗口最终只提取 `data_idx` 对应概率并恢复全局顺序。每个数据符号恰好输出一次。

为避免概率下溢，保存 `log_softmax`，概率残差用

$$\log q=\operatorname{logaddexp}\big(\log(1-r_q)+\log\widetilde q,\log r_q+\log q_{prev}\big)$$

等价实现。均值用其指数计算。

### 4.6 参数维度与可训练量

| 参数 | 维度 | 是否训练 |
|---|---|---|
| $W_{m,1}$ | $h\times6W$ | 是 |
| $b_{m,1}$ | $h$ | 是 |
| $W_{m,2}$ | $4W\times h$ | 是 |
| $b_{m,2}$ | $4W$ | 是 |
| $\lambda_{m,1},\lambda_{m,2}$ | 两个实标量/层 | 是，初值均为 1 |
| $r_v,r_q$ | 实标量 | 否，固定 0.5 |
| $W,M,h$、窗口/类别规则 | 配置 | 否 |
| $H_j,b_j,G_j$ | 样本相关 | 否，不是权重 |

$\lambda$ 的符号不作投影，输入分支的正负作用可由后续权重学习。参数总数为

$$P_{UD}=M(10Wh+h+4W+2).$$

以代码实际 `sum(p.numel())` 复核；模型大小需另计算 dtype 字节。不得把每个窗口重复使用的同一参数累计多次。

### 4.7 分类损失

对数据符号标签 $c_i$，采用所有层的平均交叉熵：

$$\mathcal L_{UD}=-\frac1{M D}\sum_{m=1}^{M}\sum_{i\in\mathcal D}\log q_{m,i,c_i}.$$

批处理时按实际有效数据符号数归一化，不把空位置或导频计入分母。one-hot 目标下 KL 与交叉熵仅相差与模型无关的常数，不能把优化方向写反。[B2]

用于模型选择的验证指标统一为未截断 LLR 的比特交叉熵；另保存符号 CE、BER 与 SER。训练目标仍是上述 UDNet 符号分类损失，不擅自换成 DU 的逐层比特损失。

### 4.8 训练配置与窗口选择

```yaml
algorithm_name: udnet
algorithm_version: v1
implementation_flavor: paper_structure_project_spec_v1
layers: 22
window_size: 32
window_stride: 32
hidden_width_factor: 8
residual_tanh: 0.5
residual_probability: 0.5
init_symbols: zf_pseudoinverse
pinv_rcond: 1.0e-10
normalization: estimated_noise_whitening
compute_dtype: float64
step_parameters_trainable: true
train:
  seed: 31001
  optimizer: Adam
  learning_rate: 0.001
  betas: [0.9, 0.999]
  eps: 1.0e-8
  weight_decay: 0.0
  effective_windows_per_step: 256
  max_epochs: 200
  gradient_clip_norm: 5.0
  early_stop_patience_epochs: 20
  model_selection: validation_bit_cross_entropy
  snr_sampling: all_fixed_training_snr
  test_access: forbidden
```

原论文表 I 的 22 层、32×32 窗口、Adam 和学习率 0.001 可作为起点；本项目改变训练分布、数值精度和批组织以匹配统一数据，不能宣称复制了原论文训练集或性能数字。[B2]

窗口候选 32、64、128 各自训练、在相同验证集比较；选定一次后写入正式 `udnet_v1.yaml`。窗口大小会改变权重形状和 checkpoint，必须纳入预测缓存键。额外重叠窗口只能另设变体。所有候选的训练预算和选择记录保存，不按测试 SNR 事后挑选模型。

## 5. 训练与 checkpoint 的共同规范

LMMSE、传统 VAMP 不训练；验证集调参记录仍要保存。UDNet 使用封存训练集，校验集只用于模型选择，不反向传播。

初始化使用固定种子、固定 dtype 和确定性算子；权重采用明确记录的 Xavier uniform，偏置为零。采样器顺序从 `training_seed` 和 epoch 派生，不使用系统时间。训练允许固定顺序随机打乱样本，但不生成新的信道或噪声。

每个 checkpoint 保存：模型结构配置、权重、优化器状态、学习率状态、epoch、global step、训练 RNG 状态、sampler 状态、train/val 内容哈希、训练源码依赖哈希、初始 seed、验证指标、选择规则和环境锁。

`best` 只是训练过程中的可变指针；正式测试必须解析成不可变文件和权重摘要，不得让评估器读取一个会变化的 `best.pt` 路径。恢复训练产生新权重后，旧测试结果保留；新权重用新的预测身份单独评估。

## 6. 实现任务分解与单元测试

### 6.1 推荐顺序

```text
1. contracts + QPSK 概率/LLR + 接收输入 loader
2. 小维度精确 LMMSE + 软解调
3. 传统 VAMP 单步 → 多步 → 精确谱分解加速
4. UDNet 单窗口 → 类别损失 → mask → 多窗口重组
5. 防标签泄漏、有限性与梯度测试
6. UDNet 训练/验证、baseline 参数选择
7. 冻结三个算法各自 manifest
8. 文档 04 对每个算法分别评价并保存
```

### 6.2 必需测试

| ID | 检查 | 通过条件 |
|---|---|---|
| BL-01 | 星座和 bit/class 映射 | 与数据资产逐项一致 |
| BL-02 | 导频消除 | 只减已知均值，保留接收行 |
| BL-03 | LMMSE 求解 | 与直接小矩阵逆参考一致 |
| BL-04 | LMMSE soft output | 去偏增益/方差及无信息极限正确 |
| BL-05 | VAMP 条件散度 | 实雅可比迹与解析式误差 ≤ 1e-6 |
| BL-06 | VAMP 投影顺序 | 精度截断不改变已得到的均值 |
| BL-07 | VAMP 两种精确求解 | 各层 $z,r,\gamma,p$ 一致 |
| BL-08 | QPSK 闭式 | Softmax 均值、tanh 均值、LLR 一致 |
| BL-09 | UDNet 形状 | 每个窗口 $[W,4]$，每行概率和为 1 |
| BL-10 | UDNet 实/复表示 | $\mathcal R(Hx)=\mathcal R(H)\mathcal R(x)$ |
| BL-11 | UDNet ZF | 全秩、欠秩与 mask 情况均符合定义 |
| BL-12 | UDNet 窗口重组 | 每个数据索引恰好出现一次 |
| BL-13 | UDNet 残差 | 凸组合公式与 log 域实现一致 |
| BL-14 | 训练梯度 | 随机非饱和小样本有限差分核对 |
| BL-15 | 防泄漏 | forward 不加载 labels/truth/test 调参字段 |
| BL-16 | 重复推理 | 同样本、同模型、同环境输出一致 |

有限差分测试只在小维度开发样本上做，不在正式测试集上重复运行性能试验。不能仅凭 loss 下降证明实现符合论文结构。

## 7. 复杂度与运行时间边界

对白噪声、$N\ge D$，稠密 LMMSE 的主要代价是 $O(ND^2+D^3)$；VAMP 的固定 Gram 分解准备为同级代价，之后每层约 $O(D^2+D)$，因此可写成 $O(ND^2+D^3+TD^2)$。若输入为任意稠密彩色噪声，还需计入协方差分解/白化代价。

UDNet 窗口数 $J=N/W$，局部预处理/伪逆为 $O(JW^3)$，22 层网络计算约 $O(JM(Wh+W^2))$。这只是数量级，不能把它直接转换成某台机器上的毫秒值。所有窗口、mask、输出解调和算法内分解均计入均衡运行时间。

共同接收估计已经在数据生成时完成，统一排除于均衡核心时间之外；可以另报共同前端成本，但四种算法使用同一个数值。磁盘 I/O、矩阵展开、设备传输、算法准备和核心计算按文档 04 分段保存，不能通过预先离线分解仅给某个算法“免费加速”。

## 8. 独立版本与缓存依赖

每个算法自己的 `dependency_manifest` 列出：本算法源码、实际调用的共享数学函数、输入/输出适配器、有效算法配置、模型结构与推理权重摘要。完整 Git commit 作为审计信息保存，但不能直接当作所有算法的唯一失效条件。

例：仅修改 `algorithms/vamp_du/`，LMMSE、VAMP、UDNet 的依赖摘要均不变，其正式结果必须复用。修改三者共同调用的 QPSK 类别映射时，需要判断变化发生在预测还是统计层，并按文档 04 重新推理或仅重新解调；不得因为仓库 commit 改变就全部重跑。

每个正式版本目录不可被不同代码/配置悄悄复用。语义版本一致而依赖摘要不同，注册时应报错，要求新版本或明确新的不可变运行身份。

## 9. 命令契约：一个命令只评价一个算法

```bash
# 验证集调参/训练；不接触正式 test
python -m vampnet.cli.select_baseline --algorithm vamp --split val
python -m vampnet.cli.train --algorithm udnet --config configs/training/udnet_v1.yaml

# 正式测试由独立 evaluation 入口负责
python -m vampnet.cli.evaluate --algorithm lmmse --version v1 --suite configs/evaluation/accuracy_v1.yaml
python -m vampnet.cli.evaluate --algorithm vamp  --version v1 --suite configs/evaluation/accuracy_v1.yaml
python -m vampnet.cli.evaluate --algorithm udnet --version v1 --suite configs/evaluation/accuracy_v1.yaml
```

第二次执行同一 `evaluate` 命令时，有效预测与统计全部命中则只返回结果路径，不调用 forward。训练完成不会自动测试其他算法；新增一个 UDNet checkpoint 也不会让 LMMSE 或 VAMP 重测。结果目录、缓存状态机、恢复策略与比较方式以文档 04 为准。

## 10. 外部依据

- [B1] S. Rangan, P. Schniter, A. K. Fletcher, *Vector Approximate Message Passing*，双模块更新、散度与外信息框架：<https://arxiv.org/abs/1610.03082>。本文复数记号、投影细节与阻尼由项目统一定义。
- [B2] H. Zhao et al., *Model-Driven Based Deep Unfolding Equalizer for Underwater Acoustic OFDM Communications*，arXiv:2207.04478v2，式 (13)–(19)、图 4–5、表 I：<https://arxiv.org/abs/2207.04478v2>；PDF：<https://arxiv.org/pdf/2207.04478>。本文件已将论文未唯一规定的工程细节显式标记，不能将这些补全误引为作者原始参数。
