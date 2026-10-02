# 仿真数据字段与轴约定

每条原始记录、每个 SNR 对应一个 HDF5 分片；一个样本是一个数据 OFDM 槽。
完整目录标识为 `ncs1_qpsk_ofdm_6khz_lfm_pair`，小规模目录追加 `_smoke`。
当前向量维数为 384，字段尺寸由 manifest 和分片中的配置决定。

矩阵实部、虚部为 float32。用 h5py 读取时，向量轴为 `[sample, subcarrier]`，
矩阵轴为 `[sample, column, row]`；解释为接收行、发送列时须交换最后两轴。
FFT 正变换除以 sqrt(512)，逆变换乘以 sqrt(512)，有符号数据载波按升序排列。

| 字段 | 含义 |
|---|---|
| y_eq_real / y_eq_imag | 扣除估计导频贡献后的 384 维接收观测 |
| H_eq_real / H_eq_imag | 数据行列上的 384×384 估计矩阵 |
| x_real / x_imag | 384 个 QPSK 真实标签，不是估计器输入 |
| class_id | 0…3 对应 (1+j)、(−1+j)、(−1−j)、(1−j)，均除 sqrt(2) |
| bits | Gray 比特 00、01、11、10；h5py 形状 [sample, subcarrier, 2] |
| ce_coeff_real / ce_coeff_imag | h5py 形状 [sample, 5, 193]，先验和校正摘要 |
| noise_var | 冻结前端后数据行的复噪声平均功率，不含 CSI 误差 |
| snr_db / measured_snr_db | 目标 SNR 与 6000 Hz 通信带内实测 SNR |
| record_id / frame_id / symbol_id | 场景标识，数据槽为 2/3/4/5/7/8/9/10 |
| tx_seed / noise_seed | 发送载荷与噪声种子 |
| source_time | 原始信道记录中帧的发送起点，秒 |
| alpha / doppler_score | 首尾 LFM 峰间距尺度、两峰中较小的归一化相关功率 |
| known_start | 仿真已知通带起点，可为非整数；LFM 不修改它 |
| lfm_first_peak / lfm_last_peak | 相对于已知起点的零基基带相关峰位置，允许亚样点 |
| data_bins / pilot_bins | 分片共享的有符号 FFT 索引，不含样本轴 |
| fit_residual | 前四槽为两个训练拟合残差均值，后四槽为第二训练残差 |
| model_residual_ratio | 无噪声观测对带噪估计矩阵预测的相对残差 |
| observed_ici_proxy | 估计数据域矩阵非对角能量比例 |
| estimated_beyond_one_ratio | 估计矩阵在物理 FFT 距离 ±1 之外的能量比例 |
| noise_induced_csi_change_ratio | 冻结前端下带噪与无噪估计之差，不是总 CSI NMSE |

五项系数依次是先验左训练中心值、左每样点变化率、右训练中心值、右每样点变化率，
及本数据槽经导频增量校正后的平均值。
前四槽使用两训练槽的相位感知 Hermite 插值；后四槽的左右项均来自第二训练槽，
每槽重新建立局部先验，不跨槽外推。系数须与保存的接收机版本共同解释。

`schema_version` 为 `1.1`。
manifest 包含配置、官方 ZIP 哈希、分片列表、来源记录、状态、代码签名和信道估计语义。
配置或代码签名不一致时，生成器拒绝续写；已有分片不可覆盖。
全部预期分片完成后，`status` 为 `complete`。
HDF5 根属性保存 `config_json`、轴约定、噪声定义和 `doppler_source`。

`provenance.json` 由 `scripts/record_provenance.py` 写入，
保存生成代码、配置、官方回放代码、资源清单和分片 SHA256。
