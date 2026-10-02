# NCS1 水声 QPSK–OFDM 仿真数据生成

项目仅用于通过官方 Watermark/NCS1 物理回放生成仿真数据。
生成流程包括发送波形、时变信道回放、接收噪声、粗多普勒补偿、信道估计和 HDF5 分片写入。
接收机只使用首尾 LFM、已知训练符号和嵌入式导频，不读取真实信道或发送数据标签。

唯一仿真配置为 `configs/simulation.json`，数据集标识为 `ncs1_qpsk_ofdm_6khz_lfm_pair`。

- 512 FFT、6 kHz 复基带和噪声带宽、14 kHz 载频、96 kHz 通带采样率。
- 两端各 32 个保护子载波；活动索引 −224…223，含直流；64 导频和 384 数据载波。
- CP 192 点（32 ms），193 抽头模型覆盖 NCS1 原始最大时延 31.875 ms。
- 128 ms 上扫频 LFM、32 ms 静默、训练 + 4 数据 + 训练 + 4 数据、32 ms 静默、相同上扫频 LFM。
- 记录 1–42 / 43–51 / 52–60 分为 train / validation / test；前两组 20 dB，test 为 5/10/15/20/25/30 dB 配对。
- 每样本保存 384 维观测、384×384 估计矩阵、QPSK 标签、噪声方差和生成元数据。

需要 MATLAB 与 Signal Processing Toolbox；资源安装和哈希记录脚本使用 Python 3.11 及以上版本的标准库。
将完整官方 `WatermarkV1.zip` 放在 `data/external/`，从项目根目录运行：

```powershell
python scripts/install_watermark.py
matlab -wait -nosplash -batch "run('scripts/check_environment.m'); generate_dataset('full');"
```

小规模生成入口为 `generate_dataset('smoke')`，也是无参数调用的默认模式。
现有 `data/datasets/ncs1_qpsk_ofdm_6khz_lfm_pair_smoke/` 保留了 8 个分片、128 个样本；完整数据尚未生成。
生成器支持按分片续写，核对配置、官方资源哈希和代码签名，并拒绝覆盖已有分片。

保留完整官方 ZIP、60 条 NCS1 原始记录、未修改的官方 `replayfilter.m` 和已生成的仿真数据。
原始资源和生成数据不纳入 Git。

运行方式见 [生成说明](docs/reproduction.md)，物理流程见 [生成器设计](docs/system_design.md)，输出格式见 [数据字段](docs/dataset_schema.md)。
