# 仿真数据生成与续写

## 环境和资源

需要 MATLAB 与 Signal Processing Toolbox，现有环境为 MATLAB R2024b。
Python 3.11 及以上版本用于资源安装和 SHA256 记录，只使用标准库。

将完整官方 `WatermarkV1.zip` 放入 `data/external/`，在项目根目录运行：

```powershell
python scripts/install_watermark.py
matlab -wait -nosplash -batch "run('scripts/check_environment.m');"
```

安装脚本从官方 ZIP 提取 60 条 NCS1 记录、生成器使用的 `replayfilter.m` 和官方说明，
记录各文件及 ZIP 的 SHA256 到 `data/external/source_manifest.json`。
保留完整 ZIP；原始记录和官方回放代码不修改。

## 完整生成

唯一配置文件为 `configs/simulation.json`。
在项目根目录运行：

```powershell
matlab -wait -nosplash -batch "run('scripts/check_environment.m'); generate_dataset('full');"
python scripts/record_provenance.py data/datasets/ncs1_qpsk_ofdm_6khz_lfm_pair
```

完整生成使用全部 60 条记录：1–42 为 train，43–51 为 validation，52–60 为 test。
前两组 SNR 为 20 dB；test 的六档 SNR 为 5/10/15/20/25/30 dB，使用配对发送载荷和回放位置。
每条记录在每个 SNR 下写一个 HDF5 分片，完整生成共 105 个分片。
输出为 `data/datasets/ncs1_qpsk_ofdm_6khz_lfm_pair/`；当前该完整数据尚未生成。

## 小规模生成

```powershell
matlab -wait -nosplash -batch "run('scripts/check_environment.m'); generate_dataset('smoke');"
python scripts/record_provenance.py data/datasets/ncs1_qpsk_ofdm_6khz_lfm_pair_smoke
```

`generate_dataset()` 默认 smoke，使用记录 1/43/52，每条最多两帧。
输出目录追加 `_smoke`，配置文件本身不变。
train 和 validation 各 16 样本，六档 test 共 96 样本，合计 8 分片、128 样本。
现有 smoke 数据已保留。

## 分片续写与来源记录

重复运行相同生成命令会跳过 manifest 中已经登记的分片。
配置、官方 ZIP 哈希或生成代码签名不一致时拒绝续写；已有分片不允许覆盖。
若存在未登记的分片，生成器会报错，需要先检查该文件。

也可以分批指定记录，例如在 MATLAB 中执行：

```matlab
addpath('matlab');
setup_paths;
generate_dataset('full', 1:10);
generate_dataset('full', 11:60);
```

只有全部预期分片完成后，manifest 的 `status` 才会变为 `complete`。
`record_provenance.py` 在选定数据目录写入 `provenance.json`，
保存当前生成代码、配置、官方回放代码、资源清单和分片的 SHA256。

数据字段见 [输出格式](dataset_schema.md)。
