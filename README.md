# Steane-code-qec

**一个方案一个 Python 文件，直接运行即可采样并画曲线。** 每个方案文件含完整电路、Stim 模拟、decoder、恢复和统计代码，可以单独复制到其他目录使用。仅需 Python、Stim 和 NumPy，已验证 Python 3.13；无需额外安装 Tsim 或绘图库。

Python 程序集中在 `programs/`，保留的最终图表在 `results/`。以下命令均在项目根目录 `Steane-code-qec/` 执行。

```bash
python -m pip install -r requirements.txt
python programs/parallel.py
```

默认在 `p=0.0001～0.01` 的 12 个点各尝试 100000 次输入，自动生成 `curve.csv`、矢量曲线 `curve.svg` 和简短运行记录 `run.txt`。结果保存在项目根目录的 `output/` 新子目录；单独复制方案文件时，保存在该文件旁的 `output/`，SVG 可用浏览器打开；没有 JSON 配置文件。

| 直接运行 | 方案 |
|---|---|
| `python programs/steane_1.py` | Steane 一轮测量后恢复 |
| `python programs/steane_2.py` | Steane 固定两轮，按第二轮恢复 |
| `python programs/shor_1.py` | Shor 一轮测量后恢复 |
| `python programs/shor_3.py` | Shor 前两轮一致则恢复，否则按第三轮恢复 |
| `python programs/shor_4.py` | Shor 相邻两轮一致才恢复，最多四轮，否则中止 |
| `python programs/sequential_flag.py` | Sequential Flag：触发后补测、恢复，继续扫描 |
| `python programs/parallel.py` | Parallel：A/B 块、选择性补测、历史查表恢复 |

自定采样量、物理错误率与种子：

```bash
python programs/parallel.py --p 0.0001 0.0003 0.001 0.003 0.01 --shots 1000000 --seed 1234
```

**汇总绘图和纯 decoder 计时在 `programs/compare_and_decoder.py`。** 默认读取保留的最终逻辑错误率数据画比较图，然后重新测量七种方案的 decoder 时间。计时默认三点、五次重复、每批 2048 次输入；`--historical-plan` 使用原五点、九次重复、每批 8192 次输入的计划。

```bash
python programs/compare_and_decoder.py
python programs/compare_and_decoder.py --mode plot
python programs/compare_and_decoder.py --mode decoder --historical-plan
python programs/compare_and_decoder.py --mode plot --logical-csv output/某次运行/curve.csv output/另一次运行/curve.csv
```

第一条命令会产生新的计时 CSV 和 SVG；第二条仅重画保留的图表。全部参数可用 `--help` 查看。同一代码、依赖版本、CPU 架构、种子和 batch 划分可重放采样；decoder 时间会随机器与运行负载变化。

保留的最终结果：

- 逻辑错误率：[图片](results/logical_error.png) · [PDF](results/logical_error.pdf) · [84 点数据](results/logical_error.csv)
- 纯 decoder 时间：[柱图](results/decoder_time.png) · [五点曲线](results/decoder_curve.pdf) · [汇总数据](results/decoder_time.csv) · [逐次计时](results/decoder_repetitions.csv)

**指标与正确性边界：** `L` 是逻辑零态在额外理想 SteaneCode 恢复后的目标翻转；`B` 是实际 QEC 输出距任意码态超过一个 Pauli 错误。历史最终曲线和默认单方案曲线中，Steane-1 使用 `L∪B`，其余六种使用 `L`；CSV 同时记录全部指标。需要所有方案统一画 `L` 时，运行各方案加 `--metric target`。分母是完成 QEC 的样本，拒绝与中止单独记录；新抽样区间为逐点 95% Wilson 区间。零次失败画上限标记，不能解释成真实错误率为零。

新脚本使用普通 Monte Carlo；保留图中两个 Flag 方案的低 p 原点使用重要性采样，抽样次数不等于普通物理尝试数，原区间按原数据保留。因此默认小规模运行用于复现方案和生成新曲线，不会逐点复刻最终图的统计数值。最终 CSV 仅统一了 SteaneCode 命名，数值保持不变。

Decoder 时间只包含恢复计算、索引与查表，按进入 QEC 的输入数归一化（包含后来中止者），误差棒为重复间标准差；不包含 Stim、测量奇偶处理、经典控制、恢复操作、终端评分和初始化，也不是硬件延迟。脚本会先检查加计时前后的同种子原始输出一致。

共同噪声：reset 后 `10p` 单比特退极化，单比特门后 `p`，每对 CNOT 后 `5p` 双比特退极化，测量报告以 `2p` 翻转；无 idle 噪声，恢复为理想 Pauli frame。输入为一次带噪、经验证的逻辑零态。Parallel 对输入 `Z(q1)`、`Z(q3)` 会留下单比特残余（从 0 编号）；Shor 单轮和 Steane 单轮有已验证的内部单故障反例。终端评分通过不等于实际 QEC 完全恢复，有限案例验证不构成完整容错证明。
