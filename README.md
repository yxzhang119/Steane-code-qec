# Steane-code-qec

基于 **Stim** 的 Steane [[7,1,3]] 量子纠错实验。每个方案一个 Python 文件，直接运行即可生成物理错误率—逻辑错误率曲线。依赖 Python、Stim 和 NumPy，已验证 Python 3.13。

在项目根目录安装依赖并运行一个方案：

```bash
python -m pip install -r requirements.txt
python programs/parallel.py
```

切换方案只需替换文件名：

| 文件（位于 `programs/`） | 方案 |
|---|---|
| `steane_1.py` | Steane 单轮 |
| `steane_2.py` | Steane 固定两轮 |
| `shor_1.py` | Shor 单轮 |
| `shor_3.py` | Shor 自适应，最多三轮 |
| `shor_4.py` | Shor 相邻两轮一致才恢复，最多四轮 |
| `sequential_flag.py` | Sequential Flag |
| `parallel.py` | Parallel |

新结果保存到 `output/`，包含 CSV 数据与 SVG 曲线。采样量、物理错误率等参数可通过 `--help` 查看。

绘制对比图并重新测量纯 decoder 时间：

```bash
python programs/compare_and_decoder.py
```

加 `--mode plot` 可只重画已有结果。

**保留的最终结果**（完整文件见 [results/](results/)）：

- 逻辑错误率：[曲线](results/logical_error.png) · [数据](results/logical_error.csv)
- 纯 decoder 时间：[柱图](results/decoder_time.png) · [五点曲线](results/decoder_curve.pdf) · [数据](results/decoder_time.csv)

<details>
<summary>指标与计时说明</summary>

- **错误率：** `L` 是额外理想 SteaneCode 恢复后的逻辑零态翻转；`B` 是实际 QEC 输出距任意码态超过一个 Pauli 错误。历史曲线和默认运行中，Steane-1 使用 `L∪B`，其余使用 `L`，分母为完成 QEC 的样本。统一比较 `L` 时，各方案加 `--metric target`。
- **历史结果：** 两个 Flag 方案的部分低错误率点来自重要性采样，抽样数不等于物理尝试数。当前脚本采用普通 Monte Carlo，生成新曲线，不保证逐点复刻历史数值；新运行给出逐点 95% Wilson 区间，零次失败显示上限。
- **纯 decoder 时间：** 只计恢复计算、索引与查表，按进入 QEC 的输入数归一化（含后来中止者）。不含 Stim、测量奇偶处理、经典控制、恢复操作、评分与初始化。误差棒为重复间标准差，时间随机器和 batch 变化，不代表硬件延迟。

理想终端评分通过不等于实际 QEC 完全恢复；这些结果不构成完整容错证明。

</details>
