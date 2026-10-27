# Steane-code-qec

Steane [[7,1,3]] quantum error correction experiments using **Stim**. Each scheme is self-contained in one Python file and produces a logical error rate curve versus physical error rate. Requires Python, Stim, and NumPy; verified with Python 3.13.

Install dependencies and run a scheme from the project root:

```bash
python -m pip install -r requirements.txt
python programs/parallel.py
```

Select another scheme by changing the filename:

| File in `programs/` | Scheme |
|---|---|
| `steane_1.py` | Steane, one round |
| `steane_2.py` | Steane, two fixed rounds |
| `shor_1.py` | Shor, one round |
| `shor_3.py` | Adaptive Shor, up to three rounds |
| `shor_4.py` | Shor, two consecutive matching rounds required, up to four rounds |
| `sequential_flag.py` | Sequential Flag |
| `parallel.py` | Parallel |

New results are saved to `output/` as CSV data and SVG curves. Use `--help` for sampling, physical error rate, and other options.

Plot comparisons and measure pure decoder time:

```bash
python programs/compare_and_decoder.py
```

Add `--mode plot` to redraw saved results only.

**Retained final results** (all files are in [results/](results/)):

- Logical error rate: [curve](results/logical_error.png) · [data](results/logical_error.csv)
- Pure decoder time: [bar chart](results/decoder_time.png) · [five-point curve](results/decoder_curve.pdf) · [data](results/decoder_time.csv)

<details>
<summary>Metrics and timing notes</summary>

- **Error rate:** `L` is a logical-zero target flip after additional ideal SteaneCode recovery; `B` means the actual QEC output is more than one Pauli error away from any code state. Retained and default curves use `L OR B` for Steane-1 and `L` for all other schemes, normalized by completed QEC samples. Add `--metric target` to each scheme to compare `L` consistently.
- **Historical results:** Some low-error-rate points for the two Flag schemes use importance sampling; importance draws are not physical attempts. Current scripts use ordinary Monte Carlo and generate new curves rather than reproduce historical values point by point. New runs report pointwise 95% Wilson intervals and mark upper bounds when no failures are observed.
- **Pure decoder time:** Includes only recovery computation, indexing, and table lookup, normalized by inputs entering QEC, including later aborts. Excludes Stim, measurement parity processing, classical control, recovery operations, scoring, and initialization. Error bars show standard deviation across repetitions. Timing depends on the machine and batch size and is not hardware latency.

Passing ideal terminal scoring does not imply complete recovery by the actual QEC protocol. These results are not a complete proof of fault tolerance.

</details>
