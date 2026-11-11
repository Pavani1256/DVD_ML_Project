tech node : 16nm_LP
VDD : 0.9
Testing * - bit parity checker
Simulator : ngspice-44.2 (`ngspice_con.exe`)

## Run order (from the project root)

```powershell
python .\src\data_generation\generate_samples.py         # 4 x 10,000 uniform input samples (seed 2026)
python .\src\data_generation\generate_edge_samples.py    # 4 x 6,000 supplementary rail-edge samples (seed 2028)
python .\src\spice\run_dataset.py                        # simulate -> data/processed/{case}.csv
python .\src\spice\run_dataset.py --suffix _edge         # simulate -> data/processed/{case}_edge.csv
python .\src\analysis\data_check.py                      # Kirchhoff check + 3 trend plots
python .\train_models.py                                 # train, compare, select, test, save
python .\src\validation\gate_validation.py --corners 200 --ngspice "C:\Ngspice\Spice64\bin\ngspice_con.exe"
```

`run_dataset.py` runs simulations in parallel (`--workers N`), and the output does
not depend on the worker count. Re-simulating the uniform samples reproduces
the original dataset exactly. To re-score new models without re-running the
3,600 gate SPICE simulations, use `gate_validation.py --corners 200 --reuse-spice`.

## Models we are trying out

Model	             Why try it?
Random Forest	     Reliable baseline; handles nonlinearities
Extra Trees	         Often effective for complex transistor behavior
Gradient Boosting	 Can model nonlinear relationships accurately
MLP Neural Network	 Can learn smooth nonlinear functions

Data Split Used : 70+15+15, seed 42 (train_models.py contains splitting, training and selection).
The uniform 10,000-row dataset and the supplementary edge dataset are split
separately and then combined, so the uniform split matches the original one.

IMPORTANT (should include in the report)
* Errors are floor-adjusted relative errors, `|pred - true| / max(|true|, 1e-18 A) * 100`.
  1e-18 A is the SPICE `abstol`, below which currents are not resolved.
* Test-set results are reported only after model selection on validation data.
* Accuracy = `100 - error`, clipped at 0, averaged over all 3,600 gate cases.

## Modelling method (shared code in `src/transistor_ml.py`)

1. **Targets**: `log10(|I| + 1e-21)` per terminal current. The sign of each terminal
   is constant where the training data agrees on it, otherwise it is predicted by
   a gradient-boosting classifier. (The previous transform used one fixed sign per
   terminal, which is wrong for ~50% of ON-state `id`/`is` samples.)
2. **Features**: the 12 inputs plus Vgs, Vds, Vbs, Vgd, Vbd, Vgb, a BSIM4-style
   effective threshold (body effect, temperature via `kt1`, DIBL via `eta0`), the
   subthreshold overdrive `(Vgs - Vth_eff)/(n kT/q)`, log10 of an EKV channel-current
   estimate, oxide-field proxies, log10(W), 1/T, and the raw vd, vs and a swap flag.
3. **Two target frames**: "raw" (netlist terminals) and "canonical" (drain/source
   relabelled so Vds >= 0 for NMOS, Vsd >= 0 for PMOS). Both are trained, because
   the 16nm LP card is not drain/source symmetric. It puts a ~1e-14 A leakage term
   on the netlist drain terminal regardless of which side is at higher voltage.
4. **Per-terminal selection**: for every terminal, the (frame, model) with the
   lowest validation P95 error is selected. The MLP is trained with full-batch
   L-BFGS (64x64x64, tanh), because sklearn's Adam has no learning-rate decay and
   stalled at ~20% error.
5. **Supplementary rail-edge samples**: inside a gate, transistors sit with Vd/Vs
   at (or within microvolts of) 0 or VDD. Under uniform sampling the chance that
   both are within 1% of VDD is ~1e-4, so the models were extrapolating exactly
   where gate leakage is evaluated. `generate_edge_samples.py` adds 6,000 samples
   per case with every parameter drawn as in the spec, except Vd/Vs (40% near 0,
   40% near VDD, 20% uniform). **This deviates from the uniform Vd/Vs spec and should be
   stated in the report.** Set `USE_EDGE_SAMPLES=0` to train without them.

### Model comparison (validation, best frame per family, all terminals pooled)

| Case | Model | Frame | R2 | Median error (%) | P95 error (%) |
|---|---|---|---:|---:|---:|
| nmos_on | RandomForest | raw | 0.9319 | 16.80 | 93.40 |
| nmos_on | ExtraTrees | raw | 0.9495 | 13.29 | 82.92 |
| nmos_on | GradientBoosting | raw | 0.9890 | 7.54 | 52.16 |
| nmos_on | MLP | raw | 0.9970 | 2.60 | 20.57 |
| nmos_off | RandomForest | raw | 0.9069 | 2.99 | 60.50 |
| nmos_off | ExtraTrees | raw | 0.9419 | 1.98 | 52.56 |
| nmos_off | GradientBoosting | raw | 0.9836 | 1.50 | 35.82 |
| nmos_off | MLP | raw | 0.9939 | 1.62 | 22.10 |
| pmos_on | RandomForest | raw | 0.9153 | 18.40 | 99.81 |
| pmos_on | ExtraTrees | raw | 0.9343 | 15.12 | 78.23 |
| pmos_on | GradientBoosting | raw | 0.9918 | 6.92 | 42.62 |
| pmos_on | MLP | raw | 0.9992 | 2.04 | 15.67 |
| pmos_off | RandomForest | canonical | 0.8508 | 15.58 | 99.93 |
| pmos_off | ExtraTrees | canonical | 0.9060 | 10.30 | 86.27 |
| pmos_off | GradientBoosting | canonical | 0.9729 | 6.22 | 55.61 |
| pmos_off | MLP | raw | 0.9913 | 3.59 | 42.06 |

### Chosen models (test set, uniform samples only)

| Case | Current | Selected | Median error (%) | P95 error (%) |
|---|---|---|---:|---:|
| nmos_on | id | raw MLP | 3.17 | 14.71 |
| nmos_on | ig | canonical MLP | 3.89 | 75.11 |
| nmos_on | is | raw MLP | 3.18 | 14.80 |
| nmos_on | ib | raw MLP | 1.13 | 4.78 |
| nmos_off | id | raw MLP | 1.46 | 16.69 |
| nmos_off | ig | raw MLP | 0.04 | 1.71 |
| nmos_off | is | raw MLP | 6.61 | 32.98 |
| nmos_off | ib | raw Gradient Boosting | 0.36 | 6.20 |
| pmos_on | id | raw MLP | 2.95 | 19.71 |
| pmos_on | ig | raw MLP | 1.92 | 9.60 |
| pmos_on | is | raw MLP | 3.16 | 26.43 |
| pmos_on | ib | raw MLP | 1.00 | 4.13 |
| pmos_off | id | raw MLP | 3.48 | 18.71 |
| pmos_off | ig | raw MLP | 3.04 | 11.66 |
| pmos_off | is | canonical MLP | 4.22 | 23.80 |
| pmos_off | ib | raw Extra Trees | 5.24 | 31.55 |

Full tables: `results/metrics/` (`model_comparison.csv`,
`model_comparison_detailed.csv`, `final_test_metrics_per_current.csv`).

### Data check

Kirchhoff residual `|id + ig + is + ib|` (`results/data_check/kcl_summary.csv`):
median relative residual is ~1e-6 in every case. Trend plots are in `results/data_check/`.

## Gate-level validation

[`src/validation/gate_validation.py`](src/validation/gate_validation.py) simulates
INV, NAND2, NOR2, AND2 and OR2 as complete static-CMOS netlists for all 18 input
cases at 200 PVT corners (corner seed 7, 3,600 SPICE runs; Wn = 2·Lmin, Wp = 4·Lmin).

ML leakage:
1. The SPICE node voltages give every transistor's Vd, Vs and Vb. The gate voltage is
   snapped to its logic level (0 or VDD), and ON/OFF follows from that logic level.
2. ON transistors in a gate sit at |Vds| ~ 1 µV, where their channel current is a tiny
   residual no regression can resolve. So ON devices contribute only their
   predicted `ig` and `ib`. Their channel currents are solved from Kirchhoff's law at the
   internal nodes, using the OFF devices' predicted currents.
3. Leakage = |sum of terminal currents at the VDD node|.

Check: feeding exact single-transistor SPICE currents into this procedure
reproduces the gate SPICE leakage to within 0.03% for all 18 cases, so the
remaining error comes from the transistor models alone.

### Results

| Gate | Inputs | SPICE median (A) | ML median (A) | Bias (A) | Median error (%) | P95 error (%) | Mean accuracy (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| INV | 0 | 1.205e-10 | 1.202e-10 | 1.001e-12 | 3.51 | 8.76 | 95.81 |
| INV | 1 | 5.115e-12 | 4.949e-12 | -4.574e-14 | 5.34 | 15.79 | 93.53 |
| NAND2 | 00 | 2.157e-10 | 2.229e-10 | 1.600e-12 | 3.20 | 8.64 | 96.27 |
| NAND2 | 01 | 1.205e-10 | 1.202e-10 | 9.750e-13 | 3.50 | 8.76 | 95.82 |
| NAND2 | 10 | 1.125e-10 | 1.139e-10 | 8.764e-13 | 3.31 | 8.55 | 95.94 |
| NAND2 | 11 | 1.023e-11 | 9.897e-12 | -9.139e-14 | 5.34 | 15.77 | 93.53 |
| NOR2 | 00 | 2.410e-10 | 2.412e-10 | 1.471e-12 | 2.77 | 7.26 | 96.73 |
| NOR2 | 01 | 1.501e-10 | 1.561e-10 | 4.878e-13 | 2.78 | 8.52 | 96.51 |
| NOR2 | 10 | 4.693e-13 | 4.665e-13 | -8.386e-16 | 4.85 | 13.90 | 94.11 |
| NOR2 | 11 | 3.032e-14 | 2.990e-14 | -3.252e-16 | 2.97 | 23.86 | 93.50 |
| AND2 | 00 | 4.384e-10 | 4.164e-10 | 5.321e-13 | 3.73 | 19.35 | 93.23 |
| AND2 | 01 | 3.312e-10 | 3.138e-10 | -3.285e-13 | 3.99 | 16.49 | 93.53 |
| AND2 | 10 | 3.300e-10 | 3.102e-10 | -6.111e-13 | 4.09 | 22.42 | 92.39 |
| AND2 | 11 | 1.712e-10 | 1.705e-10 | 5.566e-13 | 2.66 | 8.41 | 96.60 |
| OR2 | 00 | 4.289e-10 | 4.299e-10 | -1.730e-13 | 3.42 | 12.36 | 95.45 |
| OR2 | 01 | 2.683e-10 | 2.770e-10 | 1.390e-12 | 2.95 | 8.43 | 96.44 |
| OR2 | 10 | 1.250e-10 | 1.232e-10 | 1.031e-12 | 3.46 | 8.32 | 96.23 |
| OR2 | 11 | 1.206e-10 | 1.204e-10 | 1.029e-12 | 3.48 | 8.43 | 96.17 |

**Overall: 95.10% mean accuracy, 96.49% median; 65.6% of cases within ±5% and 91.7% within ±10%.**
Best: NOR2 `00`/`01` (2.8% median error). Worst: INV/NAND2 with the NMOS pull-down ON
(`1`, `11`; 5.3% median), and AND2 `10` (22% P95).

### Fixes compared with the earlier gate validation

| Problem | Effect |
|---|---|
| 1 MΩ `Rbias` resistors on internal nodes | "SPICE leakage" in 9 rows was the resistor current (e.g. NAND2 `10` = V(x)/1 MΩ exactly), and logic levels were wrong |
| OR2 netlist: NOR-stage PMOS drained into `out` instead of `z` | OR2 was not an OR gate (z = 0.32 V for `00`) |
| ON/OFF decided by exact float `vg == vdd`; internal gate voltage passed through | AND2/OR2 inverter devices were misclassified and given out-of-training gate voltages |
| Leakage summed from ON devices at Vds ≈ 1 µV | ~10^5 % errors (INV `0`, NAND2 `00`/`01`, AND2/OR2 `11`) |
| W = Lmin (edge of the training range) | Replaced by Wn = 2·Lmin, Wp = 4·Lmin |

Progression of the mean gate accuracy: original ≈ 0% (median errors up to 5×10^5 %) →
81.4% (fixed validation + new targets/features/MLP) → 94.9% (+ rail-edge samples) →
95.1% (MLP trained to convergence, 1e-21 A log floor).

### Remaining gap to 99%

The leakage procedure is exact (0.03% with oracle currents), so the remaining ~5%
comes from transistor-model error at the gate operating points (≈2–5% median for
PMOS-ON `ig`, NMOS-OFF `id`/`is` and PMOS-OFF `is`). Options for improving it:
* More rail-edge training samples, or samples at the exact gate bias points.
* An ensemble of several MLP seeds, or a separate network per terminal.
* A larger network trained in PyTorch with a learning-rate schedule (sklearn's
  L-BFGS run takes ~25 min per fit and has plateaued at this size).
