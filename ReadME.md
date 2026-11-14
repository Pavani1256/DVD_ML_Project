tech node : 16nm_LP
VDD : 0.9
Test circuit : 8-bit parity checker
Simulator : ngspice-44.2 (`ngspice_con.exe`)

## Run order (from the project root)

```powershell
python .\src\data_generation\generate_samples.py         # 4 x 10,000 uniform input samples (seed 2026)
python .\src\data_generation\generate_edge_samples.py    # 4 x 6,000 supplementary rail-edge samples (seed 2028)
python .\src\spice\run_dataset.py                        # simulate -> data/processed/{case}.csv
python .\src\spice\run_dataset.py --suffix _edge         # simulate -> data/processed/{case}_edge.csv
python .\src\analysis\data_check.py                      # Kirchhoff check + 3 trend plots
python .\train_models.py                                 # train, compare, select, test, save
python .\src\spice\generate_rail_dataset.py --workers 8  # 160,000 operating-point samples -> data/processed/rail_samples.csv.gz
python .\train_rail_models.py --jobs 8                   # operating-point sub-models -> added to saved_models/*.joblib
python .\src\spice\write_example_netlists.py --check     # deliverable 2: netlists/*.cir
python .\src\validation\gate_validation.py --corners 200 --ngspice "C:\Ngspice\Spice64\bin\ngspice_con.exe"
python .\src\validation\circuit_validation.py --corners 200 --ngspice "C:\Ngspice\Spice64\bin\ngspice_con.exe"
```

Re-running `train_models.py` rewrites the four model files without the
operating-point sub-models, so run `train_rail_models.py` again after it.
On Linux use `--ngspice ngspice` (Python 3.12 and `requirements.txt`; the saved
models only load with scikit-learn 1.5.1). Part C is documented in
[README_part_c.md](README_part_c.md).

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
6. **Operating-point sub-models** (`generate_rail_dataset.py`, `train_rail_models.py`;
   see "Reaching 99%" below). In static CMOS, a transistor's drain and source sit
   exactly at a rail, or (in a series stack) one sits at a rail and the other at an
   internal node. For these 8 configurations (`00 01 10 11` = drain/source at 0/VDD,
   `m0 m1 0m 1m` = one terminal free), small MLPs trained on 160,000 extra
   single-transistor SPICE samples at exactly those configurations replace the
   general prediction. They are kept only where they pass a held-out check, and
   they live inside the four saved model files (`model.rail_models`).
7. **Zero-bias gate rule**: `ig = 0` when the gate is within 1 mV of every other
   terminal. With no field across the oxide there is no tunnelling; the PMOS OFF model
   predicted ~1e-7 A there at high VDD, where SPICE gives ~0.

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

Every SPICE run also records the signed supply current and whether ngspice needed
its "transient op" fallback (`transient_op_fallback`). Those results are not true
DC solutions (they are not even reproducible across ngspice versions), so they are
flagged (`spice_flagged`), and accuracy is also reported without them
(`gate_validation_metrics_unflagged.csv`). 33 of 3,600 cases are flagged, all AND2.
If ngspice cannot solve a case at all (ngspice-47 fails on AND2 `00` at corner 164, even with
its fallback), the case is recorded with `solve_failed = True`, listed in the output and left
out of the metrics, so that gate's row shows 199 corners.

Check: feeding exact single-transistor SPICE currents into this procedure
reproduces the gate SPICE leakage to within 0.03% for all 18 cases, so the
remaining error comes from the transistor models alone.

### Results

| Gate | Inputs | SPICE median (A) | ML median (A) | Bias (A) | Median error (%) | P95 error (%) | Mean accuracy (%) | Flagged SPICE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| INV | 0 | 1.205e-10 | 1.208e-10 | -3.501e-14 | 0.28 | 0.99 | 99.62 | 0 |
| INV | 1 | 5.115e-12 | 5.069e-12 | -1.198e-14 | 1.03 | 3.06 | 98.81 | 0 |
| NAND2 | 00 | 2.157e-10 | 2.167e-10 | -2.844e-13 | 0.31 | 0.82 | 99.65 | 0 |
| NAND2 | 01 | 1.205e-10 | 1.208e-10 | -3.474e-14 | 0.28 | 0.99 | 99.62 | 0 |
| NAND2 | 10 | 1.125e-10 | 1.126e-10 | -1.848e-13 | 0.32 | 0.98 | 99.24 | 0 |
| NAND2 | 11 | 1.023e-11 | 1.014e-11 | -2.349e-14 | 1.03 | 3.04 | 98.81 | 0 |
| NOR2 | 00 | 2.410e-10 | 2.417e-10 | -6.941e-14 | 0.28 | 1.00 | 99.62 | 0 |
| NOR2 | 01 | 1.501e-10 | 1.493e-10 | -1.956e-13 | 0.33 | 0.98 | 99.60 | 0 |
| NOR2 | 10 | 4.693e-13 | 4.678e-13 | 1.119e-17 | 1.00 | 3.31 | 98.74 | 0 |
| NOR2 | 11 | 3.032e-14 | 3.002e-14 | -7.276e-17 | 0.50 | 2.27 | 99.24 | 0 |
| AND2 | 00 | 4.384e-10 | 4.005e-10 | -1.527e-12 | 0.55 | 1.94 | 97.52 | 22 |
| AND2 | 01 | 3.312e-10 | 3.246e-10 | -9.436e-13 | 0.59 | 1.85 | 98.52 | 2 |
| AND2 | 10 | 3.300e-10 | 3.215e-10 | -1.242e-12 | 0.67 | 2.14 | 97.39 | 9 |
| AND2 | 11 | 1.712e-10 | 1.700e-10 | -1.175e-13 | 0.34 | 1.19 | 99.59 | 0 |
| OR2 | 00 | 4.289e-10 | 4.276e-10 | -1.173e-12 | 0.48 | 1.45 | 99.43 | 0 |
| OR2 | 01 | 2.683e-10 | 2.672e-10 | -2.190e-13 | 0.30 | 0.83 | 99.66 | 0 |
| OR2 | 10 | 1.250e-10 | 1.251e-10 | -2.943e-14 | 0.28 | 1.04 | 99.62 | 0 |
| OR2 | 11 | 1.206e-10 | 1.210e-10 | -3.464e-14 | 0.28 | 0.99 | 99.62 | 0 |

**Overall: 99.13% mean accuracy (99.36% excluding the 33 flagged SPICE cases), 99.60% median;
81.8% of cases within ±1%, 99.56% within ±5% and 99.61% within ±10%.**
Best: INV `0`, NAND2 `00`/`01`, NOR2 `00`, OR2 `10`/`11` (≈0.28% median error).
Least accurate: INV `1`, NAND2 `11` and NOR2 `10` (≈1.0% median): the leakage there is
small (5e-12 to 5e-13 A) and set by stack/OFF-device currents. AND2 `00`/`10` have the
lowest mean accuracy only because of their flagged SPICE references (circled in the
parity plot). With those excluded, AND2 matches the other gates.

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

### Reaching 99%: where the error was and how it was removed

**Diagnosis.** Each transistor in every Part B gate case was re-simulated *on its own*
in SPICE at exactly the bias the ML model is given. Feeding those SPICE device currents
through the same leakage procedure gives **99.75%** (the remaining 0.25% is the 33 flagged
AND2 references), so the composition is exact and all the error came from the transistor
models at the in-gate bias points. Making one model output exact at a time showed
where. The largest gains were PMOS ON `ig` (+1.8 points; pull-up with drain = source = VDD),
PMOS OFF `is` (+1.3; drain = 0, source = VDD) and NMOS ON `ig` (+0.8; drain = source = 0).
At these exact-rail points the general models were 3–8% off: the uniform datasets almost
never sample a drain *and* source sitting exactly on a rail.

**Fix.** `generate_rail_dataset.py` simulates 160,000 extra single-transistor samples at
those operating points (8 configurations × 4 cases; VDD, T, W and process drawn as in the
main datasets, own seeds 4041/5051). **No gate or circuit SPICE result is used for
training.** `train_rail_models.py` fits one MLP per (case, configuration, terminal) on 80%
of the corners. A sub-model is kept only if, on the held-out 20%, its median error is
< 10% and its P95 is ≤ 50%. 67 of 77 are kept, most with 0.3–2% held-out median error
(`results/metrics/rail_submodels_metrics.csv`). ON-device `id`/`is` are not modelled:
the gate procedure solves them from Kirchhoff's law.

| | Before | After |
|---|---:|---:|
| Part B mean accuracy (3,600 cases) | 95.10% | **99.13%** (99.36% unflagged) |
| Part B median / P95 error | 3.51% / 12.96% | **0.40% / 1.94%** |
| Part C mean accuracy (51,199 cases, with input loading) | 92.32% | **99.21%** |
| Part C median / P95 error | 4.16% / 19.86% | **0.63% / 1.91%** |

On the original test set (`results/metrics/rail_submodels_test_effect.csv`) nothing gets
worse. The uniform samples are unchanged (only 5–10 of the 1,500 per case are within 1 mV of a rail),
and on the rail-edge samples the median errors drop (e.g. PMOS OFF `is` 6.4% → 2.8%,
NMOS ON `ig` 3.8% → 1.6%) with equal or lower P95.

**For the report:** the extra dataset deviates from the spec's uniform Vd/Vs sampling in
the same way as the rail-edge samples, so state it. All four models remain one model per case.

### Remaining limitations

* 33 AND2 SPICE references needed ngspice's transient-op fallback and are unreliable.
  Nodesets and higher iteration limits did not fix them.
* Devices with *both* drain and source at internal nodes (`mm`) still use the general
  models. They do not occur in these gates, but would in deeper stacks.
* The general models' P95 on the uniform test set is unchanged (e.g. NMOS ON `ig` 75%).
  Those errors sit at bias points that do not occur inside the gates.
