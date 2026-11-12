# Part C: Test-circuit validation (8-bit parity checker)

Script: [`src/validation/circuit_validation.py`](src/validation/circuit_validation.py)
Results: `results/circuit_validation/` · Example netlist: `netlists/parity8.cir`

## Run

```powershell
# needs Part B first (results/gate_validation/gate_spice_raw.csv, same --corners)
python .\src\validation\circuit_validation.py --corners 200 --ngspice "C:\Ngspice\Spice64\bin\ngspice_con.exe" --workers 8
# re-score new models without re-simulating:
python .\src\validation\circuit_validation.py --corners 200 --reuse-spice
```

The SPICE stage is 51,200 runs. It took 66 min on 2 cores, and the results are saved per
corner in `data/circuit_validation/corners/`, so an interrupted run resumes where it stopped.

## Circuit

Odd-parity checker `P = d0 ^ d1 ^ ... ^ d7`, built as a balanced tree of 7 XOR2 cells,
each the textbook 4 × NAND2 XOR (`n1=NAND(a,b); n2=NAND(a,n1); n3=NAND(b,n1); y=NAND(n2,n3)`).
This gives 28 NAND2 gates and 112 transistors. Every gate maps directly onto a Part B gate model.
The NAND2 subcircuit is generated from `gate_validation.GATES`, so it is
transistor-for-transistor identical to the Part B NAND2. Wn = 2·Lmin, Wp = 4·Lmin, and the
corners are the same 200 as Part B (corner seed 7). `--xor-style andor` builds
XOR = AND2(OR2, NAND2) instead (not run).
Vector labels are written `d7…d0` (e.g. `00000101` means d0 = d2 = 1).

## Method

1. **SPICE reference**: the complete transistor-level netlist for each vector and corner, solved with `.op`.
   Leakage = |I(VDD)|. Each net gets a `.nodeset` at its expected logic level. This only sets the
   Newton starting point, and the results agree with the no-nodeset runs to ~1e-6. Without it,
   nearly every point needed slow gmin stepping. If an attempt fails or falls back to ngspice's
   "transient op", the vector is re-run: first without nodesets, then with higher DC iteration
   limits (`itl1`/`itl6`), both without and with nodesets. The attempt that converged most directly
   is kept. If every attempt fails, the point is marked `solve_path = failed`, listed in the log
   and left out of all metrics; the run continues. (This happened once with ngspice-47, but never
   with ngspice-42.) Every net is checked against its expected logic value.
2. **ML, hierarchical**: a logic simulation gives every gate's input bits. Each gate model
   (Part B's transistor models + KCL) gives the gate's supply leakage and the current drawn into each input
   pin (the sum of the predicted `ig` of the transistors on that pin). Two compositions:
   * **plain sum**: Σ gate supply leakage, the literal wording of the spec.
   * **with input loading**: plain sum + the pin currents of gate inputs that sit on an
     *internal* net at logic 1 (primary inputs are ideal sources in both netlists, so they are excluded).
3. **Composition check**: the same two sums built from *SPICE* gate results (the Part B cache, plus a
   re-simulation of the identical Part B NAND2 netlists that also records the input-source currents).
   The error of this check against the full netlist is the composition error alone.

## Results (51,200 cases = 256 vectors × 200 corners)

| Prediction | SPICE median (A) | Pred. median (A) | Bias (A) | Median err (%) | P95 err (%) | Mean acc. (%) | Within ±10 % |
|---|---:|---:|---:|---:|---:|---:|---:|
| **ML, gate leakage + input loading** | 6.884e-09 | 6.728e-09 | -1.72e-11 | **4.16** | 19.86 | **92.32** | 83.1 % |
| ML, plain sum of gate leakages (spec wording) | 6.884e-09 | 3.154e-09 | -2.83e-09 | 54.21 | 88.46 | 47.38 | 2.4 % |
| SPICE gates + input loading (composition check) | 6.884e-09 | 6.885e-09 | 6.2e-13 | 0.009 | 0.040 | 99.97 | 99.98 % |
| SPICE gates, plain sum (composition check) | 6.884e-09 | 3.124e-09 | -2.82e-09 | 54.29 | 88.67 | 47.33 | 1.9 % |
| ML vs composed SPICE gates (gate-model error only) | — | — | — | 4.14 | 19.80 | 92.34 | 83.1 % |

The per-vector table (all 256 vectors, the same columns as the gate results table) is
`circuit_metrics_per_vector_with_loading.csv` (and `_plain_sum.csv`). The error is almost
independent of the vector: the per-vector median error is 4.05–4.30 % for every vector
(`circuit_metrics_by_ones.csv`). Plots: `circuit_validation_parity.png` and `circuit_error_analysis.png`.

### Main findings for the report

* **Loading is not a small effect here; it is about half the leakage.** In the 16nm LP card,
  gate-oxide tunnelling is large (input-source currents are about 27 % of the circuit leakage at the primary
  inputs alone). Inside the circuit, the current drawn by a gate input at logic 1 is supplied from
  VDD through the driving gate's pull-up. In an isolated gate it comes from the ideal input source,
  so it is not part of the gate's |I(VDD)|. The plain sum therefore misses about 54 % of the circuit leakage
  even with *exact SPICE* gate values. Adding the input-pin currents makes the SPICE composition
  exact (0.009 % median), and all of the remaining ML error is gate-model error.
* **Accuracy vs gate level**: 4.16 % median error at circuit level vs 2.7–5.3 % for the NAND2 cases at
  gate level. The mean accuracy of 92.3 % (Part B: 95.1 %) is pulled down by the issue below.
* **Largest errors: 4 corners (15, 45, 140, 146), 1,024 cases, up to 4.7×10^4 % error.** All
  four have high VDD (0.95–0.98 V). The `pmos_off` model predicts `ig ≈ 2–5×10^-7 A`
  for the NAND2 `10` pull-up PMOS p1, which has all four terminals at VDD (single-transistor SPICE gives
  ≈ 0). This is a Part A model defect that Part B could not see, because it doesn't enter the gate's
  |I(VDD)|. Without these 4 corners, the mean accuracy is 94.2 % and the P95 error is 16.7 %.
* The other 196 per-corner medians fall between 0.07 % and 51 %, with no trend vs temperature, VDD or
  NMOS/PMOS vth0 (|correlation| ≤ 0.15). The worst four (195, 122, 91, 22; 28–51 %) don't share one
  condition. Corners 22 and 195 (low VDD, below −25 °C) are also the weak corners in Part B.
* **SPICE solve quality**: all 51,200 points have every net at the correct logic level (worst
  0.22 mV from a rail), and no point has the supply absorbing current. 51,119 converged directly, 69 needed gmin
  stepping, and **12 needed the transient-op fallback** (flagged in `spice_flagged`). 8 of those 12
  are 70–94 % off the composed-SPICE value, so treat them as unreliable references. Excluding them
  changes the headline numbers by < 0.02 %.

## Environment notes

* Simulated here with **ngspice-42** (the team used 44.2). Re-simulating 108 Part B gate cases with
  ngspice-42 reproduced the cached 44.2 results to within 4.5e-6 (relative). Re-run on the team machine for the final report.
* The saved models load only with **scikit-learn 1.5.1** (the version they were pickled with).
  `requirements.txt` pins 1.9.1, which fails with `No module named '_loss'`. The gate models used here
  reproduce the Part B ML leakages to within 5e-10 %.

## Files (`results/circuit_validation/`)

| File | Content |
|---|---|
| `circuit_spice_raw.csv` | SPICE cache: leakage, signed I(VDD), input-source current, logic/rail checks, solve path |
| `circuit_validation_raw.csv` | Every case: SPICE and all four predictions, errors, corner VDD/T/vth0 |
| `circuit_validation_summary.csv` | The results table above |
| `circuit_metrics_per_vector_with_loading.csv`, `..._plain_sum.csv` | Per-vector metrics (256 rows each) |
| `circuit_metrics_by_ones.csv` | Per-vector metrics grouped by number of ones |
| `gate_model_table.csv` | Gate-model outputs per (gate, inputs, corner): ML/SPICE supply leakage and pin currents |
| `gate_pin_currents_spice.csv` | SPICE re-simulation of the Part B NAND2 gates with input-source currents |
| `gate_input_usage.csv` | How often each NAND2 input case occurs in the circuit |
| `circuit_validation_parity.png` | Parity plot (log axes), both compositions |
| `circuit_error_analysis.png` | Per-corner median error vs T, VDD, NMOS/PMOS vth0 |
