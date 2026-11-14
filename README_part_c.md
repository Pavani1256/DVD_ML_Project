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
   is kept. Each DC attempt has a 120 s limit. If every attempt fails, or the best one still used
   the transient op (which can stop before the stack nodes settle), the point is solved with a long
   transient instead: `tran 10n 2m` (gear, `reltol=1e-4`) from `.ic` at the expected logic levels.
   It is accepted only if |I(VDD)| changes by < 1e-4 (relative) between 90 % and 100 % of the run
   (`solve_path = ...->long_tran`). A point that still fails is marked `solve_path = failed` and
   left out of all metrics. Every net is checked against its expected logic value.
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
| **ML, gate leakage + input loading** | 6.884e-09 | 6.907e-09 | 1.00e-11 | **0.50** | 1.59 | **99.35** | 100.0 % |
| ML, plain sum of gate leakages (spec wording) | 6.884e-09 | 3.125e-09 | -2.83e-09 | 54.07 | 88.62 | 47.39 | 1.4 % |
| SPICE gates + input loading (composition check) | 6.884e-09 | 6.885e-09 | 6.2e-13 | 0.009 | 0.040 | 99.99 | 100.0 % |
| SPICE gates, plain sum (composition check) | 6.884e-09 | 3.124e-09 | -2.82e-09 | 54.29 | 88.67 | 47.34 | 1.9 % |
| ML vs composed SPICE gates (gate-model error only) | — | — | — | 0.50 | 1.60 | 99.34 | 100.0 % |

The per-vector table (all 256 vectors, the same columns as the gate results table) is
`circuit_metrics_per_vector_with_loading.csv` (and `_plain_sum.csv`). The error is almost
independent of the vector: the per-vector median error is 0.49–0.51 % for every vector
(`circuit_metrics_by_ones.csv`). Plots: `circuit_validation_parity.png` and `circuit_error_analysis.png`.

### Main findings for the report

* **Loading is not a small effect here; it is about half the leakage.** In the 16nm LP card,
  gate-oxide tunnelling is large (input-source currents are about 27 % of the circuit leakage at the primary
  inputs alone). Inside the circuit, the current drawn by a gate input at logic 1 is supplied from
  VDD through the driving gate's pull-up. In an isolated gate it comes from the ideal input source,
  so it is not part of the gate's |I(VDD)|. The plain sum therefore misses about 54 % of the circuit leakage
  even with *exact SPICE* gate values. Adding the input-pin currents makes the SPICE composition
  exact (0.009 % median), and all of the remaining ML error is gate-model error.
* **Accuracy vs gate level**: 0.50 % median error at circuit level vs 0.66–0.73 % for the NAND2 cases at
  gate level. The mean accuracy is 99.35 % (Part B: 99.12 %), 80.2 % of cases are within ±1 % and
  99.53 % within ±5 %. History: 92.32 % with the earliest transistor models (Part B 95.10 %), 99.19 % with
  the previous retrain (Part B 98.77 %), 99.35 % with the current models (stack samples added).
* **Largest errors.** The worst case is 7.2 % (corner 3, vector `10101000`). Per-corner median errors
  range from 0.016 % to 6.0 % (worst corners 3, 104, 100, 56, 97), with only weak trends vs temperature,
  VDD or NMOS/PMOS vth0 (|correlation| ≤ 0.23).
  * The earlier models had a Part A defect: the `pmos_off` model predicted `ig ≈ 2–5×10^-7 A` for the
    NAND2 `10` pull-up PMOS with all four terminals at VDD (up to 4.7×10^4 % error at corners 15, 45, 140 and 146).
  * The retrained models fix this. Those four corners now have 0.9–1.9 % error.
* **SPICE solve quality**: all 51,200 points have every net at the correct logic level (worst
  0.22 mV from a rail), and no point has the supply absorbing current. 51,119 converged directly, 69 needed gmin
  stepping, and 12 (10 failed, 2 transient-op; corners 101, 108, 147, 191) were solved with the long
  transient. All 12 settled and agree with the composed-SPICE value to within 0.015 %, so no point is
  flagged (`spice_flagged` = 0) and none is excluded. In the previous run these 12 used ngspice's short
  transient op, and 8 of them were 70–94 % off.

## Environment notes

* Simulated with **ngspice-46** (Linux; the team used 44.2 on Windows). The Part B gate references were
  re-simulated with ngspice-46 as well; they match the 44.2 cache to < 0.1 % except the 11 AND2/OR2
  points that 44.2 solved with the short transient op (see `ReadME.md`). On Linux the script finds
  `ngspice` on `PATH` (or `$NGSPICE`); on Windows pass `--ngspice` as above.
* The results above use the retrained transistor models (scikit-learn 1.9.1, as pinned in
  `requirements.txt`). The models are not committed: regenerate them with `train_models.py`
  (see `ReadME.md`), then run `circuit_validation.py --reuse-spice`.

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
