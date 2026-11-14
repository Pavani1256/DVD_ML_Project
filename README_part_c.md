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

## Results (51,199 cases = 256 vectors × 200 corners)

SPICE: the team's ngspice-47 run (one point, corner 147 vector `00100111`, could not be solved
by ngspice-47 and is excluded). Models: the four saved models with the operating-point
sub-models and the zero-bias gate rule (see `ReadME.md`, "Reaching 99%").

| Prediction | SPICE median (A) | Pred. median (A) | Bias (A) | Median err (%) | P95 err (%) | Mean acc. (%) | Within ±10 % |
|---|---:|---:|---:|---:|---:|---:|---:|
| **ML, gate leakage + input loading** | 6.884e-09 | 6.860e-09 | -2.49e-11 | **0.63** | 1.91 | **99.21** | 99.99 % |
| ML, plain sum of gate leakages (spec wording) | 6.884e-09 | 3.114e-09 | -2.81e-09 | 54.44 | 88.68 | 47.22 | 2.0 % |
| SPICE gates + input loading (composition check) | 6.884e-09 | 6.885e-09 | 6.2e-13 | 0.009 | 0.040 | 99.97 | 99.99 % |
| SPICE gates, plain sum (composition check) | 6.884e-09 | 3.124e-09 | -2.82e-09 | 54.29 | 88.67 | 47.33 | 1.9 % |
| ML vs composed SPICE gates (gate-model error only) | — | — | — | 0.64 | 1.91 | 99.21 | 100 % |

Excluding the 11 flagged SPICE points, every case is within ±10 % (worst 8.7 %) and 99.96 % are
within ±5 %.

The per-vector table (all 256 vectors, the same columns as the gate results table) is
`circuit_metrics_per_vector_with_loading.csv` (and `_plain_sum.csv`). The error is almost
independent of the vector: the per-vector median error is 0.60–0.66 % for every vector
(`circuit_metrics_by_ones.csv`). Plots: `circuit_validation_parity.png` and `circuit_error_analysis.png`.

### Main findings for the report

* **Loading is not a small effect here; it is about half the leakage.** In the 16nm LP card,
  gate-oxide tunnelling is large (input-source currents are about 27 % of the circuit leakage at the primary
  inputs alone). Inside the circuit, the current drawn by a gate input at logic 1 is supplied from
  VDD through the driving gate's pull-up. In an isolated gate it comes from the ideal input source,
  so it is not part of the gate's |I(VDD)|. The plain sum therefore misses about 54 % of the circuit leakage
  even with *exact SPICE* gate values. Adding the input-pin currents makes the SPICE composition
  exact (0.009 % median), and all of the remaining ML error is gate-model error.
* **Accuracy vs gate level**: 0.63 % median error and 99.21 % mean accuracy at circuit level vs
  0.28–1.03 % median for the NAND2 cases and 99.13 % overall at gate level. Composing 28 gates
  averages out part of the per-gate error, so the circuit is as accurate as the gates.
* **Progression** (same SPICE data): plain sum 47.4 % → with input loading 92.3 % → + zero-bias
  gate rule 94.5 % → + operating-point sub-models **99.2 %**.
* **Zero-bias gate rule (in `src/transistor_ml.py`)**: without it, 4 corners (15, 45, 140, 146;
  1,024 cases, all at VDD 0.95–0.98 V) had errors up to 4.7×10^4 %. The `pmos_off` model
  predicted `ig ≈ 2–5×10^-7 A` for the NAND2 `10` pull-up PMOS p1, which has all four terminals at VDD,
  where single-transistor SPICE gives ≈ 0. Part B could not see this, because that current does not
  enter a gate's |I(VDD)|. The rule sets `ig = 0` whenever the gate is within 1 mV of every other
  terminal: there is no field across the oxide, so there is no tunnelling. Effect on Part C:
  mean accuracy 92.3 → 94.5 %, P95 19.9 → 13.6 %, max error 4.7×10^4 → 94 %. Part B is unchanged.
* Per-corner median errors fall between 0.004 % and 3.4 % (before the sub-models: up to 51 % at
  cold corners), with only weak trends vs temperature (r = 0.22) and VDD (r = −0.25). The worst
  corners are 93 (58 °C, 3.4 %), 50 (85 °C, 2.9 %), 37, 104 and 63 (2.2–2.8 %). Four of these
  five have low VDD (0.82–0.85 V), where the leakage is smallest.
* **SPICE solve quality (ngspice-47)**: all 51,199 solved points have every net at the correct logic
  level (worst 0.22 mV from a rail), and no point has the supply absorbing current. 51,119 converged
  directly, 69 needed gmin stepping, and **11 needed the transient-op fallback** (flagged in
  `spice_flagged`). Most of those are 70–94 % off the composed-SPICE value, so treat them as
  unreliable references. Excluding them changes the headline numbers by < 0.02 %.

## Environment notes

* The circuit SPICE results are from the team's ngspice-47 run. ngspice-42 gave the same headline
  numbers to within 0.01 %, and reproduced the cached 44.2 Part B results to within 4.5e-6.
* The saved models load only with **scikit-learn 1.5.1** (Python 3.12; see `requirements.txt`).

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
