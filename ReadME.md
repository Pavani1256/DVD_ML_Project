tech node : 16nm_LP
VDD : 0.9
Testing * - bit parity checker 
To generate data : python .\src\spice\run_datatset.py

Models we are trying out :
Model	             Why try it?
Random Forest	     Reliable baseline; handles nonlinearities
Extra Trees	         Often effective for complex transistor behavior
Gradient Boosting	 Can model nonlinear relationships accurately
MLP Neural Network	 Can learn smooth nonlinear functions

Data Split Used : 70+15+15 (train_models.py[contains splitting, training and selection])

IMPORTANT ( should include in the report)
The relative error calculation uses a small denominator floor to avoid division by zero. Therefore, it is a floor-adjusted relative error, not an unmodified percentage error. Mention this definition in your report.

Also, your project requires the test-set results to be reported only after model selection on validation data. This script follows that order.

One final note: This script saves the models with their current transformation and feature information. Your teammate can load them later to predict transistor currents, without retraining.

Used P95 error and here are the final chosen model:
1. NMOS_ON = Random Forest 
2. NMOS_OFF = Extra Trees 
3. PMOS_ON = Gradient Boosting 
4. PMOS_OFF = MLP

## Gate-level validation

Gate validation was implemented in
[`src/validation/gate_validation.py`](src/validation/gate_validation.py). It
simulates INV, NAND2, NOR2, AND2, and OR2 as complete CMOS transistor
netlists for all 18 input cases at 200 fixed-seed PVT corners (3,600 SPICE
cases total). SPICE node voltages are then supplied to the selected saved
transistor models, and ML leakage is calculated from the predicted currents
at the VDD-connected transistor terminals.

Run it from the project root with:

```powershell
python .\src\validation\gate_validation.py --corners 200 --ngspice "C:\Ngspice\Spice64\bin\ngspice_con.exe"
```

The complete raw results, summary table, and parity plot are saved in
[`results/gate_validation/`](results/gate_validation/).

### Results

The error denominator is `max(abs(SPICE leakage), 1e-30)`, so the reported
percentages are floor-adjusted relative errors. Bias is the median of
`ML leakage - SPICE leakage`.

| Gate | Inputs | SPICE median (A) | ML median (A) | Bias (A) | Median error (%) | P95 error (%) |
|---|---:|---:|---:|---:|---:|---:|
| INV | 0 | 1.546e-11 | 6.910e-08 | 6.909e-08 | 439729.3 | 2832296.2 |
| INV | 1 | 1.315e-12 | 1.066e-12 | -2.823e-13 | 34.5 | 145.7 |
| NAND2 | 00 | 2.709e-11 | 1.382e-07 | 1.382e-07 | 520727.9 | 3174834.2 |
| NAND2 | 01 | 1.546e-11 | 6.910e-08 | 6.909e-08 | 439744.1 | 2832274.9 |
| NAND2 | 10 | 2.693e-07 | 2.139e-07 | -4.496e-08 | 68.9 | 234.3 |
| NAND2 | 11 | 2.631e-12 | 2.131e-12 | -5.644e-13 | 34.5 | 145.9 |
| NOR2 | 00 | 7.674e-07 | 9.260e-07 | 2.702e-07 | 53.3 | 123.4 |
| NOR2 | 01 | 7.674e-07 | 9.260e-07 | 2.702e-07 | 53.3 | 123.4 |
| NOR2 | 10 | 1.327e-12 | 1.098e-12 | -2.708e-13 | 33.9 | 123.9 |
| NOR2 | 11 | 1.327e-12 | 1.091e-12 | -2.696e-13 | 34.1 | 125.4 |
| AND2 | 00 | 8.345e-07 | 4.802e-06 | 3.996e-06 | 497.4 | 1102.0 |
| AND2 | 01 | 7.676e-07 | 4.485e-06 | 3.727e-06 | 512.3 | 1110.9 |
| AND2 | 10 | 9.679e-07 | 4.488e-06 | 3.554e-06 | 395.6 | 810.5 |
| AND2 | 11 | 2.265e-11 | 6.910e-08 | 6.909e-08 | 273803.7 | 2425843.5 |
| OR2 | 00 | 8.075e-07 | 1.022e-06 | 3.834e-07 | 69.0 | 154.5 |
| OR2 | 01 | 7.674e-07 | 8.731e-07 | 2.614e-07 | 55.1 | 132.4 |
| OR2 | 10 | 5.724e-07 | 6.311e-07 | 2.550e-07 | 79.3 | 237.8 |
| OR2 | 11 | 2.267e-11 | 6.910e-08 | 6.909e-08 | 273892.3 | 2429200.2 |

The best gate case was NOR2 with input `10` (33.9% median error, 123.9% P95).
The largest errors occurred in low-leakage states such as INV `0`, NAND2
`00`, and the AND2/OR2 inactive combinations. These cases are dominated by
very small SPICE currents, while the selected NMOS-ON/RF and PMOS-OFF/MLP
transistor predictions have an absolute error much larger than the reference
leakage. The gate-level target therefore did not reach the PDF's 99% mean
accuracy target with the currently saved models; retraining the weak
transistor cases and using a leakage-specific target would be the next
improvements.

### Accuracy-model update

The training transform now fits `log10(abs(I) + 1e-30)` for each terminal
current and restores the terminal sign learned from the training set. This
prevents tiny OFF currents from dominating the regression in linear-current
space. The gate validator also uses the solved SPICE intermediate voltages
as the source and drain features for stacked transistors, rather than
reusing generic OFF-state voltages.

Retrain the four packages before rerunning gate validation:

```powershell
python .\train_models.py
python .\src\validation\gate_validation.py --corners 200 --ngspice "C:\Ngspice\Spice64\bin\ngspice_con.exe"
```