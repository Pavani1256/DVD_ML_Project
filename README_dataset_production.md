# DVD Project --- Dataset Production

## 1. Objective

The dataset-production stage creates SPICE-simulated transistor data for
four operating cases:

1.  NMOS ON
2.  NMOS OFF
3.  PMOS ON
4.  PMOS OFF

The target is 10,000 samples per case, or 40,000 samples in total. Each
sample contains 12 input parameters and four simulated terminal
currents.

## 2. Environment and tools

  Component                Setup
  ------------------------ -----------------------------------
  Operating system         Windows
  Project directory        `C:\Users\hp\Desktop\DVD_project`
  Python environment       Project `.venv`
  SPICE simulator          ngspice 44.2
  Batch executable         `ngspice_con.exe`
  Technology model         `models/16nm_LP.pm`
  Nominal supply voltage   0.9 V
  Minimum channel length   16 nm

ngspice was run in batch mode, for example:

``` powershell
ngspice_con.exe -b netlists/test.cir
```

## 3. Project folders used

``` text
DVD_project/
├── .venv/
├── models/
│   └── 16nm_LP.pm
├── netlists/
│   ├── test.cir
│   └── nmos_on_smoke.cir
├── data/
│   ├── samples/
│   ├── spice_test/
│   └── processed/
└── src/
    ├── data_generation/
    │   └── generate_samples.py
    └── spice/
        ├── run_one_sample.py
        └── run_dataset.py
```

## 4. Technology model and nominal parameters

The selected model is `16nm_LP.pm`, with model names `nmos` and `pmos`.

The five process parameters varied independently for each sample are
`vth0`, `toxe`, `ndep`, `u0`, and `xj`. The nominal values used by the
sample-generation script are:

  Parameter        NMOS      PMOS
  ----------- --------- ---------
  `vth0`        0.68191   -0.6862
  `toxe`        1.20e-9   1.22e-9
  `ndep`         7.0e18    4.4e18
  `u0`            0.028    0.0075
  `xj`           5.0e-9    7.2e-9

Each parameter is sampled independently over ±10% of its nominal value.
The PMOS threshold voltage is negative, so the script orders the two
interval endpoints using `min()` and `max()`.

## 5. Input sample generation

The script is `src/data_generation/generate_samples.py`. It uses a fixed
random seed for reproducibility and generates 10,000 input
configurations for each case.

### Sampling ranges

  -----------------------------------------------------------------------
  Input                               Sampling rule
  ----------------------------------- -----------------------------------
  `vdd`                               Uniform from 0.81 V to 0.99 V

  `vd`                                Uniform from 0 to the sample's
                                      `vdd`

  `vs`                                Uniform from 0 to the sample's
                                      `vdd`

  `w`                                 Uniform from 16 nm to 128 nm

  `temp`                              Uniform from -40 °C to 125 °C

  `vth0`, `toxe`, `ndep`, `u0`, `xj`  Each uniform within ±10% of its
                                      transistor's nominal value
  -----------------------------------------------------------------------

`vdd` is drawn first because the ranges for `vd` and `vs` depend on the
sampled supply voltage.

### Fixed gate and body voltages

  Case       `vg`    `vb`
  ---------- ------- -------
  NMOS ON    `vdd`   0
  NMOS OFF   0       0
  PMOS ON    0       `vdd`
  PMOS OFF   `vdd`   `vdd`

Run the generator from the project root:

``` powershell
python .\src\data_generation\generate_samples.py
```

It creates:

``` text
data/samples/
├── nmos_on_samples.csv
├── nmos_off_samples.csv
├── pmos_on_samples.csv
└── pmos_off_samples.csv
```

Each file was verified to contain 10,000 rows and 12 columns(inputs), for 40,000
input configurations total.

The input columns are:

``` text
vdd,temp,w,vd,vs,vg,vb,vth0,toxe,ndep,u0,xj
```

## 6. Python-to-ngspice integration

The individual-sample runner is `src/spice/run_one_sample.py`.

For each row, it:

1.  Reads the sample's input parameters.
2.  Reads the original `16nm_LP.pm` file.
3.  Locates the relevant NMOS or PMOS model section.
4.  Replaces that model section's five process parameters with the
    sampled values.
5.  Writes a temporary modified model file.
6.  Generates a SPICE netlist using the sample's voltages, temperature
    and width.
7.  Runs ngspice in batch mode.
8.  Extracts the four terminal currents.

This per-sample model substitution is important: the process-parameter
values are applied to the SPICE model, not just stored as unused inputs
in the CSV.

### Netlist setup

Each transistor terminal has its own voltage source:

``` spice
VDS d 0 DC <vd>
VGS g 0 DC <vg>
VSS s 0 DC <vs>
VBS b 0 DC <vb>
```

The transistor instance uses the selected model and sampled width:

``` spice
M1 d g s b <model_name> L=16n W=<sampled_width>
```

The required numerical options are:

``` spice
.options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9
```

The sample temperature is set using `.temp`.

### Current extraction and sign convention

The netlist extracts currents using:

``` spice
let Id = -i(VDS)
let Ig = -i(VGS)
let Is = -i(VSS)
let Ib = -i(VBS)
```

The negative signs account for ngspice's voltage-source current
reference direction and provide a consistent terminal-current
convention.

## 7. Validation performed

The workflow was tested incrementally before starting the full dataset.

### 7.1 NMOS ON smoke test

A nominal NMOS ON smoke-test netlist ran successfully and produced all
four terminal currents. Its KCL residual was approximately:

``` text
-1.10015e-15 A
```

### 7.2 Python simulation of one generated sample

The first row of `nmos_on_samples.csv` was passed through the Python
runner. The script successfully substituted the sampled model
parameters, generated the netlist, ran ngspice and extracted all four
currents.

The KCL residual was:

``` text
-5.74832e-16 A
```

### 7.3 Ten-sample NMOS ON test

The batch runner successfully simulated the first 10 NMOS ON samples and
wrote a test output CSV.

### 7.4 Ten-sample test for all four cases

The first 10 samples of each of the four cases were tested. All four
case tests passed, corresponding to 40 successful test simulations.

## 8. Full dataset production

The full-run script is `src/spice/run_dataset.py`. It is intended to
process all 10,000 samples in each input CSV.

The intended final datasets are:

``` text
data/processed/
├── nmos_on.csv
├── nmos_off.csv
├── pmos_on.csv
└── pmos_off.csv
```

Each final dataset should have 10,000 rows and the 16 assignment
columns:

``` text
vdd,temp,w,vd,vs,vg,vb,vth0,toxe,ndep,u0,xj,id,ig,is,ib
```

KCL residuals are stored separately in `*_kcl_checks.csv` files. Failed
simulations, if any, are recorded separately in `*_failures.csv` files.

The full runner reports progress and writes output in batches. The final
datasets must not be considered complete until the run has finished and
the row counts, column counts, failures and KCL checks have been
verified.

## 9. Final validation checklist

-   [ ] `nmos_on.csv` has 10,000 rows and 16 columns.
-   [ ] `nmos_off.csv` has 10,000 rows and 16 columns.
-   [ ] `pmos_on.csv` has 10,000 rows and 16 columns.
-   [ ] `pmos_off.csv` has 10,000 rows and 16 columns.
-   [ ] No unexpected header rows or missing values are present.
-   [ ] Failed simulations, if any, have been inspected.
-   [ ] KCL residuals have been checked for successful simulations.
-   [ ] The original input CSVs remain unchanged.

## 10. Next stage

After dataset production and validation, the next Part A steps are:

1.  Inspect the datasets for missing, invalid or unexpected values.
2.  Produce the required current-trend plots.
3.  Split each case into training, validation and test sets using a
    fixed seed.
4.  Train and compare the specified machine-learning models.
5.  Select the best model using validation performance.
6.  Evaluate the selected model on the test set.
7.  Save the four selected models and the required metrics and plots.

------------------------------------------------------------------------

**Status note:** The four input CSVs have been generated and verified.
The single-sample test and all four 10-sample case tests passed. The
full 40,000-simulation run is in progress and is not marked complete
until its outputs are validated.
