# OPH-FPE Reproduction Notes

This repository emits finite OPH screen diagnostics, CDM-limit CAMB
regression readouts, and gated Boltzmann-input scaffolding. These reports are
not physical OPH CMB predictions and do not establish a populated 3D bulk.

## Clean Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[test,camb]'
python -m pytest -q
```

For non-CAMB diagnostics, `python -m pip install -e '.[test]'` is sufficient.

## CLI Smoke

```bash
oph-fpe --help
python -m oph_fpe.cli --help
```

The CLI and `oph_fpe.cosmology` package use lazy imports so a missing optional
Boltzmann dependency does not break unrelated diagnostics.

## Measurement-Facing Reports

Regenerate the standard LambdaCDM/CAMB benchmark receipt:

```bash
python -m oph_fpe.cli camb-baseline-report \
  --benchmark data/measurements/planck2018/COM_PowerSpect_CMB-TT-binned_R3.01.txt \
  --out runs/camb_lcdm_baseline_20260606 \
  --lmax 2600
```

Regenerate the screen-shape CMB diagnostics:

```bash
python -m oph_fpe.cli cmb-lite-compare \
  --run-dir runs/cmb_adapter_smoke_64k_20260606/e3_cosmo_proxy_screen_64k_seed20260671_420fc254 \
  --benchmark data/measurements/planck2018/COM_PowerSpect_CMB-TT-binned_R3.01.txt

python -m oph_fpe.cli cmb-lite-compare \
  --run-dir runs/cmb_adapter_smoke_64k_256k_20260606/e3_cosmo_proxy_screen_256k_seed20260671_78aa331c \
  --benchmark data/measurements/planck2018/COM_PowerSpect_CMB-TT-binned_R3.01.txt
```

Regenerate the CDM-limit and diagnostic OPH Boltzmann-input bridge:

```bash
python -m oph_fpe.cli oph-boltzmann-inputs \
  --run-dir runs/camb_lcdm_baseline_20260606 \
            runs/cmb_adapter_smoke_64k_20260606 \
            runs/cmb_adapter_smoke_64k_256k_20260606 \
  --out runs/oph_boltzmann_inputs_cmb_adapter_64k_256k_20260606
```

Regenerate the frozen transfer/likelihood closure audit from a run bundle:

```bash
python -m oph_fpe.cli frozen-transfer-likelihood \
  --run-dir runs/<run_id> \
  --out runs/<run_id>/frozen_transfer_likelihood \
  --solver CAMB \
  --solver-version-pin <camb-version> \
  --source-plugin-hash sha256:<source-plugin-hash>
```

Regenerate the comparable-data snapshot:

```bash
python -m oph_fpe.cli comparable-data \
  --run-dir runs/camb_lcdm_baseline_20260606 \
            runs/cmb_adapter_smoke_64k_20260606 \
            runs/cmb_adapter_smoke_64k_256k_20260606 \
            runs/oph_boltzmann_inputs_cmb_adapter_64k_256k_20260606 \
  --include runs/cmb_transfer_adapter_smoke_64k_to_256k_20260606 \
  --out runs/comparable_data_cmb_boltzmann_readouts_20260606
```

## Claim Boundary

The CAMB lane is a standard external LambdaCDM regression target. The
screen `C_l` lane is a normalized-axis diagnostic unless the real-ell coverage
gate passes. The OPH repair-exchange rows are finite-collar diagnostics; they
are not physical `rho_A(a)`, `rho_A_eq(a)`, `Gamma_rec(k,a)`, or `B_A(k,a)`
Boltzmann inputs.
