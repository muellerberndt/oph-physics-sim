# Official Planck likelihood install and run guide

This runtime has `camb` and `cobaya`, but it does not contain the official Planck `clik` library or the PR3 likelihood data package. The official Planck run therefore has to be executed in an environment where the following ESA products are installed:

- `COM_Likelihood_Code-v3.0_R3.01.tar.gz`
- `COM_Likelihood_Data-baseline_R3.00.tar.gz`

Official ESA product page:

```text
https://esdcdoi.esac.esa.int/doi/html/data/astronomy/planck/Cosmology.html
```

Cobaya documentation for Planck likelihoods:

```text
https://cobaya.readthedocs.io/en/latest/likelihood_planck.html
```

## Minimal official PR3 path

```bash
mkdir -p $HOME/likelihoods/planck_2018
cd $HOME/likelihoods/planck_2018

# Official PLC/clik code and baseline data from ESA PLA.
# Use browser or wget where available.
wget 'https://pla.esac.esa.int/pla-sl/data-action?COSMOLOGY.COSMOLOGY_OID=151912' -O planck2018_code.tar.bz2
# Or download COM_Likelihood_Code-v3.0_R3.01.tar.gz from the ESA DOI page.

# Build clik.
cd code/plc_3.0/plc-3.01
python waf configure --install_all_deps
python waf install

# Confirm the Python library is visible.
python - <<'PY'
import clik
print('clik import OK')
PY
```

Then unpack `COM_Likelihood_Data-baseline_R3.00.tar.gz` under the same Planck 2018 path.

## Expected baseline clik files

The official PR3 baseline evaluation normally uses:

```text
low_l/commander/commander_dx12_v3_2_29.clik
low_l/simall/simall_100x143_offlike5_EE_Aplanck_B.clik
hi_l/plik/plik_rd12_HM_v22b_TTTEEE.clik
lensing/smica_g30_ftl_full_pp.clik_lensing
```

Path names can vary by unpacked distribution. The scripts in `scripts/` accept explicit file paths so the exact local layout can be used.

## What counts as success

A completed official run should produce:

```text
loglike_LCDM_baseline
loglike_OPH_exact_scalar
Delta(-2lnL)
DeltaAIC
DeltaBIC
bestfit nuisance vector or profiled nuisance treatment
TT,TE,EE,lensing components separately
```

The v1.0 package supplies the official likelihood wrapper and configs. It does not include ESA likelihood data or `clik` binaries.
