# Official Planck Likelihood Readiness

Runtime readiness check only. Official Planck likelihood execution also needs the ESA PR3 likelihood data files and validated nuisance/path configuration.

## Gates

- CAMB available: `true`
- Cobaya available: `false`
- official clik API available: `false`
- likelihood data paths configured: `false`
- official likelihood execution ready: `false`

## Blockers

- `official_clik_api_not_available`
- `official_planck_likelihood_data_path_not_configured`
- `cobaya_not_importable`

## Data Path Environment

- `OPH_PLANCK_LIKELIHOOD_DIR`: configured `false`, exists `false`
- `PLANCK_PR3_LIKELIHOOD_DIR`: configured `false`, exists `false`
- `PLANCK_LIKELIHOOD_DIR`: configured `false`, exists `false`
- `CLIK_DATA`: configured `false`, exists `false`
