# CloseFlyBy(F) public certificates

These rows are generated from public comparator data and raw-data availability manifests.
They are not solved historical certificates until OD replay fills A_hist, A_full, A_proj, and residual tests.

| Flyby | Status | Public dVinf mm/s | Anderson mm/s | Residual mm/s | Raw receipt | Missing receipts |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| Galileo I | OD_REPLAY_PENDING | 3.92 | 4.122493 | -0.202493 | RAW_TRACKING_PARTIAL_REPLAY_PENDING | 6 |
| Galileo II | OD_REPLAY_PENDING | -4.6 | -4.673874 | 0.073874 | RAW_TRACKING_PARTIAL_REPLAY_PENDING | 6 |
| NEAR | OD_REPLAY_PENDING | 13.46 | 13.277863 | 0.182137 | RAW_TRACKING_AVAILABLE_REPLAY_PENDING | 8 |
| Cassini | OD_REPLAY_PENDING | -0.5 | -1.068046 | 0.568046 | RAW_TRACKING_TO_LOCATE_REPLAY_PENDING | 8 |
| Rosetta I | OD_REPLAY_PENDING | 1.8 | 2.066282 | -0.266282 | RAW_RSI_DATA_AVAILABLE_REPLAY_PENDING | 6 |
| MESSENGER | OD_REPLAY_PENDING | 0.02 | 0.055302 | -0.035302 | RAW_TRACKING_TO_LOCATE_REPLAY_PENDING | 7 |
| Rosetta II | OD_REPLAY_PENDING | 0.0 | 1.001276 | -1.001276 | NULL_CONTROL_ARCHIVE_REPLAY_PENDING | 6 |
| EPOXI I | OD_REPLAY_PENDING | 0.0 | 0.418065 | -0.418065 | NULL_CONTROL_ARCHIVE_REPLAY_PENDING | 6 |
| EPOXI II | OD_REPLAY_PENDING | 0.0 | 5.572116 | -5.572116 | NULL_CONTROL_ARCHIVE_REPLAY_PENDING | 6 |
| Rosetta III | OD_REPLAY_PENDING | 0.0 | 1.089468 | -1.089468 | NULL_CONTROL_ARCHIVE_REPLAY_PENDING | 6 |
| EPOXI III | OD_REPLAY_PENDING | 0.0 | -5.692015 | 5.692015 | NULL_CONTROL_ARCHIVE_REPLAY_PENDING | 6 |
| Juno | OD_REPLAY_PENDING | 0.0 | 6.425214 | -6.425214 | NULL_CONTROL_ARCHIVE_REPLAY_PENDING | 6 |

Closure rule: a non-null row can become SOLVED_PROJECTION_ARTIFACT only after real-data OD replay shows
A_hist approximately equals the public residual, A_full is zero within the replay floor,
A_proj equals A_hist - A_full and matches the public residual, and residual/covariance/holdout tests pass.

Synthetic projection fixtures validate this machinery only; they do not close historical rows.
