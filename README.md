# Monkey ECoG

**Reconstruction of the large scale connectome based on macaque ECoG data.**

## Compiler Cython-integrated MI estimator

```bash
cd mutual_information
python setup.py build_ext --inplace
# copy compiled lib (*.so) to the parent folder
cp mutual_info_cy.cpython-37m-darwin.so ..
```

## Data file structure

### Filename: `preprocessed_data.npz`
| Variable name            | Descriptions                                     |
| ------------------------ | ------------------------------------------------ |
|`'stride'                `| cumulative sum of number of channels in each area|
|`'adj_mat'               `| measured weight adjacent matrix (area-wise)      |
|`'weight'                `| measured weight adjacent matrix (channel-wise)   |
|`'data_series_raw'       `| unfiltered data                                  |
|`'data_series_delta'     `| delta band filtered data  (1-4 Hz)               |
|`'data_series_theta'     `| theta band filtered data  (5-8 Hz)               |
|`'data_series_alpha'     `| alpha band filtered data  (9-12 Hz)              |
|`'data_series_beta'      `| beta  band filtered data  (13-30 Hz)             |
|`'data_series_gamma'     `| gamma band filtered data  (31-100 Hz)            |
|`'data_series_high_gamma'`| high gamma band filtered data  (55-100 Hz)       |

---
### Filename: `tdmi_data.npz`
| Variable name | Descriptions                                             |
| ------------- | -------------------------------------------------------- |
|`'raw'       ` | tdmi data for unfiltered data                            |
|`'delta'     ` | tdmi data for delta band filtered data  (1-4 Hz)         |
|`'theta'     ` | tdmi data for theta band filtered data  (5-8 Hz)         |
|`'alpha'     ` | tdmi data for alpha band filtered data  (9-12 Hz)        |
|`'beta'      ` | tdmi data for beta  band filtered data  (13-30 Hz)       |
|`'gamma'     ` | tdmi data for gamma band filtered data  (31-100 Hz)      |
|`'high_gamma'` | tdmi data for high gamma band filtered data  (55-100 Hz) |

---
### Filename: `tdmi_data_shuffle.npz`
| Variable name | Descriptions                                                      |
| ------------- | ----------------------------------------------------------------- |
|`'raw'       ` | shuffled tdmi data for unfiltered data                            |
|`'delta'     ` | shuffled tdmi data for delta band filtered data  (1-4 Hz)         |
|`'theta'     ` | shuffled tdmi data for theta band filtered data  (5-8 Hz)         |
|`'alpha'     ` | shuffled tdmi data for alpha band filtered data  (9-12 Hz)        |
|`'beta'      ` | shuffled tdmi data for beta  band filtered data  (13-30 Hz)       |
|`'gamma'     ` | shuffled tdmi data for gamma band filtered data  (31-100 Hz)      |
|`'high_gamma'` | shuffled tdmi data for high gamma band filtered data  (55-100 Hz) |

---
### Filename: `opt_threshold_*.npz`
| Variable name | Descriptions                                                                                     |
| ------------- | ------------------------------------------------------------------------------------------------ |
|`'raw'       ` | optimal threshold chosen by Youden Index criteria for unfiltered data                            |
|`'delta'     ` | optimal threshold chosen by Youden Index criteria for delta band filtered data  (1-4 Hz)         |
|`'theta'     ` | optimal threshold chosen by Youden Index criteria for theta band filtered data  (5-8 Hz)         |
|`'alpha'     ` | optimal threshold chosen by Youden Index criteria for alpha band filtered data  (9-12 Hz)        |
|`'beta'      ` | optimal threshold chosen by Youden Index criteria for beta  band filtered data  (13-30 Hz)       |
|`'gamma'     ` | optimal threshold chosen by Youden Index criteria for gamma band filtered data  (31-100 Hz)      |
|`'high_gamma'` | optimal threshold chosen by Youden Index criteria for high gamma band filtered data  (55-100 Hz) |

---
### Filename: `gc_order_n.npz`, `gc_shuffled_order_n.npz`
| Variable name | Descriptions                                                      |
| ------------- | ----------------------------------------------------------------- |
|`'raw'       ` | GC data for unfiltered data                                       |
|`'delta'     ` | GC data for delta band filtered data  (1-4 Hz)                    |
|`'theta'     ` | GC data for theta band filtered data  (5-8 Hz)                    |
|`'alpha'     ` | GC data for alpha band filtered data  (9-12 Hz)                   |
|`'beta'      ` | GC data for beta  band filtered data  (13-30 Hz)                  |
|`'gamma'     ` | GC data for gamma band filtered data  (31-100 Hz)                 |
|`'high_gamma'` | GC data for high gamma band filtered data  (55-100 Hz)            |
