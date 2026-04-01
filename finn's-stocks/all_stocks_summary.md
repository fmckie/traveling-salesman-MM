# All Stocks Summary — Finn's Analysis

Results from running the full model pipeline on all 9 stocks. Each stock uses conformal prediction to set bid/ask spreads that are tighter than the competitor baseline (±1×RMSE).

## Results Table

| Stock | Samples | Features | Mean   | Std   | Best Model     | RMSE    | MAE     | Opt. Coverage | E[PnL] | Ensemble Pred | Bid    | Ask    | Spread |
|------:|--------:|---------:|-------:|------:|:---------------|--------:|--------:|--------------:|-------:|--------------:|-------:|-------:|-------:|
|     1 |  19,999 |        5 | 246.14 | 39.05 | Ridge_0.1      |  4.8675 |  3.8822 |          0.68 |   0.91 |      273.9293 | 269.07 | 278.79 |   9.72 |
|     2 |   1,499 |       15 | 219.37 | 48.60 | Lasso_0.01     |  9.7035 |  7.7093 |          0.69 |   2.11 |      220.8921 | 211.42 | 230.37 |  18.95 |
|     3 |      29 |        4 | 205.49 | 77.39 | Ridge_0.1      | 24.6147 | 19.0350 |          0.77 |   9.56 |      259.7488 | 235.29 | 284.21 |  48.91 |
|     4 |   9,999 |       12 | 245.40 | 27.76 | XGB_lr0.01_d5  | 24.7039 | 12.0925 |          0.92 |  15.83 |      236.3262 | 212.26 | 260.40 |  48.14 |
|     5 |     799 |       20 | 253.14 | 30.77 | Ridge_100      | 28.0916 | 15.4516 |          0.89 |  15.64 |      249.2542 | 222.30 | 276.21 |  53.91 |
|     6 |     119 |        8 | 172.24 | 54.69 | Lasso_10       | 55.0004 | 36.7889 |          0.80 |  23.88 |      173.7326 | 119.02 | 228.44 | 109.42 |
|     7 |  19,999 |       25 | 211.04 | 15.84 | XGB_lr05       | 15.4931 |  9.3736 |          0.83 |   7.29 |      213.9388 | 198.56 | 229.32 |  30.75 |
|     8 |   1,999 |       25 | 197.35 | 26.31 | Lasso_1        | 25.5278 | 16.4315 |          0.79 |   8.95 |      200.5320 | 176.36 | 224.70 |  48.34 |
|     9 |      59 |       20 | 218.60 | 43.37 | Lasso_10       | 45.3053 | 29.5820 |          0.78 |  17.84 |      223.3027 | 178.48 | 268.12 |  89.64 |

## Notes

- **RMSE** and **MAE** are computed via cross-validation matching each notebook's approach:
  - Stocks 1, 2, 5, 7: 5-fold CV
  - Stocks 3, 6: Leave-One-Out CV
  - Stocks 4, 8: 80/20 holdout split
  - Stock 9: 10-fold CV (min(n, 10))
- **Ensemble Pred / Bid / Ask / Spread** use the top-3 ensemble from each notebook (values taken from notebook outputs).
- **Conformal prediction** uses split conformal (stocks 1,2,4,5,7,8) or LOO conformal (stocks 3,6,9) to calibrate the spread.
- **Opt. Coverage** is the coverage level that maximises E[PnL] assuming competitors use ±1×RMSE as their spread.
- MAE values were computed by `run_all_analysis.py` using `sklearn.metrics.mean_absolute_error` via the same CV strategy.
