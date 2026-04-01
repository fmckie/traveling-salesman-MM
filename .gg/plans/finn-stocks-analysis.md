# Plan: Run Finn's Stock Analyses 1–9 & Compile Results Table with MAE

## Analysis

Finn's 9 stock analysis notebooks (`finn's-stocks/stock_*_analysis.ipynb`) each follow the same pipeline:
1. Load train/test CSVs
2. Fit multiple models (Ridge, Lasso, XGB, etc.)
3. Select best model by RMSE (cross-validation)
4. Compute conformal prediction residuals to set bid/ask spread
5. Output: Best Model, RMSE, Ensemble Prediction, Bid, Ask, Spread

**MAE is NOT currently computed** in any of the 9 notebooks. We need to add MAE computation. The notebooks already compute `conformal_abs_residuals` (absolute residuals from the conformal calibration set). We can also compute MAE from cross-validation alongside RMSE.

### Current Output Data (from notebook cell outputs)

| Stock | Samples | Features | Mean   | Std   | Best Model     | Best RMSE | Opt. Coverage | E[PnL] | Ensemble Pred | Bid    | Ask    | Spread |
|-------|---------|----------|--------|-------|----------------|-----------|---------------|--------|---------------|--------|--------|--------|
| 1     | 19,999  | 5        | 246.14 | 39.05 | Ridge_0.1      | 4.8675    | 0.68          | 0.91   | 273.9293      | 269.07 | 278.79 | 9.72   |
| 2     | 1,499   | 15       | 219.37 | 48.60 | Lasso_0.01     | 9.7035    | 0.69          | 2.11   | 220.8921      | 211.42 | 230.37 | 18.95  |
| 3     | 29      | 4        | 205.49 | 77.39 | Ridge_0.1      | 24.6147   | 0.77          | 9.56   | 259.7488      | 235.29 | 284.21 | 48.91  |
| 4     | 9,999   | 12       | 245.40 | 27.76 | XGB_lr0.01_d5  | 24.7039   | 0.92          | 15.83  | 236.3262      | 212.26 | 260.40 | 48.14  |
| 5     | 799     | 20       | 253.14 | 30.77 | Ridge_100      | 28.0916   | 0.89          | 15.64  | 249.2542      | 222.30 | 276.21 | 53.91  |
| 6     | 119     | 8        | 172.24 | 54.69 | Lasso_10       | 55.0004   | 0.80          | 23.88  | 173.7326      | 119.02 | 228.44 | 109.42 |
| 7     | 19,999  | 25       | 211.04 | 15.84 | XGB_lr05       | 15.4931   | 0.83          | 7.29   | 213.9388      | 198.56 | 229.32 | 30.75  |
| 8     | 1,999   | 25       | 197.35 | 26.31 | Lasso_1        | 25.5278   | 0.79          | 8.95   | 200.5320      | 176.36 | 224.70 | 48.34  |
| 9     | 59      | 20       | 218.60 | 43.37 | Lasso_10       | 45.3053   | 0.78          | 17.84  | 223.3027      | 178.48 | 268.12 | 89.64  |

### Approach for MAE

Create a single Python script that:
1. Runs each stock's model pipeline (replicating the notebook logic)
2. Computes MAE alongside RMSE during cross-validation
3. Outputs the complete table with MAE included

Alternatively, we can add MAE computation cells to each notebook and re-run them. The simpler approach is a standalone script.

## Steps

1. Create a Python script `finn's-stocks/run_all_analysis.py` that loads each stock's train CSV, fits the best model identified in each notebook, computes cross-validation MAE using `mean_absolute_error` from sklearn, and prints a summary table
2. Run the script to get MAE values for all 9 stocks
3. Create a summary notebook or markdown file `finn's-stocks/all_stocks_summary.md` compiling all data (Samples, Features, Target Mean/Std, Best Model, RMSE, MAE, Optimal Coverage, E[PnL], Ensemble Prediction, Bid, Ask, Spread) into a formatted table
