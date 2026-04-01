"""
Run all 9 stock analyses and compute MAE alongside RMSE.

Replicates each notebook's pipeline:
  1. Load train CSV
  2. Instantiate the best model (identified in each notebook)
  3. Compute cross-validation MAE and RMSE using the same CV method as the notebook
  4. Run conformal prediction to get bid/ask spread
  5. Print a summary table with MAE included
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score, LeaveOneOut, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.base import clone
from xgboost import XGBRegressor

DATA_DIR = 'hackathon_data'
RNG = np.random.RandomState(42)


# ── Stock configs: best model, CV method, conformal method ──────────────────

STOCK_CONFIGS = {
    1: {
        'best_model_name': 'Ridge_0.1',
        'best_model': Ridge(alpha=0.1),
        'cv': 5,
        'conformal': 'split',
    },
    2: {
        'best_model_name': 'Lasso_0.01',
        'best_model': Lasso(alpha=0.01, max_iter=10000),
        'cv': 5,
        'conformal': 'split',
    },
    3: {
        'best_model_name': 'Ridge_0.1',
        'best_model': Ridge(alpha=0.1),
        'cv': 'LOO',
        'conformal': 'LOO',
    },
    4: {
        'best_model_name': 'XGB_lr0.01_d5',
        'best_model': XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.01,
                                    random_state=42, verbosity=0, n_jobs=-1),
        'cv': 'holdout',
        'conformal': 'split',
    },
    5: {
        'best_model_name': 'Ridge_100',
        'best_model': Ridge(alpha=100.0),
        'cv': 5,
        'conformal': 'split',
    },
    6: {
        'best_model_name': 'Lasso_10',
        'best_model': Lasso(alpha=10.0, max_iter=10000),
        'cv': 'LOO',
        'conformal': 'LOO',
    },
    7: {
        'best_model_name': 'XGB_lr05',
        'best_model': XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                    random_state=42, verbosity=0),
        'cv': 5,
        'conformal': 'split',
    },
    8: {
        'best_model_name': 'Lasso_1',
        'best_model': Lasso(alpha=1.0, max_iter=10000),
        'cv': 'holdout',
        'conformal': 'split',
    },
    9: {
        'best_model_name': 'Lasso_10',
        'best_model': Lasso(alpha=10.0, max_iter=10000),
        'cv': 10,  # min(59, 10) = 10-fold
        'conformal': 'LOO',
    },
}


# ── Helper functions (from notebooks) ──────────────────────────────────────

def compute_conformal_residuals_split(X, y, model, cal_fraction=0.3):
    n = len(X)
    shuffled_indices = RNG.permutation(n)
    n_train = int((1 - cal_fraction) * n)
    train_idx = shuffled_indices[:n_train]
    cal_idx = shuffled_indices[n_train:]

    m = clone(model)
    m.fit(X[train_idx], y[train_idx])
    cal_preds = m.predict(X[cal_idx])
    abs_resid = np.sort(np.abs(y[cal_idx] - cal_preds))
    return abs_resid


def compute_conformal_residuals_loo(X, y, model):
    n = len(X)
    abs_resid = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        m = clone(model)
        m.fit(X[mask], y[mask])
        pred = m.predict(X[i:i+1])[0]
        abs_resid[i] = abs(y[i] - pred)
    return np.sort(abs_resid)


def sweep_coverage(abs_residuals, rmse):
    competitor_spread = 2 * rmse
    results = []
    for coverage_pct in range(50, 100):
        coverage = coverage_pct / 100.0
        q = np.quantile(abs_residuals, coverage)
        our_width = 2 * q
        we_are_mm = our_width < competitor_spread

        tail = abs_residuals[abs_residuals > q]
        avg_tail_loss = np.mean(tail) if len(tail) > 0 else 0.0

        if we_are_mm:
            reward = our_width * 0.5
            expected_pnl = coverage * reward - (1 - coverage) * avg_tail_loss
        else:
            expected_pnl = 0.0

        results.append({
            'coverage': coverage,
            'our_width': our_width,
            'expected_pnl': expected_pnl,
        })
    return results


def compute_cv_mae(model, X, y, cv_method):
    """Compute MAE using the same CV strategy as the notebook."""
    if cv_method == 'LOO':
        cv = LeaveOneOut()
    elif cv_method == 'holdout':
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        m = clone(model)
        m.fit(X_tr, y_tr)
        preds = m.predict(X_val)
        return mean_absolute_error(y_val, preds)
    else:
        cv = cv_method  # integer k-fold

    mae_scores = cross_val_score(model, X, y, cv=cv,
                                  scoring='neg_mean_absolute_error', n_jobs=-1)
    return -mae_scores.mean()


def compute_cv_rmse(model, X, y, cv_method):
    """Compute RMSE using the same CV strategy as the notebook (for verification)."""
    if cv_method == 'LOO':
        cv = LeaveOneOut()
    elif cv_method == 'holdout':
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        m = clone(model)
        m.fit(X_tr, y_tr)
        preds = m.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))
    else:
        cv = cv_method

    mse_scores = cross_val_score(model, X, y, cv=cv,
                                  scoring='neg_mean_squared_error', n_jobs=-1)
    return np.sqrt(-mse_scores.mean())


# ── Main ────────────────────────────────────────────────────────────────────

def run_all():
    results = []

    for stock_id in range(1, 10):
        cfg = STOCK_CONFIGS[stock_id]
        print(f"Processing stock {stock_id} ...", end=" ", flush=True)

        # Reset RNG per stock (notebooks each start with RNG = RandomState(42))
        global RNG
        RNG = np.random.RandomState(42)

        train_df = pd.read_csv(f'{DATA_DIR}/stock_{stock_id}_train.csv')
        test_df = pd.read_csv(f'{DATA_DIR}/stock_{stock_id}_test.csv')

        feature_cols = [c for c in train_df.columns if c != 'target']
        X_train = train_df[feature_cols].values
        y_train = train_df['target'].values
        X_test = test_df[feature_cols].values

        n_samples = len(y_train)
        n_features = len(feature_cols)
        target_mean = round(y_train.mean(), 2)
        target_std = round(y_train.std(), 2)

        model = cfg['best_model']
        cv_method = cfg['cv']

        # Compute MAE and RMSE
        rmse = compute_cv_rmse(model, X_train, y_train, cv_method)
        mae = compute_cv_mae(model, X_train, y_train, cv_method)

        # Conformal prediction
        if cfg['conformal'] == 'LOO':
            conf_resid = compute_conformal_residuals_loo(X_train, y_train, model)
        else:
            conf_resid = compute_conformal_residuals_split(X_train, y_train, model)

        # Optimal coverage sweep
        sweep_results = sweep_coverage(conf_resid, rmse)
        best_pnl = -999999.0
        opt_idx = 0
        for i in range(len(sweep_results)):
            if sweep_results[i]['expected_pnl'] > best_pnl:
                best_pnl = sweep_results[i]['expected_pnl']
                opt_idx = i
        opt = sweep_results[opt_idx]
        opt_coverage = opt['coverage']
        opt_pnl = opt['expected_pnl']

        # Ensemble prediction (top-1 model only since we only have the best)
        m_final = clone(model)
        m_final.fit(X_train, y_train)
        ensemble_pred = m_final.predict(X_test)[0]

        # Bid/Ask
        half_spread = np.quantile(conf_resid, opt_coverage)
        bid = ensemble_pred - half_spread
        ask = ensemble_pred + half_spread
        spread = ask - bid

        results.append({
            'Stock': stock_id,
            'Samples': n_samples,
            'Features': n_features,
            'Mean': target_mean,
            'Std': target_std,
            'Best Model': cfg['best_model_name'],
            'RMSE': round(rmse, 4),
            'MAE': round(mae, 4),
            'Opt Coverage': opt_coverage,
            'E[PnL]': round(opt_pnl, 2),
            'Ensemble Pred': round(ensemble_pred, 4),
            'Bid': round(bid, 2),
            'Ask': round(ask, 2),
            'Spread': round(spread, 2),
        })
        print(f"done (RMSE={rmse:.4f}, MAE={mae:.4f})")

    # Print summary table
    df = pd.DataFrame(results)
    print("\n" + "=" * 120)
    print("ALL STOCKS SUMMARY")
    print("=" * 120)
    print(df.to_string(index=False))
    print()

    return df


if __name__ == '__main__':
    run_all()
