import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox
import scipy.stats as stats

file_path = '/Users/oliviersaidoun/Documents/GitHub/SPY_allocation_HMM/SPY.csv'
df = pd.read_csv(file_path)

"""
GARCH(1,1) baseline for realized volatility comparison against HMM regimes.
"""

# ── 1. Fit ──────────────────────────────────────────────────────────────────

def fit_garch(returns, dist="normal"):
    """
    Fit a GARCH(1,1) model on returns (in %, arch expects scaled returns
    for numerical stability).
    """
    scaled_returns = returns * 100
    am = arch_model(scaled_returns, mean="Constant", vol="Garch", p=1, q=1, dist=dist)
    res = am.fit(disp="off")
    return res


# ── 2. Extract conditional volatility (annualized, back to decimal scale) ───

def get_conditional_vol(res):
    """
    Convert the fitted conditional variance into annualized volatility,
    in the same scale as your realized_volatility column.
    """
    cond_vol = res.conditional_volatility / 100  # undo the *100 scaling
    annualized = cond_vol * np.sqrt(252)
    return annualized


# ── 3. Rolling-origin forecast (out-of-sample, no lookahead) ────────────────

def rolling_forecast(returns, window=750, horizon=1, dist="normal"):
    """
    Re-fit GARCH on an expanding window and produce one-step-ahead forecasts.
    This avoids lookahead bias — each forecast only uses data available
    up to that point in time.
    """
    forecasts = pd.Series(index=returns.index, dtype=float)

    for t in range(window, len(returns)):
        train = returns.iloc[:t] * 100
        am = arch_model(train, mean="Constant", vol="Garch", p=1, q=1, dist=dist)
        res = am.fit(disp="off")
        fcast = res.forecast(horizon=horizon, reindex=False)
        var_forecast = fcast.variance.values[-1, 0]
        vol_forecast = np.sqrt(var_forecast) / 100 * np.sqrt(252)
        forecasts.iloc[t] = vol_forecast

    return forecasts.dropna()


# ── 4. Diagnostics ───────────────────────────────────────────────────────────

def run_diagnostics(res, lags=10):
    """
    Check standardized residuals for remaining ARCH effects (should be
    approximately white noise if the model is well-specified) and check
    the normality assumption.
    """
    std_resid = res.std_resid.dropna()

    # Ljung-Box on squared standardized residuals: tests if ARCH effects remain
    lb_test = acorr_ljungbox(std_resid ** 2, lags=[lags], return_df=True)
    print("Ljung-Box test on squared standardized residuals:")
    print(lb_test)
    print(
        "-> If p-value > 0.05, no significant remaining ARCH effects "
        "(model adequately captures volatility clustering).\n"
    )

    # Normality check
    jb_stat, jb_p = stats.jarque_bera(std_resid)
    print(f"Jarque-Bera test: stat={jb_stat:.3f}, p-value={jb_p:.4f}")
    print(
        "-> If p-value < 0.05, residuals are not normal "
        "(common with 'normal' dist assumption; consider dist='t').\n"
    )

    return std_resid


def plot_diagnostics(std_resid):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(std_resid.index, std_resid)
    axes[0].axhline(0, color="black", linewidth=0.7)
    axes[0].set_title("Standardized Residuals")

    stats.probplot(std_resid, dist="norm", plot=axes[1])
    axes[1].set_title("QQ Plot vs Normal")

    plt.tight_layout()
    plt.show()


def plot_vol_comparison(realized_vol, garch_vol, title="Realized vs GARCH Conditional Volatility"):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(realized_vol.index, realized_vol, label="Realized Volatility", alpha=0.7)
    ax.plot(garch_vol.index, garch_vol, label="GARCH(1,1) Conditional Volatility", alpha=0.8)
    ax.set_title(title)
    ax.legend()
    plt.show()


# ── 5. Main pipeline ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    from Data_handling import prep_data

    file_path = "/Users/oliviersaidoun/Documents/GitHub/SPY_allocation_HMM/SPY.csv"
    df = pd.read_csv(file_path)
    df = prep_data(df)

    # Full-sample fit for diagnostics
    res = fit_garch(df["logreturn"], dist="normal")
    print(res.summary())

    df["garch_vol"] = get_conditional_vol(res)
    plot_vol_comparison(df["realized_volatility"], df["garch_vol"])

    std_resid = run_diagnostics(res, lags=10)
    plot_diagnostics(std_resid)

    # Rolling out-of-sample forecast (this is the piece you'll actually
    # compare the HMM regimes against later — expanding window, no lookahead)
    rolling_vol_forecast = rolling_forecast(df["logreturn"], window=750)
    df["garch_rolling_forecast"] = rolling_vol_forecast

    df.to_csv("SPY_with_garch.csv")