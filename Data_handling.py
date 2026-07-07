import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

file_path = '/Users/oliviersaidoun/Documents/GitHub/SPY_allocation_HMM/SPY.csv'
df = pd.read_csv(file_path)


def prep_data(df):
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df['avg_price'] = df[['open', 'high', 'low', 'close']].mean(axis=1)
    df['logreturn'] = np.log(df['close'] / df['close'].shift(1))
    df['realized_volatility'] = df['logreturn'].rolling(window=21).std() * np.sqrt(252)
    df.dropna(inplace=True)
    return df


def cusum_square(series):
    series = series.dropna().values
    n = len(series)
    sq = series ** 2
    cum_sq = np.cumsum(sq)
    total = cum_sq[-1]
    C = cum_sq / total
    k = np.arange(1, n + 1) / n
    D = C - k
    stat = np.max(np.abs(D)) * np.sqrt(n / 2)
    break_idx = np.argmax(np.abs(D))
    return D, stat, break_idx


def plot_cusum(D, stat, break_idx, dates, title):
    n = len(D)
    crit_95 = 1.36 / np.sqrt(n / 2)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, D, label="CUSUM of squares deviation")
    ax.axhline(crit_95, color="red", linestyle="--", label="95% critical bound")
    ax.axhline(-crit_95, color="red", linestyle="--")
    ax.axvline(dates[break_idx], color="black", linestyle=":", label=f"Max deviation: {dates[break_idx].date()}")
    ax.set_title(f"{title} — stat = {stat:.3f} (crit ≈ 1.36)")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    file_path = "/Users/oliviersaidoun/Documents/GitHub/SPY_allocation_HMM/SPY.csv"
    df = pd.read_csv(file_path)
    df = prep_data(df)

    result1 = adfuller(df['logreturn'].dropna())
    result2 = adfuller(df['realized_volatility'].dropna())

    print('p-value for logreturn:', result1[1])
    print('p-value for realized_volatility:', result2[1])

    ret_clean = df['logreturn'].dropna()
    D_ret, stat_ret, idx_ret = cusum_square(ret_clean)
    plot_cusum(D_ret, stat_ret, idx_ret, ret_clean.index, "CUSUMSQ Log Returns")

    rv_clean = df['realized_volatility'].dropna()
    D_rv, stat_rv, idx_rv = cusum_square(rv_clean)
    plot_cusum(D_rv, stat_rv, idx_rv, rv_clean.index, "CUSUMSQ Realized Volatility")