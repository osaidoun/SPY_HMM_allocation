import hmmlearn as hmm
from hmmlearn.hmm import GaussianHMM
import pandas as pd
from Data_handling import prep_data
from garch import fit_garch, get_conditional_vol, rolling_forecast
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def fit_hmm(df, n_states):
    scaler = StandardScaler()
    df[["logreturn", "realized_volatility"]] = scaler.fit_transform(df[["logreturn", "realized_volatility"]])
    model = GaussianHMM(n_components = n_states, covariance_type = "full", n_iter = 1000)
    model.fit(df[["logreturn", "realized_volatility"]])
    hidden_states = model.predict(df[["logreturn", "realized_volatility"]])
    df["hmm_state"] = hidden_states
    return model, df


if __name__ == "__main__":
    
    file_path = fr'C:\Users\OSAIDO\Downloads\PP\SPY.csv'
    df = pd.read_csv(file_path)
    df = prep_data(df)

    n_states = 3
    model, df = fit_hmm(df, n_states)

    mask_2020 = (df.index >= "2020-01-01")
    df_2020 = df.loc[mask_2020]

    colors = {
        0: "lightgreen",
        1: "khaki",
        2: "lightcoral"
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    # Price line
    ax.plot(
        df_2020.index,
        df_2020["avg_price"],
        color="black",
        lw=1.5,
        label="SPY"
    )

    # Regime changes
    state_changes = df_2020["hmm_state"].ne(
        df_2020["hmm_state"].shift()
    ).cumsum()

    for _, group in df_2020.groupby(state_changes):

        state = group["hmm_state"].iloc[0]

        ax.axvspan(
            group.index[0],
            group.index[-1],
            color=colors[state],
            alpha=0.3
        )

    ax.set_title("SPY Price with HMM Regimes")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")

    plt.show()

