import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from itertools import combinations
from tqdm import tqdm
import seaborn as sns
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# ----------------------------- Global Plot Style -----------------------------
sns.set_style("whitegrid")
plt.style.use("seaborn-v0_8-muted")

# ----------------------------- Helper functions -----------------------------
def adf_test(series, signif=0.05):
    """Perform ADF test and return True if stationary."""
    try:
        result = adfuller(series, autolag="AIC")
        return result[1] <= signif
    except Exception:
        return False

def compute_hurst(series, window=100):
    """Compute rolling Hurst exponent to classify regime."""
    hurst_vals = []
    for i in range(window, len(series)):
        windowed = series[i-window:i]
        if windowed.std() == 0:
            hurst_vals.append(np.nan)
            continue
        R = (windowed.cumsum() - windowed.cumsum().mean()).max() - (windowed.cumsum() - windowed.cumsum().mean()).min()
        S = windowed.std()
        hurst_vals.append(np.log(R/S)/np.log(window))
    return pd.Series([np.nan]*window + hurst_vals, index=series.index)

def pairs_trading_ratio(data1, data2, t1, t2, results_folder,
                        lookback=90, entry_z=2.0, exit_long=-0.5, exit_short=0.75,
                        cost=0.001, stop_loss=-0.05, cointegration_p_threshold=0.05):
    df = pd.concat([data1, data2], axis=1).dropna()
    df.columns = [t1, t2]
    if df.empty:
        return None

    # Hedge ratio via OLS
    y = df[t1]
    x = sm.add_constant(df[t2])
    model = sm.OLS(y, x).fit()
    beta = model.params[1]
    residuals = model.resid

    # Cointegration test
    adf_result = adfuller(residuals)
    if adf_result[1] >= cointegration_p_threshold:
        return None

    # Spread and z-score
    df["Spread"] = df[t1] - beta * df[t2]
    df["RollingMean"] = df["Spread"].rolling(window=lookback).mean()
    df["RollingStd"] = df["Spread"].rolling(window=lookback).std()
    df["ZScore"] = (df["Spread"] - df["RollingMean"]) / df["RollingStd"]

    # Trading signals
    df["Signal"] = 0
    df.loc[df["ZScore"] > entry_z, "Signal"] = -1
    df.loc[df["ZScore"] < -entry_z, "Signal"] = 1
    df["Position"] = df["Signal"].replace(0, np.nan).ffill().fillna(0)

    # Asymmetric exit
    df.loc[(df["Position"].shift(1) == 1) & (df["ZScore"] > exit_long), "Signal"] = 0
    df.loc[(df["Position"].shift(1) == -1) & (df["ZScore"] < exit_short), "Signal"] = 0
    df["Position"] = df["Signal"].replace(0, np.nan).ffill().fillna(0)

    # Trades and PnL
    df["Trade"] = (df["Position"] != df["Position"].shift(1)).astype(int)
    df["PnL"] = df["Position"].shift(1) * df["Spread"].pct_change().fillna(0) - cost * df["Trade"]
    df.loc[df["PnL"] < stop_loss, "Signal"] = 0
    df["Position"] = df["Signal"].replace(0, np.nan).ffill().fillna(0)
    df["CumulativePnL"] = df["PnL"].cumsum()

    # Compute average Hurst to classify pair
    df["Hurst"] = compute_hurst(df["Spread"], window=lookback)
    mean_hurst = df["Hurst"].mean()
    pair_regime = "Mean Reverting" if mean_hurst < 0.5 else "Trend Following"

    # -------------------- Plots --------------------
    fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # 1. Spread plot with rolling mean and bands
    axs[0].plot(df.index, df["Spread"], label="Spread", color="#1f77b4", lw=1.8)
    axs[0].plot(df.index, df["RollingMean"], label="Rolling Mean", color="#ff7f0e", lw=1.6)
    axs[0].fill_between(df.index,
                        df["RollingMean"] + entry_z * df["RollingStd"],
                        df["RollingMean"] - entry_z * df["RollingStd"],
                        color="lightgray", alpha=0.3, label=f"±{entry_z}σ Band")
    axs[0].set_title(f"{t1}-{t2} Spread with Rolling Mean & Bands", fontsize=14, weight="bold")
    axs[0].set_ylabel("Spread Value")
    axs[0].legend(frameon=True, fancybox=True)
    axs[0].grid(alpha=0.3)

    # 2. Z-score plot
    axs[1].plot(df.index, df["ZScore"], label="Z-Score", color="purple", lw=1.8)  # Changed color
    axs[1].axhline(entry_z, linestyle="--", color="red", alpha=0.7, label=f"+{entry_z}")
    axs[1].axhline(-entry_z, linestyle="--", color="green", alpha=0.7, label=f"-{entry_z}")
    axs[1].axhline(0, linestyle="--", color="black", alpha=0.5)
    axs[1].set_title(f"{t1}-{t2} Z-Score ({pair_regime})", fontsize=14, weight="bold")
    axs[1].set_ylabel("Z-Score")
    axs[1].set_xlabel("Date")
    axs[1].legend(frameon=True, fancybox=True)
    axs[1].grid(alpha=0.3)

    # Statistics text box on spread plot (without Final PnL and Total Trades)
    stats_text = (
        f"Spread Mean: {df['Spread'].mean():.4f}\n"
        f"Spread Std: {df['Spread'].std():.4f}\n"
        f"Z-Score Mean: {df['ZScore'].mean():.4f}\n"
        f"Z-Score Std: {df['ZScore'].std():.4f}\n"
        f"Average Hurst: {mean_hurst:.2f}\n"
        f"Regime: {pair_regime}"
    )
    axs[0].text(1.01, 0.5, stats_text, transform=axs[0].transAxes,
                fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", alpha=0.9))

    plt.tight_layout()
    sns.despine()
    plot_file = os.path.join(results_folder, f"pairs_{t1}_{t2}.png")
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close()

    return df, pair_regime

# ----------------------------- Main Execution -----------------------------

tickers = ["SAC.JO", "SAP.JO", "SBK.JO", "SHC.JO", "SHP.JO", "SLM.JO", "SNT.JO", "SOL.JO", "SPG.JO", "SPP.JO", "SRE.JO", "SRI.JO", "SSS.JO",
           "SSU.JO", "SSW.JO", "SUI.JO", "TBS.JO", "TFG.JO", "TGA.JO", "TKG.JO", "TRU.JO", "TSG.JO", "VAL.JO", "VKE.JO", "VOD.JO", "WBC.JO", "WHL.JO"]

#tickers_universe = ["ABG.JO", "ADH.JO", "AEL.JO", "AFE.JO", "AFH.JO", "AFT.JO", "AGL.JO", "AHR.JO", "AIP.JO", "ANG.JO", "ANH.JO", "APN.JO", "ARI.JO",
#         "ARL.JO", "ATT.JO", "AVI.JO", "BAW.JO", "BHG.JO", "BID.JO", "BLU.JO", "BOX.JO", "BTI.JO", "BTN.JO", "BVT.JO", "BYI.JO", "CFR.JO", "CLS.JO",
#         "CML.JO", "COH.JO", "CPI.JO", "CSB.JO", "DCP.JO", "DRD.JO", "DSY.JO", "DTC.JO", "EMI.JO", "EQU.JO", "EXX.JO", "FBR.JO", "FFB.JO", "FSR.JO",
#         "FTB.JO", "GFI.JO", "GLN.JO", "GND.JO", "GRT.JO", "HAR.JO", "HCI.JO", "HDC.JO", "HMN.JO", "HYP.JO", "IMP.JO", "INL.JO", "INP.JO", "ITE.JO",
#         "JSE.JO", "KAP.JO", "KIO.JO", "KRO.JO", "KST.JO", "LHC.JO", "LTE.JO", "MCG.JO", "MKR.JO", "MNP.JO", "MRP.JO", "MSP.JO", "MTH.JO", "MTM.JO",
#         "MTN.JO", "N91.JO", "NED.JO", "NPH.JO", "NPN.JO", "NRP.JO", "NTC.JO", "NY1.JO", "OCE.JO", "OMN.JO", "OMU.JO", "OUT.JO", "PAN.JO", "PHP.JO",
#         "PIK.JO", "PMR.JO", "PPC.JO", "PPH.JO", "PRX.JO", "QLT.JO", "RBX.JO", "RCL.JO", "RDF.JO", "REM.JO", "RES.JO", "RLO.JO", "RNI.JO", "S32.JO",
#         "SAC.JO", "SAP.JO", "SBK.JO", "SHC.JO", "SHP.JO", "SLM.JO", "SNT.JO", "SOL.JO", "SPG.JO", "SPP.JO", "SRE.JO", "SRI.JO", "SSS.JO",
#         "SSU.JO", "SSW.JO", "SUI.JO", "TBS.JO", "TFG.JO", "TGA.JO", "TKG.JO", "TRU.JO", "TSG.JO", "VAL.JO", "VKE.JO", "VOD.JO", "WBC.JO", "WHL.JO"]

results_folder = "pairs_trading_results"
os.makedirs(results_folder, exist_ok=True)

print("Downloading all tickers...")
start_date = "2024-01-01"
end_date = datetime.today().strftime("%Y-%m-%d")
all_data = yf.download(tickers, start=start_date, end=end_date)["Close"]
print("Download completed!\n")

results = []
pairs_list = list(combinations(tickers, 2))

for t1, t2 in tqdm(pairs_list, desc="Processing pairs"):
    if t1 not in all_data.columns or t2 not in all_data.columns:
        continue
    df1 = all_data[t1].dropna()
    df2 = all_data[t2].dropna()
    if df1.empty or df2.empty:
        continue

    result = pairs_trading_ratio(df1, df2, t1, t2, results_folder)
    if result is not None:  # Only unpack if not None
        df, regime = result
        results.append({
            "Pair": f"{t1}-{t2}",
            "Regime": regime
        })

# Save summary
summary = pd.DataFrame(results)
summary.to_csv(os.path.join(results_folder, "pairs_summary3.csv"), index=False)

print("\nTop Processed Pairs:")
print(summary.head())
print(f"All plots saved in folder: {results_folder}")