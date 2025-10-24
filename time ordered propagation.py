
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests

# -------------------- Config --------------------
DATA_FILES = {
    "anxiety": "anxiety_2024_features_tfidf_256.csv",
    "depression": "depression_2024_features_tfidf_256.csv",
    "suicidewatch": "suicidewatch_2024_features_tfidf_256.csv",
}
MAX_LAG = 4
N_PLACEBO = 1000
OUT_DIR = "data"


def load_and_aggregate_weekly(data_files: dict) -> pd.DataFrame:
    dfs = []
    for label, path in data_files.items():
        df = pd.read_csv(path, encoding="ISO-8859-1", low_memory=False)
        df["subreddit"] = label
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df["week"] = df["date"].dt.to_period("W").dt.to_timestamp()
        w = df.groupby("week").size().rename(label).to_frame()
        dfs.append(w)
    weekly = pd.concat(dfs, axis=1).fillna(0).astype(float)
    weekly = weekly.sort_index()
    return weekly

def granger_min_p(weekly: pd.DataFrame, a: str, b: str, max_lag: int) -> float:

    try:
        res = grangercausalitytests(weekly[[b, a]], maxlag=max_lag, verbose=False)
        pvals = [res[k][0]['ssr_ftest'][1] for k in res.keys() if 'ssr_ftest' in res[k][0]]
        return float(np.nanmin(pvals)) if len(pvals) else np.nan
    except Exception:
        return np.nan

def all_pairs_granger(weekly: pd.DataFrame, max_lag: int) -> pd.DataFrame:
    cols = list(weekly.columns)
    rows = []
    for a in cols:
        for b in cols:
            if a == b: 
                continue
            p = granger_min_p(weekly, a, b, max_lag)
            rows.append({"from": a, "to": b, "min_p": p})
    return pd.DataFrame(rows)

def circular_shift(arr: np.ndarray, k: int) -> np.ndarray:

    k = int(k) % len(arr)
    if k == 0:
        return arr.copy()
    return np.concatenate([arr[-k:], arr[:-k]])

def placebo_min_p(weekly: pd.DataFrame, a: str, b: str, max_lag: int, n_iter: int=1000, seed: int=42) -> dict:

    rng = np.random.default_rng(seed)
    y = weekly[b].values.astype(float)
    x = weekly[a].values.astype(float)
    obs_minp = granger_min_p(weekly, a, b, max_lag)
    null_minps = []
    for _ in range(n_iter):

        k = int(rng.integers(1, len(x)))
        x_shift = circular_shift(x, k)
        df = pd.DataFrame({b: y, a: x_shift}, index=weekly.index)
        p = granger_min_p(df, a, b, max_lag)
        null_minps.append(p if not np.isnan(p) else 1.0)
    null_minps = np.array(null_minps)

    emp_p = float((null_minps <= (obs_minp if not np.isnan(obs_minp) else 0)).mean())
    return {"observed_min_p": float(obs_minp) if obs_minp is not None else np.nan,
            "placebo_empirical_p": emp_p,
            "placebo_min_p_distribution": null_minps.tolist()}

def save_fig_s1(weekly: pd.DataFrame, granger_df: pd.DataFrame, out_path: str):

    lines = []
    for _, r in granger_df.iterrows():
        a, b, p = r["from"], r["to"], r["min_p"]
        if pd.isna(p):
            lines.append(f"{a}→{b}: p=NA")
        else:
            lines.append(f"{a}→{b}: p={p:.3f}")
    summary_text = "Granger min p (lags 1–4):\n" + "\n".join(lines)


    ax = weekly.plot(figsize=(10,6))
    ax.set_xlabel("Week")
    ax.set_ylabel("Posts per week")
    ax.set_title("Supplementary Fig. S1: Weekly posts and Granger lead–lag summary")
    ax.legend(loc="best")
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, va='top', ha='left',
            bbox=dict(boxstyle='round', alpha=0.2))
    import matplotlib.pyplot as plt
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    weekly = load_and_aggregate_weekly(DATA_FILES)
    weekly.to_csv(os.path.join(OUT_DIR, "weekly counts.csv"))
    

    granger_df = all_pairs_granger(weekly, MAX_LAG)
    granger_df.to_csv(os.path.join(OUT_DIR, "granger min pvalues.csv"), index=False)


    placebo = {}
    if all(c in weekly.columns for c in ["depression", "suicidewatch"]):
        placebo["depression_to_suicidewatch"] = placebo_min_p(weekly, "depression", "suicidewatch", MAX_LAG, N_PLACEBO, 42)

    with open(os.path.join(OUT_DIR, "granger_placebo_minp.json"), "w", encoding="utf-8") as f:
        import json; json.dump(placebo, f, ensure_ascii=False, indent=2)


    save_fig_s1(weekly, granger_df, os.path.join(OUT_DIR, "weekly posts granger.png"))

if __name__ == "__main__":
    main()
