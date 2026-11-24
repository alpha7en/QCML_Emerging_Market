"""Moscow Exchange equity panel, corporate-action adjustments, causal inter-session news features,
and purged walk-forward calendar (Sec. 3.1-3.2).

Constructs the active trading calendar tau in {0, ..., 1250} (T = 1,251 sessions across
2020-04-17..2025-04-17 after dropping 18 zero-close dates), applies backward split factors across
seven corporate actions keeping ruble turnover VALUE invariant, selects the daily liquid Top-50
universe U_t by trailing 63-session median VALUE (>= 42 active sessions), computes the 12 market
features and 6 inter-session news features timestamped strictly in [t_{prev}, t - 1], and slices
six semi-annual expanding folds F1-F6 (2022-04-01..2025-03-27, 37,048 out-of-sample stock-days)
plus the excised February-March 2022 crisis window Q_{OOD}.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RAW_PARQUET = DATA_DIR / "raw" / "moex_equities.parquet"
NEWS_AGG_NPZ = DATA_DIR / "news_aggregates.npz"
CACHE_DIR = DATA_DIR / "cache"
DEFAULT_BUNDLE_NPZ = CACHE_DIR / "panel_folds.npz"
DEFAULT_PANEL_NPZ = CACHE_DIR / "market_aux.npz"

N_FOLDS = 6
DATA_SPLIT_SEED = 42
SEEDS10 = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
P_MARKET = 12
P_NEWS = 6
L_HAC_DEFAULT = 12
PRIMARY_HORIZON = 5
SECONDARY_HORIZON = 15
G_PURGE = 5
G_EMBARGO = 5
TOP_K_UNIVERSE = 50

HALT_PRE_DATE_ISO = "2022-02-25"
SHORT_BAN_START_ISO = "2022-02-24"
SHORT_BAN_END_ISO = "2022-05-31"
POST_2022_START_ISO = "2022-04-01"
QUARANTINE_START = pd.Timestamp("2022-02-16")
QUARANTINE_END = pd.Timestamp("2022-03-31")
HALT_LAST_PRE_DATE = pd.Timestamp("2022-02-25")

# Backward split adjustments on CLOSE (multiplied by factor) and VOLUME (divided by factor)
# for all sessions strictly preceding the effective date (Sec. 3.1).
SPLIT_ADJUSTMENTS: list[tuple[str, str, float, str]] = [
    ("VTBR", "2024-07-15", 5000.0, "1:5000 reverse split"),
    ("TRNFP", "2024-02-21", 1.0 / 100.0, "100:1 split"),
    ("GMKN", "2024-04-08", 1.0 / 100.0, "100:1 split"),
    ("PLZL", "2025-03-27", 1.0 / 10.0, "10:1 split"),
    ("ROLO", "2023-01-19", 1.0 / 10.0, "10:1 split"),
    ("GEMA", "2024-02-08", 1.0 / 10.0, "10:1 split"),
    ("BELU", "2024-08-22", 1.0 / 8.0, "7-for-1 bonus issue"),
]

# Official 27-step Central Bank of Russia key-rate schedule (% per annum) across 2020-2025 (Sec. 3.3).
CBR_KEY_RATE_SCHEDULE: list[tuple[str, float]] = [
    ("2020-02-10", 6.00),
    ("2020-04-27", 5.50),
    ("2020-06-22", 4.50),
    ("2020-07-27", 4.25),
    ("2021-03-22", 4.50),
    ("2021-04-26", 5.00),
    ("2021-06-15", 5.50),
    ("2021-07-26", 6.50),
    ("2021-09-13", 6.75),
    ("2021-10-25", 7.50),
    ("2021-12-20", 8.50),
    ("2022-02-14", 9.50),
    ("2022-02-28", 20.00),
    ("2022-04-11", 17.00),
    ("2022-05-04", 14.00),
    ("2022-05-27", 11.00),
    ("2022-06-14", 9.50),
    ("2022-07-25", 8.00),
    ("2022-09-19", 7.50),
    ("2023-07-24", 8.50),
    ("2023-08-15", 12.00),
    ("2023-09-18", 13.00),
    ("2023-10-30", 15.00),
    ("2023-12-18", 16.00),
    ("2024-07-29", 18.00),
    ("2024-09-16", 19.00),
    ("2024-10-28", 21.00),
]

# Regular dividend-paying blue chips for the Supplementary Material S3 ex-dividend sensitivity check.
KNOWN_DIV_PAYERS: set[str] = {
    "SBER", "SBERP", "LKOH", "GAZP", "ROSN", "NVTK", "TATN", "TATNP",
    "GMKN", "NLMK", "CHMF", "MAGN", "PLZL", "MTSS", "PHOR", "TRNFP",
    "SNGS", "SNGSP", "ALRS", "IRAO", "HYDR", "SIBN", "BANE", "BANEP",
    "RASP", "FLOT", "MGNT", "BELU", "LSNGP", "MOEX", "BSPB", "RTKM",
    "RTKMP", "FEES", "UPRO", "OGKB", "MSRS", "MSNG", "NMTP",
}

MARKET_FEATURE_NAMES: list[str] = [
    "ret_1d",
    "ret_5d",
    "ret_21d",
    "ret_63d",
    "vol_21d",
    "turnover_shock",
    "beta_252d",
    "cs_rank_ret_21d",
    "cs_rank_vol_21d",
    "cs_rank_adv_21d",
    "imoex_ret_5d",
    "imoex_vol_21d",
]

NEWS_FEATURE_NAMES: list[str] = [
    "news_vol_shock_21d",
    "news_lex_polarity",
    "news_lex_uncertainty",
    "news_svd_0",
    "news_svd_1",
    "news_svd_2",
]

FOLD_TEST_WINDOWS: list[tuple[str, str, str]] = [
    ("2022-04-01", "2022-09-30", "F1_2022H2"),
    ("2022-10-01", "2023-03-31", "F2_2022Q4_2023Q1"),
    ("2023-04-01", "2023-09-30", "F3_2023H2"),
    ("2023-10-01", "2024-03-31", "F4_2023Q4_2024Q1"),
    ("2024-04-01", "2024-09-30", "F5_2024H2"),
    ("2024-10-01", "2025-03-27", "F6_2024Q4_2025Q1"),
]

POS_STEMS: tuple[str, ...] = (
    "рост", "вырос", "подорож", "прибыл", "укреп", "рекорд", "повыс", "повыш",
    "дивиденд", "восстанов", "увелич", "профицит", "спрос", "поддерж",
    "инвестиц", "стабилиз", "оптимиз", "улучш",
)
NEG_STEMS: tuple[str, ...] = (
    "паден", "упал", "сниз", "снижен", "обвал", "кризис", "убыт", "дефицит",
    "инфляц", "дефолт", "банкрот", "сократ", "ослаб", "рецесс", "риск",
    "подешев", "девальвац", "потер", "долг", "штраф",
)
UNCERT_STEMS: tuple[str, ...] = (
    "кризис", "обвал", "дефолт", "банкрот", "рецесс", "девальвац", "волатильн",
    "паник", "угроз", "риск", "неопределен", "инфляц", "дефицит", "убыт",
)

_TOKEN_RE = re.compile(r"[а-яёa-z]{3,}", flags=re.IGNORECASE)
_CACHE: dict[str, Any] = {}


def _tokenize_ru(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _compute_cbr_daily_curve(active_dates_iso: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise-constant CBR policy rate R_t^{CBR} (% p.a.) and causal ACT/365 daily accrual
    r_{f,t} = (1 + R_{t-1}^{CBR}/100)^{Delta_cal(t)/365} - 1 (Sec. 3.3).
    """
    T = len(active_dates_iso)
    rates_pct = np.empty(T, dtype=np.float64)
    for i, d_str in enumerate(active_dates_iso):
        r = CBR_KEY_RATE_SCHEDULE[0][1]
        for eff_str, val in CBR_KEY_RATE_SCHEDULE:
            if d_str >= eff_str:
                r = val
            else:
                break
        rates_pct[i] = r

    dt_days = active_dates_iso.astype("datetime64[D]")
    cal_diff = np.ones(T, dtype=np.float64)
    cal_diff[1:] = np.maximum(1.0, (dt_days[1:] - dt_days[:-1]).astype(np.float64))

    rate_prev = np.empty(T, dtype=np.float64)
    rate_prev[0] = rates_pct[0]
    rate_prev[1:] = rates_pct[:-1]
    rf_daily = np.power(1.0 + rate_prev / 100.0, cal_diff / 365.0) - 1.0
    return rates_pct, rf_daily


def _load_market_matrices(raw_parquet: Path = RAW_PARQUET) -> dict[str, Any]:
    raw_cols = ["TRADEDATE", "SECID", "CLOSE", "VOLUME", "VALUE", "IMOEX_CLOSE"]
    df = pd.read_parquet(raw_parquet, columns=raw_cols)
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"]).dt.normalize()
    df["SECID"] = df["SECID"].astype(str)

    n_close_by_date = df.groupby("TRADEDATE")["CLOSE"].apply(lambda s: int(s.notna().sum()))
    active_dates = n_close_by_date[n_close_by_date > 0].index.sort_values()
    df = df[df["TRADEDATE"].isin(active_dates)].copy()

    for secid, eff_date_str, factor, _ in SPLIT_ADJUSTMENTS:
        eff_dt = pd.Timestamp(eff_date_str)
        mask = (df["SECID"] == secid) & (df["TRADEDATE"] < eff_dt)
        df.loc[mask, "CLOSE"] = df.loc[mask, "CLOSE"] * factor
        df.loc[mask, "VOLUME"] = df.loc[mask, "VOLUME"].astype(np.float64) / factor

    secids = np.array(sorted(df["SECID"].unique()), dtype=str)
    close_df = df.pivot(index="TRADEDATE", columns="SECID", values="CLOSE").reindex(
        index=active_dates, columns=secids
    )
    value_df = df.pivot(index="TRADEDATE", columns="SECID", values="VALUE").reindex(
        index=active_dates, columns=secids
    )
    imoex_s = (
        df.groupby("TRADEDATE")["IMOEX_CLOSE"]
        .first()
        .reindex(active_dates)
        .astype(np.float64)
    )
    return {
        "active_dates": active_dates,
        "secids": secids,
        "close_df": close_df,
        "value_df": value_df,
        "imoex_s": imoex_s,
    }


def _compute_intersession_news_from_csv(lenta_csv: Path, active_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Inter-session news aggregation over tau_news in [t_{prev}, t - 1] using pre-2020 Russian
    financial stem lexicons (Sec. 3.1).
    """
    lenta = pd.read_csv(lenta_csv, usecols=["date", "headline"])
    lenta["date"] = pd.to_datetime(lenta["date"]).dt.normalize()
    lenta["headline"] = lenta["headline"].fillna("").astype(str)
    by_date = lenta.groupby("date")["headline"].apply(list).to_dict()
    all_dates = sorted(by_date.keys())

    records: list[dict[str, Any]] = []
    for k, t_curr in enumerate(active_dates):
        if k == 0:
            records.append(
                {
                    "TRADEDATE": t_curr,
                    "daily_rate": np.nan,
                    "news_lex_polarity": np.nan,
                    "news_lex_uncertainty": np.nan,
                    "doc_text": "",
                }
            )
            continue
        t_prev = active_dates[k - 1]
        span_days = max(1, int((t_curr - t_prev).days))
        headlines: list[str] = []
        for d in all_dates:
            if d < t_prev:
                continue
            if d >= t_curr:
                break
            headlines.extend(by_date[d])

        n_h = len(headlines)
        pos_hits = sum(1 for h in headlines if any(st in h.lower() for st in POS_STEMS))
        neg_hits = sum(1 for h in headlines if any(st in h.lower() for st in NEG_STEMS))
        unc_hits = sum(1 for h in headlines if any(st in h.lower() for st in UNCERT_STEMS))
        records.append(
            {
                "TRADEDATE": t_curr,
                "daily_rate": n_h / float(span_days),
                "news_lex_polarity": (pos_hits - neg_hits) / float(max(1, pos_hits + neg_hits)),
                "news_lex_uncertainty": unc_hits / float(max(1, n_h)),
                "doc_text": " ".join(headlines),
            }
        )

    news_df = pd.DataFrame(records).set_index("TRADEDATE")
    log_rate = np.log1p(news_df["daily_rate"])
    med_21 = log_rate.rolling(window=21, min_periods=15).median()
    news_df["news_vol_shock_21d"] = log_rate - med_21
    return news_df


def _fit_fold_news_svd(
    news_df: pd.DataFrame,
    train_cal_indices: np.ndarray,
    n_svd: int = 3,
) -> np.ndarray:
    """Fold-causal TF-IDF + rank-3 truncated SVD fitted strictly on training-fold sessions (Sec. 3.1)."""
    docs_all = news_df["doc_text"].tolist()
    docs_train = [docs_all[i] for i in train_cal_indices]
    vec = TfidfVectorizer(
        tokenizer=_tokenize_ru,
        token_pattern=None,
        min_df=5,
        max_df=0.85,
        max_features=3000,
        sublinear_tf=True,
    )
    X_tfidf_train = vec.fit_transform(docs_train)
    svd = TruncatedSVD(n_components=n_svd, random_state=DATA_SPLIT_SEED).fit(X_tfidf_train)
    svd_all = svd.transform(vec.transform(docs_all))
    for j in range(n_svd):
        comp = svd.components_[j]
        if comp[int(np.argmax(np.abs(comp)))] < 0:
            svd_all[:, j] *= -1.0
    return np.column_stack(
        [
            news_df["news_vol_shock_21d"].to_numpy(dtype=np.float64),
            news_df["news_lex_polarity"].to_numpy(dtype=np.float64),
            news_df["news_lex_uncertainty"].to_numpy(dtype=np.float64),
            svd_all,
        ]
    )


def _compute_market_features_and_targets(mkt: dict[str, Any]) -> dict[str, Any]:
    active_dates: pd.DatetimeIndex = mkt["active_dates"]
    secids: np.ndarray = mkt["secids"]
    close_df: pd.DataFrame = mkt["close_df"]
    value_df: pd.DataFrame = mkt["value_df"]
    imoex_s: pd.Series = mkt["imoex_s"]

    T, N = close_df.shape
    log_close = np.log(close_df.where(close_df > 0))
    log_imoex = np.log(imoex_s)

    val_pos = value_df.fillna(0.0).clip(lower=0.0)
    valid_trade = ((close_df > 0) & (value_df > 0)).astype(np.float64)
    valid_cnt_63 = valid_trade.rolling(window=63, min_periods=42).sum()
    adv63_med = val_pos.rolling(window=63, min_periods=42).median()
    eligible = (valid_trade > 0) & (valid_cnt_63 >= 42) & (adv63_med > 0)
    adv63_ranked = adv63_med.where(eligible).rank(axis=1, ascending=False, method="first")
    top50_mask = (adv63_ranked <= TOP_K_UNIVERSE).fillna(False)

    ret_1d = log_close - log_close.shift(1)
    ret_5d = log_close - log_close.shift(5)
    ret_21d = log_close - log_close.shift(21)
    ret_63d = log_close - log_close.shift(63)
    imoex_ret_1d = log_imoex - log_imoex.shift(1)
    imoex_ret_5d = log_imoex - log_imoex.shift(5)

    vol_21d = ret_1d.rolling(window=21, min_periods=15).std(ddof=1)
    imoex_vol_21d = imoex_ret_1d.rolling(window=21, min_periods=15).std(ddof=1)

    adv21_mean = val_pos.rolling(window=21, min_periods=15).mean()
    adv63_mean = val_pos.rolling(window=63, min_periods=42).mean()
    turnover_shock = np.log1p(val_pos) - np.log1p(adv63_mean)

    # Trailing 252-session OLS market beta beta_{i,t} = Cov(r_i, r_m) / Var(r_m) (>= 126 valid pairs).
    r_stock = ret_1d.to_numpy(dtype=np.float64)
    r_mkt = imoex_ret_1d.to_numpy(dtype=np.float64)[:, None]
    valid_pair = np.isfinite(r_stock) & np.isfinite(r_mkt)
    rs_z = np.where(valid_pair, r_stock, 0.0)
    rm_z = np.where(valid_pair, r_mkt, 0.0)
    cnt_z = valid_pair.astype(np.float64)

    def _roll_sum_252(arr: np.ndarray) -> np.ndarray:
        cs = np.cumsum(arr, axis=0)
        out = cs.copy()
        out[252:] = cs[252:] - cs[:-252]
        return out

    s_cnt = _roll_sum_252(cnt_z)
    s_x = _roll_sum_252(rm_z)
    s_y = _roll_sum_252(rs_z)
    s_xx = _roll_sum_252(rm_z * rm_z)
    s_xy = _roll_sum_252(rm_z * rs_z)

    denom = s_xx - (s_x * s_x) / np.maximum(s_cnt, 1.0)
    numer = s_xy - (s_x * s_y) / np.maximum(s_cnt, 1.0)
    beta_arr = np.full_like(numer, np.nan, dtype=np.float64)
    np.divide(numer, denom, out=beta_arr, where=(s_cnt >= 126.0) & (denom > 1e-12))
    beta_252d = pd.DataFrame(beta_arr, index=active_dates, columns=secids)

    def _cs_rank_top50(frame: pd.DataFrame) -> pd.DataFrame:
        return 2.0 * frame.where(top50_mask).rank(axis=1, pct=True, method="average") - 1.0

    imoex_ret_5d_df = pd.DataFrame(
        np.repeat(imoex_ret_5d.to_numpy()[:, None], N, axis=1), index=active_dates, columns=secids
    )
    imoex_vol_21d_df = pd.DataFrame(
        np.repeat(imoex_vol_21d.to_numpy()[:, None], N, axis=1), index=active_dates, columns=secids
    )

    market_frames = {
        "ret_1d": ret_1d,
        "ret_5d": ret_5d,
        "ret_21d": ret_21d,
        "ret_63d": ret_63d,
        "vol_21d": vol_21d,
        "turnover_shock": turnover_shock,
        "beta_252d": beta_252d,
        "cs_rank_ret_21d": _cs_rank_top50(ret_21d),
        "cs_rank_vol_21d": _cs_rank_top50(vol_21d),
        "cs_rank_adv_21d": _cs_rank_top50(adv21_mean),
        "imoex_ret_5d": imoex_ret_5d_df,
        "imoex_vol_21d": imoex_vol_21d_df,
    }

    # Forward h-day log-returns r_{i, t -> t+h} masked across the February 28-March 23, 2022 halt.
    halt_pre_idx = int(active_dates.get_loc(HALT_LAST_PRE_DATE))
    y_h5_raw = log_close.shift(-PRIMARY_HORIZON) - log_close
    crosses_halt_h5 = np.array(
        [(idx <= halt_pre_idx) and (idx + PRIMARY_HORIZON > halt_pre_idx) for idx in range(T)],
        dtype=bool,
    )
    y_h5_raw.iloc[crosses_halt_h5, :] = np.nan
    y_h5_masked = y_h5_raw.where(top50_mask)
    y_h5_cs = y_h5_masked.sub(y_h5_masked.mean(axis=1), axis=0)

    y_h15_raw = log_close.shift(-SECONDARY_HORIZON) - log_close
    crosses_halt_h15 = np.array(
        [(idx <= halt_pre_idx) and (idx + SECONDARY_HORIZON > halt_pre_idx) for idx in range(T)],
        dtype=bool,
    )
    y_h15_raw.iloc[crosses_halt_h15, :] = np.nan
    y_h15_masked = y_h15_raw.where(top50_mask)
    y_h15_cs = y_h15_masked.sub(y_h15_masked.mean(axis=1), axis=0)

    fwd_sq_stack = np.stack([ret_1d.shift(-s).to_numpy() ** 2 for s in range(1, PRIMARY_HORIZON + 1)], axis=0)
    fwd_vol_5d = pd.DataFrame(np.log(1e-4 + np.sqrt(np.mean(fwd_sq_stack, axis=0))), index=active_dates, columns=secids)
    fwd_vol_5d.iloc[crosses_halt_h5, :] = np.nan

    fwd_cum_stack = np.stack(
        [(log_close.shift(-s) - log_close).to_numpy() for s in range(1, PRIMARY_HORIZON + 1)], axis=0
    )
    fwd_downside_5d = pd.DataFrame(np.min(fwd_cum_stack, axis=0), index=active_dates, columns=secids)
    fwd_downside_5d.iloc[crosses_halt_h5, :] = np.nan

    X_market_3d = np.stack([market_frames[name].to_numpy(dtype=np.float64) for name in MARKET_FEATURE_NAMES], axis=-1)
    Y_state_3d = np.stack(
        [
            y_h5_cs.to_numpy(dtype=np.float64),
            y_h5_raw.to_numpy(dtype=np.float64),
            fwd_vol_5d.to_numpy(dtype=np.float64),
            fwd_downside_5d.to_numpy(dtype=np.float64),
        ],
        axis=-1,
    )
    imoex_roll_vol_5d = imoex_ret_1d.rolling(window=5, min_periods=3).std(ddof=1).to_numpy()

    return {
        "top50_mask": top50_mask.to_numpy(dtype=bool),
        "adv63_med": adv63_med.fillna(0.0).to_numpy(dtype=np.float32),
        "top30_mask": (adv63_ranked <= 30).fillna(False).to_numpy(dtype=bool),
        "adv21_mean": adv21_mean.to_numpy(dtype=np.float64),
        "X_market_3d": X_market_3d,
        "y_h5_cs": y_h5_cs.to_numpy(dtype=np.float64),
        "y_h5_raw": y_h5_raw.to_numpy(dtype=np.float64),
        "y_h15_cs": y_h15_cs.to_numpy(dtype=np.float64),
        "Y_state_3d": Y_state_3d,
        "imoex_ret_1d": imoex_ret_1d.to_numpy(dtype=np.float64),
        "imoex_roll_vol_5d": imoex_roll_vol_5d,
    }


def _fit_winsor_scaler(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Expanding-train [1%, 99%] winsorization thresholds and z-score moments (mu_{tr}, sigma_{tr})."""
    q_lo = np.quantile(X_train, 0.01, axis=0)
    q_hi = np.quantile(X_train, 0.99, axis=0)
    Xc = np.clip(X_train, q_lo, q_hi)
    mean = Xc.mean(axis=0)
    std = Xc.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return q_lo, q_hi, mean, std


def _apply_winsor_scaler(
    X: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    Xc = np.clip(X, q_lo, q_hi)
    return ((Xc - mean) / std).astype(np.float32)


def _extract_panel_rows(
    date_indices: np.ndarray,
    panel: dict[str, Any],
    daily_news: np.ndarray,
    require_target: bool = True,
) -> dict[str, np.ndarray]:
    top50 = panel["top50_mask"]
    X_mkt_3d = panel["X_market_3d"]
    y_h5_cs = panel["y_h5_cs"]
    y_h5_raw = panel["y_h5_raw"]
    y_h15_cs = panel["y_h15_cs"]
    Y_state_3d = panel["Y_state_3d"]

    d_list, s_list, xm_list, xn_list, yh5c_list, yh5r_list, yh15c_list, yst_list = (
        [], [], [], [], [], [], [], []
    )
    for t in date_indices:
        news_row = daily_news[t]
        if not np.all(np.isfinite(news_row)):
            continue
        stock_ok = top50[t] & np.all(np.isfinite(X_mkt_3d[t]), axis=-1)
        if require_target:
            stock_ok = stock_ok & np.isfinite(y_h5_cs[t]) & np.all(np.isfinite(Y_state_3d[t]), axis=-1)
        s_idx = np.flatnonzero(stock_ok)
        if len(s_idx) == 0:
            continue
        k = len(s_idx)
        d_list.append(np.full(k, t, dtype=np.int32))
        s_list.append(s_idx.astype(np.int32))
        xm_list.append(X_mkt_3d[t, s_idx])
        xn_list.append(np.repeat(news_row[None, :], k, axis=0))
        yh5c_list.append(y_h5_cs[t, s_idx])
        yh5r_list.append(y_h5_raw[t, s_idx])
        yh15c_list.append(y_h15_cs[t, s_idx])
        yst_list.append(Y_state_3d[t, s_idx])

    return {
        "date_idx": np.concatenate(d_list),
        "secid_idx": np.concatenate(s_list),
        "X_market": np.concatenate(xm_list, axis=0),
        "X_news": np.concatenate(xn_list, axis=0),
        "y_h5_cs": np.concatenate(yh5c_list).astype(np.float32),
        "y_h5_raw": np.concatenate(yh5r_list).astype(np.float32),
        "y_h15_cs": np.concatenate(yh15c_list).astype(np.float32),
        "Y_state_raw": np.concatenate(yst_list, axis=0).astype(np.float32),
    }


def build_panels_from_raw(
    raw_parquet: Path = RAW_PARQUET,
    news_agg_npz: Path = NEWS_AGG_NPZ,
    lenta_csv: Path | None = None,
    out_bundle_npz: Path = DEFAULT_BUNDLE_NPZ,
    out_panel_npz: Path = DEFAULT_PANEL_NPZ,
) -> tuple[Path, Path]:
    """Build fold-standardized walk-forward matrices and auxiliary portfolio tables from
    MOEX equities (`raw_parquet`) and shipped derived news aggregates (`news_agg_npz`) or an
    optional user-supplied headline file (`lenta_csv`).
    """
    if not raw_parquet.is_file():
        raise FileNotFoundError(
            f"Missing MOEX quote table {raw_parquet}. Retrieve it first via: "
            "python -I experiments/fetch_moex.py"
        )

    mkt = _load_market_matrices(raw_parquet)
    active_dates: pd.DatetimeIndex = mkt["active_dates"]
    secids: np.ndarray = mkt["secids"]
    T = len(active_dates)
    N = len(secids)
    active_dates_iso = np.array([d.strftime("%Y-%m-%d") for d in active_dates], dtype="<U10")

    panel = _compute_market_features_and_targets(mkt)
    news_df = _compute_intersession_news_from_csv(lenta_csv, active_dates) if lenta_csv is not None else None
    shipped_news = None if news_df is not None else np.load(news_agg_npz, allow_pickle=False)

    quarantine_mask = (active_dates >= QUARANTINE_START) & (active_dates <= QUARANTINE_END)
    quarantine_idx_set = set(int(i) for i in np.flatnonzero(quarantine_mask))
    first_usable_idx = 127
    val_window_days = 63
    purge_plus_embargo = G_PURGE + G_EMBARGO

    npz_payload: dict[str, np.ndarray] = {
        "cal_dates": active_dates_iso,
        "secids": secids.astype("<U16"),
    }

    for f_idx, (test_start_str, test_end_str, _) in enumerate(FOLD_TEST_WINDOWS):
        test_start_dt = pd.Timestamp(test_start_str)
        test_end_dt = pd.Timestamp(test_end_str)
        test_cal = np.array(
            [
                i
                for i in range(first_usable_idx, T)
                if (test_start_dt <= active_dates[i] <= test_end_dt) and (i not in quarantine_idx_set)
            ],
            dtype=np.int32,
        )
        t_test_min = int(test_cal.min())
        pre_eligible = np.array(
            [i for i in range(first_usable_idx, t_test_min - purge_plus_embargo) if i not in quarantine_idx_set],
            dtype=np.int32,
        )
        val_cal = pre_eligible[-val_window_days:]
        v_min = int(val_cal.min())
        train_cal = np.array([i for i in pre_eligible if i <= v_min - G_PURGE - 1], dtype=np.int32)

        if news_df is not None:
            daily_news_fold = _fit_fold_news_svd(news_df, train_cal, n_svd=3)
        else:
            daily_news_fold = shipped_news[f"daily_news_fold_{f_idx}"].astype(np.float64)

        splits_raw = {
            "train": _extract_panel_rows(train_cal, panel, daily_news_fold, require_target=True),
            "val": _extract_panel_rows(val_cal, panel, daily_news_fold, require_target=True),
            "test": _extract_panel_rows(test_cal, panel, daily_news_fold, require_target=True),
        }

        mkt_params = _fit_winsor_scaler(splits_raw["train"]["X_market"])
        news_params = _fit_winsor_scaler(splits_raw["train"]["X_news"])
        pfx = f"fold_{f_idx}"

        for sp_name, sp in splits_raw.items():
            Xm_s = _apply_winsor_scaler(sp["X_market"], *mkt_params)
            Xn_s = _apply_winsor_scaler(sp["X_news"], *news_params)
            h15_valid = np.isfinite(sp["y_h15_cs"])
            if sp_name == "train":
                h15_valid = h15_valid & (sp["date_idx"] + SECONDARY_HORIZON < int(val_cal.min()))
            elif sp_name == "val":
                h15_valid = h15_valid & (sp["date_idx"] + SECONDARY_HORIZON < int(test_cal.min()))

            npz_payload[f"{pfx}_X_market_{sp_name}"] = Xm_s
            npz_payload[f"{pfx}_X_news_{sp_name}"] = Xn_s
            npz_payload[f"{pfx}_y_h5_cs_{sp_name}"] = sp["y_h5_cs"]
            npz_payload[f"{pfx}_y_h5_raw_{sp_name}"] = sp["y_h5_raw"]
            npz_payload[f"{pfx}_y_h15_cs_{sp_name}"] = np.where(h15_valid, sp["y_h15_cs"], np.nan).astype(np.float32)
            npz_payload[f"{pfx}_h15_valid_{sp_name}"] = h15_valid
            npz_payload[f"{pfx}_Y_state_raw_{sp_name}"] = sp["Y_state_raw"]
            npz_payload[f"{pfx}_date_idx_{sp_name}"] = sp["date_idx"]
            npz_payload[f"{pfx}_secid_idx_{sp_name}"] = sp["secid_idx"]

    ood_train_cal = np.arange(first_usable_idx, 401, dtype=np.int32)
    ood_eval_cal = np.arange(406, 516, dtype=np.int32)
    if news_df is not None:
        daily_news_ood = _fit_fold_news_svd(news_df, ood_train_cal, n_svd=3)
    else:
        daily_news_ood = shipped_news["daily_news_ood"].astype(np.float64)

    ood_tr = _extract_panel_rows(ood_train_cal, panel, daily_news_ood, require_target=True)
    ood_ev = _extract_panel_rows(ood_eval_cal, panel, daily_news_ood, require_target=False)
    ood_mkt_params = _fit_winsor_scaler(ood_tr["X_market"])
    ood_news_params = _fit_winsor_scaler(ood_tr["X_news"])

    ood_ev_dates = active_dates[ood_ev["date_idx"]]
    ood_regime = np.zeros(len(ood_ev["date_idx"]), dtype=np.int32)
    ood_regime[(ood_ev_dates >= QUARANTINE_START) & (ood_ev_dates <= QUARANTINE_END)] = 1
    ood_regime[ood_ev_dates > QUARANTINE_END] = 2

    npz_payload["ood_X_market_train"] = _apply_winsor_scaler(ood_tr["X_market"], *ood_mkt_params)
    npz_payload["ood_X_news_train"] = _apply_winsor_scaler(ood_tr["X_news"], *ood_news_params)
    npz_payload["ood_y_h5_cs_train"] = ood_tr["y_h5_cs"]
    npz_payload["ood_date_idx_train"] = ood_tr["date_idx"]
    npz_payload["ood_secid_idx_train"] = ood_tr["secid_idx"]

    _, _, mkt_mu, mkt_sd = ood_mkt_params
    _, _, news_mu, news_sd = ood_news_params
    npz_payload["ood_X_market_eval"] = ((ood_ev["X_market"] - mkt_mu) / mkt_sd).astype(np.float32)
    npz_payload["ood_X_news_eval"] = ((ood_ev["X_news"] - news_mu) / news_sd).astype(np.float32)
    npz_payload["ood_X_market_eval_winsor"] = _apply_winsor_scaler(ood_ev["X_market"], *ood_mkt_params)
    npz_payload["ood_X_news_eval_winsor"] = _apply_winsor_scaler(ood_ev["X_news"], *ood_news_params)
    npz_payload["ood_y_h5_cs_eval"] = ood_ev["y_h5_cs"]
    npz_payload["ood_y_h5_raw_eval"] = ood_ev["y_h5_raw"]
    npz_payload["ood_date_idx_eval"] = ood_ev["date_idx"]
    npz_payload["ood_secid_idx_eval"] = ood_ev["secid_idx"]
    npz_payload["ood_regime_eval"] = ood_regime
    npz_payload["ood_imoex_ret_1d_by_cal"] = panel["imoex_ret_1d"].astype(np.float32)
    npz_payload["ood_imoex_roll_vol_5d_by_cal"] = panel["imoex_roll_vol_5d"].astype(np.float32)

    out_bundle_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_bundle_npz, **npz_payload)

    # Auxiliary portfolio & robustness arrays on the full (T = 1251, N = 253) grid.
    close = mkt["close_df"].to_numpy(dtype=np.float64)
    value_rub = mkt["value_df"].fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float64)
    imoex = mkt["imoex_s"].to_numpy(dtype=np.float64)
    cbr_pct, cbr_daily = _compute_cbr_daily_curve(active_dates_iso)
    short_ban_mask = (active_dates_iso >= SHORT_BAN_START_ISO) & (active_dates_iso <= SHORT_BAN_END_ISO)

    log_close = np.where(close > 0, np.log(np.where(close > 0, close, 1.0)), np.nan)
    ret_1d = np.full((T, N), np.nan, dtype=np.float64)
    ret_1d[1:] = log_close[1:] - log_close[:-1]
    excess_1d = ret_1d - panel["imoex_ret_1d"][:, None]

    months = np.array([int(d[5:7]) for d in active_dates_iso], dtype=np.int32)
    div_season = np.isin(months, [5, 6, 7, 10, 11, 12])
    is_div_payer = np.array([s in KNOWN_DIV_PAYERS for s in secids], dtype=bool)
    raw_drop4 = np.isfinite(excess_1d) & (excess_1d < -0.04)
    isolated_day = (raw_drop4.sum(axis=1) < 5)[:, None]
    exdiv_sens_mask = raw_drop4 & div_season[:, None] & is_div_payer[None, :] & isolated_day
    gazp_cancel = np.where(active_dates_iso == "2022-06-30")[0]
    gazp_col = np.where(secids == "GAZP")[0]
    if len(gazp_cancel) and len(gazp_col):
        exdiv_sens_mask[gazp_cancel[0], gazp_col[0]] = False

    fwd_val_5 = np.full((T, N), np.nan, dtype=np.float64)
    for t in range(T - 5):
        fwd_val_5[t] = value_rub[t + 1 : t + 6].mean(axis=0)
    fwd_vol_shock_5d = np.log1p(fwd_val_5) - np.log1p(panel["adv21_mean"])
    halt_idx = int(np.where(active_dates_iso == HALT_PRE_DATE_ISO)[0][0])
    for t in range(max(0, halt_idx - 4), halt_idx + 1):
        fwd_vol_shock_5d[t, :] = np.nan

    has_close = np.isfinite(close) & (close > 0)
    pre2020_continuous_mask = (has_close[:63].mean(axis=0) >= 0.90) & (has_close.sum(axis=0) >= 1100)

    out_panel_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_panel_npz,
        close=close.astype(np.float32),
        value_rub=value_rub.astype(np.float32),
        adv63_med=panel["adv63_med"],
        top30_mask=panel["top30_mask"],
        imoex=imoex.astype(np.float64),
        active_dates=active_dates_iso,
        secids=secids.astype("<U16"),
        cbr_key_rate_pct=cbr_pct.astype(np.float64),
        cbr_key_rate_daily=cbr_daily.astype(np.float64),
        short_ban_mask=short_ban_mask.astype(bool),
        exdiv_sens_mask=exdiv_sens_mask.astype(bool),
        fwd_vol_shock_5d=fwd_vol_shock_5d.astype(np.float32),
        pre2020_continuous_mask=pre2020_continuous_mask.astype(bool),
    )
    return out_bundle_npz, out_panel_npz


def ensure_panels(
    bundle_path: Path | str | None = None,
    panel_path: Path | str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    b_path = Path(bundle_path) if bundle_path is not None else DEFAULT_BUNDLE_NPZ
    p_path = Path(panel_path) if panel_path is not None else DEFAULT_PANEL_NPZ
    cache_key = f"{b_path.resolve()}::{p_path.resolve()}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    if not b_path.is_file() or not p_path.is_file():
        build_panels_from_raw(out_bundle_npz=b_path, out_panel_npz=p_path)

    with np.load(b_path, allow_pickle=False) as zb:
        bundle_data = {k: zb[k] for k in zb.files}
    with np.load(p_path, allow_pickle=False) as zp:
        panel_data = {k: zp[k] for k in zp.files}

    close64 = panel_data["close"].astype(np.float64)
    panel_data["log_close"] = np.where(close64 > 0, np.log(np.where(close64 > 0, close64, 1.0)), np.nan)
    log_im = np.log(panel_data["imoex"].astype(np.float64))
    panel_data["log_imoex"] = log_im
    im_ret1 = np.full(len(log_im), np.nan, dtype=np.float64)
    im_ret1[1:] = log_im[1:] - log_im[:-1]
    panel_data["imoex_ret_1d"] = im_ret1

    _CACHE[cache_key] = (bundle_data, panel_data)
    return bundle_data, panel_data


def _assemble_X(bundle: dict[str, np.ndarray], pfx: str, split: str, modality: str) -> np.ndarray:
    mod = modality.lower().strip()
    if mod in ("market_only", "mkt"):
        return bundle[f"{pfx}_X_market_{split}"].astype(np.float32)
    if mod in ("market_news", "mkt_news"):
        return np.concatenate(
            [bundle[f"{pfx}_X_market_{split}"], bundle[f"{pfx}_X_news_{split}"]], axis=1
        ).astype(np.float32)
    raise ValueError(f"Unknown modality: {modality!r}")


def _build_4factor_ortho(X_mkt: np.ndarray) -> np.ndarray:
    """Four-factor cross-sectional control matrix [Mom21, LowVol, Mom5, TurnoverShock] (Sec. 3.3)."""
    return np.column_stack([X_mkt[:, 2], -X_mkt[:, 4], X_mkt[:, 1], X_mkt[:, 5]]).astype(np.float32)


def get_fold_data(
    f: int,
    modality: str,
    target: str = "y_h5_cs",
    horizon: int = 5,
    train_window: int | None = None,
    post2022_only: bool = False,
) -> dict[str, Any]:
    """Purged expanding (or rolling W_{train}) fold slice enforcing max(tau_{tr}) + h < min(tau_{val})
    and max(tau_{val}) + h < min(tau_{test}) for horizon h in {5, 15} (Sec. 3.2, Supplementary Material S3).
    """
    if not (0 <= f < N_FOLDS):
        raise ValueError(f"Fold f must be in 0..{N_FOLDS - 1}, got {f}")
    bundle, panel = ensure_panels()
    pfx = f"fold_{f}"
    eff_horizon = max(int(horizon), 15) if target == "y_h15_cs" else int(horizon)

    splits: dict[str, dict[str, np.ndarray]] = {}
    for sp in ("train", "val", "test"):
        X = _assemble_X(bundle, pfx, sp, modality)
        X_mkt = bundle[f"{pfx}_X_market_{sp}"].astype(np.float32)
        di = bundle[f"{pfx}_date_idx_{sp}"].astype(np.int32)
        si = bundle[f"{pfx}_secid_idx_{sp}"].astype(np.int32)

        if target == "y_h5_cs":
            y = bundle[f"{pfx}_y_h5_cs_{sp}"].astype(np.float32)
            ok = np.isfinite(y)
        elif target == "y_h5_raw":
            y = bundle[f"{pfx}_y_h5_raw_{sp}"].astype(np.float32)
            ok = np.isfinite(y)
        elif target == "y_h15_cs":
            y = bundle[f"{pfx}_y_h15_cs_{sp}"].astype(np.float32)
            ok = bundle[f"{pfx}_h15_valid_{sp}"].astype(bool) & np.isfinite(y)
        elif target == "Y_state":
            yst_old = bundle[f"{pfx}_Y_state_raw_{sp}"].astype(np.float32)
            vshock = panel["fwd_vol_shock_5d"][di, si].astype(np.float32)
            y = np.column_stack([yst_old[:, 0], yst_old[:, 2], yst_old[:, 3], vshock]).astype(np.float32)
            ok = np.all(np.isfinite(y), axis=1)
        else:
            raise ValueError(f"Unsupported target: {target!r}")

        splits[sp] = {"X": X[ok], "X_mkt": X_mkt[ok], "y": y[ok], "di": di[ok], "si": si[ok]}

    te_min = int(splits["test"]["di"].min())
    va_keep = splits["val"]["di"] + eff_horizon < te_min
    for k in ("X", "X_mkt", "y", "di", "si"):
        splits["val"][k] = splits["val"][k][va_keep]

    va_min = int(splits["val"]["di"].min())
    tr_keep = splits["train"]["di"] + eff_horizon < va_min
    if post2022_only:
        post_idx = int(np.where(panel["active_dates"] >= POST_2022_START_ISO)[0][0])
        tr_keep = tr_keep & (splits["train"]["di"] >= post_idx)
    if train_window is not None:
        uniq_tr_days = np.unique(splits["train"]["di"][tr_keep])
        if len(uniq_tr_days) > int(train_window):
            cutoff_day = int(uniq_tr_days[-int(train_window)])
            tr_keep = tr_keep & (splits["train"]["di"] >= cutoff_day)

    for k in ("X", "X_mkt", "y", "di", "si"):
        splits["train"][k] = splits["train"][k][tr_keep]

    extra: dict[str, Any] = {}
    if target == "Y_state":
        y_tr_raw = splits["train"]["y"].astype(np.float64)
        mu = np.array([0.0, y_tr_raw[:, 1].mean(), y_tr_raw[:, 2].mean(), y_tr_raw[:, 3].mean()], dtype=np.float32)
        sd_raw = np.where(y_tr_raw[:, 1:].std(axis=0) < 1e-8, 1.0, y_tr_raw[:, 1:].std(axis=0))
        sd = np.array([1.0, sd_raw[0], sd_raw[1], sd_raw[2]], dtype=np.float32)
        extra["y_state_raw_te"] = splits["test"]["y"].copy()
        extra["y_state_mean"] = mu
        extra["y_state_std"] = sd
        for sp in ("train", "val", "test"):
            splits[sp]["y"] = ((splits[sp]["y"] - mu) / sd).astype(np.float32)
        tr_mean: Any = splits["train"]["y"].mean(axis=0).astype(np.float64)
    else:
        tr_mean = float(np.mean(splits["train"]["y"], dtype=np.float64))

    out: dict[str, Any] = {
        "X_tr": splits["train"]["X"],
        "y_tr": splits["train"]["y"],
        "di_tr": splits["train"]["di"],
        "si_tr": splits["train"]["si"],
        "X_va": splits["val"]["X"],
        "y_va": splits["val"]["y"],
        "di_va": splits["val"]["di"],
        "si_va": splits["val"]["si"],
        "X_te": splits["test"]["X"],
        "y_te": splits["test"]["y"],
        "di_te": splits["test"]["di"],
        "si_te": splits["test"]["si"],
        "ortho_te": _build_4factor_ortho(splits["test"]["X_mkt"]),
        "tr_mean": tr_mean,
        "adv_rank_te": splits["test"]["X_mkt"][:, 9].copy(),
        "top30_te": panel["top30_mask"][splits["test"]["di"], splits["test"]["si"]].copy(),
        "fold": int(f),
        "modality": modality,
        "target": target,
        "horizon": eff_horizon,
    }
    out.update(extra)
    return out


def get_pooled_oos_meta(
    modality: str = "market_only",
    target: str = "y_h5_cs",
    horizon: int = 5,
    train_window: int | None = None,
    post2022_only: bool = False,
) -> dict[str, Any]:
    ys, dis, sis, folds, orthos, tr_means, X_tes = [], [], [], [], [], [], []
    for f in range(N_FOLDS):
        fd = get_fold_data(
            f, modality=modality, target=target, horizon=horizon,
            train_window=train_window, post2022_only=post2022_only,
        )
        n = len(fd["di_te"])
        ys.append(fd["y_te"])
        dis.append(fd["di_te"])
        sis.append(fd["si_te"])
        folds.append(np.full(n, f, dtype=np.int32))
        orthos.append(fd["ortho_te"])
        X_tes.append(fd["X_te"])
        if target == "Y_state":
            tr_means.append(np.repeat(np.asarray(fd["tr_mean"], dtype=np.float64)[None, :], n, axis=0))
        else:
            tr_means.append(np.full(n, float(fd["tr_mean"]), dtype=np.float64))
    return {
        "y_te": np.concatenate(ys, axis=0),
        "di_te": np.concatenate(dis, axis=0),
        "si_te": np.concatenate(sis, axis=0),
        "fold_te": np.concatenate(folds, axis=0),
        "ortho_te": np.concatenate(orthos, axis=0),
        "tr_mean_per_row": np.concatenate(tr_means, axis=0),
        "X_te": np.concatenate(X_tes, axis=0),
    }


def get_ood2022_data(modality: str, winsor: bool = False) -> dict[str, Any]:
    """Pre-crisis training slice (<= 2021-11-16) and unclipped February-March 2022 crisis evaluation
    slice Q_{OOD} with pre-crisis training mean benchmark y_bar_{tr} (Sec. 3.2, 4.3).
    """
    bundle, _ = ensure_panels()
    mod = modality.lower().strip()
    suf = "_winsor" if winsor else ""
    if mod in ("market_only", "mkt"):
        X_tr = bundle["ood_X_market_train"].astype(np.float32)
        X_ev = bundle[f"ood_X_market_eval{suf}"].astype(np.float32)
    elif mod in ("market_news", "mkt_news"):
        X_tr = np.concatenate([bundle["ood_X_market_train"], bundle["ood_X_news_train"]], axis=1).astype(np.float32)
        X_ev = np.concatenate(
            [bundle[f"ood_X_market_eval{suf}"], bundle[f"ood_X_news_eval{suf}"]], axis=1
        ).astype(np.float32)
    else:
        raise ValueError(f"Unsupported ood2022 modality: {modality!r}")
    y_tr = bundle["ood_y_h5_cs_train"].astype(np.float32)
    return {
        "X_tr": X_tr,
        "y_tr": y_tr,
        "di_tr": bundle["ood_date_idx_train"].astype(np.int32),
        "si_tr": bundle["ood_secid_idx_train"].astype(np.int32),
        "X_ev": X_ev,
        "y_ev": bundle["ood_y_h5_cs_eval"].astype(np.float32),
        "di_ev": bundle["ood_date_idx_eval"].astype(np.int32),
        "si_ev": bundle["ood_secid_idx_eval"].astype(np.int32),
        "regime_ev": bundle["ood_regime_eval"].astype(np.int32),
        "tr_mean": float(np.mean(y_tr, dtype=np.float64)),
    }


def permute_news_by_date(
    X: np.ndarray,
    date_idx: np.ndarray,
    p_mkt: int = P_MARKET,
    seed: int = 101,
) -> np.ndarray:
    """Date-level permutation pi(t) of news coordinates X[:, p_mkt:] preserving cross-sectional
    constancy within each trading day t (Table 2, Panel C).
    """
    X_out = np.array(X, copy=True)
    if X_out.ndim != 2 or X_out.shape[1] <= p_mkt:
        return X_out
    di = np.asarray(date_idx)
    u_dates, inv = np.unique(di, return_inverse=True)
    n_days = len(u_dates)
    if n_days <= 1:
        return X_out

    day_news = np.empty((n_days, X_out.shape[1] - p_mkt), dtype=X_out.dtype)
    for k in range(n_days):
        day_news[k] = X_out[inv == k, p_mkt:][0]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_days)
    if np.all(perm == np.arange(n_days)):
        perm = np.roll(perm, 1)
    X_out[:, p_mkt:] = day_news[perm[inv]]
    return X_out
