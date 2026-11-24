"""Retrieve Moscow Exchange TQBR daily equity quotes and the IMOEX benchmark index (Sec. 3.1).

Queries the public MOEX Informational & Statistical Server (ISS) for the 253 common and preferred
equities in `data/universe.json` and the broad-market `IMOEX` index across 2020-04-17..2025-04-17,
writing `data/raw/moex_equities.parquet` required by `qcml_moex.build_panels_from_raw` for full
model training from scratch. All tables and figures in the paper can be reproduced directly from the
derived aggregate outputs in `data/runs/` without running this script.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_JSON = REPO_ROOT / "data" / "universe.json"
DEFAULT_OUT = REPO_ROOT / "data" / "raw" / "moex_equities.parquet"
ISS_BASE = "https://iss.moex.com/iss/history/engines/stock/markets"


def _fetch_iss_series(url_base: str, date_from: str, date_till: str, fields: tuple[str, ...]) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    start = 0
    while True:
        query = urllib.parse.urlencode(
            {
                "from": date_from,
                "till": date_till,
                "start": start,
                "iss.meta": "off",
                "iss.only": "history",
                "history.columns": ",".join(fields),
            }
        )
        url = f"{url_base}?{query}"
        payload = None
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "qcml-moex-research/1.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(0.5 * (2**attempt))
        cols = payload["history"]["columns"]
        data = payload["history"]["data"]
        if not data:
            break
        for item in data:
            rows_out.append(dict(zip(cols, item)))
        if len(data) < 100:
            break
        start += len(data)
    return rows_out


def fetch_moex_panel(out_parquet: Path = DEFAULT_OUT, jobs: int = 8) -> Path:
    meta = json.loads(UNIVERSE_JSON.read_text(encoding="utf-8"))
    board = meta["board"]
    index_ticker = meta["index_ticker"]
    date_from, date_till = meta["date_range"]
    secids: list[str] = meta["secids"]

    idx_url = f"{ISS_BASE}/index/securities/{index_ticker}.json"
    idx_rows = _fetch_iss_series(idx_url, date_from, date_till, ("TRADEDATE", "CLOSE"))
    imoex_map = {str(r["TRADEDATE"]): float(r["CLOSE"]) for r in idx_rows if r["CLOSE"] is not None}

    def _one_secid(secid: str) -> list[dict[str, Any]]:
        eq_url = f"{ISS_BASE}/shares/boards/{board}/securities/{secid}.json"
        return _fetch_iss_series(
            eq_url, date_from, date_till, ("TRADEDATE", "SECID", "CLOSE", "VOLUME", "VALUE")
        )

    equity_rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(jobs))) as pool:
        for batch in pool.map(_one_secid, secids):
            equity_rows.extend(batch)

    df = pd.DataFrame(equity_rows)
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"]).dt.strftime("%Y-%m-%d")
    df["SECID"] = df["SECID"].astype(str)
    df["CLOSE"] = pd.to_numeric(df["CLOSE"], errors="coerce")
    df["VOLUME"] = pd.to_numeric(df["VOLUME"], errors="coerce")
    df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
    df["IMOEX_CLOSE"] = df["TRADEDATE"].map(imoex_map).astype("float64")
    df = df.sort_values(["TRADEDATE", "SECID"]).reset_index(drop=True)

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    return out_parquet


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    out = fetch_moex_panel(out_parquet=args.out, jobs=args.jobs)
    print(f"Saved MOEX daily panel -> {out}")


if __name__ == "__main__":
    main()
