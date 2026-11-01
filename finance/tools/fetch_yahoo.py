#!/usr/bin/env python3
"""Download daily bars from Yahoo's chart endpoint into one Parquet file per symbol.

Data acquisition only: the Rust pipeline reads the Parquet files; nothing in training
or inference imports Python.

    python3 finance/tools/fetch_yahoo.py --out finance/data/bars [--start 1999-01-01]

Writes `<out>/<SYMBOL>.parquet` with columns
    date (int32 yyyymmdd, exchange-local), open, high, low, close, adj_close (float64), volume (float64)
and `<out>/universe.json` (symbols, sectors, market index, download time).

Limitations, written into universe.json as well: the stock list is today's large caps
(survivorship bias), and the sector map is today's (not point-in-time).
"""
import argparse
import datetime as dt
import json
import pathlib
import time
import urllib.error
import urllib.request

import pyarrow as pa
import pyarrow.parquet as pq

MARKET = "SPY"
SECTORS = {
    "XLK": ["AAPL", "MSFT", "INTC", "CSCO", "ORCL", "IBM", "TXN", "QCOM", "ADBE", "AMAT"],
    "XLF": ["JPM", "BAC", "WFC", "C", "AXP", "USB", "PNC"],
    "XLV": ["JNJ", "PFE", "MRK", "ABT", "AMGN", "UNH", "LLY", "BMY", "MDT"],
    "XLE": ["XOM", "CVX", "COP", "SLB", "OXY"],
    "XLP": ["PG", "KO", "PEP", "WMT", "CL", "KMB"],
    "XLY": ["HD", "MCD", "NKE", "LOW", "TGT"],
    "XLI": ["GE", "MMM", "CAT", "HON", "UNP", "BA", "LMT", "EMR"],
    "XLU": ["DUK", "SO", "AEP"],
    "XLB": ["APD", "ECL", "NEM"],
}


def fetch(symbol: str, start: dt.date) -> dict:
    p1 = int(dt.datetime(start.year, start.month, start.day, tzinfo=dt.timezone.utc).timestamp())
    p2 = int(time.time()) + 86400
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1={p1}&period2={p2}&interval=1d&events=div%2Csplit&includeAdjustedClose=true"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        payload = json.load(r)
    res = payload["chart"]["result"][0]
    gmtoffset = res["meta"].get("gmtoffset", 0)
    ts = res["timestamp"]
    q = res["indicators"]["quote"][0]
    adj = res["indicators"]["adjclose"][0]["adjclose"]
    rows = {k: [] for k in ["date", "open", "high", "low", "close", "adj_close", "volume"]}
    for i, t in enumerate(ts):
        vals = [q["open"][i], q["high"][i], q["low"][i], q["close"][i], adj[i], q["volume"][i]]
        if any(v is None for v in vals):
            continue  # a missing bar stays missing; the Rust side treats it as absent
        d = dt.datetime.fromtimestamp(t + gmtoffset, tz=dt.timezone.utc).date()
        rows["date"].append(d.year * 10000 + d.month * 100 + d.day)
        for k, v in zip(["open", "high", "low", "close", "adj_close", "volume"], vals):
            rows[k].append(float(v))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="1999-01-01")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    start = dt.date.fromisoformat(args.start)
    symbols = [MARKET] + list(SECTORS) + [s for v in SECTORS.values() for s in v]
    schema = pa.schema(
        [("date", pa.int32())] + [(k, pa.float64()) for k in ["open", "high", "low", "close", "adj_close", "volume"]]
    )
    counts = {}
    missing = []
    for sym in symbols:
        rows = None
        for attempt in range(4):
            try:
                rows = fetch(sym, start)
                break
            except urllib.error.HTTPError as err:
                if err.code == 404:  # unknown symbol: record it, do not guess a substitute
                    break
                print(f"{sym}: {err}; retrying")
                time.sleep(2 + 4 * attempt)
            except Exception as err:  # network hiccup: back off and retry
                if attempt == 3:
                    raise
                print(f"{sym}: {err}; retrying")
                time.sleep(2 + 4 * attempt)
        if rows is None:
            print(f"{sym}: not found, skipped")
            missing.append(sym)
            continue
        table = pa.table(rows, schema=schema)
        pq.write_table(table, out / f"{sym}.parquet")
        counts[sym] = table.num_rows
        print(f"{sym}: {table.num_rows} bars {rows['date'][0]}..{rows['date'][-1]}")
        time.sleep(0.5)
    meta = {
        "market": MARKET,
        "sectors": SECTORS,
        "rows": counts,
        "missing": missing,
        "source": "Yahoo Finance v8 chart API, daily, adjclose",
        "downloaded_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "limitations": [
            "survivorship bias: the stock list is today's large caps",
            "sector map is today's, not point-in-time",
            "adj_close is back-adjusted for later splits and dividends; only ratios are used",
        ],
    }
    (out / "universe.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
