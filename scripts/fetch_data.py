"""Populate the data cache. The only script that makes a network call.

    uv run --group data python scripts/fetch_data.py

Deliberately a separate entry point rather than something a test can trigger. The gate
suite is offline by construction and stays that way; this is run by hand when the cache
needs filling, and every later run reads the cached parquet and verifies it against the
manifest.

The manifest is committed. The cache is not -- it is regenerable from the manifest's
own record of what to ask for, and a repository is not a data store.
"""

from __future__ import annotations

import sys

from falsify.data.loaders import (
    DEFAULT_CACHE,
    DEFAULT_MANIFEST,
    FetchSpec,
    describe_biases,
    load,
)
from falsify.data.panel import SECTORS
from falsify.data.rates import DEFAULT_TICKER, RateSpec, load_rate

# SPY only, per 03 Part H decision 1: a single ticker removes survivorship bias
# entirely as a confound, which keeps the statistical argument clean.
SPECS = (
    FetchSpec("SPY", "2015-01-01", "2025-01-01", "total_return"),
    FetchSpec("SPY", "2015-01-01", "2025-01-01", "raw"),
)

# 03 Part H decision 4: the risk-free term must exist, set to the period mean of
# 3-month T-bills, with the constant recorded in the manifest.
RATE_SPEC = RateSpec(DEFAULT_TICKER, "2015-01-01", "2025-01-01")

# Phase 7 opens the universe, under the condition 03 Part H decision 1 named for it:
# "the engine is certified and you want factor attribution". Nine sector funds rather
# than nine stocks, because yfinance returns currently-listed tickers only and a stock
# universe picked today is survivorship-biased over a 2015 start.
SECTOR_SPECS = tuple(
    FetchSpec(ticker, "2015-01-01", "2025-01-01", "total_return") for ticker in SECTORS
)


def main() -> int:
    print("fetching -- this is the project's deliberate network access")
    for spec in (*SPECS, *SECTOR_SPECS):
        try:
            bars = load(
                spec,
                allow_network=True,
                cache_dir=DEFAULT_CACHE,
                manifest_path=DEFAULT_MANIFEST,
            )
        except Exception as exc:
            print(f"  FAILED {spec.cache_key}: {type(exc).__name__}: {exc}")
            return 1
        print(
            f"  ok {spec.cache_key}: {len(bars)} bars, "
            f"{bars.ts[0]} .. {bars.ts[-1]}, auto_adjust={spec.auto_adjust}"
        )

    print("\nrisk-free rate (03 Part H decision 4):")
    try:
        rate = load_rate(
            RATE_SPEC,
            allow_network=True,
            cache_dir=DEFAULT_CACHE,
            manifest_path=DEFAULT_MANIFEST,
        )
    except Exception as exc:
        print(f"  FAILED {RATE_SPEC.cache_key}: {type(exc).__name__}: {exc}")
        return 1
    print(f"  {rate.describe()}")

    print("\nbiases stated rather than fixed:")
    for name, text in describe_biases().items():
        print(f"  {name}: {text}")
    print(f"\nmanifest: {DEFAULT_MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
