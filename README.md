# Quant Research Studio

Python research tools for event-driven futures backtesting, market-regime analysis, and options pricing. Built with AI assistance. No broker connection or automatic live orders.

## Explore the backtester

- [Event engine source](quantcore/events.py)
- [Streamlit interface](event_backtester/app.py)
- [Execution and numerical tests](tests/test_models.py)
- [Generate reproducible demonstration candles](generate_demo.py)

A FIFO queue processes market-open, market-close, signal, and order events. Signals see completed candles; entries use the next contiguous opening price. The engine models fees, adverse slippage, target penetration, stop-first resolution for ambiguous OHLC bars, and missing-data cancellations. Planned trade loss, daily realized loss, and realized drawdown limits gate entries. Trade, rejection, exit, and accounting events can be exported.

The included strategy is an EMA crossover engineering benchmark. It is separate from the owner's private trading bots. These tools demonstrate implementation and validation, not trading profitability.

## Run the backtester

Use Python 3.11 or newer. From this folder:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python generate_demo.py
.venv/bin/python -m streamlit run event_backtester/app.py --server.address 127.0.0.1
```

On Windows use `.venv\Scripts\python.exe` instead of `.venv/bin/python`.
Open the local URL printed by Streamlit. Select cached local candles, then run the historical comparison. The generated candles are synthetic and must not be treated as market observations. The CSV upload control also accepts your own appropriately licensed 15-minute OHLCV history.

To explore all three projects, replace the last command with `.venv/bin/python launch.py`.

## Verify

```sh
.venv/bin/python -m unittest discover -s tests -v
```

Tests cover numerical option pricing, implied volatility, Greek derivatives, chronological regime fitting, event ordering, future-data mutation checks, costs, and selected Streamlit behavior. Tests requiring private licensed caches are skipped when those caches are absent. The shared research suite contains 50 tests. The public-copy check with generated demonstration candles passed 49 tests and skipped one private-cache case. Without local candles it passed 48 tests and skipped two cache-dependent cases.

## Other research tools

**Market regimes:** rolling features, training-window scaling and K-means, chronological evaluation, and gap-aware warm-up. Regime labels describe volatility; they are not validated buy/sell signals.

**Options:** Black-Scholes pricing, five Greeks, implied volatility, and fitted volatility surfaces using generated analytic quotes. Imported quotes require validation. Missing-expiry interpolation is not established as accurate; no globally arbitrage-free guarantee is made.

## Scope and assumptions

OHLC simulation does not reconstruct tick paths, spread, queue position, or real broker fills. Risk limits use realized equity; adverse opening gaps can exceed planned loss. Zero fills under strict limits are a valid risk-gate result. No profit, predictive edge, or live execution performance is established.

No licensed historical market-data exports, credentials, model weights, account records, or private bot code are included. Source implementation dates: October 2026.
