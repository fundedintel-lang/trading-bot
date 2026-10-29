# APEX-SMC — XAU/USD Institutional Algo Trading Engine

Paper-first **Smart Money Concepts (SMC) / ICT** research and trading engine for XAU/USD.
Async-native Python 3.11+, externalised configuration, structured JSON logging,
hard risk limits, OANDA v20 execution (plus a deterministic paper broker),
backtesting, walk-forward analysis and a Streamlit dashboard.

> **Risk warning.** Leveraged gold trading is high risk and most retail algo
> accounts lose money. This repository ships in **PAPER / SEMI_AUTO** mode with
> live execution disabled behind three independent interlocks. Backtest output —
> especially on synthetic data — is a plumbing check, **not** a profitability
> claim.

### Execution and AI safety upgrade

See **[docs/UPGRADE.md](docs/UPGRADE.md)** for optional Groq **or** xAI/Grok
review, secure environment-key setup, stricter risk/approval guards and corrected
backtesting. Rotate any key shared in chat; no disclosed credential is stored in
this repository. AI is off by default and can never override risk limits.
Historical synthetic results below/linked elsewhere predate the replay fix and
are **not** current performance evidence. Profitability is unproven.

---

## 1. Quick start

```bash
git clone <repo> trading-bot && cd trading-bot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env && chmod 600 .env     # fill in credentials later

python main.py validate-config             # sanity-check config + secrets
python main.py selftest                    # offline smoke test of every module
python main.py demo                        # one full decision cycle on synthetic data
python main.py dashboard --port 8501       # Streamlit dashboard
```

Nothing above needs an API key, a network connection or a broker account.

> **What `run` actually trades without credentials.** With no OANDA/Twelve Data
> keys, `build_feed()` falls back to the deterministic synthetic feed. That feed
> now advances with the wall clock (new 5M bars appear every 5 minutes and the
> history is never rewritten), so `python main.py run` behaves like a live
> session — but the prices are generated, not gold. For a forward test on real
> prices, create a free **OANDA practice** account and put the token + account id
> in `.env`; that is the only change needed.

### CLI

| Command | Purpose |
| --- | --- |
| `python main.py run` | Live/paper loop driven by APScheduler (5M trading, 30s management, 30min news) |
| `python main.py once` | Exactly one decision cycle |
| `python main.py demo` | One cycle on deterministic synthetic data (offline) |
| `python main.py status` | Engine, account, risk and session snapshot |
| `python main.py backtest [--days N] [--csv F] [--seed S] [--research]` | Backtest + performance report → `reports/backtest_*.json` |
| `python main.py walkforward [--days N] [--folds K] [--json OUT]` | Rolling in-sample / out-of-sample stability analysis |
| `python main.py approve <signal_id>` / `reject <signal_id>` | Semi-auto gate |
| `python main.py dashboard [--port 8501]` | Read-only Streamlit dashboard |
| `python main.py validate-config` | Config validation (exits 1 on failure) |
| `python main.py selftest` | Offline component smoke test |

Global flags: `--config PATH`, `--env-file PATH`, `--log-level LEVEL`.

---

## 2. Architecture

```
apex_smc_bot/
├── config.py              frozen dataclasses, .env loading, validation
├── logging_setup.py       JSON rotating logs (engine / signals / trades / errors)
├── data/
│   ├── feed_base.py       retry + backoff helpers shared by every feed
│   ├── feed_oanda.py      OANDA v20 candles/pricing (primary)
│   ├── feed_twelve.py     Twelve Data (fallback)
│   ├── feed_mt5.py        MetaTrader 5 (tertiary, optional Windows dep)
│   ├── candle_manager.py  multi-timeframe store, pip maths, resampling
│   ├── database.py        SQLite: candles, signals, trades, equity, runtime state
│   ├── synthetic.py       deterministic regime-switching market generator
│   └── replay_feed.py     cursor-driven feed for backtests (no look-ahead)
├── analysis/
│   ├── structure.py       swings (N=3), BOS/CHoCH by close, dual (8/3) structure
│   ├── order_blocks.py    OB detection, quality 1-5, mitigation / flip / breaker
│   ├── fvg.py             3-candle imbalances + fill lifecycle + IFVG
│   ├── liquidity.py       EQH/EQL (5-pip tolerance), sweeps, inducement, Judas
│   ├── session.py         ICT kill-zone table (UTC) + dead zones
│   ├── volume.py          volume expansion, delta, HVN/LVN
│   ├── candle_psych.py    rejection/wick/body psychology
│   ├── premium_discount.py dealing range, equilibrium, OTE 61.8-79 %
│   ├── htf_bias.py        4H + 1H bias and alignment gate
│   └── confluence.py      10-factor scoring matrix (min 7/10)
├── signals/
│   ├── signal_generator.py 15M arming + 7-check 5M trigger
│   ├── signal_validator.py 16 validity checks before any order
│   ├── signal_formatter.py JSON, text, Telegram, Discord, console
│   ├── alternate_paths.py  Plan A / Plan B
│   ├── smart_money.py      narrative simulation
│   └── models.py           Signal, PendingSetup, TriggerResult (fingerprints)
├── execution/
│   ├── broker.py           Broker protocol, Position, OrderRequest/Result
│   ├── paper_broker.py     deterministic fills, partials, immutable trade ledger
│   ├── oanda_executor.py   OANDA v20 REST client (retry, partial close, OCO)
│   ├── order_manager.py    routing, exposure accounting, semi-auto approvals
│   ├── position_manager.py TP1/TP2 partials, breakeven, trailing, breaker close
│   ├── trail_stop.py       breakeven / protective / ATR / structure trails
│   └── lot_calculator.py   1 % risk sizing with broker lot constraints
├── risk/
│   ├── risk_engine.py      the single gate every order must pass
│   ├── drawdown_monitor.py daily/D/D breaker, loss cooldowns (persisted)
│   ├── spread_monitor.py   absolute limit + spike detection
│   ├── news_filter.py      FOMC/NFP/CPI/PPI/JOLTS/ISM blackout ±30 min
│   └── caution_zones.py    chop-box / missed-entry / stale-zone vetoes
├── notifications/          Telegram (commands), Discord, email, console
├── core/
│   ├── engine.py           the decision loop (one function, fully typed)
│   ├── scheduler.py        APScheduler jobs
│   ├── state_manager.py    fingerprint ledger, counters, pause/resume
│   └── selftest.py         offline end-to-end smoke test
├── backtesting/
│   ├── backtest_engine.py  replays history through the *live* engine
│   ├── performance_metrics.py win rate, PF, expectancy, DD, Sharpe, Sortino
│   └── walk_forward.py     rolling IS/OOS folds
└── dashboard/
    ├── chart_renderer.py   plotly candlesticks, equity, drawdown, histogram
    └── streamlit_app.py    read-only operator dashboard
```

### The decision pipeline

```
data refresh ─▶ hard gates (session, weekend, breaker, news)
             ─▶ position management (TP1/TP2 partials, breakeven, trail, stop)
             ─▶ 4H bias ─▶ 1H bias ─▶ alignment gate
             ─▶ 15M structure ─▶ OB / FVG / sweep ─▶ zone selection
             ─▶ 10-factor confluence (≥7) ─▶ PendingSetup armed
             ─▶ per 5M bar: 7-check trigger (≥6, checks 1 & 7 mandatory)
             ─▶ validator ─▶ optional advisory/veto AI review
             ─▶ fresh quote + final risk/approval checks ─▶ broker
```

Every stage writes a structured log line, and every rejection carries a reason
code (`HTF_NOT_ALIGNED`, `NO_CONFIRMED_SWEEP`, `MISSED_ENTRY`,
`SESSION_NOT_TRADEABLE:…`, `RISK:…`, `VALIDATION:…`).

---

## 3. Strategy rules implemented

**Structure** — fractal swings with `N=3` (internal) and `N=8` (external);
BOS/CHoCH confirmed on candle **close**, never on a wick.

**Order blocks** — last opposite candle before displacement, ATR guard on the
origin candle, quality 1–5 (displacement, volume, freshness, confluence),
lifecycle `UNMITIGATED → PARTIALLY_MITIGATED → FULLY_MITIGATED`, plus
breaker/flip inversion.

**Fair value gaps** — 3-candle imbalance with `min_gap_price` filter, fill
ratio tracking, inverse-FVG support.

**Liquidity** — EQH/EQL clustering at 5 pips, sweep detection (raid + close
back inside), double inducement, Judas swing inside the first 30 min of the NY
kill zone.

**Sessions (UTC)** — London KZ 07:00–10:00, NY KZ 12:00–15:00 (12:00–12:30 =
`NY_JUDAS_WINDOW`, non-tradeable); dead zones 00/05/12/21/23; London low 11,
NY KZ high 13, NY afternoon low 15, NY close 18.

**Entry model** — 15M zone (OB ± FVG) filtered to 4–30 pips wide and quality
≥3; entry capped at 60 pips from price; premium/discount veto (no buys in
premium, no sells in discount); OTE 61.8–79 %.

**Trigger (5M, ≥6/7)** — **1** price traded into the zone within the last 3
bars (a wick counts: that is what fills a limit order) and the market is not
already beyond `max_entry_distance_pips` away, **2** rejection candle, **3**
micro-BOS, **4** volume/delta, **5** wick ratio, **6** no opposing quality OB
within 25/40 pips, **7** spread. Checks **1 and 7 are mandatory** — 6/7 without
them never fires, and a stale zone can never confirm a fresh entry.

**Targets** — TP1 1.5R (close 50 %, stop to breakeven), TP2 3R (close 25 %),
TP3 5R runner with ATR/structure trail.

**Risk** — 1 % per trade (0.5 % after 3 % daily drawdown), 5 % daily circuit
breaker (24 h), max 2 concurrent trades **including resting orders and pending
approvals**, spread ≤3.5 pips (spike ×2 → 15 min pause), SL 5–50 pips,
lots 0.01–10, 2 h/4 h loss cool-downs, ≥30 min news blackout.

---

## 4. Configuration

Everything operational is in [`config.yaml`](config.yaml); secrets only ever in
`.env`. `Settings` is a frozen dataclass tree with a `validate()` pass that is
run by `main.py validate-config` and again at engine start.

```bash
APEX_CONFIG=config.yaml
APEX_EXECUTION_MODE=SEMI_AUTO     # overrides bot.execution_mode
APEX_ALLOW_LIVE=false             # extra interlock for a live account
```

Live OANDA requires `execution.broker: oanda`, `oanda_environment: live`,
`execution.allow_live_trading: true`, **and** independently
`APEX_ALLOW_LIVE=true`, with valid account credentials. The YAML and environment
switches cannot enable one another. Selecting OANDA with missing credentials
fails closed rather than silently switching to paper. Synthetic fallback is
allowed only with the paper broker.

Optional AI credentials: `GROQ_API_KEY` for Groq, **or** `XAI_API_KEY` for
xAI/Grok; these are different services. Set `APEX_AI_ENABLED=true`, the correct
`APEX_AI_PROVIDER` and `APEX_AI_MODEL` in `.env` to enable review. Start with
`APEX_AI_MODE=advisory` in paper mode. [Full setup and limitations](docs/UPGRADE.md).

---

## 5. Testing

```bash
python -m pytest apex_smc_bot/tests -q
```

The suite is offline and deterministic (synthetic data, pinned clock):
structure, order blocks, FVG, liquidity, signals/validator/formatter, risk,
execution, plus `python main.py selftest` for the integrated pipeline.

---

## 6. Backtesting & walk-forward

```bash
python main.py backtest --days 90 --seed 11
python main.py walkforward --days 120 --folds 4 --json reports/wf.json
```

Both replay history through the **production** `TradingEngine` and
`PositionManager` — not a re-implementation — so execution, partial closes and
risk vetoes behave exactly as they do live. The feed is cursor-based: no bar
after the cycle timestamp is ever visible.

* reports are written to `reports/backtest_*.json`
* fewer than 20 trades is flagged as below the reporting threshold, not statistical proof
* replay uses closed HTF bars and conservative OHLC stop-first execution, with spread, slippage and configurable per-side commissions
* AI and the current news calendar are always disabled in historical replay
* confidence intervals and explicit assumptions are included; runtime errors invalidate the run
* `--research` relaxes the confluence/5M gates; it is for synthetic-data
  studies only and is never a default

Supply real data with `--csv` (columns: `timestamp,open,high,low,close,volume[,spread]`).

---

## 7. Deployment (systemd)

```bash
sudo ./deploy/install.sh          # user, venv, units, logrotate, secrets file
sudoedit /etc/apex-smc/apex.env   # OANDA + Telegram credentials
sudo systemctl enable --now apex-smc apex-smc-dashboard
journalctl -u apex-smc -f
```

`deploy/apex-smc.service` runs the engine with `ProtectSystem=strict`,
`NoNewPrivileges`, a restricted `ReadWritePaths` set and automatic restart.
`deploy/apex-smc-dashboard.service` binds Streamlit to `127.0.0.1:8501` — put it
behind an authenticating reverse proxy. `deploy/logrotate.conf` rotates the JSON
logs daily (30 days).

Health checks:

```bash
python main.py status                    # engine snapshot
curl -s localhost:8501/_stcore/health    # dashboard health
```

---

## 8. Logging & notifications

Structured JSON to `logs/` with rotation (`engine`, `signals`, `trades`,
`errors`), plus a console stream. Every signal is emitted **simultaneously** as
structured JSON and human-readable text to console, log, SQLite, Telegram and
(optionally) Discord. Telegram also accepts `/status`, `/positions`,
`/approve <id>`, `/reject <id>`, `/close <id>`, `/closeall`, `/pause`,
`/resume`, `/signals`.

---

## 9. Evaluation, not profitability claims

The previous synthetic measurements are preserved in
[docs/FREQUENCY.md](docs/FREQUENCY.md) as **legacy** records. They predate the
closed-HTF replay fix, cost corrections and production-cadence replay; do not
compare them with current output or use them as evidence of a trading edge.

```bash
python main.py backtest --days 90 --seed 11 --end 2026-10-28T16:00Z
python main.py --config configs/high_frequency.yaml backtest --days 90 --seed 11 --end 2026-10-28T16:00Z
```

These commands use **generated prices**, not market gold. They test execution
and throughput, not whether the strategy can make money. The high-frequency
preset relaxes selectivity; more trades are not an upgrade to expectancy.
Neither the strategy nor the optional AI reviewer has an established live edge.

For evaluation, supply real M5 candles from your intended broker with `--csv`,
set realistic spread/slippage/commission costs, retain enough history for HTF
warmup, and use fixed out-of-sample walk-forward windows. Follow with a
prospective paper/practice test. [Assumptions and verification](docs/UPGRADE.md).

## 10. Status & known limitations

* Paper broker and synthetic generator are deterministic — reproducible tests,
  no claim of realism. Without broker credentials `run` trades this synthetic
  feed, not the real market (see §1).
* Trade frequency is configurable, not fixed. The conservative profile can
  legitimately produce zero trades in a three-day paper session — that is
  selectivity, not a stalled engine. `configs/high_frequency.yaml` raises the
  cadence with smaller trades; there is no fixed trades/day promise. `python main.py status` and
  the dashboard's Risk tab show the live rejection counters, so you can always
  see which gate is binding.
* Frequency is not the same thing as edge. The high-frequency profile does not
  establish a positive expectancy. Validate on real data first.

* The economic-calendar fetch needs outbound network; the default now fails
  **closed** (`news.fail_closed: true`) and refuses new entries without a usable
  calendar. Offline `demo` / `selftest` disable the calendar explicitly.
* AI is an optional second opinion, not a price feed, autonomous trader, or
  guarantee. Historical backtests do not validate its live effect. See the
  dashboard's AI Review tab and [AI setup](docs/UPGRADE.md).
* Approvals and the paper broker's cash/positions are in-memory. Restarting does
  not restore them. Use Telegram approvals in the running engine; a fresh CLI
  process cannot recover another process's approval queue.
* The simulator excludes financing/margin liquidation and tick-level queue
  effects. OANDA needs practice-account verification; use a dedicated USD account.
* MetaTrader 5 is optional and Windows-only; the feed degrades gracefully.
* Gold CFDs have no central exchange — broker feed differences (spread, gaps)
  matter more than any parameter in this repository.

## 11. License

See [LICENSE](LICENSE).
