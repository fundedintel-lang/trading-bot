# Safety, AI review, and evaluation upgrade

This upgrade improves execution discipline and the reliability of measurements.
It does **not** establish a profitable strategy. More trades, a higher AI
confidence number, and a passing test suite are not evidence of a trading edge.
Live execution is still disabled by default.

## 1. Rotate the disclosed key; configure a replacement locally

Any key posted in chat must be revoked at its provider and replaced. Do not
reuse it, put it in a YAML file, commit it, or send the replacement in chat.
The disclosed credential was **not** copied into this repository or used.

**Groq is not Grok.** Groq's API serves models such as Llama; Grok is supplied
by xAI. A credential for one does not authenticate to the other. The integration
supports both services and never guesses the provider from a key string.

```bash
cp .env.example .env         # only if you do not already have a .env
chmod 600 .env
# Edit .env locally, never in a public issue/chat.
```

For **Groq**, set these values in `.env`:

```dotenv
GROQ_API_KEY=your_new_groq_key
APEX_AI_ENABLED=true
APEX_AI_PROVIDER=groq
APEX_AI_MODEL=llama-3.3-70b-versatile
APEX_AI_MODE=advisory
```

For **xAI / Grok**, use this instead:

```dotenv
XAI_API_KEY=your_new_xai_key
APEX_AI_ENABLED=true
APEX_AI_PROVIDER=xai
APEX_AI_MODEL=grok-3-mini
APEX_AI_MODE=advisory
```

Model names are configurable examples, not a guarantee of current availability
or access on your account. Select an available chat-completion model supporting
JSON-object output. No additional AI SDK is required. Requests use the fixed
HTTPS provider endpoint and a bearer header; redirects and automatic retries
are disabled. Real-provider authentication was not exercised in the offline
verification; verify it on your own machine with a rotated key.

Run `python main.py validate-config` to check configuration and redacted
credential-presence flags. `python main.py status` and the dashboard's **AI
Review** tab show configuration and audit results. A configured key alone does
not make requests: only a deterministic signal that passed strategy/validation
is reviewed. `demo`, `selftest`, backtests and walk-forward runs do not call AI.

## 2. What the AI can and cannot do

The default `ai.enabled: false` keeps the original deterministic engine usable
without a provider. Start with **advisory** review in paper trading:

- `advisory`: record the second opinion; it cannot allow or reject an order.
- `veto`: only a schema-valid `PASS` at/above `ai.min_confidence` passes the AI
  layer. `WAIT`, explicit `VETO`, and low confidence block the trade.
- In veto mode, missing credentials, timeouts, malformed output and quota
  exhaustion block new entries when `ai.fail_closed: true` (default). If you
  deliberately set it false, only unavailability fails open; explicit vetoes
  still block.

The model **cannot** choose a direction, invent an order, change prices or
size, raise risk, bypass a limit, or call tools. Confidence is certainty in a
review, **not** a calibrated win probability. An approval never overrides a
failed technical, freshness, drawdown, spread, capacity or execution check.

Only an allowlisted snapshot is sent: instrument, direction, prices/targets,
spread, technical score/check booleans and the last 24 closed M5 candles. Account
identifiers, credentials, equity, position sizes and arbitrary news/web text
are not sent. Consider your provider's data-retention policies before enabling.

There is one bounded request attempt per uncached review, an 8-second default
timeout, a 400-output-token limit, a 60-second outage cooldown and a persisted
100-request UTC-day budget. Changed prices/candles invalidate cached opinions;
cache size is bounded. The budget is per database/engine, not a provider-wide
**dollar** limit; also configure spend limits with the provider. `AI_REVIEW`
events record verdicts in SQLite. A vetoed setup is retired for its lifetime,
not retried on every bar to fish for an approval.

## 3. Risk and execution changes

- Minimum-lot sizing now **rejects** a trade if the minimum lot exceeds the
  budget. The former 10% risk overshoot allowance is removed.
- Position sizing reserves the quoted spread plus twice the configured maximum
  slippage (entry/exit). Entries dominated by costs are rejected using
  `risk.max_cost_to_stop_ratio` (default 0.25). This is a reserve, **not a
  guarantee** against gaps, financing costs or worse-than-assumed liquidity.
- Stated risk is the actual rounded position exposure, not an unspent budget.
  Units/lot consistency and actual monetary risk are validated independently.
- Both TP1 and TP2 configured minimum R multiples are enforced from actual
  prices. Market-entry rebasing keeps the full target ladder and Plan A/B
  consistent; it does not prove those targets will be reached.
- Order submission and approval capacity reservations are serialized. A delayed
  approval rechecks current equity, drawdown/reduced-risk band, cooldown,
  expiry, spread, exposure and prices. Asynchronous broker pending orders are
  counted correctly; protective SL/TP orders are not counted as new entries.
- Quote/history freshness is enforced. OANDA execution/management uses its
  native broker quote rather than an auxiliary feed's assumed spread. Quotes
  are refreshed after AI review,
  and a final current-clock check rejects stale or expired preflight results.
  The 30-second management job fetches a new quote instead of recycling the
  last 5-minute mark. Missing history does not suppress protective management.
- Halts, news/session blocks and pauses cancel pending **entry** exposure, not
  protective stops. Calendar fetch timestamps survive restart, and a non-empty
  but expired calendar does not bypass fail-closed mode. Broker/account lookup
  failures cannot masquerade as zero exposure or permission to trade cached data.
- Synthetic/replay prices cannot drive an OANDA account, even a practice
  account. Selecting OANDA without credentials no longer silently becomes a
  paper broker.
- Live OANDA needs the YAML allow-live switch **and** independent
  `APEX_ALLOW_LIVE=true`, plus the live environment and credentials. One switch
  cannot turn on the other.
- OANDA has a native stop and final-runner (TP3) take profit, a market-order
  `priceBound`, and persisted trade-plan identity for TP1/TP2 management.
  Read requests retry; ambiguous writes do **not** automatically replay.
- An unknown entry outcome consumes that signal and pauses the engine. An
  unknown management operation is persisted and prevents repeating the partial
  close or opening further entries. Check the actual broker holdings and audit
  trail before resuming. If a position has uncertain management state, use the
  broker interface or an explicit close to flatten and reconcile it; restarting
  does not erase the uncertainty. Broker-side protection stays attached.
- A failed close is no longer reported as a successful close; management state
  and the loss counter are only finalized after acknowledgement. Management
  progress, partial P/L and open/closed trades are written to the audit DB.

Use a dedicated USD account: sizing is USD-specific and no FX conversion is
implemented. There is no independent portfolio-wide dollar-risk cap summing
all open/pending stops; position-count and per-trade budgets are separate
limits. OANDA integration still needs practice-account forward testing;
mocked tests do not verify broker permissions, units precision or fills. Pending
human approvals intentionally live only in the running process. Use the running
engine's Telegram `/approve` flow; a new CLI process cannot recover its queue.
The paper broker is in-memory and does not restore positions/account cash after
restart; use isolated paper-test databases and do not treat restarts as one
continuous performance record.

## 4. More honest, faster replay

Previously a precomputed H4 candle could be exposed from its **open** timestamp
while containing hours of future prices. Replay now requires
`bar_open + timeframe_duration <= current_M5_close`. Synthetic and Twelve Data
adapters also filter forming higher-timeframe bars. Cached future bars cannot
enter a current analysis window. Duplicate or non-M5-aligned CSV timestamps are
rejected; CSV times must represent bar opens, not bar closes.

Backtests now:

1. Force the paper broker / automatic execution, strip credentials, disable AI
   and the current news calendar. Today's LLM/calendar must not leak future
   knowledge into a historical result. Replay does **not** evaluate those live
   overlays; testing them needs a timestamped historical review/calendar archive
   or a prospective paper experiment.
2. Include 90 days of **synthetic warmup history** outside the scoring window.
   CSV uses whatever preceding history is supplied; provide enough for HTF bias.
3. Respect `--days` and the UTC `--end` **close-time boundary**: bars open at or
   after the start and close by the end. Adjacent walk-forward windows do not
   overlap. Dataset injection is local, not a global monkey-patch.
4. Follow the production M5/15M analysis cadence rather than force-rearming on
   every M5 candle. Only changed candles are written to SQLite; duplicate
   persistence work has been removed.
5. Pay bid/ask spread, adverse entry/market-exit slippage and proportional
   per-side commission (`backtest.commission_per_lot`). Stop triggers use the
   executable side, not the midpoint. Limits need the correct bid/ask touch.
6. Use **stop-first** if a bar contains both a stop and target. A new limit fill
   may stop out immediately but cannot claim a high/low target that may have
   preceded its fill. This is deliberately conservative, not tick reconstruction.
7. Report source/cost assumptions, runtime-error count, independent dollar and
   percentage drawdown maxima, a Wilson win-rate interval and seeded IID
   bootstrap expectancy interval. Small samples are labeled insufficient.
   Intervals do not correct for correlated trades or strategy-selection bias.
   Sharpe/Sortino currently use days with realized closes, not a full daily NAV
   series. Combined OOS P/L aggregates independently reset folds, not a single
   compounded account or intrabar OOS drawdown.

A legitimate zero-trade test succeeds; a test with runtime cycle errors does
not. Overnight swaps/financing, margin liquidation rules, variable real latency
and tick-level queue/fill paths are **not** modeled. Reports are ignored local
artifacts under `reports/`, not committed datasets.

```bash
python main.py backtest --days 90 --seed 11 --end 2026-10-28T16:00Z
python main.py backtest --csv /path/to/xauusd_m5.csv --days 730 --end 2026-10-28T16:00Z
python main.py walkforward --csv /path/to/xauusd_m5.csv --days 730 --folds 4 \
  --end 2026-10-28T16:00Z --json reports/walkforward.json
```

`--research` deliberately relaxes quality gates and is refused with CSV input.
The old README/FREQUENCY synthetic performance table predates corrected replay
and must not be compared to these runs or used as evidence of profitability.
Do not tune the preset until a synthetic run turns green. Obtain real gold data
from the intended broker, check costs and data quality, require a substantial
out-of-sample sample (e.g. 100+ trades, not a universal statistical guarantee),
and forward-test in paper/practice with no parameter changes before considering
any live deployment.

## 5. Verification

```bash
python -m pytest
python main.py validate-config
python main.py selftest
python main.py demo
```

The regression suite mocks providers/brokers and covers untrusted AI output,
budgets/caching, secret redaction, independent live switches, timing/leakage,
real exposure, concurrent approvals, cancellation, commissions, ambiguous
writes, failed closes and restart-safe management. It never requires the
disclosed key or accesses a trading account.

### Recorded offline verification (2026-10-29)

- **214 tests passed**; config validation, **11/11** self-test checks and offline
  demo passed. The Streamlit application rendered all seven tabs in AppTest.
- Both presets completed **6,288 cycles** for a 30-day synthetic run with seed
  11, ending `2026-10-28T16:00Z`, with **zero cycle errors**. A two-fold,
  30-day walk-forward smoke test also completed (zero completed OOS trades —
  insufficient evidence, not a failure of the plumbing).
- Groq/xAI and OANDA transport behavior was **mock-tested only**; no exposed key
  was used and no live account was accessed.

| Synthetic preset | Completed trades | Net P/L | Return | Evidence |
|---|---:|---:|---:|---|
| Conservative | 2 | +$61.98 | +0.62% | Insufficient sample |
| High-frequency | 41 | −$123.16 | −1.23% | Inconclusive |

Assumptions: $10,000 initial equity, 2-pip spread, 0.5-pip adverse slippage per
side, zero per-side commission (configure your actual broker's fee). No swap or
financing costs. Both expectancy confidence intervals include zero.

The conservative preset's tiny sample and the high-frequency preset's negative
synthetic result do not demonstrate an edge. No performance-based threshold
optimization or live activation was done. Local detailed reports are in
`reports/upgrade_conservative.json`, `reports/upgrade_high_frequency.json`, and
`reports/upgrade_walkforward.json` (ignored runtime artifacts).
