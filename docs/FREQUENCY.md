# Trade frequency: what actually limits it, and what to do about it

This document records every measurement behind the frequency work, so the
numbers can be re-derived instead of taken on trust. All runs use the
deterministic synthetic gold feed (`--days 90 --seed 11 --end 2026-10-07T14:00Z`),
1 pip = $0.10, $10,000 starting equity.

> **Read this first.** The synthetic feed is regime-switching noise, not gold.
> It is good for measuring *throughput* (how many setups reach the market and
> why the others die) and useless for measuring *edge*. Every P/L figure below
> is a sample of one; 10–20 trades cannot separate a profitable system from a
> lucky one.

---

## 1. The funnel, end to end (90 days = 18,276 five-minute cycles)

| Stage | Count / 90 d | Gates that removed the rest |
|---|---|---|
| 5M cycles | 18,276 | — |
| Inside a tradeable session window | 8,600 | `SESSION_NOT_TRADEABLE:*`, `WEEKEND`, `NY_JUDAS_WINDOW` |
| HTF alignment passes | 6,000–6,500 | `HTF_NOT_ALIGNED` |
| A zone + sweep + structure candidate exists | ~500 | `MISSED_ENTRY`, `15M_STRUCTURE_CONFLICT`, `NO_VALID_ZONE`, `SELL_IN_DISCOUNT` |
| Unique setups armed (`armed`) | 50–160 | `DUPLICATE_FINGERPRINT` (same zone re-armed) |
| 5M trigger fires (`EXECUTE`) | 100–400 evaluations | `NO_ENTRY_PROXIMITY`, `Only 4/7 checks`, `OPPOSING_HTF_OB` |
| Orders submitted | 5–160 | `RISK: caution_zones`, `max_open_trades`, `CONFLUENCE:*`, `TP1_BELOW_1R` |

The single most misleading number in the old build was **"orders"**: a limit
order that rests at the zone edge and is never touched is counted as an order
but is not a trade. On the 90-day sample ~70 % of orders never filled. That is
why the v1.0 result read "5 trades" while the funnel showed ten times more
signals.

## 2. The five defects that were capping frequency

Each was fixed in code, and each is configurable, so the conservative behaviour
can be restored from `config.yaml`.

| # | Defect | Fix | Config key |
|---|---|---|---|
| 1 | Sweeps were only detected on **equal** high/low clusters with a 5-pip ($0.50) tolerance. On gold that is far tighter than a 15M candle, so 92 % of aligned setups died at `NO_CONFIRMED_SWEEP`. | Single swing highs/lows are raidable pools too (the classic ICT previous-swing raid). | `strategy.single_swing_lookback_bars` (0 = old behaviour) |
| 2 | `HTF_NOT_ALIGNED` demanded 4H **and** 1H to point the same way. That is stricter than the law ("never trade against the 4H") and blocks every with-trend pullback entry. | Added `4H_PULLBACK` mode: trade with the 4H while the 1H pulls back; and `RANGE` mode for a ranging 4H (correct half of the dealing range only). | `strategy.allow_1h_pullback_alignment`, `strategy.allow_range_alignment` |
| 3 | Entries rested as limit orders at the zone edge. Most zones were never revisited inside the 4-hour signal life. | `entry_mode: MARKET` fills at the 5M trigger, and TP1 is floored at `min_rr_tp1` so the R:R law still holds. | `strategy.entry_mode` |
| 4 | `risk.max_position_age_hours` was a **dead key**: positions were held until SL/TP and occupied the two portfolio slots for up to two days. | Time stop (`TIME_EXIT`) implemented in the position manager. | `risk.max_position_age_hours` |
| 5 | Arming required price to already be within the entry radius, so the bot could never *watch* a zone from a distance. | Arming radius separated from the trigger radius. | `strategy.arm_entry_distance_pips` |

Two smaller ones: a 20-pip chopbox threshold vetoed most of London (gold's
10-bar 15M range is routinely wider than $2.00) and the 15M structure veto was
applied twice — once at arming and again at execution time, even when the
operator had disabled it.

## 3. Measured trade frequency and P/L, 90 days, seed 11

| Profile | Trades | Trades/day | Win rate | Profit factor | Net P/L | Max DD |
|---|---|---|---|---|---|---|
| v1.0 (before this work) | 5 | 0.06 | 60.0 % | 2.18 | +$147.90 | 4.07 % |
| Default `config.yaml` (this build) | 15 | 0.17 | 26.7 % | 0.60 | −$259.26 | 4.52 % |
| `configs/high_frequency.yaml` (4 slots × 0.5 %) | 157 | **1.74** | 47.8 % | 0.93 | −$174.66 | 10.07 % |
| High-frequency + quality floor (`min_fivem_checks: 6`, confluence 6.0) | 91 | 1.01 | 44.0 % | 0.90 | −$153.53 | 6.07 % |

**The honest reading.** Frequency is now a knob, and 2 trades/day is reachable
(1.74 measured). What is *not* demonstrated is that the edge survives at that
frequency: at 1.74 trades/day the same signal set returns PF 0.93 and a 10 %
drawdown on this sample, and tightening the quality floors back to 6/6 gives
1.01 trades/day at PF 0.90 — i.e. adding trades did not *remove* an edge that
was there, the synthetic sample has no edge to find at any cadence. The
conservative profile loses too. The difference between +$148 (5 trades) and
−$259 (15 trades) is noise, not a verdict.

What the numbers *do* prove: 157 orders, 4 concurrent slots, 2 % total exposure,
5 % daily halt, spread cap and news blackout all held for 90 days without a
single risk breach, and the engine never stalled.

## 4. Choosing a profile

```bash
# conservative (default): ~1 trade every 5-6 days on this sample
python main.py backtest --days 90 --seed 11 --end 2026-10-07T14:00Z

# high frequency: ~1.7 trades/day, half size per trade, 2 % total exposure
python main.py --config configs/high_frequency.yaml backtest --days 90 --seed 11 --end 2026-10-07T14:00Z

# live/paper with a profile
python main.py --config configs/high_frequency.yaml run
```

`configs/high_frequency.yaml` keeps every hard law: no counter-4H trades, a
liquidity sweep is still mandatory, SL stays outside the OB, spread cap, news
blackout, 5 % daily circuit breaker and the **2 % total exposure cap**
(4 slots × 0.5 %). What it relaxes is quality grading — one fewer 5M
confirmation, confluence 5.0 (which is still half size: full size starts at 9),
and the optional 15M structure / premium-discount vetoes.

## 5. If you want more than 2 trades/day

The remaining ceiling is structural: a 15M order-block retest that also has a
sweep, a 5M confirmation and a session window happens ~1–2 times per day even
when every filter is relaxed. Going beyond that is not a threshold change, it is
a different strategy — an intraday scalper that trades 5M supply/demand with
5–15 pip stops. The plumbing for it now exists (`entry_mode: MARKET`, time stop,
multiple slots), but it needs its own entry module and its own validation.

## 6. What to measure before running real money

1. Re-run everything with `--csv` and 2+ years of real XAU/USD M5 candles.
2. Walk-forward: `python main.py walkforward --days 365 --csv data/xauusd_m5.csv`.
3. Minimum 100 trades per profile before believing any profit factor.
4. Compare `reports/*.json` — every run writes the full metric set, the equity
   curve and the per-trade journal.
