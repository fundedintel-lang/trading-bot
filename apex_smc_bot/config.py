"""Typed, validated configuration with YAML file + environment variable support.

Precedence (low → high):
    1. dataclass defaults below
    2. ``config.yaml`` (path overridable via ``APEX_CONFIG``)
    3. explicit environment variables for secrets/operational switches

Secrets never live in YAML: they are read from the environment (``.env`` is
loaded by the CLI before settings are constructed).
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import time
from pathlib import Path
from typing import Any

class ConfigError(ValueError):
    """Raised when configuration is missing or internally inconsistent."""


def _time(value: str | time) -> time:
    if isinstance(value, time):
        return value
    try:
        hours, minutes = str(value).split(":")[:2]
        return time(hour=int(hours), minute=int(minutes))
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"Invalid HH:MM time value {value!r}") from exc


@dataclass(frozen=True)
class BotConfig:
    name: str = "APEX-SMC"
    version: str = "1.0.0"
    instrument: str = "XAU_USD"
    execution_mode: str = "SEMI_AUTO"  # SEMI_AUTO | FULL_AUTO
    timezone: str = "UTC"


@dataclass(frozen=True)
class ExecutionConfig:
    broker: str = "paper"  # paper | oanda
    allow_live_trading: bool = False
    oanda_environment: str = "practice"  # practice | live
    max_retries: int = 3
    retry_backoff_seconds: float = 1.5
    order_timeout_seconds: float = 20.0
    pending_order_ttl_minutes: int = 120
    paper_starting_equity: float = 10_000.0
    paper_auto_approve: bool = True  # paper fills may execute without human approval


@dataclass(frozen=True)
class RiskConfig:
    max_risk_per_trade_percent: float = 1.0
    max_daily_loss_percent: float = 5.0
    reduce_risk_threshold_percent: float = 3.0
    reduced_risk_percent: float = 0.5
    max_open_trades: int = 2
    max_spread_pips: float = 3.5
    max_slippage_pips: float = 2.0
    max_sl_pips: float = 50.0
    min_sl_pips: float = 5.0
    pip_size: float = 0.10
    pip_value_per_lot: float = 10.0
    contract_size: float = 100.0
    min_lot: float = 0.01
    max_lot: float = 10.0
    daily_halt_hours: float = 24.0
    cooldown_after_loss_hours: float = 2.0
    cooldown_after_two_losses_hours: float = 4.0
    max_consecutive_losses: int = 2
    spread_spike_multiplier: float = 2.0
    spread_pause_minutes: int = 15
    min_equity: float = 100.0
    starting_equity: float = 10_000.0
    trailing_breakeven_buffer_pips: float = 1.0
    # Move the stop to breakeven as soon as TP1 fills (spec default).  Setting
    # this False keeps the original stop alive until TP2, so the runner is not
    # scratched at entry by a normal pullback — measured on the 90-day sample,
    # the breakeven-after-TP1 rule was capping the realised payoff at ~1:1
    # while the plan is 1.5R/3R/5R.
    breakeven_after_tp1: bool = True
    # The time stop closes stale positions.  When True (default) it only closes
    # the ones that are not in profit, so a runner cannot be cut at +0.4R just
    # because it is old; winners get a hard cap at twice max_position_age_hours.
    time_exit_only_losers: bool = True
    max_position_age_hours: float = 48.0
    max_cost_to_stop_ratio: float = 0.25  # spread + slippage reserve / stop distance

    def __post_init__(self) -> None:
        if self.max_open_trades < 1:
            raise ConfigError("risk.max_open_trades must be at least 1")
        exposure = self.max_open_trades * self.max_risk_per_trade_percent
        if exposure > 2.0 + 1e-9:
            raise ConfigError(
                "risk.max_open_trades x risk.max_risk_per_trade_percent must stay "
                f"within the 2% portfolio cap (got {exposure:.2f}%)"
            )


@dataclass(frozen=True)
class StrategyConfig:
    swing_lookback: int = 3
    external_swing_lookback: int = 8
    min_confluence_score: float = 7.0
    min_fivem_checks: int = 6
    ob_lookback_candles: int = 100
    fvg_lookback_candles: int = 100
    ote_low_fib: float = 0.618
    ote_high_fib: float = 0.79
    equal_level_tolerance_pips: float = 5.0
    max_signal_age_minutes: int = 240
    sl_buffer_pips: float = 5.0
    min_sl_atr_fraction: float = 0.8   # stops tighter than this fraction of 15M ATR are noise
    min_rr_tp1: float = 1.5
    min_rr_tp2: float = 3.0
    target_rr_tp3: float = 5.0
    ob_freshness_hours: float = 48.0
    fvg_freshness_hours: float = 24.0
    sweep_lookahead_candles: int = 16
    # Individual swings within this many bars are raidable liquidity pools in
    # addition to equal-high/low clusters (0 disables the classic swing raid).
    single_swing_lookback_bars: int = 96
    inducement_lookback_candles: int = 12
    min_zone_ob_quality: int = 3        # order blocks below this are not entry zones
    min_zone_width_pips: float = 4.0
    max_zone_width_pips: float = 30.0
    tp1_close_fraction: float = 0.50
    tp2_close_fraction: float = 0.25
    # Maximum distance (pips) between live price and a pending limit entry.
    # Beyond this the zone is stale and the pending setup is discarded.
    max_entry_distance_pips: float = 60.0
    # Arming radius.  0 = same as max_entry_distance_pips.  A wider arming
    # radius lets the bot *watch* a zone from further away while the 5M trigger
    # still enforces "price is at the zone" before an order is sent.
    arm_entry_distance_pips: float = 0.0
    opposing_ob_warning_pips: float = 25.0
    opposing_ob_abort_pips: float = 40.0
    min_obstacle_quality: int = 3     # ignore weak opposing blocks when checking the path
    zone_touch_bars: int = 3          # an entry zone counts as "live" for N bars after the touch
    chopbox_lookback: int = 10
    # A 10-bar 15M range narrower than this is chop.  $2.00 was too tight for
    # gold's normal intraday noise and vetoed most valid setups.
    chopbox_max_range_pips: float = 35.0

    # ---- selectivity / frequency controls --------------------------------
    # When the 4H is ranging there is no HTF bias to trade against, so a setup
    # may follow the 1H instead — but only from the correct half of the 4H
    # dealing range (buy the discount, sell the premium).  This is range
    # trading, never counter-trend trading.
    # The 4H is the law ("never trade against 4H bias"), but the 1H does not
    # have to agree: a 1H pullback *inside* a 4H trend is the classic discount
    # entry.  When False only fully aligned 4H+1H setups may arm.
    allow_1h_pullback_alignment: bool = True
    # Hard daily trend filter.  When enabled a setup may only be taken in the
    # direction of the D1 trend (close vs its 20-day EMA and the EMA's slope,
    # derived from the 4H series).  Counter-trend intraday bounces inside a
    # daily trend are the most expensive trades in the sample.
    daily_trend_filter: bool = False
    allow_range_alignment: bool = True
    # A ranging 15M is "no structure", not "conflicting structure": the sweep,
    # the zone and the HTF bias already carry the edge.
    allow_15m_ranging: bool = True
    # A 15M structure pointing the other way rejects the setup.  Strong-trend
    # profiles may keep it on; higher-frequency profiles turn it off and let
    # the sweep + zone + 5M trigger carry the decision.
    require_15m_structure: bool = True
    # Only buy the discount / sell the premium of the 15M dealing range.
    require_premium_discount: bool = True
    # A 15M event in the trade direction counts as structure this many bars back.
    structure_event_window_bars: int = 36
    # Accept a raid that has not yet produced its BOS.  The sweep still has to
    # exist (a raid beyond an equal high/low that closed back inside).
    require_sweep_confirmation: bool = False
    min_sweep_quality: str = "MEDIUM"  # HIGH | MEDIUM | LOW
    # LIMIT  — rest at the zone edge and wait for the pullback (highest R:R,
    #          lowest fill rate: most zones are never revisited inside the
    #          signal lifetime).
    # MARKET — enter at the 5M trigger price once price is already inside the
    #          confirmed entry window (fills immediately, same structural stop).
    entry_mode: str = "LIMIT"

    def __post_init__(self) -> None:
        mode = str(self.entry_mode).upper()
        if mode not in {"LIMIT", "MARKET"}:
            raise ConfigError(f"strategy.entry_mode must be LIMIT or MARKET, got {self.entry_mode!r}")
        object.__setattr__(self, "entry_mode", mode)
        quality = str(self.min_sweep_quality).upper()
        if quality not in {"HIGH", "MEDIUM", "LOW"}:
            raise ConfigError(
                f"strategy.min_sweep_quality must be HIGH, MEDIUM or LOW, got {self.min_sweep_quality!r}"
            )
        object.__setattr__(self, "min_sweep_quality", quality)


@dataclass(frozen=True)
class SessionConfig:
    london_kill_zone_start: time = time(7, 0)
    london_kill_zone_end: time = time(10, 0)
    ny_kill_zone_start: time = time(12, 0)
    ny_kill_zone_end: time = time(15, 0)
    ny_close_start: time = time(17, 0)
    ny_close_end: time = time(20, 0)
    judas_window_minutes: int = 30
    allow_outside_kill_zone: bool = False
    # Kill zones are the highest-quality window, but the London afternoon
    # (10:00-12:00) and the London close / NY afternoon (15:00-17:00) are liquid
    # sessions where reversal setups also print.  When True, any session marked
    # tradeable may arm; dead zones and the Judas window stay blocked.
    trade_secondary_sessions: bool = True

    def __post_init__(self) -> None:
        for start, end, label in (
            (self.london_kill_zone_start, self.london_kill_zone_end, "london"),
            (self.ny_kill_zone_start, self.ny_kill_zone_end, "ny"),
            (self.ny_close_start, self.ny_close_end, "ny_close"),
        ):
            if _time(start) >= _time(end):
                raise ConfigError(f"{label} session start must be before end")


@dataclass(frozen=True)
class NotificationConfig:
    telegram_enabled: bool = False
    discord_enabled: bool = False
    email_enabled: bool = False
    console_enabled: bool = True
    timeout_seconds: float = 10.0
    max_retries: int = 3


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    directory: str = "logs"
    json_logs: bool = True
    max_bytes: int = 10 * 1024 * 1024
    backup_count: int = 5
    signal_log: str = "signals.log"
    trade_log: str = "trades.log"
    error_log: str = "errors.log"
    analysis_log: str = "analysis.log"

    def __post_init__(self) -> None:
        if self.level.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ConfigError(f"Invalid log level {self.level!r}")


@dataclass(frozen=True)
class FeedConfig:
    primary: str = "oanda"  # oanda | synthetic | twelve | mt5
    secondary: str = "twelve"
    tertiary: str = "mt5"
    historical_bars: int = 500
    request_timeout_seconds: float = 15.0
    max_retries: int = 3
    synthetic_seed: int = 7
    max_candle_age_seconds: float = 600.0
    max_quote_age_seconds: float = 30.0


@dataclass(frozen=True)
class DatabaseConfig:
    enabled: bool = True
    path: str = "apex_state.sqlite3"


@dataclass(frozen=True)
class NewsConfig:
    enabled: bool = True
    calendar_url: str = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    cache_minutes: int = 30
    block_minutes_before: int = 30
    block_minutes_after: int = 30
    fail_closed: bool = True  # no new entries without a usable calendar
    high_impact_keywords: tuple[str, ...] = (
        "FOMC",
        "NFP",
        "Non-Farm",
        "CPI",
        "PPI",
        "JOLTS",
        "ISM",
        "Powell",
        "Federal Funds",
        "GDP",
        "PCE",
        "Unemployment",
        "Retail Sales",
    )


@dataclass(frozen=True)
class BacktestConfig:
    days: int = 60
    initial_equity: float = 10_000.0
    spread_pips: float = 2.0
    slippage_pips: float = 0.5
    commission_per_lot: float = 0.0
    risk_percent: float = 1.0
    seed: int = 42
    data_csv: str | None = None
    min_trades_for_report: int = 20


@dataclass(frozen=True)
class AIConfig:
    """Optional second opinion, never an order generator or risk authority."""

    enabled: bool = False
    provider: str = "groq"  # groq | xai (Grok)
    model: str = ""        # blank selects the provider default below
    mode: str = "advisory"  # advisory | veto
    timeout_seconds: float = 8.0
    max_tokens: int = 400
    max_calls_per_day: int = 100
    cache_ttl_seconds: float = 60.0
    failure_cooldown_seconds: float = 60.0
    min_confidence: float = 0.65
    fail_closed: bool = True  # applies only in veto mode

    @property
    def model_name(self) -> str:
        return self.model or {"groq": "llama-3.3-70b-versatile", "xai": "grok-3-mini"}.get(self.provider, "")


@dataclass(frozen=True, repr=False)
class Secrets:
    """Runtime credentials — env only, never serialised into logs."""

    groq_api_key: str = ""
    xai_api_key: str = ""
    oanda_account_id: str = ""
    oanda_access_token: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    discord_webhook_url: str = ""
    twelve_data_api_key: str = ""
    mt5_login: str = ""
    mt5_password: str = ""
    mt5_server: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    email_to: str = ""

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Secrets":
        environ = os.environ if environ is None else environ

        def env(name: str, default: str = "") -> str:
            return environ.get(name, default).strip()

        try:
            smtp_port = int(env("SMTP_PORT", "587") or 587)
        except ValueError:
            smtp_port = 587
        return cls(
            groq_api_key=env("GROQ_API_KEY"),
            xai_api_key=env("XAI_API_KEY"),
            oanda_account_id=env("OANDA_ACCOUNT_ID"),
            oanda_access_token=env("OANDA_ACCESS_TOKEN"),
            telegram_bot_token=env("TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=env("TELEGRAM_CHAT_ID"),
            discord_webhook_url=env("DISCORD_WEBHOOK_URL"),
            twelve_data_api_key=env("TWELVE_DATA_API_KEY"),
            mt5_login=env("MT5_LOGIN"),
            mt5_password=env("MT5_PASSWORD"),
            mt5_server=env("MT5_SERVER"),
            smtp_host=env("SMTP_HOST"),
            smtp_port=smtp_port,
            smtp_user=env("SMTP_USER"),
            smtp_password=env("SMTP_PASSWORD"),
            email_to=env("EMAIL_TO"),
        )

    @property
    def has_oanda(self) -> bool:
        return bool(self.oanda_account_id and self.oanda_access_token)

    @property
    def has_telegram(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)

    def redacted(self) -> dict[str, Any]:
        """Safe representation for logs/health checks."""

        return {
            "groq_configured": bool(self.groq_api_key),
            "xai_configured": bool(self.xai_api_key),
            "oanda_configured": self.has_oanda,
            "telegram_configured": self.has_telegram,
            "discord_configured": bool(self.discord_webhook_url),
            "twelve_data_configured": bool(self.twelve_data_api_key),
            "mt5_configured": bool(self.mt5_login),
            "smtp_configured": bool(self.smtp_host),
        }


    def __repr__(self) -> str:
        return f"Secrets(configured={self.redacted()!r})"


@dataclass(frozen=True)
class Settings:
    bot: BotConfig = field(default_factory=BotConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    sessions: SessionConfig = field(default_factory=SessionConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    feed: FeedConfig = field(default_factory=FeedConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    news: NewsConfig = field(default_factory=NewsConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    secrets: Secrets = field(default_factory=Secrets, repr=False)
    live_trading_confirmed: bool = False  # environment-only third interlock
    project_root: str = "."

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: str | Path | None = None, env: dict[str, str] | None = None) -> "Settings":
        """Load settings from YAML + environment.

        Args:
            path: Optional config file. Defaults to ``$APEX_CONFIG`` or ``config.yaml``.
            env: Optional environment mapping (defaults to ``os.environ``); used by tests.
        """

        environ = dict(os.environ if env is None else env)
        config_path = Path(path or environ.get("APEX_CONFIG", "config.yaml"))
        raw: dict[str, Any] = {}
        if config_path.exists():
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover
                raise ConfigError("PyYAML is required to read config.yaml") from exc
            try:
                loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                raise ConfigError(f"Could not parse {config_path}: {exc}") from exc
            if loaded is not None and not isinstance(loaded, dict):
                raise ConfigError("config.yaml must contain a mapping at the top level")
            raw = loaded or {}

        settings = cls._from_dict(raw, environ)
        settings.validate()
        return settings

    @classmethod
    def _from_dict(cls, raw: dict[str, Any], environ: dict[str, str]) -> "Settings":
        allowed_sections = {"bot", "execution", "risk", "strategy", "sessions", "notifications",
                            "logging", "feed", "database", "news", "backtest", "ai"}
        unknown = set(raw) - allowed_sections
        if unknown:
            raise ConfigError(f"Unknown config sections: {sorted(unknown)}; secrets belong in the environment")

        def section(name: str) -> dict[str, Any]:
            value = raw.get(name) or {}
            if not isinstance(value, dict):
                raise ConfigError(f"config section {name!r} must be a mapping")
            return value

        def build(factory: Any, values: dict[str, Any], cast: dict[str, Any] | None = None) -> Any:
            allowed = factory.__dataclass_fields__  # type: ignore[attr-defined]
            defaults = factory()
            kwargs: dict[str, Any] = {}
            cast = cast or {}
            for key, value in values.items():
                if key not in allowed:
                    raise ConfigError(f"Unknown config key {key!r} for {factory.__name__}")
                if key in cast:
                    value = cast[key](value)
                default = getattr(defaults, key)
                if isinstance(default, bool) and not isinstance(value, bool):
                    raise ConfigError(f"{factory.__name__}.{key} must be a YAML boolean")
                if isinstance(default, (int, float)) and not isinstance(default, bool):
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ConfigError(f"{factory.__name__}.{key} must be a finite number")
                    if isinstance(default, int) and not isinstance(value, int):
                        raise ConfigError(f"{factory.__name__}.{key} must be an integer")
                kwargs[key] = value
            return factory(**kwargs)

        try:
            sessions_raw = section("sessions")
            sessions = build(
                SessionConfig,
                sessions_raw,
                cast={
                    "london_kill_zone_start": _time,
                    "london_kill_zone_end": _time,
                    "ny_kill_zone_start": _time,
                    "ny_kill_zone_end": _time,
                    "ny_close_start": _time,
                    "ny_close_end": _time,
                },
            )
            news_raw = section("news")
            if "high_impact_keywords" in news_raw and isinstance(news_raw["high_impact_keywords"], list):
                news_raw = dict(news_raw, high_impact_keywords=tuple(news_raw["high_impact_keywords"]))
            settings = cls(
                bot=build(BotConfig, section("bot")),
                execution=build(ExecutionConfig, section("execution")),
                risk=build(RiskConfig, section("risk")),
                strategy=build(StrategyConfig, section("strategy")),
                sessions=sessions,
                notifications=build(NotificationConfig, section("notifications")),
                logging=build(LoggingConfig, section("logging")),
                feed=build(FeedConfig, section("feed")),
                database=build(DatabaseConfig, section("database")),
                news=build(NewsConfig, news_raw),
                backtest=build(BacktestConfig, section("backtest")),
                ai=build(AIConfig, section("ai")),
                project_root=str(Path.cwd()),
            )
        except TypeError as exc:  # dataclass constructor mismatch
            raise ConfigError(str(exc)) from exc

        def env_bool(name: str, default: bool = False) -> bool:
            if name not in environ:
                return default
            value = environ[name].strip().lower()
            if value not in {"1", "true", "yes", "0", "false", "no"}:
                raise ConfigError(f"{name} must be true or false")
            return value in {"1", "true", "yes"}

        # The environment cannot turn on the YAML live switch: BOTH are required.
        bot = replace(settings.bot, execution_mode=environ.get("APEX_EXECUTION_MODE", settings.bot.execution_mode).upper())
        ai = replace(
            settings.ai,
            enabled=env_bool("APEX_AI_ENABLED", settings.ai.enabled),
            provider=environ.get("APEX_AI_PROVIDER", settings.ai.provider).strip().lower(),
            model=environ.get("APEX_AI_MODEL", settings.ai.model).strip(),
            mode=environ.get("APEX_AI_MODE", settings.ai.mode).strip().lower(),
        )
        return replace(settings, bot=bot, ai=ai, secrets=Secrets.from_env(environ),
                       live_trading_confirmed=env_bool("APEX_ALLOW_LIVE"))

    # ------------------------------------------------------------------ helpers
    def validate(self) -> None:
        """Raise ``ConfigError`` when values are inconsistent or unsafe."""

        risk, strategy = self.risk, self.strategy
        # Also validate directly constructed settings (tests and library callers).
        for section in (self.risk, self.strategy, self.execution, self.feed, self.backtest, self.ai,
                        self.news, self.notifications, self.database, self.sessions):
            for key, value in asdict_shallow(section).items():
                default = section.__dataclass_fields__[key].default
                if isinstance(default, bool) and not isinstance(value, bool):
                    raise ConfigError(f"{type(section).__name__}.{key} must be boolean")
                if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(value):
                    raise ConfigError(f"{type(section).__name__}.{key} must be finite")
        if not isinstance(self.live_trading_confirmed, bool):
            raise ConfigError("live_trading_confirmed must be boolean")
        if not 0 < risk.max_risk_per_trade_percent <= 2.0:
            raise ConfigError("risk.max_risk_per_trade_percent must be within (0, 2]")
        if not 0 < risk.max_daily_loss_percent <= 20:
            raise ConfigError("risk.max_daily_loss_percent must be within (0, 20]")
        if not 0 < risk.reduced_risk_percent < risk.max_risk_per_trade_percent:
            raise ConfigError("risk.reduced_risk_percent must be below max risk per trade")
        if self.bot.execution_mode not in {"SEMI_AUTO", "FULL_AUTO"}:
            raise ConfigError("bot.execution_mode must be SEMI_AUTO or FULL_AUTO")
        if self.execution.broker not in {"paper", "oanda"}:
            raise ConfigError("execution.broker must be 'paper' or 'oanda'")
        if self.execution.oanda_environment not in {"practice", "live"}:
            raise ConfigError("execution.oanda_environment must be 'practice' or 'live'")
        if self.execution.broker == "oanda" and self.execution.oanda_environment == "live":
            if not self.execution.allow_live_trading or not self.live_trading_confirmed:
                raise ConfigError("Live OANDA requires BOTH execution.allow_live_trading: true and APEX_ALLOW_LIVE=true")
        if self.execution.broker == "oanda" and self.feed.primary == "synthetic":
            raise ConfigError("Synthetic prices must never drive an OANDA account")
        if risk.max_sl_pips <= risk.min_sl_pips:
            raise ConfigError("risk.max_sl_pips must be greater than min_sl_pips")
        if risk.min_lot <= 0 or risk.max_lot < risk.min_lot:
            raise ConfigError("Lot limits are invalid")
        if risk.pip_size <= 0 or risk.pip_value_per_lot <= 0:
            raise ConfigError("pip_size and pip_value_per_lot must be positive")
        if risk.contract_size <= 0 or not math.isclose(risk.pip_value_per_lot, risk.pip_size * risk.contract_size):
            raise ConfigError("pip_value_per_lot must equal pip_size * contract_size for USD-quoted sizing")
        if not 0 < risk.max_cost_to_stop_ratio <= 1:
            raise ConfigError("risk.max_cost_to_stop_ratio must be within (0, 1]")
        if risk.min_sl_pips <= 0 or risk.max_slippage_pips < 0 or risk.max_spread_pips <= 0:
            raise ConfigError("Stop/spread limits must be positive and slippage non-negative")
        if not 0 < risk.reduce_risk_threshold_percent < risk.max_daily_loss_percent:
            raise ConfigError("Reduced-risk drawdown threshold must be below the daily halt")
        if self.execution.pending_order_ttl_minutes <= 0 or self.execution.paper_starting_equity <= 0:
            raise ConfigError("Pending order TTL and paper equity must be positive")
        if self.feed.max_quote_age_seconds <= 0 or self.feed.max_candle_age_seconds <= 0:
            raise ConfigError("Feed freshness limits must be positive")

        if not 0 < strategy.ote_low_fib < strategy.ote_high_fib < 1:
            raise ConfigError("OTE Fibonacci bounds must satisfy 0 < low < high < 1")
        if not 0 <= strategy.min_confluence_score <= 10:
            raise ConfigError("strategy.min_confluence_score must be within [0, 10]")
        if not 0 <= strategy.min_fivem_checks <= 7:
            raise ConfigError("strategy.min_fivem_checks must be within [0, 7]")
        if strategy.max_entry_distance_pips <= 0:
            raise ConfigError("strategy.max_entry_distance_pips must be positive")
        if self.feed.historical_bars < 250:
            raise ConfigError("feed.historical_bars must be at least 250 for HTF analysis")
        if not (0 < strategy.tp1_close_fraction < 1 and 0 < strategy.tp2_close_fraction < 1
                and strategy.tp1_close_fraction + strategy.tp2_close_fraction < 1):
            raise ConfigError("TP fractions must be positive and leave a runner (sum < 1)")
        if not 0 < strategy.min_rr_tp1 < strategy.min_rr_tp2 <= strategy.target_rr_tp3:
            raise ConfigError("Target R multiples must increase: TP1 < TP2 <= TP3")
        if self.ai.provider not in {"groq", "xai"} or self.ai.mode not in {"advisory", "veto"}:
            raise ConfigError("ai.provider must be groq or xai; ai.mode must be advisory or veto")
        if not 0 <= self.ai.min_confidence <= 1 or not 0 < self.ai.timeout_seconds <= 30:
            raise ConfigError("AI confidence must be in [0, 1] and timeout in (0, 30]")
        if not 32 <= self.ai.max_tokens <= 2000 or self.ai.max_calls_per_day < 1:
            raise ConfigError("AI token limit must be 32–2000 and daily call limit positive")
        if self.ai.cache_ttl_seconds < 0 or self.ai.failure_cooldown_seconds <= 0:
            raise ConfigError("AI cache TTL must be non-negative and cooldown positive")
        if self.backtest.days < 1 or self.backtest.initial_equity <= 0:
            raise ConfigError("Backtest days and equity must be positive")
        if min(self.backtest.spread_pips, self.backtest.slippage_pips, self.backtest.commission_per_lot) < 0:
            raise ConfigError("Backtest costs cannot be negative")
        if self.database.enabled and not self.database.path:
            raise ConfigError("database.path is required when database.enabled is true")

    @property
    def logs_dir(self) -> Path:
        path = Path(self.logging.directory)
        if not path.is_absolute():
            path = Path(self.project_root) / path
        return path

    @property
    def database_path(self) -> Path:
        path = Path(self.database.path)
        if not path.is_absolute():
            path = Path(self.project_root) / path
        return path

    def ensure_runtime_dirs(self) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        if self.database.enabled:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)

    def public_dict(self) -> dict[str, Any]:
        """Serialisable config without secrets (for logs/dashboard)."""

        data = asdict_shallow(self)
        data.pop("secrets", None)
        data["project_root"] = "."
        for key, value in list(data.items()):
            if hasattr(value, "__dataclass_fields__"):
                data[key] = asdict_shallow(value)
        data["secrets_configured"] = self.secrets.redacted()
        return data


def asdict_shallow(obj: Any) -> dict[str, Any]:
    """``dataclasses.asdict`` without deep-copying nested dataclasses."""

    return {field_name: getattr(obj, field_name) for field_name in obj.__dataclass_fields__}


def load_settings(path: str | Path | None = None) -> Settings:
    """Convenience wrapper used by the entry point and tests."""

    return Settings.load(path)
