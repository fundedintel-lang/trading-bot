"""OANDA v20 execution engine (async REST, no SDK required).

Safety:
  * refuses to construct in ``live`` environment unless ``allow_live`` is True,
  * verifies credentials are present,
  * every REST call has timeout + 3-attempt exponential backoff,
  * order payloads always carry a broker-side stop loss,
  * take-profit is attached at **TP2** (the runner target).  TP1 is managed by the
    position manager as a partial close so a 50% partial can never become a full
    exit if the process dies mid-management.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

import aiohttp

from apex_smc_bot.config import Secrets
from apex_smc_bot.data.candle_manager import to_utc
from apex_smc_bot.execution.broker import (
    AccountSnapshot,
    BrokerError,
    BrokerNotConfigured,
    OrderRequest,
    OrderResult,
    Position,
)

logger = logging.getLogger("apex.execution.oanda")

PRACTICE_URL = "https://api-fxpractice.oanda.com"
LIVE_URL = "https://api-fxtrade.oanda.com"


class OANDAExecutionEngine:
    """Order routing and position management against OANDA v20."""

    name = "oanda"

    def __init__(
        self,
        account_id: str,
        access_token: str,
        environment: str = "practice",
        allow_live: bool = False,
        timeout_seconds: float = 20.0,
        max_retries: int = 3,
        backoff_seconds: float = 1.5,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        if not account_id or not access_token:
            raise BrokerNotConfigured(
                "OANDA credentials missing: set OANDA_ACCOUNT_ID and OANDA_ACCESS_TOKEN"
            )
        if environment not in {"practice", "live"}:
            raise ValueError("environment must be 'practice' or 'live'")
        if environment == "live" and not allow_live:
            raise BrokerNotConfigured(
                "Refusing to construct a LIVE OANDA executor without allow_live=True"
            )
        self.account_id = account_id
        self._token = access_token
        self.environment = environment
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self._session = session
        self._owns_session = session is None

    @classmethod
    def from_settings(cls, settings: Any) -> "OANDAExecutionEngine":
        secrets: Secrets = settings.secrets
        return cls(
            account_id=secrets.oanda_account_id,
            access_token=secrets.oanda_access_token,
            environment=settings.execution.oanda_environment,
            allow_live=settings.execution.allow_live_trading,
            timeout_seconds=settings.execution.order_timeout_seconds,
            max_retries=settings.execution.max_retries,
            backoff_seconds=settings.execution.retry_backoff_seconds,
        )

    # ------------------------------------------------------------------ plumbing
    @property
    def base_url(self) -> str:
        return LIVE_URL if self.environment == "live" else PRACTICE_URL

    @property
    def configured(self) -> bool:
        return bool(self.account_id and self._token)

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept-Datetime-Format": "RFC3339",
            "Content-Type": "application/json",
        }

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout_seconds),
                headers=self.headers,
            )
        return self._session

    async def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        session = await self._get_session()
        url = f"{self.base_url}{path}"
        last_error: str = ""
        for attempt in range(self.max_retries):
            try:
                async with session.request(method, url, json=payload) as response:
                    text = await response.text()
                    if response.status == 429 or response.status >= 500:
                        last_error = f"HTTP {response.status}: {text[:200]}"
                        raise BrokerError(last_error)
                    try:
                        data = await response.json(content_type=None)
                    except Exception:  # noqa: BLE001
                        data = {}
                    if response.status >= 400:
                        message = data.get("errorMessage") or text[:300]
                        raise BrokerError(f"OANDA rejected {method} {path}: {message}")
                    return data
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_error = str(exc)
            except BrokerError as exc:
                last_error = str(exc)
                if "HTTP 4" in last_error and "429" not in last_error:
                    raise
            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.backoff_seconds * (2 ** attempt))
        raise BrokerError(f"OANDA request failed after {self.max_retries} attempts: {last_error}")

    @staticmethod
    def _price(instrument: str, price: float) -> str:
        decimals = 2 if "XAU" in instrument.upper() or "GOLD" in instrument.upper() else 5
        return f"{price:.{decimals}f}"

    @staticmethod
    def _units(units: float) -> str:
        return str(int(round(abs(units))))

    @staticmethod
    def _rfc3339(moment: datetime) -> str:
        return to_utc(moment).isoformat().replace("+00:00", "Z")

    def _account_path(self, suffix: str) -> str:
        return f"/v3/accounts/{self.account_id}{suffix}"

    # ------------------------------------------------------------------ orders
    async def submit_order(
        self, request: OrderRequest, market_price: float = 0.0, spread_price: float = 0.0
    ) -> OrderResult:
        order_type = request.order_type.upper()
        if order_type not in {"MARKET", "LIMIT"}:
            return OrderResult(
                success=False,
                status="REJECTED",
                message=f"Unsupported order type {request.order_type}",
                signal_id=request.signal_id,
            )
        order: dict[str, Any] = {
            "type": order_type,
            "instrument": request.instrument,
            "units": str(request.signed_units),
            "positionFill": "DEFAULT",
            "timeInForce": "FOK" if order_type == "MARKET" else "GTC",
            "clientExtensions": {
                "id": request.signal_id,
                "tag": request.client_tag[:128],
                "comment": f"APEX-SMC {request.signal_id}"[:128],
            },
            "stopLossOnFill": {
                "price": self._price(request.instrument, request.stop_loss),
                "timeInForce": "GTC",
            },
        }
        if request.take_profit:
            order["takeProfitOnFill"] = {
                "price": self._price(request.instrument, request.take_profit),
                "timeInForce": "GTC",
            }
        if order_type == "LIMIT":
            order["price"] = self._price(request.instrument, request.entry_price)
            if request.expires_at:
                order["timeInForce"] = "GTD"
                order["gtdTime"] = self._rfc3339(request.expires_at)
        data = await self._request("POST", self._account_path("/orders"), {"order": order})
        return self._parse_order_response(data, request)

    def _parse_order_response(self, data: dict[str, Any], request: OrderRequest) -> OrderResult:
        if data.get("orderRejectTransaction") or data.get("errorMessage"):
            reject = data.get("orderRejectTransaction") or {}
            reason = reject.get("rejectReason") or data.get("errorMessage") or "rejected"
            return OrderResult(
                success=False,
                status="REJECTED",
                message=str(reason),
                raw=data,
                signal_id=request.signal_id,
            )
        fill = data.get("orderFillTransaction")
        if fill:
            opened = fill.get("tradeOpened") or {}
            trade_id = str(opened.get("tradeID") or "")
            price = float(fill.get("price", 0) or 0)
            units = abs(float(opened.get("units", fill.get("units", 0)) or 0))
            logger.info(
                "OANDA FILL %s %s %s units @ %s (trade %s)",
                request.instrument,
                request.side,
                units,
                price,
                trade_id,
            )
            return OrderResult(
                success=True,
                status="FILLED",
                order_id=str(fill.get("id", "")),
                trade_id=trade_id,
                filled_price=price,
                units=units,
                message="filled",
                raw=data,
                signal_id=request.signal_id,
            )
        created = data.get("orderCreateTransaction")
        if created:
            return OrderResult(
                success=True,
                status="PENDING",
                order_id=str(created.get("id", "")),
                message="order accepted and resting",
                raw=data,
                signal_id=request.signal_id,
            )
        cancel = data.get("orderCancelTransaction")
        if cancel:
            return OrderResult(
                success=False,
                status="CANCELLED",
                order_id=str(cancel.get("id", "")),
                message=str(cancel.get("reason", "cancelled")),
                raw=data,
                signal_id=request.signal_id,
            )
        return OrderResult(
            success=False,
            status="REJECTED",
            message=f"Unrecognised OANDA response: {str(data)[:200]}",
            raw=data,
            signal_id=request.signal_id,
        )

    async def cancel_order(self, order_id: str) -> OrderResult:
        data = await self._request(
            "PUT", self._account_path(f"/orders/{order_id}/cancel"), {"order": {}}
        )
        cancelled = data.get("orderCancelTransaction") or {}
        return OrderResult(
            success=bool(cancelled),
            status="CANCELLED" if cancelled else "REJECTED",
            order_id=order_id,
            message=str(cancelled.get("reason", data.get("errorMessage", "cancel failed"))),
            raw=data,
        )

    # ------------------------------------------------------------------ management
    async def modify_stop(self, trade_id: str, stop_loss: float, instrument: str = "XAU_USD") -> OrderResult:
        payload = {
            "stopLoss": {
                "price": self._price(instrument, stop_loss),
                "timeInForce": "GTC",
            }
        }
        data = await self._request(
            "PUT", self._account_path(f"/trades/{trade_id}/orders"), payload
        )
        if data.get("stopLossOrderTransaction") or data.get("tradeCRCDOTransaction"):
            return OrderResult(
                success=True,
                status="MODIFIED",
                trade_id=trade_id,
                message=f"stop moved to {stop_loss}",
                raw=data,
            )
        return OrderResult(
            success=False,
            status="REJECTED",
            trade_id=trade_id,
            message=str(data.get("errorMessage", "stop modification failed")),
            raw=data,
        )

    async def partial_close(self, trade_id: str, units: float, price: float | None = None) -> OrderResult:
        payload = {"units": self._units(units)}
        data = await self._request(
            "PUT", self._account_path(f"/trades/{trade_id}/close"), payload
        )
        fill = data.get("orderFillTransaction") or {}
        if fill:
            closed_units = abs(float(fill.get("units", units) or units))
            realized = float(fill.get("pl", 0) or 0)
            return OrderResult(
                success=True,
                status="PARTIAL_CLOSE",
                trade_id=trade_id,
                filled_price=float(fill.get("price", 0) or 0) or price,
                units=closed_units,
                message=f"closed {closed_units} units, P/L {realized:+.2f}",
                raw=data,
                realized_pnl=realized,
            )
        return OrderResult(
            success=False,
            status="REJECTED",
            trade_id=trade_id,
            message=str(data.get("errorMessage", "partial close failed")),
            raw=data,
        )

    async def close_position(
        self, trade_id: str, price: float | None = None, reason: str = ""
    ) -> OrderResult:
        data = await self._request("PUT", self._account_path(f"/trades/{trade_id}/close"), {"units": "ALL"})
        fill = data.get("orderFillTransaction") or {}
        if fill:
            return OrderResult(
                success=True,
                status="CLOSED",
                trade_id=trade_id,
                filled_price=float(fill.get("price", 0) or 0) or price,
                message=f"closed ({reason})" if reason else "closed",
                raw=data,
                realized_pnl=float(fill.get("pl", 0) or 0),
            )
        return OrderResult(
            success=False,
            status="REJECTED",
            trade_id=trade_id,
            message=str(data.get("errorMessage", "close failed")),
            raw=data,
        )

    async def close_all(self, instrument: str | None = None, reason: str = "CIRCUIT_BREAKER") -> list[OrderResult]:
        if instrument:
            payload = {"longUnits": "ALL", "shortUnits": "ALL"}
            data = await self._request(
                "PUT", self._account_path(f"/positions/{instrument}/close"), payload
            )
            return [
                OrderResult(
                    success=True,
                    status="CLOSED",
                    message=f"position {instrument} closed ({reason})",
                    raw=data,
                )
            ]
        results = []
        for position in await self.get_positions():
            results.append(await self.close_position(position.trade_id, reason=reason))
        return results

    # ------------------------------------------------------------------ reads
    async def get_positions(self) -> list[Position]:
        data = await self._request("GET", self._account_path("/openTrades"))
        trades = data.get("trades") or []
        positions: list[Position] = []
        for trade in trades:
            try:
                stop = trade.get("stopLossOrder") or {}
                take = trade.get("takeProfitOrder") or {}
                client = trade.get("clientExtensions") or {}
                current_units = float(trade.get("currentUnits", trade.get("initialUnits", 0)) or 0)
                positions.append(
                    Position(
                        trade_id=str(trade.get("id", "")),
                        signal_id=str(client.get("id", "")),
                        instrument=str(trade.get("instrument", "XAU_USD")),
                        direction="BULLISH" if current_units > 0 else "BEARISH",
                        units=abs(current_units),
                        initial_units=abs(float(trade.get("initialUnits", current_units) or 0)),
                        entry_price=float(trade.get("price", 0) or 0),
                        stop_loss=float(stop.get("price", 0) or 0),
                        take_profit=float(take.get("price", 0)) if take.get("price") else None,
                        opened_at=datetime.fromisoformat(str(trade.get("openTime")).replace("Z", "+00:00")),
                        current_price=float(trade.get("price", 0) or 0),
                        metadata={"unrealized_pnl": float(trade.get("unrealizedPL", 0) or 0)},
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping malformed OANDA trade: %s", exc)
        return positions

    async def get_closed_trades(self, count: int = 20) -> list[dict[str, Any]]:
        """Recently closed trades with realized P/L (used for reconciliation)."""

        data = await self._request(
            "GET", f"/v3/accounts/{self.account_id}/trades?state=CLOSED&count={max(1, int(count))}"
        )
        return data.get("trades") or []

    async def get_pending_orders(self) -> list[dict[str, Any]]:
        data = await self._request("GET", self._account_path("/pendingOrders"))
        return data.get("orders") or []

    async def get_account(self) -> AccountSnapshot:
        data = await self._request("GET", self._account_path("/summary"))
        account = data.get("account") or {}
        balance = float(account.get("balance", 0) or 0)
        nav = float(account.get("NAV", balance) or balance)
        return AccountSnapshot(
            balance=balance,
            equity=nav,
            margin_available=float(account.get("marginAvailable", 0) or 0),
            open_trade_count=int(account.get("openTradeCount", 0) or 0),
            currency=str(account.get("currency", "USD")),
        )

    async def get_quote(self, instrument: str) -> tuple[float, float]:
        data = await self._request(
            "GET", f"/v3/accounts/{self.account_id}/pricing?instruments={instrument}"
        )
        prices = data.get("prices") or []
        if not prices:
            raise BrokerError("OANDA returned no pricing data")
        bid = float(prices[0]["bids"][0]["price"])
        ask = float(prices[0]["asks"][0]["price"])
        return bid, ask

    async def close(self) -> None:
        if self._session is not None and not self._session.closed and self._owns_session:
            await self._session.close()
        self._session = None

    def __repr__(self) -> str:
        suffix = self.account_id[-4:] if len(self.account_id) > 4 else "****"
        return f"OANDAExecutionEngine(account=***{suffix}, env={self.environment})"
