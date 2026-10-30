"""
ASTRA Mutual Funds — Indian mutual fund NAVs, analytics and paper investing.

Data: mfapi.in (free, mirrors AMFI's daily NAV file for every scheme).
  GET https://api.mfapi.in/mf/search?q=<text>   → [{schemeCode, schemeName}]
  GET https://api.mfapi.in/mf/<code>            → {meta: {...}, data: [{date: dd-mm-yyyy, nav}]}

Analytics are computed from the NAV history: trailing returns (absolute < 1y,
CAGR ≥ 1y), annualised volatility, max drawdown, rolling 3-year returns, and a
historical SIP simulator (XIRR). Paper investing books units at the NAV of the
transaction date, like a real fund; SIP instalments are booked lazily.
Paper trading only — nothing is ever sent to an AMC, RTA or broker.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE_URL = "https://api.mfapi.in/mf"
NAV_TTL_SEC = 6 * 3600          # NAVs publish once a day
SEARCH_TTL_SEC = 24 * 3600


class MFError(Exception):
    pass


MAX_ACTIVE_SIPS = 25


def canon_code(code) -> str:
    """AMFI scheme codes are plain ASCII digits; normalise ' 0120503' → '120503'."""
    import re as _re
    c = str(code or "").strip()
    if not _re.fullmatch(r"[0-9]{1,8}", c):
        raise MFError("Scheme code must be a number")
    return str(int(c))


def _default_fetch(url: str, params: Optional[dict] = None) -> object:
    import requests
    r = requests.get(url, params=params, timeout=12, headers={"User-Agent": "ASTRA/1.14"})
    if r.status_code == 404:
        raise MFError("Scheme not found")
    r.raise_for_status()
    return r.json()


class MutualFundService:
    def __init__(self, fetch: Callable[[str, Optional[dict]], object] = _default_fetch):
        self._fetch = fetch
        self._nav_cache: dict = {}          # code → (fetched_at, meta, nav Series)
        self._search_cache: dict = {}       # query → (fetched_at, results)
        self._lock = threading.Lock()

    # ── data ─────────────────────────────────────────────────────────────
    def _disk_dir(self) -> Path:
        from app.core.config import DATA_DIR
        d = Path(os.getenv("ASTRA_OFFLINE_DATA_DIR") or DATA_DIR) / "mf"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _parse(payload: dict) -> tuple[dict, pd.Series]:
        if not isinstance(payload, dict) or not payload.get("data"):
            raise MFError("No NAV data for this scheme")
        rows = payload["data"]
        idx = pd.to_datetime([r["date"] for r in rows], format="%d-%m-%Y", errors="coerce")
        vals = pd.to_numeric([r.get("nav") for r in rows], errors="coerce")
        s = pd.Series(vals, index=idx).dropna()
        s = s[s > 0]
        s = s[~s.index.duplicated(keep="first")].sort_index()
        if s.empty:
            raise MFError("No valid NAVs for this scheme")
        return payload.get("meta") or {}, s

    def nav_history(self, code: str) -> tuple[dict, pd.Series]:
        code = canon_code(code)
        hit = self._nav_cache.get(code)
        if hit and time.time() - hit[0] < NAV_TTL_SEC:
            return hit[1], hit[2]
        path = self._disk_dir() / f"{code}.json"
        offline = bool(os.getenv("ASTRA_OFFLINE_DATA_DIR"))
        payload = None
        if not offline:
            try:
                payload = self._fetch(f"{BASE_URL}/{code}", None)
                try:
                    path.write_text(json.dumps(payload))
                except Exception:
                    pass
            except MFError:
                raise
            except Exception as e:
                logger.warning(f"[mf] live fetch failed for {code}: {e}")
        if payload is None and path.exists():
            payload = json.loads(path.read_text())
        if payload is None:
            raise MFError("Mutual fund data is unavailable right now (mfapi.in unreachable)")
        meta, s = self._parse(payload)
        with self._lock:
            self._nav_cache[code] = (time.time(), meta, s)
        return meta, s

    def search(self, q: str, limit: int = 25) -> list[dict]:
        q = (q or "").strip()
        if len(q) < 2:
            return []
        key = q.lower()
        hit = self._search_cache.get(key)
        if hit and time.time() - hit[0] < SEARCH_TTL_SEC:
            res = hit[1]
        else:
            res = self._fetch(f"{BASE_URL}/search", {"q": q}) or []
            if not isinstance(res, list):
                res = []
            self._search_cache[key] = (time.time(), res)
        out = [{"scheme_code": str(r.get("schemeCode")), "scheme_name": r.get("schemeName", "")}
               for r in res if r.get("schemeCode")]
        # Direct-plan growth options first (no commission, no payouts)
        def rank(r):
            n = r["scheme_name"].lower()
            return (0 if "direct" in n else 1, 0 if "growth" in n else 1, n)
        return sorted(out, key=rank)[:limit]

    # ── analytics ────────────────────────────────────────────────────────
    @staticmethod
    def nav_for_order(s: pd.Series, d) -> tuple[Optional[pd.Timestamp], Optional[float]]:
        """NAV an order placed on date d gets: the first NAV dated on/after d
        (real funds price orders at the NAV declared for the order date or later,
        never at an NAV already known). None if not published yet → PENDING."""
        sub = s.loc[pd.Timestamp(d).normalize():]
        if sub.empty:
            return None, None
        return sub.index[0], float(sub.iloc[0])

    @staticmethod
    def nav_on(s: pd.Series, d) -> tuple[Optional[pd.Timestamp], Optional[float]]:
        """NAV applicable on date d (last published NAV on or before d)."""
        sub = s.loc[:pd.Timestamp(d)]
        if sub.empty:
            return None, None
        return sub.index[-1], float(sub.iloc[-1])

    @classmethod
    def trailing_returns(cls, s: pd.Series) -> dict:
        last_d, last = s.index[-1], float(s.iloc[-1])
        out = {}
        for label, days in (("1m", 30), ("3m", 91), ("6m", 182), ("1y", 365), ("3y", 1095), ("5y", 1826), ("10y", 3652)):
            d0, v0 = cls.nav_on(s, last_d - timedelta(days=days))
            if v0 is None or (last_d - d0).days < days * 0.95:
                out[label] = None
                continue
            yrs = (last_d - d0).days / 365.25
            r = last / v0
            out[label] = round(((r ** (1 / yrs)) - 1) * 100 if days >= 365 else (r - 1) * 100, 2)
        yrs_all = (last_d - s.index[0]).days / 365.25
        out["since_inception"] = round(((last / float(s.iloc[0])) ** (1 / yrs_all) - 1) * 100, 2) if yrs_all >= 1 else None
        return out

    @staticmethod
    def risk_stats(s: pd.Series) -> dict:
        tail = s[s.index >= s.index[-1] - timedelta(days=3 * 365)]
        r = tail.pct_change().dropna()
        vol = float(r.std() * np.sqrt(250) * 100) if len(r) > 30 else None
        dd = float((s / s.cummax() - 1).min() * 100)
        roll = None
        if (s.index[-1] - s.index[0]).days > 4 * 365:
            w = s.resample("W").last().dropna()
            r3 = (w / w.shift(156)) ** (1 / 3) - 1
            r3 = r3.dropna() * 100
            if len(r3):
                roll = {"min": round(float(r3.min()), 2), "median": round(float(r3.median()), 2),
                        "max": round(float(r3.max()), 2), "pct_positive": round(float((r3 > 0).mean() * 100), 1)}
        return {"volatility_3y_pct": round(vol, 2) if vol is not None else None,
                "max_drawdown_pct": round(dd, 2), "rolling_3y_cagr": roll}

    @staticmethod
    def xirr(cashflows: list[tuple[date, float]]) -> Optional[float]:
        """Annualised IRR for dated cashflows (negative = invested)."""
        if len(cashflows) < 2 or not any(c > 0 for _, c in cashflows) or not any(c < 0 for _, c in cashflows):
            return None
        t0 = min(d for d, _ in cashflows)
        ts = np.array([(d - t0).days / 365.25 for d, _ in cashflows])
        cf = np.array([c for _, c in cashflows])

        def npv(r):
            return float(np.sum(cf / (1 + r) ** ts))
        lo, hi = -0.99, 10.0
        f_lo, f_hi = npv(lo), npv(hi)
        if f_lo * f_hi > 0:
            return None
        for _ in range(200):
            mid = (lo + hi) / 2
            f_mid = npv(mid)
            if abs(f_mid) < 1e-6:
                break
            if f_lo * f_mid < 0:
                hi, f_hi = mid, f_mid
            else:
                lo, f_lo = mid, f_mid
        return round(mid * 100, 2)

    @classmethod
    def sip_projection(cls, s: pd.Series, amount: float, years: int, day: int = 5) -> dict:
        """What a monthly SIP over the last `years` would be worth today."""
        end = s.index[-1]
        start = end - pd.DateOffset(years=years)
        if start < s.index[0]:
            raise MFError(f"Fund history is shorter than {years} years")
        units = invested = 0.0
        flows = []
        d = pd.Timestamp(start.year, start.month, min(day, 28))
        if d < start:
            d += pd.DateOffset(months=1)
        while d <= end:
            nd, nav = cls.nav_on(s, d)
            if nav:
                units += amount / nav
                invested += amount
                flows.append((d.date(), -amount))
            d += pd.DateOffset(months=1)
        value = units * float(s.iloc[-1])
        flows.append((end.date(), value))
        return {"years": years, "monthly_amount": amount, "instalments": len(flows) - 1,
                "invested": round(invested, 2), "value": round(value, 2),
                "gain": round(value - invested, 2), "xirr_pct": cls.xirr(flows)}

    def scheme(self, code: str) -> dict:
        meta, s = self.nav_history(code)
        return {
            "scheme_code": str(code),
            "scheme_name": meta.get("scheme_name"),
            "fund_house": meta.get("fund_house"),
            "category": meta.get("scheme_category"),
            "scheme_type": meta.get("scheme_type"),
            "isin": meta.get("isin_growth"),
            "nav": round(float(s.iloc[-1]), 4),
            "nav_date": s.index[-1].strftime("%Y-%m-%d"),
            "first_nav_date": s.index[0].strftime("%Y-%m-%d"),
            "returns": self.trailing_returns(s),
            "risk": self.risk_stats(s),
            "nav_stale_days": (pd.Timestamp(date.today()) - s.index[-1]).days,
        }

    def chart(self, code: str, period: str = "5y") -> list[dict]:
        _, s = self.nav_history(code)
        days = {"1y": 365, "3y": 1095, "5y": 1826, "10y": 3652}.get(period)
        if days:
            s = s[s.index >= s.index[-1] - timedelta(days=days)]
        if len(s) > 800:   # thin out long histories for the chart
            s = s.resample("W").last().dropna()
        return [{"time": d.strftime("%Y-%m-%d"), "value": round(float(v), 4)} for d, v in s.items()]


mf_service = MutualFundService()


# ── paper portfolio (DB) ─────────────────────────────────────────────────────

def _today() -> date:
    return date.today()


def _holding(db, user_id: int, code: str, name: str):
    from app.models.database import MFHolding
    h = db.query(MFHolding).filter(MFHolding.user_id == user_id, MFHolding.scheme_code == code).first()
    if h is None:
        h = MFHolding(user_id=user_id, scheme_code=code, scheme_name=name, units=0.0, invested=0.0)
        db.add(h)
        db.flush()
    return h


MAX_NAV_LAG_DAYS = 7      # a fund whose latest NAV is older than this isn't accepting orders
PENDING_EXPIRY_DAYS = 10  # a pending order with no NAV after this many days is cancelled


def _check_priced(s: pd.Series):
    lag = (pd.Timestamp(_today()) - s.index[-1]).days
    if lag > MAX_NAV_LAG_DAYS:
        raise MFError(f"This fund hasn't published a NAV for {lag} days (closed or merged?) — "
                      "it isn't accepting orders")


def _cancel(db, tx, reason: str):
    """Expire an order that can't be priced; a redemption's reserved units go back."""
    if tx.kind == "SELL" and tx.units:
        h = _holding(db, tx.user_id, tx.scheme_code, tx.scheme_name)
        h.units = round((h.units or 0) + tx.units, 4)
        h.invested = round((h.invested or 0) + (tx.cost_basis or 0.0), 2)
    tx.status = "CANCELLED"
    logger.info(f"[mf] order {tx.id} cancelled: {reason}")


def _settle(db, tx, s: pd.Series) -> bool:
    """Price a PENDING transaction once the NAV for its order date is out.
    A NAV that only appears long after the order date (data gap) is not used."""
    nd, nav = MutualFundService.nav_for_order(s, tx.order_date)
    od = pd.Timestamp(tx.order_date)
    if nav is not None and (nd - od).days > MAX_NAV_LAG_DAYS:
        _cancel(db, tx, f"first NAV after the order date is {nd.date()}")
        return True
    if nav is None:
        if (pd.Timestamp(_today()) - od).days > PENDING_EXPIRY_DAYS:
            _cancel(db, tx, "no NAV published")
            return True
        return False
    h = _holding(db, tx.user_id, tx.scheme_code, tx.scheme_name)
    if tx.kind in ("BUY", "SIP"):
        units = round(tx.amount / nav, 4)
        h.units = round((h.units or 0) + units, 4)
        h.invested = round((h.invested or 0) + tx.amount, 2)
        tx.units = units
    else:  # SELL — units were reserved when the order was placed
        tx.amount = round(tx.units * nav, 2)
        tx.realised_pnl = round(tx.amount - (tx.cost_basis or 0.0), 2)
    tx.nav, tx.nav_date, tx.status = nav, nd.strftime("%Y-%m-%d"), "SETTLED"
    return True


def book_purchase(db, user_id: int, code: str, amount: float, kind: str = "BUY",
                  on: Optional[date] = None, sip_id: Optional[int] = None):
    from app.models.database import MFTransaction
    code = canon_code(code)
    if amount is None or not (amount >= 100):
        raise MFError("Minimum investment is ₹100")
    if amount > 1e8:
        raise MFError("Amount too large for a paper investment")
    meta, s = mf_service.nav_history(code)
    if on is None:
        _check_priced(s)
    d = on or _today()
    if pd.Timestamp(d) < s.index[0]:
        raise MFError("The fund didn't exist on that date")
    tx = MFTransaction(user_id=user_id, scheme_code=code, scheme_name=meta.get("scheme_name"),
                       kind=kind, amount=round(amount, 2), units=0.0, nav=None, nav_date=None,
                       sip_id=sip_id, status="PENDING", order_date=pd.Timestamp(d).strftime("%Y-%m-%d"))
    db.add(tx)
    db.flush()
    _settle(db, tx, s)
    return tx


def book_redemption(db, user_id: int, code: str, units: Optional[float] = None,
                    amount: Optional[float] = None):
    from app.models.database import MFHolding, MFTransaction
    code = canon_code(code)
    h = db.query(MFHolding).filter(MFHolding.user_id == user_id, MFHolding.scheme_code == code).first()
    if h is None or (h.units or 0) <= 0:
        raise MFError("You don't hold this fund")
    meta, s = mf_service.nav_history(code)
    _check_priced(s)
    if units is None:
        if amount is None or amount <= 0:
            raise MFError("Give units or an amount to redeem")
        units = amount / float(s.iloc[-1])           # estimate; priced at the order date's NAV
    units = round(min(float(units), h.units), 4)
    if units <= 0:
        raise MFError("Nothing to redeem")
    avg_cost = (h.invested or 0) / h.units
    cost = round(units * avg_cost, 2)
    h.invested = round((h.invested or 0) - cost, 2)
    h.units = round(h.units - units, 4)
    if h.units < 1e-4:
        h.units, h.invested = 0.0, 0.0
    tx = MFTransaction(user_id=user_id, scheme_code=code, scheme_name=meta.get("scheme_name"),
                       kind="SELL", amount=0.0, units=units, nav=None, nav_date=None,
                       status="PENDING", order_date=_today().isoformat(), cost_basis=cost)
    db.add(tx)
    db.flush()
    _settle(db, tx, s)
    return tx


def settle_pending(db, user_id: int) -> int:
    from app.models.database import MFTransaction
    n = 0
    for tx in db.query(MFTransaction).filter(MFTransaction.user_id == user_id, MFTransaction.status == "PENDING").all():
        try:
            _, s = mf_service.nav_history(tx.scheme_code)
            n += int(_settle(db, tx, s))
        except MFError:
            continue
    if n:
        db.commit()
    return n


def run_due_sips(db, user_id: int) -> int:
    """Book every SIP instalment due up to today (each priced at its own date's NAV).
    Each instalment date is claimed atomically, so concurrent loads can't double-book."""
    from app.models.database import MFSip
    booked = 0
    today = _today()
    for sip in db.query(MFSip).filter(MFSip.user_id == user_id, MFSip.active == True).all():  # noqa: E712
        try:
            _, s = mf_service.nav_history(sip.scheme_code)
        except MFError:
            continue
        first_nav = s.index[0].date()
        start = date.fromisoformat(sip.start_date)
        d = date(start.year, start.month, sip.day_of_month)
        if d < start:
            d = (pd.Timestamp(d) + pd.DateOffset(months=1)).date()
        guard = 0
        while d <= today and guard < 600:
            guard += 1
            last = sip.last_run_date
            if d >= first_nav and (last is None or d.isoformat() > last):
                claimed = db.query(MFSip).filter(
                    MFSip.id == sip.id,
                    (MFSip.last_run_date == None) | (MFSip.last_run_date < d.isoformat())  # noqa: E711
                ).update({MFSip.last_run_date: d.isoformat()}, synchronize_session=False)
                db.commit()
                db.refresh(sip)
                if claimed:
                    try:
                        book_purchase(db, user_id, sip.scheme_code, sip.amount, kind="SIP", on=d, sip_id=sip.id)
                        db.commit()
                        booked += 1
                    except MFError as e:
                        logger.info(f"[mf] SIP {sip.id} instalment {d} skipped: {e}")
            d = (pd.Timestamp(d) + pd.DateOffset(months=1)).date()
    return booked


def portfolio(db, user_id: int) -> dict:
    from app.models.database import MFHolding, MFSip, MFTransaction
    run_due_sips(db, user_id)
    settle_pending(db, user_id)
    rows, total_inv, total_val = [], 0.0, 0.0
    for h in db.query(MFHolding).filter(MFHolding.user_id == user_id, MFHolding.units > 0).all():
        try:
            meta, s = mf_service.nav_history(h.scheme_code)
            nav, nav_date = float(s.iloc[-1]), s.index[-1].strftime("%Y-%m-%d")
        except MFError:
            nav, nav_date = None, None
        value = round(h.units * nav, 2) if nav else None
        flows = [(datetime.strptime(t.nav_date, "%Y-%m-%d").date(), -t.amount if t.kind != "SELL" else t.amount)
                 for t in db.query(MFTransaction).filter(MFTransaction.user_id == user_id,
                                                         MFTransaction.scheme_code == h.scheme_code,
                                                         MFTransaction.status == "SETTLED").all()]
        if value:
            flows.append((_today(), value))
        rows.append({
            "scheme_code": h.scheme_code, "scheme_name": h.scheme_name, "units": round(h.units, 4),
            "invested": round(h.invested, 2), "nav": nav, "nav_date": nav_date, "value": value,
            "gain": round(value - h.invested, 2) if value is not None else None,
            "gain_pct": round((value / h.invested - 1) * 100, 2) if value and h.invested else None,
            "xirr_pct": MutualFundService.xirr(flows) if value else None,
        })
        total_inv += h.invested
        total_val += value or h.invested
    sips = [{"id": x.id, "scheme_code": x.scheme_code, "scheme_name": x.scheme_name, "amount": x.amount,
             "day_of_month": x.day_of_month, "start_date": x.start_date, "last_run_date": x.last_run_date}
            for x in db.query(MFSip).filter(MFSip.user_id == user_id, MFSip.active == True).all()]  # noqa: E712
    txs = [{"date": t.nav_date or t.order_date, "order_date": t.order_date, "status": t.status or "SETTLED",
            "scheme_name": t.scheme_name, "kind": t.kind, "amount": t.amount,
            "units": t.units, "nav": t.nav, "realised_pnl": t.realised_pnl}
           for t in db.query(MFTransaction).filter(MFTransaction.user_id == user_id)
           .order_by(MFTransaction.id.desc()).limit(50).all()]
    pending = [t for t in txs if t["status"] == "PENDING"]
    return {"holdings": rows, "sips": sips, "transactions": txs, "pending_orders": pending,
            "total_invested": round(total_inv, 2), "total_value": round(total_val, 2),
            "total_gain": round(total_val - total_inv, 2),
            "total_gain_pct": round((total_val / total_inv - 1) * 100, 2) if total_inv else None,
            "paper": True}
