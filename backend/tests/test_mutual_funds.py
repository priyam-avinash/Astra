"""Mutual fund analytics + paper investing (offline: fake mfapi.in responses)."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest


def _fake_payload(code="120503", years=6, cagr=0.12, name="Test Flexi Cap Fund - Direct Plan - Growth"):
    end = date.today()
    days = pd.bdate_range(end - timedelta(days=int(365.25 * years)), end)
    n = len(days)
    t = np.arange(n) / 250.0
    nav = 10 * (1 + cagr) ** t
    rows = [{"date": d.strftime("%d-%m-%Y"), "nav": f"{v:.4f}"} for d, v in zip(days[::-1], nav[::-1])]
    return {"meta": {"scheme_code": int(code), "scheme_name": name, "fund_house": "Test AMC",
                     "scheme_category": "Equity Scheme - Flexi Cap Fund", "scheme_type": "Open Ended Schemes"},
            "data": rows, "status": "SUCCESS"}


@pytest.fixture()
def svc(monkeypatch, tmp_path):
    from app.services import mutual_funds as mf
    monkeypatch.setenv("ASTRA_DATA_DIR", str(tmp_path))
    payloads = {"120503": _fake_payload()}

    def fetch(url, params=None):
        if url.endswith("/search"):
            return [{"schemeCode": 120503, "schemeName": "Test Flexi Cap Fund - Direct Plan - Growth"},
                    {"schemeCode": 999, "schemeName": "Test Flexi Cap Fund - Regular Plan - IDCW"}]
        code = url.rsplit("/", 1)[-1]
        if code not in payloads:
            raise mf.MFError("Scheme not found")
        return payloads[code]

    s = mf.MutualFundService(fetch=fetch)
    monkeypatch.setattr(mf, "mf_service", s)
    import app.api.mf_router as r
    monkeypatch.setattr(r, "mf_service", s)
    return s


def test_trailing_returns_match_constant_growth(svc):
    info = svc.scheme("120503")
    assert info["fund_house"] == "Test AMC"
    for k in ("1y", "3y", "5y"):
        assert abs(info["returns"][k] - 12.0) < 0.6, (k, info["returns"][k])
    assert info["risk"]["max_drawdown_pct"] > -0.5          # monotonic NAV → no drawdown


def test_search_prefers_direct_growth(svc):
    res = svc.search("flexi")
    assert res[0]["scheme_code"] == "120503"


def test_sip_projection_xirr_close_to_cagr(svc):
    _, s = svc.nav_history("120503")
    p = svc.sip_projection(s, 10000, 5)
    assert p["instalments"] in (59, 60, 61)
    assert p["value"] > p["invested"]
    assert abs(p["xirr_pct"] - 12.0) < 0.8


def test_xirr_simple_case():
    from app.services.mutual_funds import MutualFundService
    r = MutualFundService.xirr([(date(2020, 1, 1), -100.0), (date(2021, 1, 1), 110.0)])
    assert abs(r - 10.0) < 0.1


def test_invalid_code_rejected(svc):
    from app.services.mutual_funds import MFError
    with pytest.raises(MFError):
        svc.nav_history("../etc/passwd")


def test_paper_invest_redeem_and_sip_flow(svc, jwt_client, random_username):
    c = jwt_client
    c.post("/api/auth/register", json={"username": random_username, "password": "pw_test_123"})
    tok = c.post("/api/auth/token", data={"username": random_username, "password": "pw_test_123"}).json()["access_token"]
    h = {"Authorization": f"Bearer {tok}"}
    assert c.get("/api/mf/portfolio").status_code == 401        # login required

    r = c.post("/api/mf/invest", json={"scheme_code": "120503", "amount": 50000}, headers=h)
    assert r.status_code == 200, r.text
    assert r.json()["paper"] is True and r.json()["units"] > 0

    # back-dated SIP books past instalments at their own NAVs
    start = (date.today() - timedelta(days=200)).isoformat()
    r = c.post("/api/mf/sip", json={"scheme_code": "120503", "amount": 5000, "day_of_month": 5,
                                    "start_date": start}, headers=h)
    assert r.status_code == 200, r.text
    pf = c.get("/api/mf/portfolio", headers=h).json()
    assert len(pf["holdings"]) == 1
    kinds = [t["kind"] for t in pf["transactions"]]
    assert kinds.count("SIP") >= 6 and "BUY" in kinds
    assert pf["total_value"] >= pf["total_invested"]            # rising NAV

    units = pf["holdings"][0]["units"]
    r = c.post("/api/mf/redeem", json={"scheme_code": "120503", "units": units / 2}, headers=h)
    assert r.status_code == 200 and r.json()["realised_pnl"] >= 0
    pf2 = c.get("/api/mf/portfolio", headers=h).json()
    assert abs(pf2["holdings"][0]["units"] - units / 2) < 0.01

    # another user can't see or stop this user's SIP
    sip_id = pf2["sips"][0]["id"]
    other = random_username + "z"
    c.post("/api/auth/register", json={"username": other, "password": "pw_test_123"})
    tok2 = c.post("/api/auth/token", data={"username": other, "password": "pw_test_123"}).json()["access_token"]
    h2 = {"Authorization": f"Bearer {tok2}"}
    assert c.get("/api/mf/portfolio", headers=h2).json()["holdings"] == []
    assert c.delete(f"/api/mf/sip/{sip_id}", headers=h2).status_code == 404


def test_minimum_amount(svc, jwt_client, random_username):
    c = jwt_client
    c.post("/api/auth/register", json={"username": random_username, "password": "pw_test_123"})
    tok = c.post("/api/auth/token", data={"username": random_username, "password": "pw_test_123"}).json()["access_token"]
    r = c.post("/api/mf/invest", json={"scheme_code": "120503", "amount": 50},
               headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 400
