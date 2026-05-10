"""Natural language stock finder — searches across the S&P 500.

On first use, fetches sector/industry metadata from yfinance for all 500
constituents and caches to data/sp500_metadata.json (TTL 7 days).
Subsequent searches are instant (pure in-memory scoring).
"""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import yfinance as yf

logger = logging.getLogger(__name__)

_CACHE_PATH = Path(__file__).parent.parent / "data" / "sp500_metadata.json"
_CACHE_TTL_DAYS = 7
_FETCH_WORKERS = 25

# ── S&P 500 tickers (as of early 2025) ───────────────────────────────────────
SP500_TICKERS: list[str] = [
    "MMM","AOS","ABT","ABBV","ACN","ADBE","AMD","AES","AFL","A","APD","ABNB",
    "AKAM","ALB","ARE","ALGN","ALLE","LNT","ALL","GOOGL","GOOG","MO","AMZN",
    "AMCR","AEE","AEP","AXP","AIG","AMT","AWK","AMP","AME","AMGN","APH","ADI",
    "ANSS","AON","APA","AAPL","AMAT","APTV","ACGL","ADM","ANET","AJG","AIZ",
    "T","ATO","ADSK","ADP","AZO","AVB","AVY","AXON","BKR","BALL","BAC","BAX",
    "BDX","BRK-B","BBY","BIO","BIIB","BLK","BX","BA","BKNG","BWA","BSX","BMY",
    "AVGO","BR","BRO","BLDR","CBOE","CDNS","CPT","CPB","COF","CAH","KMX","CCL",
    "CARR","CAT","CBRE","CDW","CE","COR","CNC","CDAY","CF","CRL","SCHW","CHTR",
    "CVX","CMG","CB","CHD","CI","CINF","CTAS","CSCO","C","CFG","CLX","CME",
    "CMS","KO","CTSH","CL","CMCSA","CAG","COP","ED","STZ","CEG","COO","CPRT",
    "GLW","CPAY","CTVA","CSGP","COST","CTRA","CCI","CSX","CMI","CVS","DHI",
    "DHR","DRI","DVA","DECK","DE","DAL","DVN","DXCM","FANG","DLR","DFS","DG",
    "DLTR","D","DPZ","DOV","DOW","DTE","DUK","DD","EMN","ETN","EBAY","ECL",
    "EIX","EW","EA","ELV","LLY","EMR","ENPH","ETR","EOG","EPAM","EQT","EFX",
    "EQIX","EQR","ESS","EL","ETSY","EG","ES","EXC","EXPE","EXPD","EXR","XOM",
    "FFIV","FDS","FICO","FAST","FRT","FDX","FIS","FITB","FSLR","FE","FI","FMC",
    "F","FTNT","FTV","FOXA","FOX","BEN","FCX","GRMN","IT","GE","GEHC","GEN",
    "GNRC","GD","GIS","GM","GPC","GILD","GS","HAL","HIG","HAS","HCA","HSIC",
    "HSY","HES","HPE","HLT","HOLX","HD","HON","HRL","HST","HWM","HPQ","HUBB",
    "HUM","HBAN","HII","IBM","IEX","IDXX","ITW","INCY","INTC","ICE","IFF","IP",
    "IPG","INTU","ISRG","IVZ","INVH","IQV","IRM","JKHY","J","JBL","K","KVUE",
    "KDP","KEY","KEYS","KMB","KIM","KMI","KLAC","KHC","KR","LHX","LH","LRCX",
    "LW","LVS","LDOS","LEN","LIN","LYV","LKQ","LMT","L","LOW","LULU","LYB",
    "MTB","MRO","MPC","MKTX","MAR","MMC","MLM","MAS","MA","MKC","MCD","MCK",
    "MDT","MRK","META","MET","MTD","MGM","MCHP","MU","MSFT","MAA","MRNA","MHK",
    "MOH","TAP","MDLZ","MPWR","MNST","MCO","MS","MOS","MSI","MSCI","NDAQ","NTAP",
    "NFLX","NEM","NEE","NKE","NI","NDSN","NSC","NTRS","NOC","NCLH","NRG","NUE",
    "NVDA","NVR","NXPI","ORLY","OXY","ODFL","OMC","ON","OKE","ORCL","OTIS",
    "PCAR","PKG","PANW","PH","PAYX","PAYC","PYPL","PNR","PEP","PFE","PCG","PM",
    "PSX","PNW","PNC","POOL","PPG","PPL","PFG","PG","PGR","PLD","PRU","PEG",
    "PTC","PSA","PHM","PWR","QCOM","DGX","RL","RJF","RTX","O","REG","REGN",
    "RF","RSG","RMD","RVTY","ROK","ROL","ROP","ROST","RCL","SPGI","CRM","SBAC",
    "SLB","STX","SEE","SRE","NOW","SHW","SPG","SWKS","SJM","SNA","SO","LUV",
    "SWK","SBUX","STT","STLD","STE","SYK","SYF","SNPS","SYY","TMUS","TROW",
    "TTWO","TPR","TRGP","TGT","TEL","TDY","TFX","TER","TSLA","TXN","TXT","TMO",
    "TJX","TSCO","TT","TDG","TRV","TRMB","TFC","TYL","TSN","USB","UBER","UDR",
    "ULTA","UNP","UAL","UPS","URI","UNH","UHS","VLO","VTR","VRSN","VRSK","VZ",
    "VRTX","V","VST","VFC","VICI","WRB","GWW","WAB","WBA","WMT","DIS","WBD",
    "WM","WAT","WEC","WFC","WELL","WST","WDC","WY","WMB","WTW","WYNN","XEL",
    "XYL","YUM","ZBRA","ZBH","ZTS","SOLV","VLTO","NWSA","NWS","MTCH",
]

# ── Sector → extra search tags (yfinance sector names as keys) ───────────────
# yfinance uses: Technology, Healthcare, Financial Services, Consumer Defensive,
# Consumer Cyclical, Energy, Industrials, Basic Materials, Real Estate,
# Utilities, Communication Services
_SECTOR_EXTRA: dict[str, list[str]] = {
    "Technology":              ["tech", "digital", "software", "platform", "enterprise"],
    "Healthcare":              ["healthcare", "health", "medical", "pharma", "drug"],
    "Financial Services":      ["financial", "finance", "bank", "banking", "dividend", "income"],
    "Consumer Defensive":      ["staples", "defensive", "stable", "dividend", "income",
                                 "consumer staples", "brand", "recession proof"],
    "Consumer Cyclical":       ["consumer", "retail", "brand", "lifestyle", "discretionary"],
    "Energy":                  ["oil", "gas", "petroleum", "energy", "commodity", "dividend"],
    "Industrials":             ["industrial", "manufacturing", "cyclical", "defense", "aerospace"],
    "Basic Materials":         ["materials", "mining", "commodity", "cyclical", "chemical"],
    "Real Estate":             ["real estate", "reit", "property", "dividend", "income", "yield"],
    "Utilities":               ["utility", "utilities", "electric", "power", "regulated",
                                 "defensive", "stable", "dividend", "income", "yield"],
    "Communication Services":  ["media", "communication", "telecom", "content", "advertising",
                                 "streaming", "wireless"],
}

# ── Query expansion: user shorthand → canonical search tokens ─────────────────
_QUERY_EXPANSIONS: dict[str, list[str]] = {
    "defensive":     ["defensive", "stable", "consumer defensive", "recession proof"],
    "safe":          ["defensive", "stable", "consumer defensive"],
    "staples":       ["consumer defensive", "staples", "defensive", "food"],
    "consumer staples": ["consumer defensive", "staples", "defensive"],
    "income":        ["dividend", "income", "yield"],
    "dividend":      ["dividend", "income", "yield"],
    "yield":         ["dividend", "yield", "income"],
    "value":         ["value", "dividend", "undervalued"],
    "growth":        ["growth", "high growth"],
    "quality":       ["quality", "profitable", "margins"],
    "tech":          ["technology", "tech", "software"],
    "software":      ["software", "saas", "cloud", "technology"],
    "ai":            ["artificial intelligence", "machine learning", "ai"],
    "semiconductor": ["semiconductor", "semiconductors", "chip", "chips"],
    "chip":          ["semiconductor", "semiconductors", "chip", "chips"],
    "gpu":           ["semiconductor", "gpu", "graphics"],
    "pharma":        ["pharmaceutical", "pharmaceuticals", "drug", "healthcare"],
    "pharmaceutical":["pharmaceutical", "pharmaceuticals", "drug", "healthcare"],
    "biotech":       ["biotechnology", "biologic", "drug", "pharmaceutical"],
    "healthcare":    ["healthcare", "health", "medical", "drug"],
    "health":        ["healthcare", "health", "medical"],
    "bank":          ["bank", "banks", "banking", "financial services"],
    "financial":     ["financial services", "bank", "banks", "banking"],
    "payment":       ["payment", "transaction", "credit services"],
    "fintech":       ["financial services", "payment", "transaction", "credit services"],
    "cloud":         ["cloud computing", "saas", "cloud", "software"],
    "energy":        ["energy", "oil", "gas"],
    "oil":           ["oil", "gas", "petroleum", "energy"],
    "retail":        ["retail", "consumer cyclical", "store", "discount"],
    "ev":            ["electric vehicle", "auto", "battery"],
    "electric vehicle": ["electric vehicle", "auto", "battery"],
    "telecom":       ["telecommunications", "wireless", "telecom", "communication services"],
    "streaming":     ["streaming", "media", "entertainment", "content"],
    "defense":       ["defense", "aerospace", "military", "industrials"],
    "aerospace":     ["aerospace", "defense", "military", "industrials"],
    "reit":          ["real estate", "reit", "property", "dividend"],
}


# ── Metadata cache ────────────────────────────────────────────────────────────

def _cache_is_fresh() -> bool:
    if not _CACHE_PATH.exists():
        return False
    mtime = datetime.fromtimestamp(_CACHE_PATH.stat().st_mtime)
    return datetime.now() - mtime < timedelta(days=_CACHE_TTL_DAYS)


def _load_cache() -> dict[str, dict]:
    try:
        with open(_CACHE_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(data: dict[str, dict]) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_PATH, "w") as f:
        json.dump(data, f)


def _fetch_one(ticker: str) -> tuple[str, dict]:
    try:
        info = yf.Ticker(ticker).info
        return ticker, {
            "name":     info.get("longName") or info.get("shortName") or ticker,
            "sector":   info.get("sector") or "",
            "industry": info.get("industry") or "",
        }
    except Exception as exc:
        logger.debug("metadata fetch failed for %s: %s", ticker, exc)
        return ticker, {"name": ticker, "sector": "", "industry": ""}


def build_sp500_metadata(
    progress_callback=None,
) -> dict[str, dict]:
    """Fetch metadata for all S&P 500 tickers in parallel and cache result.

    progress_callback(done, total) is called after each completed ticker.
    """
    existing = _load_cache() if _cache_is_fresh() else {}
    missing = [t for t in SP500_TICKERS if t not in existing]

    if not missing:
        return existing

    metadata = dict(existing)
    total = len(missing)
    done = 0

    with ThreadPoolExecutor(max_workers=_FETCH_WORKERS) as pool:
        futures = {pool.submit(_fetch_one, t): t for t in missing}
        for future in as_completed(futures):
            ticker, data = future.result()
            metadata[ticker] = data
            done += 1
            if progress_callback:
                progress_callback(done, total)

    _save_cache(metadata)
    return metadata


# ── Scoring ───────────────────────────────────────────────────────────────────

@dataclass
class StockSuggestion:
    ticker: str
    name: str
    sector: str = ""
    industry: str = ""
    match_reason: str = ""
    score: float = 0.0


def _expand_query(query: str) -> set[str]:
    q = query.lower()
    tokens: set[str] = set()

    words = re.findall(r"\b\w+\b", q)
    tokens.update(words)
    for i in range(len(words) - 1):
        tokens.add(f"{words[i]} {words[i+1]}")

    # Use word-boundary matching so "tech" doesn't fire inside "fintech" / "biotech"
    for keyword, expansions in _QUERY_EXPANSIONS.items():
        pattern = r"\b" + re.escape(keyword) + r"\b"
        if re.search(pattern, q):
            tokens.update(expansions)

    return tokens


def _build_tag_groups(ticker: str, meta: dict) -> dict[str, list[str]]:
    """Build categorised tag groups so scoring can weight by specificity."""
    sector   = meta.get("sector", "")
    industry = meta.get("industry", "")
    name     = meta.get("name", ticker)

    _skip = {"inc", "corp", "company", "the", "and", "group", "holdings",
              "international", "limited", "ltd", "plc", "co", "llc", "class",
              "corporation", "systems", "solutions", "services", "technologies"}

    # Words in the sector name are not specific — don't credit them as industry tags
    sector_words = set(re.findall(r"\b[a-z]{4,}\b", sector.lower())) if sector else set()

    industry_tags: list[str] = []
    if industry:
        ind_lower = industry.lower()
        industry_tags.append(ind_lower)
        for word in re.findall(r"\b[a-z]{4,}\b", ind_lower):
            if word not in sector_words:   # skip words that are also in the sector name
                industry_tags.append(word)

    sector_tags: list[str] = []
    if sector:
        sector_tags.append(sector.lower())
        sector_tags.extend(_SECTOR_EXTRA.get(sector, []))

    name_tags: list[str] = []
    for word in re.findall(r"\b[a-z]{4,}\b", name.lower()):
        if word not in _skip:
            name_tags.append(word)

    return {
        "industry": industry_tags,
        "sector":   sector_tags,
        "name":     name_tags,
        "ticker":   [ticker.lower()],
    }


def _score_stock(ticker: str, tag_groups: dict[str, list[str]], tokens: set[str]) -> tuple[float, list[str]]:
    """Score a stock across four tiers.

    Tier weights (designed so higher-specificity signals dominate):
      ticker  exact  +8   (override everything)
      sector  name   +8   (e.g. "energy" in query and sector == "Energy")
      industry kw    +3 each, uncapped
      sector extra   +1 each, capped at 3 total
      company name   +0.5 each, capped at 1 total
    """
    matched: list[str] = []
    sector_name_tag = tag_groups["sector"][0] if tag_groups["sector"] else ""

    # ── Ticker exact match ────────────────────────────────────────────────────
    ticker_score = 0.0
    if ticker.lower() in tokens:
        ticker_score = 8.0
        matched.insert(0, f"ticker:{ticker}")

    # ── Sector name match (first tag in sector group is always the sector name)
    sector_name_score = 0.0
    if sector_name_tag and sector_name_tag in tokens and sector_name_tag not in matched:
        sector_name_score = 8.0
        matched.append(sector_name_tag)

    # ── Industry keyword matches (+3 each, uncapped) ──────────────────────────
    industry_score = 0.0
    for tag in tag_groups["industry"]:
        if tag in tokens and tag not in matched:
            industry_score += 3.0
            matched.append(tag)

    # ── Sector-extra tag matches (+1 each, capped at 3) ───────────────────────
    extra_score = 0.0
    for tag in tag_groups["sector"][1:]:   # skip index 0 (sector name, already handled)
        if extra_score >= 3.0:
            break
        if tag in tokens and tag not in matched:
            extra_score += 1.0
            matched.append(tag)

    # ── Company name keyword matches (+0.5 each, capped at 1) ────────────────
    name_score = 0.0
    for tag in tag_groups["name"]:
        if name_score >= 1.0:
            break
        if tag in tokens and tag not in matched:
            name_score += 0.5
            matched.append(tag)

    total = ticker_score + sector_name_score + industry_score + extra_score + name_score
    return total, matched


def search_stocks(
    query: str,
    max_results: int = 6,
    metadata: dict[str, dict] | None = None,
) -> list[StockSuggestion]:
    """Score every S&P 500 stock against the query and return ranked suggestions.

    Pass pre-loaded metadata to avoid re-loading the cache on every call.
    If metadata is None, loads from cache (fast after first build).
    """
    if not query or not query.strip():
        return []

    if metadata is None:
        metadata = _load_cache()
        if not metadata:
            # Cache empty — trigger a fetch (blocking)
            metadata = build_sp500_metadata()

    tokens = _expand_query(query.strip())
    scored: list[tuple[float, str, list[str]]] = []

    for ticker in SP500_TICKERS:
        meta = metadata.get(ticker, {})
        tag_groups = _build_tag_groups(ticker, meta)
        score, matched = _score_stock(ticker, tag_groups, tokens)
        if score > 0:
            scored.append((score, ticker, matched))

    scored.sort(key=lambda x: x[0], reverse=True)

    results: list[StockSuggestion] = []
    for score, ticker, matched in scored[:max_results]:
        meta = metadata.get(ticker, {})
        reason_tags = [t for t in matched if not t.startswith("ticker:")][:3]
        reason = ", ".join(reason_tags) if reason_tags else meta.get("sector", "")
        results.append(StockSuggestion(
            ticker=ticker,
            name=meta.get("name", ticker),
            sector=meta.get("sector", ""),
            industry=meta.get("industry", ""),
            match_reason=reason,
            score=score,
        ))

    return results
