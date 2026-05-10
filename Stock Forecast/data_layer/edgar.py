"""SEC EDGAR + yfinance news integration.

Fetches:
  - Recent headlines via yfinance .news
  - 8-K / 10-Q / 10-K filings via EDGAR EFTS (free, no API key)
  - Risk Factors section from the most recent 10-K
  - Extracts catalyst signals from combined headline / filing data
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import requests
import yfinance as yf

logger = logging.getLogger(__name__)

# ── EDGAR endpoints ────────────────────────────────────────────────────────────
_EFTS_BASE   = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_BASE  = "https://www.sec.gov"
_HEADERS     = {
    "User-Agent": "StockAnalyzerApp research@example.com",
    "Accept":     "application/json",
}

# ── Catalyst keyword patterns ──────────────────────────────────────────────────
_CATALYST_PATTERNS: dict[str, list[str]] = {
    "earnings_beat":    ["beat", "exceeded", "surpassed", "above consensus", "above estimates",
                         "earnings beat", "topped estimates"],
    "earnings_miss":    ["miss", "below consensus", "below estimates", "earnings miss",
                         "fell short", "disappointing earnings"],
    "guidance_raise":   ["raised guidance", "raised outlook", "raised forecast",
                         "increased guidance", "raised its guidance", "boosted guidance",
                         "upped guidance"],
    "guidance_cut":     ["lowered guidance", "cut guidance", "reduced guidance",
                         "lowered outlook", "lowered forecast", "warns on", "profit warning"],
    "buyback":          ["share repurchase", "buyback", "buy back", "repurchase program",
                         "stock repurchase"],
    "dividend":         ["dividend", "quarterly dividend", "special dividend",
                         "dividend increase", "raises dividend"],
    "ma":               ["merger", "acquisition", "acquires", "acquired by", "takeover",
                         "deal", "buyout", "combine with", "strategic partnership"],
    "management":       ["ceo", "cfo", "appoints", "names new", "resign", "stepping down",
                         "departure", "executive change"],
    "regulatory":       ["fda approval", "cleared", "approved", "authorization",
                         "regulatory approval", "consent order", "sec investigation",
                         "doj probe", "antitrust"],
    "product_launch":   ["launches", "unveils", "introduces", "new product", "announced launch",
                         "breakthrough", "innovation"],
}


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass
class NewsItem:
    title: str
    publisher: str
    published_at: Optional[datetime]
    url: str
    catalysts: list[str] = field(default_factory=list)
    source: str = "yfinance"


@dataclass
class Filing:
    form_type: str          # e.g. "8-K", "10-Q", "10-K"
    filed_at: Optional[datetime]
    description: str
    filing_url: str
    document_url: str = ""
    catalysts: list[str] = field(default_factory=list)
    source: str = "edgar"


@dataclass
class NewsAndFilings:
    ticker: str
    news: list[NewsItem] = field(default_factory=list)
    filings: list[Filing] = field(default_factory=list)
    risk_factors: list[str] = field(default_factory=list)   # parsed from 10-K
    fetched_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def all_catalysts(self) -> list[str]:
        cats: list[str] = []
        for n in self.news:
            cats.extend(n.catalysts)
        for f in self.filings:
            cats.extend(f.catalysts)
        return list(dict.fromkeys(cats))  # deduplicated, order preserved


# ── Internal helpers ───────────────────────────────────────────────────────────

def _detect_catalysts(text: str) -> list[str]:
    """Return a list of catalyst-type labels found in *text* (case-insensitive)."""
    text_lower = text.lower()
    found: list[str] = []
    for label, patterns in _CATALYST_PATTERNS.items():
        if any(p in text_lower for p in patterns):
            found.append(label)
    return found


def _safe_get(url: str, params: dict | None = None, timeout: int = 10) -> dict | list | None:
    try:
        r = requests.get(url, params=params, headers=_HEADERS, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        logger.debug("EDGAR request failed %s: %s", url, exc)
        return None


def _cik_for_ticker(ticker: str) -> Optional[str]:
    """Resolve a ticker to its SEC CIK number via the EDGAR company search API."""
    data = _safe_get(
        f"{_EDGAR_BASE}/cgi-bin/browse-edgar",
        params={"action": "getcompany", "company": "", "CIK": ticker,
                "type": "", "dateb": "", "owner": "include",
                "count": "1", "search_text": "", "output": "atom"},
        timeout=8,
    )
    # The atom response is XML; parse it minimally for the CIK
    if data is None:
        return None
    # Fall back to the company facts endpoint which is JSON
    data2 = _safe_get(f"{_EDGAR_BASE}/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}"
                      f"&type=&dateb=&owner=include&count=1&output=atom", timeout=8)
    return None  # resolved below via direct JSON endpoint


def _get_cik_json(ticker: str) -> Optional[str]:
    """Use the EDGAR company search JSON endpoint to get CIK."""
    data = _safe_get(
        f"{_EDGAR_BASE}/cgi-bin/browse-edgar",
        params={"action": "getcompany", "CIK": ticker,
                "type": "10-K", "dateb": "", "owner": "include",
                "count": "5", "search_text": "", "output": "atom"},
    )
    # fallback: use submissions endpoint after we know CIK via tickers.json
    return None


def _get_cik_from_tickers(ticker: str) -> Optional[str]:
    """Look up CIK from EDGAR's company_tickers.json — a public JSON file."""
    data = _safe_get(f"{_EDGAR_BASE}/files/company_tickers.json")
    if not isinstance(data, dict):
        return None
    ticker_upper = ticker.upper()
    for _idx, info in data.items():
        if isinstance(info, dict) and info.get("ticker", "").upper() == ticker_upper:
            cik_raw = str(info.get("cik_str", ""))
            return cik_raw.zfill(10)
    return None


def _get_recent_filings(cik: str, form_types: list[str], limit: int = 10) -> list[dict]:
    """Fetch recent filings from the EDGAR submissions API."""
    data = _safe_get(f"{_EDGAR_BASE}/cgi-bin/browse-edgar?action=getcompany"
                     f"&CIK={cik}&type={form_types[0]}&dateb=&owner=include"
                     f"&count={limit}&search_text=&output=atom")
    # prefer the JSON submissions endpoint
    subs = _safe_get(f"{_EDGAR_BASE}/cgi-bin/browse-edgar?action=getcompany"
                     f"&CIK={cik}&type=&dateb=&owner=include"
                     f"&count=1&search_text=&output=atom")

    # Use submissions API
    submissions = _safe_get(f"https://data.sec.gov/submissions/CIK{cik}.json")
    if not isinstance(submissions, dict):
        return []

    recent = submissions.get("filings", {}).get("recent", {})
    forms      = recent.get("form", [])
    dates      = recent.get("filingDate", [])
    accs       = recent.get("accessionNumber", [])
    descriptions = recent.get("primaryDocument", [])

    results = []
    for form, date, acc, doc in zip(forms, dates, accs, descriptions):
        if form in form_types:
            acc_clean = acc.replace("-", "")
            filing_url = f"{_EDGAR_BASE}/Archives/edgar/data/{int(cik)}/{acc_clean}/{doc}"
            results.append({
                "form":        form,
                "date":        date,
                "accession":   acc,
                "document":    doc,
                "filing_url":  filing_url,
                "index_url":   f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
                               f"&CIK={cik}&type={form}&dateb=&owner=include&count=10",
            })
            if len(results) >= limit:
                break
    return results


def _parse_risk_factors(html: str, max_factors: int = 15) -> list[str]:
    """Extract the Risk Factors section from raw 10-K HTML/text, returning bullet list."""
    # Strip tags for plain text extraction
    clean = re.sub(r"<[^>]+>", " ", html)
    clean = re.sub(r"&nbsp;", " ", clean)
    clean = re.sub(r"&amp;", "&", clean)
    clean = re.sub(r"&#\d+;", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()

    # Locate "Risk Factors" heading
    match = re.search(
        r"(?i)(item\s+1a[\.\s]+risk\s+factors|risk\s+factors)",
        clean,
    )
    if not match:
        return []

    start = match.end()
    # End at next Item heading
    end_match = re.search(
        r"(?i)item\s+1b[\.\s]|item\s+2[\.\s]|unresolved\s+staff\s+comments",
        clean[start:],
    )
    section = clean[start : start + end_match.start()] if end_match else clean[start : start + 40_000]

    # Split into individual risk factor paragraphs by looking for sentence-terminal
    # patterns that often start a new risk: "The ", "We ", "Our ", "If " at the start
    # of a new paragraph-like block (preceded by period + space).
    sentences = re.split(r"(?<=[.!?])\s{2,}|\n{2,}", section)

    risks: list[str] = []
    buffer = ""
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        # A new risk typically starts with a bold/capitalized phrase followed by prose
        if len(sent) > 60:
            if buffer:
                risks.append(buffer[:400].strip())
                buffer = ""
            buffer = sent
        else:
            buffer = (buffer + " " + sent).strip() if buffer else sent

    if buffer:
        risks.append(buffer[:400].strip())

    # Deduplicate and trim
    seen: set[str] = set()
    unique: list[str] = []
    for r in risks:
        key = r[:80]
        if key not in seen and len(r) > 40:
            seen.add(key)
            unique.append(r)
        if len(unique) >= max_factors:
            break

    return unique


# ── Public API ─────────────────────────────────────────────────────────────────

def fetch_news(ticker: str, max_items: int = 20) -> list[NewsItem]:
    """Pull recent news headlines from yfinance."""
    try:
        tk = yf.Ticker(ticker)
        raw_news = tk.news or []
    except Exception as exc:
        logger.warning("yfinance news fetch failed for %s: %s", ticker, exc)
        return []

    items: list[NewsItem] = []
    for article in raw_news[:max_items]:
        title = article.get("title") or article.get("content", {}).get("title", "")
        publisher = article.get("publisher") or article.get("content", {}).get("provider", {}).get("displayName", "")
        url = article.get("link") or article.get("content", {}).get("canonicalUrl", {}).get("url", "")

        # Timestamp can be epoch int or nested
        ts = article.get("providerPublishTime") or article.get("content", {}).get("pubDate")
        published_at: Optional[datetime] = None
        if isinstance(ts, (int, float)):
            try:
                published_at = datetime.utcfromtimestamp(ts)
            except Exception:
                pass
        elif isinstance(ts, str):
            for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"):
                try:
                    published_at = datetime.strptime(ts, fmt)
                    break
                except ValueError:
                    pass

        catalysts = _detect_catalysts(title)
        items.append(NewsItem(
            title=title,
            publisher=publisher,
            published_at=published_at,
            url=url,
            catalysts=catalysts,
        ))
    return items


def fetch_filings(ticker: str, form_types: list[str] | None = None,
                  limit: int = 8) -> list[Filing]:
    """Fetch recent SEC filings for a ticker from EDGAR."""
    if form_types is None:
        form_types = ["8-K", "10-Q", "10-K"]

    cik = _get_cik_from_tickers(ticker)
    if not cik:
        logger.warning("Could not resolve CIK for %s", ticker)
        return []

    raw = _get_recent_filings(cik, form_types, limit=limit * 3)
    filings: list[Filing] = []
    for r in raw:
        if len(filings) >= limit:
            break
        try:
            filed_at = datetime.strptime(r["date"], "%Y-%m-%d")
        except Exception:
            filed_at = None

        description = f"{r['form']} — {r['document']}"
        catalysts = _detect_catalysts(r["document"])

        filings.append(Filing(
            form_type=r["form"],
            filed_at=filed_at,
            description=description,
            filing_url=r["index_url"],
            document_url=r["filing_url"],
            catalysts=catalysts,
        ))
    return filings


def fetch_risk_factors(ticker: str) -> list[str]:
    """Download the most recent 10-K and extract its Risk Factors section."""
    cik = _get_cik_from_tickers(ticker)
    if not cik:
        return []

    raw = _get_recent_filings(cik, ["10-K"], limit=3)
    if not raw:
        return []

    # Try the primary document of the most recent 10-K
    for filing in raw[:2]:
        doc_url = filing.get("filing_url", "")
        if not doc_url:
            continue
        try:
            resp = requests.get(doc_url, headers=_HEADERS, timeout=20)
            if resp.status_code == 200:
                risks = _parse_risk_factors(resp.text)
                if risks:
                    return risks
        except Exception as exc:
            logger.debug("Failed to fetch 10-K doc %s: %s", doc_url, exc)
            continue

    return []


def fetch_news_and_filings(ticker: str) -> NewsAndFilings:
    """Master fetch function — returns all news, filings, and risk factors."""
    news     = fetch_news(ticker)
    filings  = fetch_filings(ticker)
    risks    = fetch_risk_factors(ticker)
    return NewsAndFilings(ticker=ticker, news=news, filings=filings, risk_factors=risks)
