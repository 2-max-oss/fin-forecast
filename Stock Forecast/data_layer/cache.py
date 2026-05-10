"""DuckDB + Parquet cache manager with TTL enforcement."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from config import (
    DUCKDB_PATH,
    PARQUET_DIR,
    PRICE_TTL_SECONDS,
    FUNDAMENTALS_TTL_SECONDS,
    ESTIMATES_TTL_SECONDS,
    MACRO_TTL_SECONDS,
    INFO_TTL_SECONDS,
)
from core.exceptions import CacheError

logger = logging.getLogger(__name__)

TTL_MAP: dict[str, int] = {
    "price":        PRICE_TTL_SECONDS,
    "fundamentals": FUNDAMENTALS_TTL_SECONDS,
    "estimates":    ESTIMATES_TTL_SECONDS,
    "macro":        MACRO_TTL_SECONDS,
    "info":         INFO_TTL_SECONDS,
}


def _ttl(data_type: str) -> int:
    return TTL_MAP.get(data_type, PRICE_TTL_SECONDS)


class CacheManager:
    _instance: Optional["CacheManager"] = None

    def __init__(self, db_path: Path = DUCKDB_PATH):
        self._db_path = db_path
        self._conn = duckdb.connect(str(db_path))
        self._init_schema()

    # ── Singleton ──────────────────────────────────────────────────────────────
    @classmethod
    def get(cls) -> "CacheManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Schema ─────────────────────────────────────────────────────────────────
    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cache_metadata (
                cache_key   VARCHAR PRIMARY KEY,
                data_type   VARCHAR NOT NULL,
                ticker      VARCHAR,
                fetched_at  TIMESTAMP NOT NULL,
                expires_at  TIMESTAMP NOT NULL,
                file_path   VARCHAR
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS financials (
                ticker      VARCHAR,
                period_end  DATE,
                period_type VARCHAR,
                statement   VARCHAR,
                item        VARCHAR,
                value       DOUBLE,
                PRIMARY KEY (ticker, period_end, period_type, statement, item)
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS macro_series (
                series_id   VARCHAR,
                date        DATE,
                value       DOUBLE,
                PRIMARY KEY (series_id, date)
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS company_info (
                ticker          VARCHAR PRIMARY KEY,
                name            VARCHAR,
                sector          VARCHAR,
                industry        VARCHAR,
                market_cap      DOUBLE,
                shares_outstanding DOUBLE,
                enterprise_value DOUBLE,
                dividend_yield  DOUBLE,
                currency        VARCHAR,
                exchange        VARCHAR,
                description     VARCHAR,
                website         VARCHAR,
                country         VARCHAR,
                employees       INTEGER,
                fetched_at      TIMESTAMP,
                expires_at      TIMESTAMP
            )
        """)
        self._ensure_column("company_info", "dividend_yield", "DOUBLE")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS analyst_estimates (
                ticker              VARCHAR PRIMARY KEY,
                forward_revenue     DOUBLE,
                forward_ebitda      DOUBLE,
                forward_eps         DOUBLE,
                forward_pe          DOUBLE,
                target_mean_price   DOUBLE,
                target_median_price DOUBLE,
                recommendation      VARCHAR,
                num_analysts        INTEGER,
                fetched_at          TIMESTAMP,
                expires_at          TIMESTAMP
            )
        """)

    def _ensure_column(self, table: str, column: str, column_type: str) -> None:
        cols = self._conn.execute(f"PRAGMA table_info('{table}')").fetchdf()
        if column not in set(cols["name"]):
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")

    # ── TTL helpers ────────────────────────────────────────────────────────────
    def _is_valid(self, cache_key: str) -> bool:
        row = self._conn.execute(
            "SELECT expires_at FROM cache_metadata WHERE cache_key = ?", [cache_key]
        ).fetchone()
        if row is None:
            return False
        return datetime.utcnow() < row[0]

    def _upsert_metadata(
        self, cache_key: str, data_type: str, ticker: str | None, file_path: str | None = None
    ) -> None:
        now = datetime.utcnow()
        expires = now + timedelta(seconds=_ttl(data_type))
        self._conn.execute("""
            INSERT OR REPLACE INTO cache_metadata
                (cache_key, data_type, ticker, fetched_at, expires_at, file_path)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [cache_key, data_type, ticker, now, expires, file_path])

    # ── Price (Parquet) ────────────────────────────────────────────────────────
    def get_price(self, ticker: str) -> pd.DataFrame | None:
        key = f"price:{ticker}"
        if not self._is_valid(key):
            return None
        row = self._conn.execute(
            "SELECT file_path FROM cache_metadata WHERE cache_key = ?", [key]
        ).fetchone()
        if row is None or not row[0]:
            return None
        path = Path(row[0])
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except Exception as e:
            logger.warning("Failed to read price cache for %s: %s", ticker, e)
            return None

    def set_price(self, ticker: str, df: pd.DataFrame) -> None:
        path = PARQUET_DIR / f"{ticker.upper()}_price.parquet"
        df.to_parquet(path)
        key = f"price:{ticker}"
        self._upsert_metadata(key, "price", ticker, str(path))

    # ── Fundamentals (DuckDB financials table) ────────────────────────────────
    def get_financials(self, ticker: str) -> dict[str, pd.DataFrame] | None:
        key = f"fundamentals:{ticker}"
        if not self._is_valid(key):
            return None
        rows = self._conn.execute(
            "SELECT period_end, period_type, statement, item, value "
            "FROM financials WHERE ticker = ?",
            [ticker],
        ).fetchdf()
        if rows.empty:
            return None
        return self._financials_df_to_dict(rows)

    def set_financials(self, ticker: str, data: dict[str, pd.DataFrame]) -> None:
        # data = {"income_annual": df, "income_quarterly": df, ...}
        # Flatten to long form
        rows = []
        for key_name, df in data.items():
            parts = key_name.split("_", 1)
            stmt = parts[0]
            period_type = parts[1] if len(parts) > 1 else "annual"
            for item in df.columns:
                for period_end, value in df[item].items():
                    if pd.notna(value):
                        rows.append((ticker, period_end, period_type, stmt, item, float(value)))
        if rows:
            self._conn.execute("DELETE FROM financials WHERE ticker = ?", [ticker])
            self._conn.executemany(
                "INSERT INTO financials VALUES (?, ?, ?, ?, ?, ?)", rows
            )
        cache_key = f"fundamentals:{ticker}"
        self._upsert_metadata(cache_key, "fundamentals", ticker)

    def _financials_df_to_dict(self, long_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        result = {}
        for (stmt, period_type), group in long_df.groupby(["statement", "period_type"]):
            pivot = group.pivot_table(index="period_end", columns="item", values="value", aggfunc="first")
            pivot.index = pd.to_datetime(pivot.index)
            pivot = pivot.sort_index(ascending=False)
            result[f"{stmt}_{period_type}"] = pivot
        return result

    # ── Macro ─────────────────────────────────────────────────────────────────
    def get_macro(self, series_id: str) -> pd.Series | None:
        key = f"macro:{series_id}"
        if not self._is_valid(key):
            return None
        rows = self._conn.execute(
            "SELECT date, value FROM macro_series WHERE series_id = ? ORDER BY date",
            [series_id],
        ).fetchdf()
        if rows.empty:
            return None
        s = pd.Series(rows["value"].values, index=pd.to_datetime(rows["date"]))
        s.name = series_id
        return s

    def set_macro(self, series_id: str, series: pd.Series) -> None:
        rows = [(series_id, str(d.date()), float(v)) for d, v in series.items() if pd.notna(v)]
        if rows:
            self._conn.execute("DELETE FROM macro_series WHERE series_id = ?", [series_id])
            self._conn.executemany(
                "INSERT OR REPLACE INTO macro_series VALUES (?, ?, ?)", rows
            )
        key = f"macro:{series_id}"
        self._upsert_metadata(key, "macro", None)

    # ── Company Info ──────────────────────────────────────────────────────────
    def get_info(self, ticker: str) -> dict | None:
        now = datetime.utcnow()
        cols = [
            "ticker", "name", "sector", "industry", "market_cap",
            "shares_outstanding", "enterprise_value", "dividend_yield",
            "currency", "exchange", "description", "website", "country",
            "employees", "fetched_at", "expires_at",
        ]
        row = self._conn.execute(
            "SELECT "
            + ", ".join(cols)
            + " FROM company_info WHERE ticker = ? AND expires_at > ?",
            [ticker, now],
        ).fetchone()
        if row is None:
            return None
        return dict(zip(cols, row))

    def set_info(self, ticker: str, info: dict) -> None:
        now = datetime.utcnow()
        expires = now + timedelta(seconds=_ttl("info"))
        self._conn.execute("""
            INSERT OR REPLACE INTO company_info (
                ticker, name, sector, industry, market_cap, shares_outstanding,
                enterprise_value, dividend_yield, currency, exchange, description,
                website, country, employees, fetched_at, expires_at
            )
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, [
            ticker,
            info.get("name"), info.get("sector"), info.get("industry"),
            info.get("market_cap"), info.get("shares_outstanding"),
            info.get("enterprise_value"), info.get("dividend_yield"),
            info.get("currency", "USD"),
            info.get("exchange"), info.get("description"), info.get("website"),
            info.get("country"), info.get("employees"),
            now, expires,
        ])

    # ── Analyst Estimates ─────────────────────────────────────────────────────
    def get_estimates(self, ticker: str) -> dict | None:
        now = datetime.utcnow()
        row = self._conn.execute(
            "SELECT * FROM analyst_estimates WHERE ticker = ? AND expires_at > ?",
            [ticker, now],
        ).fetchone()
        if row is None:
            return None
        cols = ["ticker","forward_revenue","forward_ebitda","forward_eps","forward_pe",
                "target_mean_price","target_median_price","recommendation","num_analysts",
                "fetched_at","expires_at"]
        return dict(zip(cols, row))

    def set_estimates(self, ticker: str, est: dict) -> None:
        now = datetime.utcnow()
        expires = now + timedelta(seconds=_ttl("estimates"))
        self._conn.execute("""
            INSERT OR REPLACE INTO analyst_estimates VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, [
            ticker,
            est.get("forward_revenue"), est.get("forward_ebitda"), est.get("forward_eps"),
            est.get("forward_pe"), est.get("target_mean_price"), est.get("target_median_price"),
            est.get("recommendation"), est.get("num_analysts"),
            now, expires,
        ])

    def invalidate(self, ticker: str) -> None:
        """Force refresh of all cached data for a ticker."""
        self._conn.execute(
            "DELETE FROM cache_metadata WHERE ticker = ?", [ticker]
        )
        self._conn.execute("DELETE FROM financials WHERE ticker = ?", [ticker])
        self._conn.execute("DELETE FROM company_info WHERE ticker = ?", [ticker])
        self._conn.execute("DELETE FROM analyst_estimates WHERE ticker = ?", [ticker])
        for path in PARQUET_DIR.glob(f"{ticker.upper()}_*.parquet"):
            path.unlink(missing_ok=True)
        logger.info("Cache invalidated for %s", ticker)

    def close(self) -> None:
        self._conn.close()
