"""Abstract DataProvider protocol."""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from core.types import (
    AnalystEstimates,
    CompanyInfo,
    Financials,
    PriceHistory,
)


@runtime_checkable
class DataProvider(Protocol):
    """Interface that all equity data providers must satisfy."""

    def get_price_history(self, ticker: str, years: int = 10) -> PriceHistory: ...
    def get_financials(self, ticker: str) -> Financials: ...
    def get_estimates(self, ticker: str) -> Optional[AnalystEstimates]: ...
    def get_info(self, ticker: str) -> CompanyInfo: ...


@runtime_checkable
class MacroProvider(Protocol):
    """Interface for macro-economic data providers."""

    def get_series(self, series_id: str) -> "pd.Series": ...  # noqa: F821
