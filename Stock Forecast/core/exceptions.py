"""Custom exception hierarchy for the Stock Analyzer."""


class StockAnalyzerError(Exception):
    """Base exception for all application errors."""


class DataFetchError(StockAnalyzerError):
    """Raised when data cannot be fetched from a provider after retries."""


class DataValidationError(StockAnalyzerError):
    """Raised when fetched data fails quality checks."""


class CacheError(StockAnalyzerError):
    """Raised when the cache layer encounters an unrecoverable error."""


class AnalysisError(StockAnalyzerError):
    """Raised when an analysis module cannot produce any output."""


class ConfigurationError(StockAnalyzerError):
    """Raised for missing or invalid configuration (e.g., missing API key)."""
