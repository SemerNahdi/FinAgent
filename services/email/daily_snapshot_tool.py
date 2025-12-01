# services/email/daily_snapshot_tool.py

from services.tools.portfolio_tool import PortfolioTool
from services.email.html_templates import snapshot_html_template
from datetime import datetime, timedelta


class DailySnapshotTool:
    """
    Builds daily HTML snapshot reports with:
    - Weekend/holiday handling
    - Fallback to previous trading days when data missing
    - Graceful handling of partial or failed yfinance downloads
    """

    def __init__(self, portfolio_csv: str, metadata_csv: str):
        self.portfolio_tool = PortfolioTool(portfolio_csv, metadata_csv)

    # --------------------------------------------
    # Helper: roll back to last trading day (skip weekends)
    # --------------------------------------------
    def _last_trading_day(self, date: datetime) -> datetime:
        while date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            date -= timedelta(days=1)
        return date

    # --------------------------------------------
    # Helper: validate an analysis object returned by portfolio_tool
    # --------------------------------------------
    def _valid_analysis(self, analysis) -> bool:
        return (
            isinstance(analysis, dict)
            and isinstance(analysis.get("profit_loss"), dict)
            and len(analysis.get("profit_loss")) > 0
        )

    # --------------------------------------------
    # Main build function
    # --------------------------------------------
    def build_snapshot(self) -> str:
        today = datetime.today()

        # get last valid trading days
        latest_day = self._last_trading_day(today)
        prev_day = self._last_trading_day(latest_day - timedelta(days=1))

        date_latest = latest_day.strftime("%Y-%m-%d")
        date_prev = prev_day.strftime("%Y-%m-%d")

        # fetch latest and previous analyses
        analysis_latest = self.portfolio_tool.analyze(include_changes=date_latest)
        analysis_prev = self.portfolio_tool.analyze(include_changes=date_prev)

        # -----------------------------------------
        # If "today" data missing, fallback automatically
        # -----------------------------------------
        if not self._valid_analysis(analysis_latest):
            # fallback to yesterday
            latest_day = prev_day
            prev_day = self._last_trading_day(prev_day - timedelta(days=1))

            date_latest = latest_day.strftime("%Y-%m-%d")
            date_prev = prev_day.strftime("%Y-%m-%d")

            analysis_latest = self.portfolio_tool.analyze(include_changes=date_latest)
            analysis_prev = self.portfolio_tool.analyze(include_changes=date_prev)

        # -----------------------------------------
        # If still missing → market closed or yfinance failed
        # -----------------------------------------
        if not self._valid_analysis(analysis_latest) or not self._valid_analysis(
            analysis_prev
        ):
            return """
            <div style='font-family: Arial; padding: 20px;'>
                <h2>📉 Market Data Unavailable</h2>
                <p>
                    Live price data could not be retrieved. This usually happens when:
                </p>
                <ul>
                    <li>The market was closed (weekend or holiday)</li>
                    <li>Data was not yet published</li>
                    <li>Some tickers did not return valid results</li>
                </ul>
                <p>I’ll try again next cycle.</p>
            </div>
            """

        # -----------------------------------------
        # Merge analyses into a combined snapshot dictionary
        # (Your HTML template expects a single dict)
        # -----------------------------------------
        final_analysis = {
            "date_latest": date_latest,
            "date_previous": date_prev,
            "latest": analysis_latest,
            "previous": analysis_prev,
        }

        return snapshot_html_template(final_analysis)
