from datetime import datetime


from datetime import datetime
import math


def snapshot_html_template(snapshot: dict, user_name="Semer") -> str:
    """
    Perfect HTML email for daily snapshot:
    - Personalized greeting
    - Partial data handling
    - Trend arrows
    - Volatility/risk
    - Skip tables if no valid data
    """

    latest = snapshot.get("latest", {})
    previous = snapshot.get("previous", {})

    # Personalized greeting
    greeting = f"Good morning {user_name}! Here's your daily portfolio snapshot. Have a good day!"

    # Extract latest data
    total_value = latest.get("total_value")
    total_cost = latest.get("total_cost")
    total_gain_loss = latest.get("total_gain_loss")
    sector_alloc = {
        k: v
        for k, v in latest.get("sector_allocation", {}).items()
        if k and str(k).lower() != "nan"
    }
    pl_details = latest.get("profit_loss", {})

    # Previous day comparison
    prev_total_value = previous.get("total_value")
    value_change = (
        (total_value - prev_total_value)
        if total_value is not None and prev_total_value is not None
        else None
    )
    value_change_pct = (
        (value_change / prev_total_value * 100)
        if value_change is not None and prev_total_value
        else None
    )
    value_arrow = "▲" if value_change and value_change >= 0 else "▼"

    # Biggest mover
    biggest_mover = max(
        pl_details.items(),
        key=lambda x: (
            abs(x[1]["profit_loss"] / x[1]["cost_basis"] * 100)
            if x[1].get("cost_basis")
            else 0
        ),
        default=(None, None),
    )
    ticker = biggest_mover[0]
    biggest_change = (
        round(biggest_mover[1]["profit_loss"] / biggest_mover[1]["cost_basis"] * 100, 2)
        if biggest_mover[1] and biggest_mover[1].get("cost_basis")
        else 0
    )
    mover_arrow = "▲" if biggest_change >= 0 else "▼"

    # Volatility / risk: only valid tickers
    pl_percentages = [
        (info["profit_loss"] / info["cost_basis"] * 100)
        for info in pl_details.values()
        if info.get("cost_basis") and info.get("current_price") is not None
    ]
    volatility = (
        math.sqrt(
            sum(
                (x - sum(pl_percentages) / len(pl_percentages)) ** 2
                for x in pl_percentages
            )
            / len(pl_percentages)
        )
        if pl_percentages
        else 0
    )

    # Warnings for failed tickers
    warnings = latest.get("warnings", {})

    # Sector allocation rows
    sectors = "".join(
        f"<tr style='background-color:{'#f2f2f2' if i%2==0 else 'white'}'><td>{sector}</td><td>{pct:.2f}%</td></tr>"
        for i, (sector, pct) in enumerate(sector_alloc.items())
    )

    # Per-stock rows
    pl_rows = "".join(
        f"""
        <tr style='background-color:{'#f9f9f9' if i%2==0 else 'white'}'>
            <td>{ticker}</td>
            <td>{info['quantity']}</td>
            <td>{info['current_price']:.2f}</td>
            <td>{info['cost_basis']:.2f}</td>
            <td style='color: {"green" if info["profit_loss"]>=0 else "red"}; font-weight:bold'>
                {info['profit_loss']:.2f} USD {"▲" if info["profit_loss"]>=0 else "▼"}
            </td>
        </tr>
        """
        for i, (ticker, info) in enumerate(pl_details.items())
        if info.get("current_price") is not None and info.get("cost_basis") is not None
    )

    # Determine if we have valid ticker data
    has_data = bool(pl_rows)

    # Warnings HTML
    warnings_html = ""
    if warnings:
        warnings_list = "".join(f"<li>{t}: {msg}</li>" for t, msg in warnings.items())
        warnings_html = f"""
        <div style='color:red; margin-top:10px;'>
            ⚠ Some tickers failed to fetch data:
            <ul>{warnings_list}</ul>
        </div>
        """

    # If no valid data
    if not has_data:
        return f"""
        <html>
            <body style='font-family: Arial; padding: 20px;'>
                <h2>{greeting}</h2>
                <p>📉 Market Data Unavailable</p>
                <p>Live price data could not be retrieved. Usually this happens on weekends, holidays, or if Yahoo Finance failed.</p>
            </body>
        </html>
        """

    # Otherwise full email
    return f"""
    <html>
        <body style="font-family: Arial; padding: 20px;">
            <h2>{greeting}</h2>
            <p><b>Latest Day:</b> {snapshot['date_latest']}</p>
            <p><b>Previous Day:</b> {snapshot['date_previous']}</p>

            <h3>Overview (Latest)</h3>
            <p><b>Total Value:</b> ${total_value:,.2f} {value_arrow if value_change else ""}</p>
            <p><b>Total Cost:</b> ${total_cost:,.2f}</p>
            <p><b>Total Gain/Loss:</b> ${total_gain_loss:,.2f}</p>

            <h3>Change vs Previous Day</h3>
            {f"<p><b>Value Change:</b> <span style='color:{'green' if value_change>=0 else 'red'}'>${value_change:,.2f} ({value_change_pct:.2f}%) {value_arrow}</span></p>" if value_change is not None else ""}

            <h3>Biggest Mover</h3>
            {f"<p style='font-size:1.2em; color:orange;'><b>{ticker}</b> — {biggest_change}% {mover_arrow}</p>" if ticker else "<p>None</p>"}

            <h3>Sector Allocation</h3>
            {f"<table border='1' cellspacing='0' cellpadding='6' style='border-collapse: collapse; width: 50%;'>"
             "<tr style='background-color: #d9d9d9;'><th>Sector</th><th>Allocated %</th></tr>"
             f"{sectors}</table>" if sectors else "<p>No sector allocation data.</p>"}

            <h3>Per-Stock Breakdown</h3>
            {f"<table border='1' cellspacing='0' cellpadding='6' style='border-collapse: collapse; width: 100%;'>"
             "<tr style='background-color: #d9d9d9;'><th>Ticker</th><th>Qty</th><th>Current Price</th><th>Cost Basis</th><th>P/L</th></tr>"
             f"{pl_rows}</table>"}

            <h3>Portfolio Volatility / Risk</h3>
            <p>Standard Deviation of P/L %: {volatility:.2f}%</p>

            {warnings_html}

            <p style="margin-top: 20px; color: gray;">Generated automatically by your portfolio assistant.</p>
        </body>
    </html>
    """
