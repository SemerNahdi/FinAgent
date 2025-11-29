# services/email/html_templates.py


def snapshot_html_template(snapshot: dict) -> str:
    """
    Builds a clean HTML email for the daily snapshot with tables and color coding.
    """

    total_value = snapshot["total_value"]
    total_cost = snapshot["total_cost"]
    total_gain_loss = snapshot["total_gain_loss"]
    sector_alloc = {
        k: v
        for k, v in snapshot["sector_allocation"].items()
        if k and str(k).lower() != "nan"
    }
    pl_details = snapshot["profit_loss_details"]

    # Find biggest mover
    biggest_mover = max(
        pl_details.items(),
        key=lambda x: abs(x[1]["profit_loss"] / x[1]["cost_basis"] * 100),
        default=(None, None),
    )
    biggest_ticker = biggest_mover[0]
    biggest_change = (
        round(biggest_mover[1]["profit_loss"] / biggest_mover[1]["cost_basis"] * 100, 2)
        if biggest_mover[1]
        else 0
    )

    # Sector allocation rows
    sectors = "".join(
        f"<tr style='background-color:{'#f2f2f2' if i%2==0 else 'white'}'><td>{sector}</td><td>{pct:.2f}%</td></tr>"
        for i, (sector, pct) in enumerate(sector_alloc.items())
    )

    # Individual stock P/L rows with color coding
    pl_rows = "".join(
        f"""
        <tr style='background-color:{'#f9f9f9' if i%2==0 else 'white'}'>
            <td>{ticker}</td>
            <td>{info['quantity']}</td>
            <td>{info['current_price']:.2f}</td>
            <td>{info['cost_basis']:.2f}</td>
            <td style='color: {"green" if info["profit_loss"]>=0 else "red"}; font-weight:bold'>
                {info['profit_loss']:.2f} USD
            </td>
        </tr>
        """
        for i, (ticker, info) in enumerate(pl_details.items())
    )

    return f"""
    <html>
        <body style="font-family: Arial; padding: 20px;">
            <h2>📊 Daily Portfolio Snapshot</h2>

            <h3>Overview</h3>
            <p><b>Total Value:</b> ${total_value:,.2f}</p>
            <p><b>Total Cost:</b> ${total_cost:,.2f}</p>
            <p><b>Total Gain/Loss:</b> ${total_gain_loss:,.2f}</p>

            <h3>Biggest Mover</h3>
            <p style='font-size:1.2em; color:orange;'><b>{biggest_ticker}</b> — {biggest_change}%</p>

            <h3>Sector Allocation</h3>
            <table border="1" cellspacing="0" cellpadding="6" style="border-collapse: collapse; width: 50%;">
                <tr style="background-color: #d9d9d9;"><th>Sector</th><th>Allocated %</th></tr>
                {sectors}
            </table>

            <h3>Per-Stock Breakdown</h3>
            <table border="1" cellspacing="0" cellpadding="6" style="border-collapse: collapse; width: 100%;">
                <tr style="background-color: #d9d9d9;">
                    <th>Ticker</th>
                    <th>Qty</th>
                    <th>Current Price</th>
                    <th>Cost Basis</th>
                    <th>P/L</th>
                </tr>
                {pl_rows}
            </table>

            <p style="margin-top: 20px; color: gray;">Generated automatically by your portfolio assistant.</p>
        </body>
    </html>
    """
