import pandas as pd
import numpy as np

np.random.seed(42)

# --------------------------------------------------
# Date range
# --------------------------------------------------
dates = pd.date_range(
    start="1998-01-01",
    end=(pd.Timestamp.today() - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    freq="D"
)

N = len(dates)
df_dates = pd.DataFrame({"date": dates})
df_dates["year"] = df_dates["date"].dt.year

# --------------------------------------------------
# Cashflow drivers (unchanged)
# --------------------------------------------------
cash_inflow = np.random.uniform(10_000, 500_000, N)
cash_outflow = np.random.uniform(8_000, 480_000, N)
net_cashflow = cash_inflow - cash_outflow

# --------------------------------------------------
# 🔥 YEARLY CASH CONTINUITY LOGIC
# --------------------------------------------------
opening_cash = np.zeros(N)
closing_cash = np.zeros(N)

for year in df_dates["year"].unique():
    idx = df_dates[df_dates["year"] == year].index

    # Opening balance for Jan 1 of the year
    if year == df_dates["year"].min():
        opening_cash[idx[0]] = np.random.uniform(500_000, 5_000_000)
    else:
        opening_cash[idx[0]] = closing_cash[idx[0] - 1]  # Dec → Jan carry forward

    # Daily propagation
    for i in idx:
        if i != idx[0]:
            opening_cash[i] = closing_cash[i - 1]

        closing_cash[i] = opening_cash[i] + net_cashflow[i]

# --------------------------------------------------
# Cash source
# --------------------------------------------------
cash_source = np.random.choice(
    ["customer_payment", "loan_disbursement", "investment", "asset_sale"],
    N,
    p=[0.7, 0.15, 0.1, 0.05]
)

# --------------------------------------------------
# Invoice behavior
# --------------------------------------------------
payment_behavior = np.random.choice(
    ["full", "partial", "unpaid"],
    N,
    p=[0.65, 0.25, 0.10]
)

invoice_amount = np.random.uniform(5_000, 300_000, N)
invoice_paid = np.zeros(N)

full_mask = payment_behavior == "full"
invoice_paid[full_mask] = invoice_amount[full_mask]

partial_mask = payment_behavior == "partial"
invoice_paid[partial_mask] = (
    invoice_amount[partial_mask] *
    np.random.uniform(0.3, 0.9, partial_mask.sum())
)

unpaid_mask = payment_behavior == "unpaid"
invoice_paid[unpaid_mask] = 0

partial_payment_flag = payment_behavior == "partial"
bad_debt_flag = payment_behavior == "unpaid"

# --------------------------------------------------
# Payment timing
# --------------------------------------------------
days_payment_delay = np.random.randint(-5, 60, N)
invoice_due_date = dates
invoice_payment_date = dates + pd.to_timedelta(days_payment_delay, unit="D")

customer_payment_usd = invoice_paid

# --------------------------------------------------
# Expenses
# --------------------------------------------------
vendor_payment = np.random.uniform(5_000, 200_000, N)
salary_payment = np.random.uniform(20_000, 150_000, N)
rent = np.random.uniform(5_000, 50_000, N)
tax_payment = np.random.uniform(3_000, 100_000, N)
loan_emi = np.random.uniform(0, 80_000, N)
operational_expense = np.random.uniform(2_000, 120_000, N)

ppe_expense = np.where(
    np.random.rand(N) < 0.15,
    np.random.uniform(20_000, 1_000_000, N),
    0
)

expense_source = np.random.choice(
    ["operations", "payroll", "rent", "tax", "loan", "ppe_capex"],
    N,
    p=[0.35, 0.25, 0.1, 0.1, 0.1, 0.1]
)

# --------------------------------------------------
# Time features
# --------------------------------------------------
day_of_week = dates.dayofweek + 1
week_of_month = (dates.day - 1) // 7 + 1
month = dates.month
quarter = dates.quarter
is_month_end = dates.is_month_end
is_holiday = np.random.rand(N) < 0.1

# --------------------------------------------------
# External indicators
# --------------------------------------------------
interest_rate = np.random.uniform(2.0, 8.0, N)
inflation = np.random.uniform(1.0, 6.0, N)
fx_index = np.random.uniform(90, 120, N)
economic_sentiment = np.random.uniform(-1, 1, N)

# --------------------------------------------------
# Lag & rolling
# --------------------------------------------------
cashflow_lag_1d = np.roll(net_cashflow, 1)
cashflow_lag_7d = np.roll(net_cashflow, 7)
cashflow_lag_30d = np.roll(net_cashflow, 30)

rolling_avg_7d = pd.Series(net_cashflow).rolling(7).mean().fillna(0).values
rolling_std_30d = pd.Series(net_cashflow).rolling(30).std().fillna(0).values

# --------------------------------------------------
# Agentic fields
# --------------------------------------------------
forecast_version = np.random.choice(["v1", "v2", "v3"], N)
reforecast_trigger = np.random.choice(
    ["delay_detected", "expense_spike", "none"],
    N,
    p=[0.2, 0.1, 0.7]
)

confidence_score = np.random.uniform(0.7, 0.99, N)
alert_flag = closing_cash < 100_000
recommended_action = np.where(alert_flag, "short_term_borrowing", "no_action")

# --------------------------------------------------
# DataFrame
# --------------------------------------------------
df = pd.DataFrame({
    "date": dates,
    "opening_cash_usd": opening_cash,
    "cash_inflow_usd": cash_inflow,
    "cash_outflow_usd": cash_outflow,
    "net_cashflow_usd": net_cashflow,
    "closing_cash_usd": closing_cash,

    "cash_source": cash_source,
    "expense_source": expense_source,
    "ppe_expense_usd": ppe_expense,

    "customer_payment_usd": customer_payment_usd,
    "invoice_amount_usd": invoice_amount,
    "invoice_paid_usd": invoice_paid,
    "invoice_due_date": invoice_due_date,
    "invoice_payment_date": invoice_payment_date,
    "days_payment_delay": days_payment_delay,
    "partial_payment_flag": partial_payment_flag,
    "bad_debt_flag": bad_debt_flag,

    "vendor_payment_usd": vendor_payment,
    "salary_payment_usd": salary_payment,
    "rent_usd": rent,
    "tax_payment_usd": tax_payment,
    "loan_emi_usd": loan_emi,
    "operational_expense_usd": operational_expense,

    "day_of_week": day_of_week,
    "week_of_month": week_of_month,
    "month": month,
    "quarter": quarter,
    "is_month_end": is_month_end,
    "is_holiday": is_holiday,

    "interest_rate_pct": interest_rate,
    "inflation_pct": inflation,
    "fx_rate_usd_index": fx_index,
    "economic_sentiment_score": economic_sentiment,

    "cashflow_lag_1d_usd": cashflow_lag_1d,
    "cashflow_lag_7d_usd": cashflow_lag_7d,
    "cashflow_lag_30d_usd": cashflow_lag_30d,
    "rolling_avg_7d_usd": rolling_avg_7d,
    "rolling_std_30d_usd": rolling_std_30d,

    "forecast_version": forecast_version,
    "reforecast_trigger": reforecast_trigger,
    "confidence_score": confidence_score,
    "alert_flag": alert_flag,
    "recommended_action": recommended_action
})

# --------------------------------------------------
# Save
# --------------------------------------------------
df.to_csv("cashflow_prediction_1998_2025_v1.csv", index=False)

print("✅ Dataset generated with yearly cash continuity")
print("Date range:", df["date"].min(), "→", df["date"].max())

