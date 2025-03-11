import yfinance as yf
import pandas as pd
import json

# Select company ticker (Change as needed)
ticker = "AAPL"  # Example: Apple Inc.
stock = yf.Ticker(ticker)

# Fetch financial statements
income_stmt = stock.financials.T  # Transpose for readability
balance_sheet = stock.balance_sheet.T
cash_flow = stock.cashflow.T

# Save as CSV
income_stmt.to_csv(f"{ticker}_income_statement.csv")
balance_sheet.to_csv(f"{ticker}_balance_sheet.csv")
cash_flow.to_csv(f"{ticker}_cash_flow.csv")

# Display preview
print("Income Statement:\n", income_stmt.head())
print("\nBalance Sheet:\n", balance_sheet.head())
print("\nCash Flow Statement:\n", cash_flow.head())

# ===========================
# 🔹 Preprocessing for RAG Model
# ===========================

# ✅ Remove Null/NaN Values
income_stmt.dropna(inplace=True)
balance_sheet.dropna(inplace=True)
cash_flow.dropna(inplace=True)

# ✅ Convert Data Types (Ensure numerical values are floats)
income_stmt = income_stmt.apply(pd.to_numeric, errors="coerce")
balance_sheet = balance_sheet.apply(pd.to_numeric, errors="coerce")
cash_flow = cash_flow.apply(pd.to_numeric, errors="coerce")

# ✅ Rename Columns for Clarity
income_stmt.rename(columns={"totalRevenue": "Revenue", "netIncome": "Net Income"}, inplace=True)
balance_sheet.rename(columns={"totalAssets": "Total Assets", "totalLiabilities": "Total Liabilities"}, inplace=True)
cash_flow.rename(columns={"operatingCashflow": "Operating Cash Flow"}, inplace=True)

# ✅ Standardize Date Format & Convert Index to String (Fix JSON Serialization Error)
income_stmt.index = income_stmt.index.astype(str)
balance_sheet.index = balance_sheet.index.astype(str)
cash_flow.index = cash_flow.index.astype(str)

# ✅ Save Cleaned Data
income_stmt.to_csv("cleaned_income_statement.csv")
balance_sheet.to_csv("cleaned_balance_sheet.csv")
cash_flow.to_csv("cleaned_cash_flow.csv")

# ===========================
# 🔹 Structure Data for Retrieval (RAG Model)
# ===========================

# ✅ Create structured JSON format
structured_data = {
    "ticker": ticker,
    "income_statement": income_stmt.to_dict(),
    "balance_sheet": balance_sheet.to_dict(),
    "cash_flow": cash_flow.to_dict()
}

# ✅ Save as JSON
with open("financial_data.json", "w") as f:
    json.dump(structured_data, f, indent=4)

print("\nFinancial data saved as JSON for retrieval!")
