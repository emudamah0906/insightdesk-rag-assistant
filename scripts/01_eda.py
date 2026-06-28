"""
Exploratory Data Analysis on support tickets.

Demonstrates the data-science muscle a consultant uses in discovery: understand the
data before modelling. Class balance drives metric choice; text length
informs chunking/token budgets.

Run:  python scripts/01_eda.py
"""
from __future__ import annotations
import pathlib
import pandas as pd

DATA = pathlib.Path(__file__).resolve().parent.parent / "data" / "tickets.csv"


def main():
    df = pd.read_csv(DATA)
    print(f"Rows: {len(df)}   Columns: {list(df.columns)}\n")

    print("== Intent distribution (class balance) ==")
    dist = df["intent"].value_counts()
    print(dist.to_string())
    print(f"Imbalance ratio (max/min): {dist.max() / dist.min():.1f}x  "
          f"-> favor macro-F1 over accuracy\n")

    print("== Priority distribution ==")
    print(df["priority"].value_counts().to_string(), "\n")

    print("== Priority by intent (crosstab) ==")
    print(pd.crosstab(df["intent"], df["priority"]).to_string(), "\n")

    df["n_words"] = df["text"].str.split().str.len()
    print("== Ticket length (words) ==")
    print(df["n_words"].describe().round(1).to_string())
    print(f"\nLongest ticket ({df['n_words'].max()} words): "
          f"{df.loc[df['n_words'].idxmax(), 'text']!r}")

    # a tiny business KPI: share of high-priority tickets per intent (where to staff)
    print("\n== High-priority share by intent (staffing signal) ==")
    hp = (df.assign(high=df["priority"].eq("high"))
            .groupby("intent")["high"].mean().sort_values(ascending=False))
    print((hp * 100).round(0).astype(int).astype(str).add("%").to_string())

    print("\nConsulting takeaway: 'billing' and 'complaint' skew high-priority -> route "
          "those to senior agents; automate low-priority 'account'/'shipping' FAQs with RAG.")


if __name__ == "__main__":
    main()
