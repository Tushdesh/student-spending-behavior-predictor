"""
generate_data.py
----------------
Creates a realistic synthetic dataset of Indian college students' monthly
spending behaviour (amounts in INR) and saves it to data/student_spending.csv.

Run:  python generate_data.py --rows 2000
"""
import argparse
import os

import numpy as np
import pandas as pd


def generate(n_rows: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 25, n_rows)
    gender = rng.choice(["Male", "Female", "Other"], n_rows, p=[0.52, 0.45, 0.03])
    year_of_study = np.clip(age - 17 + rng.integers(-1, 1, n_rows), 1, 4)
    course = rng.choice(
        ["Engineering", "Commerce", "Arts", "Science", "Management", "Medical"],
        n_rows, p=[0.32, 0.18, 0.12, 0.16, 0.12, 0.10],
    )
    city_tier = rng.choice(["Tier 1", "Tier 2", "Tier 3"], n_rows, p=[0.40, 0.38, 0.22])
    living = rng.choice(["Hostel", "PG / Rented", "With Family"], n_rows, p=[0.35, 0.30, 0.35])

    tier_mult = pd.Series(city_tier).map({"Tier 1": 1.35, "Tier 2": 1.0, "Tier 3": 0.75}).values
    living_mult = pd.Series(living).map({"Hostel": 1.0, "PG / Rented": 1.25, "With Family": 0.7}).values

    monthly_allowance = np.round(
        rng.lognormal(mean=np.log(8000), sigma=0.45, size=n_rows) * tier_mult * living_mult, -2
    ).clip(2000, 50000)

    has_part_time_job = rng.random(n_rows) < (0.10 + 0.06 * (year_of_study - 1))
    part_time_income = np.where(
        has_part_time_job, np.round(rng.normal(6000, 2500, n_rows), -2).clip(1500, 20000), 0
    )

    transport = rng.choice(["Walk / Cycle", "Public Transport", "Own Bike", "Cab / Auto"],
                           n_rows, p=[0.25, 0.35, 0.28, 0.12])
    food_orders_per_week = rng.poisson(2.5 * tier_mult, n_rows).clip(0, 15)
    outings_per_month = rng.poisson(3, n_rows).clip(0, 12)
    online_shopping_orders = rng.poisson(2, n_rows).clip(0, 10)
    subscriptions = rng.integers(0, 6, n_rows)          # OTT, Spotify, gym, etc.
    social_media_hours = np.round(rng.gamma(2.2, 1.4, n_rows), 1).clip(0.2, 10)
    tracks_budget = rng.random(n_rows) < 0.35
    uses_bnpl = rng.random(n_rows) < 0.18                # Buy-Now-Pay-Later / credit
    upi_txn_per_week = rng.poisson(12, n_rows).clip(0, 50)

    # ---- spending model (ground truth) --------------------------------------
    base_living = np.select(
        [living == "Hostel", living == "PG / Rented", living == "With Family"],
        [2500, 5500, 800],
    ) * tier_mult
    transport_cost = pd.Series(transport).map(
        {"Walk / Cycle": 150, "Public Transport": 700, "Own Bike": 1500, "Cab / Auto": 2800}
    ).values
    food = food_orders_per_week * 4.3 * 260 * tier_mult
    outings = outings_per_month * 450 * tier_mult
    shopping = online_shopping_orders * rng.normal(900, 300, n_rows).clip(200)
    subs = subscriptions * 220
    impulse = social_media_hours * 180 + upi_txn_per_week * 25

    total_income = monthly_allowance + part_time_income
    lifestyle_factor = 1 + 0.18 * uses_bnpl - 0.15 * tracks_budget

    lifestyle_cost = base_living + transport_cost + food + outings + shopping + subs + impulse
    # Students with more money spend more (income effect) + lifestyle-driven costs
    monthly_spending = 0.45 * total_income + 0.40 * lifestyle_cost * lifestyle_factor
    monthly_spending += rng.normal(0, 900, n_rows)        # noise
    monthly_spending = np.round(monthly_spending.clip(1000), -1)

    ratio = monthly_spending / total_income
    category = np.select([ratio < 0.80, ratio <= 1.05], ["Saver", "Balanced"], "Overspender")

    return pd.DataFrame({
        "age": age,
        "gender": gender,
        "year_of_study": year_of_study,
        "course": course,
        "city_tier": city_tier,
        "living_situation": living,
        "monthly_allowance": monthly_allowance.astype(int),
        "part_time_income": part_time_income.astype(int),
        "transport_mode": transport,
        "food_orders_per_week": food_orders_per_week,
        "outings_per_month": outings_per_month,
        "online_shopping_orders": online_shopping_orders,
        "subscriptions": subscriptions,
        "social_media_hours": social_media_hours,
        "tracks_budget": np.where(tracks_budget, "Yes", "No"),
        "uses_bnpl": np.where(uses_bnpl, "Yes", "No"),
        "upi_txn_per_week": upi_txn_per_week,
        "monthly_spending": monthly_spending.astype(int),
        "spending_category": category,
    })


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs("data", exist_ok=True)
    df = generate(args.rows, args.seed)
    df.to_csv("data/student_spending.csv", index=False)
    print(f"Saved {len(df)} rows -> data/student_spending.csv")
    print(df["spending_category"].value_counts())
