"""
predict.py
----------
Predict a single student's monthly spending and spending category from the
command line, using the models saved by train.py.

Run:  python predict.py
      (edit the `student` dictionary below to try different profiles)
"""
import joblib
import pandas as pd

reg = joblib.load("models/spending_regressor.pkl")
clf = joblib.load("models/spending_classifier.pkl")


def predict_student(student: dict) -> dict:
    X = pd.DataFrame([student])
    spending = float(reg.predict(X)[0])
    category = clf.predict(X)[0]
    probs = {c: round(float(p), 3) for c, p in zip(clf.classes_, clf.predict_proba(X)[0])}
    income = student["monthly_allowance"] + student["part_time_income"]
    return {
        "predicted_spending": round(spending),
        "total_income": income,
        "expected_savings": round(income - spending),
        "category": category,
        "probabilities": probs,
    }


def tips(student: dict, result: dict) -> list:
    t = []
    if result["category"] == "Overspender":
        t.append("You are likely to spend more than you earn — set a weekly cap.")
    if student["food_orders_per_week"] >= 4:
        t.append("Cut food-delivery orders by half to save roughly ₹1,500–2,500/month.")
    if student["subscriptions"] >= 3:
        t.append("Share or cancel unused OTT/music subscriptions.")
    if student["uses_bnpl"] == "Yes":
        t.append("Avoid Buy-Now-Pay-Later for non-essentials; it inflates spending.")
    if student["tracks_budget"] == "No":
        t.append("Start tracking expenses (e.g., a simple spreadsheet or app).")
    return t or ["Good habits — keep it up and consider a small monthly SIP/RD."]


if __name__ == "__main__":
    student = {
        "age": 20, "gender": "Male", "year_of_study": 2, "course": "Engineering",
        "city_tier": "Tier 1", "living_situation": "PG / Rented",
        "monthly_allowance": 12000, "part_time_income": 0,
        "transport_mode": "Own Bike", "food_orders_per_week": 5,
        "outings_per_month": 4, "online_shopping_orders": 3, "subscriptions": 3,
        "social_media_hours": 4.5, "tracks_budget": "No", "uses_bnpl": "Yes",
        "upi_txn_per_week": 15,
    }
    r = predict_student(student)
    print("\n===== Student Spending Prediction =====")
    print(f"Total monthly income : ₹{r['total_income']:,}")
    print(f"Predicted spending   : ₹{r['predicted_spending']:,}")
    print(f"Expected savings     : ₹{r['expected_savings']:,}")
    print(f"Spending category    : {r['category']}")
    print(f"Probabilities        : {r['probabilities']}")
    print("\nTips:")
    for tip in tips(student, r):
        print(" -", tip)
