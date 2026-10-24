"""
app.py
------
Interactive web app for the Student Spending Behaviour Predictor.

Run:  streamlit run app.py
"""
import streamlit as st

from predict import predict_student, tips

st.set_page_config(page_title="Student Spending Predictor", page_icon="💸", layout="centered")
st.title("💸 Student Spending Behaviour Predictor")
st.caption("Enter your details to predict your monthly spending and spending type.")

with st.form("student_form"):
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age", 18, 24, 20)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        year = st.selectbox("Year of study", [1, 2, 3, 4], index=1)
        course = st.selectbox("Course", ["Engineering", "Commerce", "Arts",
                                         "Science", "Management", "Medical"])
        city = st.selectbox("City tier", ["Tier 1", "Tier 2", "Tier 3"])
        living = st.selectbox("Living situation", ["Hostel", "PG / Rented", "With Family"])
        allowance = st.number_input("Monthly allowance (₹)", 1000, 60000, 10000, step=500)
        part_time = st.number_input("Part-time income (₹)", 0, 30000, 0, step=500)
    with c2:
        transport = st.selectbox("Transport", ["Walk / Cycle", "Public Transport",
                                               "Own Bike", "Cab / Auto"])
        food = st.slider("Food delivery orders / week", 0, 15, 3)
        outings = st.slider("Outings / month", 0, 12, 3)
        shopping = st.slider("Online shopping orders / month", 0, 10, 2)
        subs = st.slider("Paid subscriptions", 0, 5, 2)
        social = st.slider("Social media hours / day", 0.0, 10.0, 3.0, 0.5)
        upi = st.slider("UPI transactions / week", 0, 50, 12)
        budget = st.radio("Do you track your budget?", ["Yes", "No"], horizontal=True)
        bnpl = st.radio("Use Buy-Now-Pay-Later / credit?", ["Yes", "No"], index=1,
                        horizontal=True)
    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    student = {
        "age": age, "gender": gender, "year_of_study": year, "course": course,
        "city_tier": city, "living_situation": living,
        "monthly_allowance": allowance, "part_time_income": part_time,
        "transport_mode": transport, "food_orders_per_week": food,
        "outings_per_month": outings, "online_shopping_orders": shopping,
        "subscriptions": subs, "social_media_hours": social,
        "tracks_budget": budget, "uses_bnpl": bnpl, "upi_txn_per_week": upi,
    }
    r = predict_student(student)
    colour = {"Saver": "green", "Balanced": "orange", "Overspender": "red"}[r["category"]]

    m1, m2, m3 = st.columns(3)
    m1.metric("Income", f"₹{r['total_income']:,}")
    m2.metric("Predicted spend", f"₹{r['predicted_spending']:,}")
    m3.metric("Expected savings", f"₹{r['expected_savings']:,}")
    st.markdown(f"### You are likely a :{colour}[**{r['category']}**]")
    st.bar_chart(r["probabilities"])
    st.subheader("Tips")
    for t in tips(student, r):
        st.write("•", t)
