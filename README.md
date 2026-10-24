# 💸 Student Spending Behaviour Predictor

A Machine Learning project that predicts **how much a college student will spend per month** and **what kind of spender they are** (Saver / Balanced / Overspender) from their income, lifestyle and digital-payment habits.

---

## 1. Problem Statement
Many college students run out of money before the month ends because of food delivery, online shopping, subscriptions and easy UPI / Buy-Now-Pay-Later payments. This project builds ML models that:

1. **Regression** – predict a student's monthly spending (₹).
2. **Classification** – classify the student as a **Saver**, **Balanced** spender or **Overspender**.
3. Give **personalised money-saving tips** based on the prediction.

## 2. Dataset
`generate_data.py` creates a realistic synthetic dataset of **2,000 Indian college students** (amounts in INR).

| Feature | Description |
|---|---|
| age, gender, year_of_study, course | Student profile |
| city_tier | Tier 1 / Tier 2 / Tier 3 city |
| living_situation | Hostel, PG / Rented, With Family |
| monthly_allowance, part_time_income | Income sources (₹) |
| transport_mode | Walk/Cycle, Public Transport, Own Bike, Cab/Auto |
| food_orders_per_week | Swiggy/Zomato-type orders |
| outings_per_month | Movies, cafés, trips |
| online_shopping_orders | Orders per month |
| subscriptions | OTT, music, gym, etc. |
| social_media_hours | Daily hours |
| upi_txn_per_week | Digital payment frequency |
| tracks_budget, uses_bnpl | Financial habits (Yes/No) |
| **monthly_spending** | 🎯 Regression target (₹) |
| **spending_category** | 🎯 Classification target |

**Labelling rule:** spending ÷ income < 0.80 → *Saver*; 0.80 – 1.05 → *Balanced*; > 1.05 → *Overspender*.

> You can replace the synthetic data with a real survey (e.g., a Google Form filled by classmates) as long as the column names match.

## 3. Methodology
1. **EDA** – distributions, correlation heatmap, spending by group.
2. **Preprocessing** – `StandardScaler` for numeric columns, `OneHotEncoder` for categorical columns, combined in a `ColumnTransformer` + `Pipeline`.
3. **Train/test split** – 80 / 20, stratified.
4. **Models compared** (with 5-fold cross-validation)
   - Regression: Linear Regression, Random Forest, Gradient Boosting
   - Classification: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
5. **Best model** chosen by Test R² (regression) and macro F1 (classification) and saved with `joblib`.

## 4. Results

**Regression (monthly spending)**

| Model | CV R² | Test R² | MAE (₹) | RMSE (₹) |
|---|---|---|---|---|
| **Linear Regression** | 0.935 | **0.939** | 817 | 1014 |
| Random Forest | 0.870 | 0.899 | 1041 | 1303 |
| Gradient Boosting | 0.919 | 0.929 | 859 | 1093 |

**Classification (spending category)**

| Model | CV Accuracy | Test Accuracy | Macro F1 |
|---|---|---|---|
| **Logistic Regression** | 0.759 | **0.775** | **0.770** |
| Decision Tree | 0.633 | 0.668 | 0.667 |
| Random Forest | 0.690 | 0.690 | 0.676 |
| Gradient Boosting | 0.726 | 0.728 | 0.724 |

**Top drivers of spending behaviour:** monthly allowance, part-time income, living in PG/rented housing, food-delivery orders, and living in a Tier 1 city.

All plots are saved in `outputs/`:
`01_target_distribution.png`, `02_correlation_heatmap.png`, `03_spending_by_group.png`, `04_actual_vs_predicted.png`, `05_confusion_matrix.png`, `06_feature_importance.png`.

## 5. Project Structure
```
student_spending_predictor/
├── data/student_spending.csv     # dataset
├── models/                       # saved best models (.pkl)
├── outputs/                      # plots, metrics, reports
├── generate_data.py              # creates the dataset
├── train.py                      # EDA + training + evaluation
├── predict.py                    # command-line prediction + tips
├── app.py                        # Streamlit web app
├── requirements.txt
└── README.md
```

## 6. How to Run
```bash
pip install -r requirements.txt
python generate_data.py        # 1. create dataset
python train.py                # 2. train models, save plots & models
python predict.py              # 3. predict for a sample student
streamlit run app.py           # 4. open the web app in your browser
```

## 7. Conclusion
Simple linear models performed best because spending in this data depends mostly linearly on income and lifestyle habits. The model predicts monthly spending within about **₹800** on average and identifies spending type with **~78% accuracy**. The *Balanced* class is hardest to predict because it sits between the other two.

## 8. Future Scope
- Collect real survey data from students.
- Add monthly time-series data to forecast end-of-month balance.
- Connect to bank/UPI statements for automatic expense categorisation.
- Deploy the app on Streamlit Cloud.
