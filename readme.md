# Machine Failure Impact Prediction

## Problem Statement
Machine failures cause unexpected downtime in industries, leading to productivity loss. 
This project predicts the impact time (in minutes) of failures using machine learning.

---

## Dataset Overview
The dataset includes:
- failure_name
- failure_category
- mtbf_hours
- time-based features (month, hour, dayofweek)

---

## Approach
- Performed data cleaning and feature engineering
- Encoded categorical variables
- Applied log transformation on target variable
- Trained multiple models

---

## Model Used
Final Model: Random Forest Regressor

Reason:
- Reduced overfitting
- Stable performance on training and testing data

---

## Performance
- Training R²: 0.93  
- Testing R²: 0.936  
- MAE: ~7.7  
- RMSE: ~28.5  

---

## Deployment
The model is deployed using Streamlit.

To run locally:
```bash
pip install -r requirements.txt
streamlit run app.py