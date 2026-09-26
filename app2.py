import streamlit as st
import pandas as pd
import numpy as np
import joblib

# LOAD FILES

model = joblib.load("random_forest_model.pkl")
columns = joblib.load("columns.pkl")
name_to_enc = joblib.load("name_to_enc.pkl")

# if you used scaler
try:
    scaler = joblib.load("scaler.pkl")
    use_scaler = True
except:
    use_scaler = False

x_template = pd.DataFrame(columns=columns)

# APP TITLE

st.title("Failure Impact Prediction")

st.write("Enter details to predict impact_min")

# USER INPUTS

mtbf_hours = st.number_input("MTBF Hours", min_value=0.0, step=0.1)
month = st.selectbox("Month", list(range(1, 3)))
hour = st.slider("Hour", 0, 23)
dayofweek = st.selectbox("Day of Week (0=Mon, 6=Sun)", list(range(7)))

failure_category = st.selectbox(
    "Failure Category",
    ["Electrical", "Idle/External", "Logistics/Organization", "Mechanical", "Other"]
)

failure_name = st.text_input("Failure Name")

# =========================
# PREDICTION FUNCTION
# =========================
def predict_impact():
    input_dict = {c: 0 for c in columns}

    input_dict['mtbf_hours'] = mtbf_hours
    input_dict['month'] = month
    input_dict['hour'] = hour
    input_dict['dayofweek'] = dayofweek

    # encode failure_name
    fname = failure_name.strip()

    if fname in name_to_enc:
        input_dict['failure_name_enc'] = name_to_enc[fname]
    else:
        matched = None
        for name in name_to_enc:
            if name.lower() == fname.lower():
                matched = name
                break

        if matched:
            input_dict['failure_name_enc'] = name_to_enc[matched]
        else:
            input_dict['failure_name_enc'] = np.mean(list(name_to_enc.values()))
            st.warning("Unknown failure name, using default encoding")

    # one-hot category
    dummy_col = f'failure_category_{failure_category}'
    if dummy_col in input_dict:
        input_dict[dummy_col] = 1

    # dataframe
    input_df = pd.DataFrame([input_dict])[columns]

    # scaling if used
    if use_scaler:
        input_df[['mtbf_hours']] = scaler.transform(input_df[['mtbf_hours']])

    # prediction
    pred_log = model.predict(input_df)[0]
    pred_actual = np.expm1(pred_log)

    return round(pred_actual, 2)

# =========================
# BUTTON
# =========================
if st.button("Predict"):
    result = predict_impact()
    st.success(f"Predicted impact_min: {result} minutes")