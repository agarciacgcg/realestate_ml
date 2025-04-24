
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Title
st.title("Johnson County Home Price Predictor")

# 1. Load the trained pipeline
@st.cache_resource
def load_model(path='joco_rf_pipeline.joblib'):
    return joblib.load(path)

model = load_model()

# 2. Sidebar inputs for features
st.sidebar.header("Input Features")
bed = st.sidebar.slider('Bedrooms', min_value=1, max_value=6, value=3)
bath = st.sidebar.slider('Bathrooms', min_value=1, max_value=5, value=2)
acre_lot = st.sidebar.number_input('Lot size (acres)', min_value=0.0, format="%.2f", value=0.2)
house_size = st.sidebar.number_input('House size (sq ft)', min_value=300, max_value=10000, value=1500)

# Derived feature
log_house_siz = np.log(house_size)

population = st.sidebar.number_input('Population', min_value=1000, max_value=200000, value=50000)
median_income = st.sidebar.number_input('Median income', min_value=20000, max_value=200000, value=80000)
pct_bachelor = st.sidebar.slider('Pct. Bachelor+', min_value=0.0, max_value=100.0, value=40.0)
num_schools = st.sidebar.slider('Num. Public Schools', min_value=0, max_value=50, value=10)

# Categorical inputs
City = st.sidebar.selectbox('City', options=['Overland Park', 'Olathe', 'Shawnee', 'Leawood', 'Lenexa'])
Metro = st.sidebar.selectbox('Metro Area', options=['Kansas City'])

# Interaction feature
size_income = house_size * median_income

# 3. Build DataFrame for prediction
features = {
    'bed': bed,
    'bath': bath,
    'acre_lot': acre_lot,
    'house_size': house_size,
    'log_house_size': log_house_siz,
    'population': population,
    'median_income': median_income,
    'pct_bachelor_plus': pct_bachelor,
    'num_public_schools': num_schools,
    'City': City,
    'Metro': Metro,
    'size_income': size_income
}
input_df = pd.DataFrame([features])
# 3.a Ensure all expected features are in the DataFrame
#    model.named_steps['preprocess'] is your ColumnTransformer
expected = model.named_steps['preprocess'].feature_names_in_

for col in expected:
    if col not in input_df.columns:
        input_df[col] = 0   # or fill with a median/default

# 3.b Reorder to exactly match training order
input_df = input_df[expected]

# 4. Predict
if st.sidebar.button('Predict Price'):
    pred_log = model.predict(input_df)[0]
    pred_price = np.exp(pred_log)
    st.metric(label="Predicted Price (USD)", value=f"${pred_price:,.2f}")

# 5. Optionally show raw input data
with st.expander("Show input data"):
    st.write(input_df)

