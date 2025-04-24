
import streamlit as st
import joblib
import numpy as np
import pandas as pd


# --- Page Config ---
st.set_page_config(
    page_title="Odyyn's Insights - JoCo Home Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# --- CSS for styling ---
# --- Custom CSS Styling ---
odyyn_primary_color = "#c10604"
odyyn_secondary_color = "#04c9c8"
custom_css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Ubbuntu:wght@400;500;700&display=swap');

html, body, [class*="css"]  {{
    font-family: 'Ubuntu', ubuntu, sans-serif;
    color: #333;
}}

h1 {{
    color: #800000;
}}

h2, h3 {{
    color: {odyyn_secondary_color};
}}

.stButton button {{
    background-color: {odyyn_primary_color};
    color: white;
    border-radius: 5px;
}}

.stButton button:hover {{
    background-color: {odyyn_secondary_color};
    color: black;
    border-radius: 5px;
}}

.stSidebar {{
    background-color:  #fdf5e6;
}}

.stnumber_input input {{
    background-color: #fdf5e6;
    color: #333;
    border-radius: 5px;
}}

stNumber_input:hover {{
    color: {odyyn_primary_color};
}}

st.slider {{
    color: {odyyn_primary_color};
}}

footer {{
    visibility: hidden;
}}
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# --- Logo & Branded Header ---
st.image("/Users/alexandrogarcia/Documents/Real-World Case Studies/Real_estate_prototype/odyyn copy.png", width=400,  )  # replace with your logo URL or local path

# Title
st.title("Johnson County Home Price Predictor")

# 1. Load the trained pipeline
@st.cache_resource
def load_model(path='joco_rf_pipeline.joblib'):
    return joblib.load(path)

model = load_model()

# 2. Sidebar inputs for features
st.sidebar.image("/Users/alexandrogarcia/Documents/Real-World Case Studies/Real_estate_prototype/odyyn copy.png", width=200,  )  # replace with your logo URL or local path
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

st.title("How it works")
st.text(
    """
This predictive model is powered by machine learning AI, a data-driven 
technique that identifies hidden relationships in large datasets.
Here's the simplified process we follow to build these models:

1. Data Collection & Cleaning:
   • We gather relevant data points—real estate listings, local demographics,
     economic indicators, and market trends.
   • We clean and enrich data (removing outliers, filling gaps, 
     standardizing formats) for reliable results.

2. Feature Engineering:
   • We create meaningful predictive variables—like price per sq. ft.,
     demographic income levels, or recent market growth rates—to help
     the model recognize nuanced patterns.

3. Model Training & Validation:
   • We train machine-learning algorithms to predict home prices based on historical data.
   • We rigorously validate models using cross-validation and unseen 
     hold-out sets to ensure accurate and generalizable predictions.

4. Deployment & Monitoring:
   • Models are packaged into intuitive dashboards and APIs, allowing 
     non-technical users to interact easily.
   • Predictions are continuously monitored, and models updated 
     regularly with new data to maintain accuracy.

Applications beyond Real Estate
-------------------------------
• Construction & Project Management:
   - Predict costs and timelines based on historical project data,
     materials pricing trends, labor market fluctuations, and 
     regional permits.
   - Scenario analysis helps contractors adjust budgets and 
     schedules proactively, significantly reducing project risks 
     and budget overruns.

• Digital Marketing & Customer Insights:
   - Analyze customer behavior and market sentiment to forecast 
     campaign success.
   - Predict customer lifetime value, optimize marketing spend, 
     and tailor advertising content based on granular audience 
     segmentation.

This model illustrates the transformative potential of machine learning 
across industries—delivering rapid, reliable insights to elevate 
decision-making, reduce risk, and optimize business outcomes."""
)


st.title("Why this matters")

st.text("""
• Faster, data-driven valuations – machine-learning models like the one behind
  this demo learn from thousands of Kansas-City-area deals and 50+ local
  features (size, income, education, market trends), letting you surface an
  evidence-based price in seconds rather than days.

• Accuracy that compounds – recent industry studies show up to 73 % of CRE
  firms already apply ML for decision-making, and more than 80 % plan to
  boost ML budgets in the next 2-3 years. Early adopters report valuation
  error reductions of 20-40 %.

• Smarter risk management – by fusing census, permit and demographic
  signals, the model spots over- or under-priced assets long before they
  hit your books, helping investors and lenders avoid seven-figure mistakes.

• Competitive differentiation – clients increasingly expect Redfin- or
  Zillow-level instant estimates. Packaging your own ML pipeline behind a
  simple UI positions you as a tech-forward partner, not just another
  brokerage or contractor.

- A small, 5% improvement in predictive accuracy can save tens of thousands per transaction. On a $1M property, that’s potentially $50K of value generated

Try adjusting the sliders on the left; each change re-runs the model in
real time, showing how seemingly small differences (an extra bathroom, a
higher-income zip code) move the projected sale price.
""".strip())

