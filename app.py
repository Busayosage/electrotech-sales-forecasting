import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="ElectroTech Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown(
    """
    <style>
   .block-container {
     max-width: 1400px;
     margin: auto;
     padding-top: 2rem;
     padding-bottom: 2rem;
     padding-left: 2rem;
     padding-right: 2rem;
     }

    .hero {
        background: linear-gradient(135deg, #0f172a, #1e3a8a);
        padding: 32px;
        border-radius: 18px;
        color: white;
        margin-bottom: 30px;
    }

    .hero h1 {
        font-size: 40px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .hero p {
        font-size: 17px;
        color: #dbeafe;
        line-height: 1.6;
    }

    .kpi-card {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 14px rgba(0,0,0,0.06);
        min-height: 135px;
    }

    .kpi-label {
        font-size: 14px;
        color: #64748b;
        margin-bottom: 8px;
    }

    .kpi-value {
        font-size: 30px;
        font-weight: 800;
        color: #0f172a;
    }

    .small-note {
        font-size: 13px;
        color: #64748b;
        margin-top: 6px;
    }

    .section-card {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 14px rgba(0,0,0,0.05);
        margin-bottom: 25px;
    }

    .prediction-card {
        background: linear-gradient(135deg, #ecfdf5, #d1fae5);
        padding: 30px;
        border-radius: 18px;
        border: 1px solid #86efac;
        margin-top: 10px;
    }

    .prediction-number {
        font-size: 46px;
        font-weight: 900;
        color: #065f46;
    }

    .insight-box {
        background-color: #eff6ff;
        padding: 20px;
        border-radius: 14px;
        border-left: 6px solid #2563eb;
        margin-top: 18px;
    }

    .recommendation-box {
        background-color: #fff7ed;
        padding: 20px;
        border-radius: 14px;
        border-left: 6px solid #f97316;
        margin-top: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# LOAD MODEL + DATA
# =========================
model = joblib.load("notebooks/rf_model.pkl")

df = pd.read_csv("data/raw/ElectroTech Forecasting Data.csv")
df["Date"] = pd.to_datetime(df["Date"])

daily_sales = df.groupby("Date")["Sales_Volume"].sum().reset_index()
category_sales = df.groupby("Category")["Sales_Volume"].sum().reset_index()
season_sales = df.groupby("Season")["Sales_Volume"].mean().reset_index()

importance_df = pd.DataFrame({
    "Feature": model.feature_names_in_,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False).head(10)

# =========================
# HEADER
# =========================
st.markdown(
    """
    <div class="hero">
        <h1>📊 ElectroTech Sales Forecasting Dashboard</h1>
        <p>
        Forecast sales volume for fast-moving consumer electronics using machine learning.
        This dashboard supports inventory planning, pricing strategy, demand analysis,
        and data-driven business decisions.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# SIDEBAR INPUTS
# =========================
with st.sidebar:
    st.header("🔧 Forecast Inputs")

    month = st.slider("Month", 1, 12, 6)
    price = st.number_input("Price", value=100.0)

    competitor = st.slider("Competitor Activity Score", -5.0, 5.0, 0.0)
    confidence = st.slider("Consumer Confidence Index", 0.0, 100.0, 50.0)
    market = st.slider("Market Trend Index", -5.0, 5.0, 0.0)

    day = st.slider("Day", 1, 31, 15)
    year = st.slider("Year", 2009, 2025, 2020)
    dayofweek = st.slider("Day of Week (0 = Monday)", 0, 6, 3)

    category = st.selectbox("Category", ["Accessories", "Laptop", "Smartphone", "Tablet"])
    season = st.selectbox("Season", ["Fall", "Spring", "Summer", "Winter"])
    spec_1 = st.selectbox("Product Specification 1", ["Spec_A", "Spec_B", "Spec_C"])
    spec_2 = st.selectbox(
        "Product Specification 2",
        ["High-Resolution", "Lightweight", "Long-Battery-Life"]
    )

# =========================
# KPI SECTION
# =========================
total_sales = int(df["Sales_Volume"].sum())
avg_sales = round(df["Sales_Volume"].mean(), 2)
best_category = category_sales.sort_values("Sales_Volume", ascending=False)["Category"].iloc[0]

st.markdown("## 📌 Executive Overview")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Sales Volume</div>
            <div class="kpi-value">{total_sales:,}</div>
            <div class="small-note">Across all records</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with kpi2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Average Sales</div>
            <div class="kpi-value">{avg_sales}</div>
            <div class="small-note">Mean sales volume</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with kpi3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Top Category</div>
            <div class="kpi-value">{best_category}</div>
            <div class="small-note">Highest total sales</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with kpi4:
    st.markdown(
        """
        <div class="kpi-card">
            <div class="kpi-label">Selected Model</div>
            <div class="kpi-value">Random Forest</div>
            <div class="small-note">Best performing ML model</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# =========================
# CREATE INPUT DATA
# =========================
input_data = pd.DataFrame(columns=model.feature_names_in_)
input_data.loc[0] = 0.0

input_data.loc[0, "Price"] = price
input_data.loc[0, "Market_Trend_Index"] = market
input_data.loc[0, "Competitor_Activity_Score"] = competitor
input_data.loc[0, "Consumer_Confidence_Index"] = confidence
input_data.loc[0, "Year"] = year
input_data.loc[0, "Month"] = month
input_data.loc[0, "Day"] = day
input_data.loc[0, "DayOfWeek"] = dayofweek

for col in input_data.columns:
    if col == f"Category_{category}":
        input_data.loc[0, col] = 1
    if col == f"Season_{season}":
        input_data.loc[0, col] = 1
    if col == f"Product_Specification_1_{spec_1}":
        input_data.loc[0, col] = 1
    if col == f"Product_Specification_2_{spec_2}":
        input_data.loc[0, col] = 1

# =========================
# FORECAST RESULT
# =========================
st.markdown("## 📈 Forecast Result")

forecast_left, forecast_right = st.columns([1, 1.5])

with forecast_left:
    st.markdown(
        f"""
        <div class="section-card">
            <h3>Selected Scenario</h3>
            <p><b>Category:</b> {category}</p>
            <p><b>Season:</b> {season}</p>
            <p><b>Month:</b> {month}</p>
            <p><b>Price:</b> {price}</p>
            <p><b>Model:</b> Random Forest</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with forecast_right:
    predict_clicked = st.button("🚀 Predict Sales", use_container_width=True)

    if predict_clicked:
        prediction = model.predict(input_data)[0]
        demand_level = "high" if prediction > avg_sales else "moderate"

        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="kpi-label">Predicted Sales Volume</div>
                <div class="prediction-number">{int(prediction)} units</div>
                <div class="small-note">Forecast generated from selected business inputs</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="insight-box">
                <b>Forecast Insight:</b><br>
                Based on the selected inputs, demand is estimated to be <b>{demand_level}</b>.
                The model is mainly influenced by <b>Month</b> and <b>Price</b>, which were identified
                as the strongest sales drivers.
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.info("Use the sidebar inputs, then click **Predict Sales** to generate a forecast.")

st.markdown("<br>", unsafe_allow_html=True)

# =========================
# SALES PERFORMANCE DASHBOARD
# =========================
st.markdown("## 📊 Sales Performance Dashboard")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("📈 Historical Sales Trend")

    fig_trend = px.line(
        daily_sales,
        x="Date",
        y="Sales_Volume",
        title="Sales Volume Over Time"
    )

    fig_trend.update_layout(
        height=430,
        xaxis_title="Date",
        yaxis_title="Sales Volume",
        margin=dict(l=20, r=20, t=50, b=20)
    )

    st.plotly_chart(fig_trend, use_container_width=True)

with chart_col2:
    st.subheader("📦 Sales by Product Category")

    fig_category = px.bar(
        category_sales,
        x="Category",
        y="Sales_Volume",
        title="Total Sales by Category"
    )

    fig_category.update_layout(
        height=430,
        xaxis_title="Category",
        yaxis_title="Sales Volume",
        margin=dict(l=20, r=20, t=50, b=20)
    )

    st.plotly_chart(fig_category, use_container_width=True)

chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    st.subheader("🔁 Average Sales by Season")

    fig_season = px.bar(
        season_sales,
        x="Season",
        y="Sales_Volume",
        title="Average Sales by Season"
    )

    fig_season.update_layout(
        height=430,
        xaxis_title="Season",
        yaxis_title="Average Sales Volume",
        margin=dict(l=20, r=20, t=50, b=20)
    )

    st.plotly_chart(fig_season, use_container_width=True)

with chart_col4:
    st.subheader("🧠 Top Sales Drivers")

    fig_importance = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 10 Feature Importance"
    )

    fig_importance.update_layout(
        height=430,
        yaxis=dict(autorange="reversed"),
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        margin=dict(l=20, r=20, t=50, b=20)
    )

    st.plotly_chart(fig_importance, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================
# BUSINESS INSIGHTS
# =========================
st.markdown("## 💼 Business Insights")

insight_col1, insight_col2, insight_col3 = st.columns(3)

with insight_col1:
    st.metric("Main Sales Driver", "Month")

with insight_col2:
    st.metric("Second Key Driver", "Price")

with insight_col3:
    st.metric("Model Type", "Machine Learning")

st.markdown(
    """
    <div class="section-card">
        <h3>Key Findings</h3>
        <ul>
            <li>Sales are strongly influenced by <b>seasonality</b>, especially the month of the year.</li>
            <li><b>Price</b> is the second most important factor, showing that demand is price-sensitive.</li>
            <li>Random Forest outperformed ARIMA because it uses multiple business variables, not only historical sales trends.</li>
            <li>External indicators such as market trends and competitor activity have lower influence in this dataset.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="recommendation-box">
        <h3>Business Recommendations</h3>
        <ul>
            <li>Increase inventory before high-demand months.</li>
            <li>Use pricing and promotional strategies during peak periods.</li>
            <li>Monitor seasonal demand patterns to reduce stockouts and overstocking.</li>
            <li>Use this dashboard as a decision-support tool for sales, marketing, and inventory planning.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.caption("ElectroTech Innovations | Sales Forecasting Machine Learning Dashboard")