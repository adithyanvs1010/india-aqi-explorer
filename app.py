import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(page_title="India AQI Explorer", page_icon="🌫️", layout="wide")

# ---------- Styling (dark) ----------
st.markdown("""
<style>
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1250px; }
.hero { background: linear-gradient(120deg, #12355B 0%, #1F7A8C 100%);
        color: white; padding: 28px 34px; border-radius: 16px; margin-bottom: 18px; }
.hero h1 { color: white; margin: 0; padding: 0; font-size: 2rem; font-weight: 700; }
.hero p { margin: 6px 0 0 0; opacity: .9; font-size: 1rem; }
div[data-testid="stMetric"] { background: #172230; border: 1px solid #263648; border-radius: 12px;
        padding: 14px 18px; box-shadow: 0 1px 3px rgba(0,0,0,.35); }
div[data-testid="stMetricLabel"] { color: #9FB0C3; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { font-weight: 600; padding: 8px 18px; }
.note { background: #172230; border-left: 4px solid #4FC3D9; padding: 10px 14px;
        border-radius: 6px; color: #C9D6E3; font-size: .9rem; margin-top: 8px; }
.advice { color: white; padding: 16px 20px; border-radius: 12px; font-size: 1rem; margin-top: 8px; }
</style>
""", unsafe_allow_html=True)

# ---------- Constants ----------
CLUSTER_NAMES = {
    0: "Extreme outlier",
    1: "Moderate, lower-pollution cities",
    2: "Chronic particulate hotspots",
    3: "Winter-spike eastern cities",
}
GROUP_COLORS = {
    "Extreme outlier": "#B388FF",
    "Chronic particulate hotspots": "#FF5252",
    "Winter-spike eastern cities": "#FFA726",
    "Moderate, lower-pollution cities": "#4FC3D9",
}
SEASONS = ["Winter", "Summer", "Monsoon", "Post-Monsoon"]
FULL_COVERAGE = ["Delhi", "Lucknow", "Chennai", "Hyderabad", "Bengaluru", "Ahmedabad"]

ADVICE = {
    "Good": "Air quality is good. Enjoy outdoor activities.",
    "Satisfactory": "Acceptable. Very sensitive people may notice minor breathing discomfort.",
    "Moderate": "People with lung or heart disease, children and older adults should reduce long outdoor exertion.",
    "Poor": "Everyone may feel discomfort. Limit prolonged outdoor activity.",
    "Very Poor": "Respiratory illness on prolonged exposure. Avoid outdoor exertion; sensitive groups stay indoors.",
    "Severe": "Serious health effects. Avoid going outdoors and keep windows closed.",
}

def bucket(a):
    if a <= 50:  return "Good", "#2e7d32"
    if a <= 100: return "Satisfactory", "#7cb342"
    if a <= 200: return "Moderate", "#f9a825"
    if a <= 300: return "Poor", "#ef6c00"
    if a <= 400: return "Very Poor", "#c62828"
    return "Severe", "#6a1b1a"

def style(fig, h=420):
    fig.update_layout(template="plotly_dark", height=h,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=55, b=10),
                      title_font=dict(size=16), legend_title_text="")
    return fig

# ---------- Data ----------
@st.cache_data
def load_data():
    daily = pd.read_csv("data/daily_all.csv", parse_dates=["Date"])
    model_df = pd.read_csv("data/clean.csv", parse_dates=["Date"])
    clusters = pd.read_csv("data/city_clusters.csv")
    return daily, model_df, clusters

@st.cache_resource
def load_model():
    m = XGBRegressor()
    m.load_model("models/xgb.json")
    return m, joblib.load("models/features.pkl")

daily, model_df, clusters = load_data()

# ---------- Header ----------
st.markdown("""
<div class="hero">
  <h1>India Air Quality Explorer</h1>
  <p>Seasonal patterns, city clusters and next-day AQI prediction &middot;
     CPCB city-level daily data, Jan 2015 to Jul 2020</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Hotspots", "Patterns & Clusters", "Predict"])

# ---------- Overview ----------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cities", daily.City.nunique())
    c2.metric("Date range", f"{daily.Date.min():%b %Y} to {daily.Date.max():%b %Y}")
    c3.metric("Highest mean AQI", daily.groupby("City").AQI.mean().idxmax())
    c4.metric("Daily records", f"{len(daily):,}")

    left, right = st.columns(2)
    trend = (daily[daily.City.isin(FULL_COVERAGE)]
             .groupby("Year").AQI.mean().round(1).reset_index())
    fig = px.line(trend, x="Year", y="AQI", markers=True,
                  title="Yearly mean AQI (6 full-coverage cities)")
    fig.update_traces(line_color="#4FC3D9")
    left.plotly_chart(style(fig, 380))

    season = daily.groupby("Season").AQI.mean().reindex(SEASONS).round(1).reset_index()
    fig = px.bar(season, x="Season", y="AQI", text="AQI", title="Mean AQI by season (all cities)")
    fig.update_traces(marker_color="#1F7A8C")
    fig.update_layout(showlegend=False)
    right.plotly_chart(style(fig, 380))

    st.markdown("<div class='note'>The yearly trend uses only cities with data for the whole period, "
                "because other cities join later and would distort it. 2020 covers Jan to Jul only and "
                "includes the COVID-19 lockdown.</div>", unsafe_allow_html=True)

# ---------- Hotspots ----------
with tab2:
    col1, col2 = st.columns(2)
    yr = col1.selectbox("Year", ["All"] + sorted(daily.Year.unique().tolist()))
    ssn = col2.selectbox("Season", ["All"] + SEASONS)

    d = daily
    if yr != "All":
        d = d[d.Year == yr]
    if ssn != "All":
        d = d[d.Season == ssn]

    rank = (d.groupby("City")
              .agg(Mean_AQI=("AQI", "mean"), Days=("AQI", "size"),
                   Unsafe_pct=("Unsafe", lambda s: s.mean() * 100))
              .reset_index())
    rank = rank[rank.Days >= 30].sort_values("Mean_AQI", ascending=False)

    if rank.empty:
        st.warning("Not enough data for this selection.")
    else:
        fig = px.bar(rank, x="City", y="Mean_AQI", color="Mean_AQI",
                     color_continuous_scale="YlOrRd", hover_data=["Days", "Unsafe_pct"],
                     title="Cities ranked by mean AQI")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style(fig, 440))
        st.dataframe(rank.round(1), hide_index=True)
        st.markdown("<div class='note'>Cities with fewer than 30 days of data in the selection are hidden. "
                    "Some AQI values in the dataset exceed the official 500 maximum and were kept as given.</div>",
                    unsafe_allow_html=True)

# ---------- Patterns & Clusters ----------
with tab3:
    pivot = daily.pivot_table(index="City", columns="Month", values="AQI", aggfunc="mean")
    fig = px.imshow(pivot, color_continuous_scale="YlOrRd", aspect="auto",
                    labels=dict(x="Month", color="Mean AQI"),
                    title="Average AQI by city and month")
    st.plotly_chart(style(fig, 620))

    feat_cols = [c for c in clusters.columns if c not in ("City", "Cluster")]
    pts = PCA(2, random_state=42).fit_transform(StandardScaler().fit_transform(clusters[feat_cols]))
    cl = clusters.copy()
    cl["PC1"], cl["PC2"] = pts[:, 0], pts[:, 1]
    cl["Group"] = cl.Cluster.map(CLUSTER_NAMES)

    a, b = st.columns([1.3, 1])
    fig = px.scatter(cl, x="PC1", y="PC2", color="Group", text="City",
                     color_discrete_map=GROUP_COLORS, title="City clusters (PCA view)")
    fig.update_traces(textposition="top center", marker=dict(size=11))
    a.plotly_chart(style(fig, 480))
    b.dataframe(cl[["City", "Group", "mean_aqi", "unsafe_share", "winter_monsoon_ratio"]]
                .sort_values("Group").round(2), hide_index=True, height=480)
    st.markdown("<div class='note'>Cities with under 250 days of data were left out of clustering. "
                "Ahmedabad forms its own group because of its extreme AQI values.</div>",
                unsafe_allow_html=True)

# ---------- Predict ----------
with tab4:
    model, features = load_model()
    left, right = st.columns([1, 1.3])

    with left:
        city = st.selectbox("City", sorted(model_df.City.unique()))
        cd = model_df[model_df.City == city].sort_values("Date")
        dates = cd.Date.dt.date.tolist()[::-1]          # latest first
        picked = st.selectbox("Date (the prediction is for the next day)", dates)
        row = cd[cd.Date.dt.date == picked].iloc[[0]].copy()

        with st.expander("What-if: adjust today's readings"):
            k = f"{city}_{picked}"
            aqi_now = st.number_input("Today's AQI", 0.0, 3000.0, float(row.AQI.iloc[0]), key=f"aqi_{k}")
            pm25 = st.number_input("PM2.5", 0.0, 2000.0, float(row["PM2.5"].iloc[0]), key=f"pm_{k}")
            co = st.number_input("CO", 0.0, 500.0, float(row.CO.iloc[0]), key=f"co_{k}")
        row["AQI"], row["PM2.5"], row["CO"] = aqi_now, pm25, co

    pred = float(model.predict(row[features])[0])
    label, color = bucket(pred)
    actual = float(row.AQI_next.iloc[0])

    with right:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=pred, number={"font": {"size": 44}},
            title={"text": "Predicted next-day AQI"},
            gauge={"axis": {"range": [0, 500]}, "bar": {"color": "#E6EDF3"},
                   "steps": [{"range": [0, 50], "color": "#2e7d32"},
                             {"range": [50, 100], "color": "#7cb342"},
                             {"range": [100, 200], "color": "#f9a825"},
                             {"range": [200, 300], "color": "#ef6c00"},
                             {"range": [300, 400], "color": "#c62828"},
                             {"range": [400, 500], "color": "#6a1b1a"}]}))
        gauge.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                            height=300, margin=dict(l=30, r=30, t=70, b=10))
        st.plotly_chart(gauge)

    m1, m2 = st.columns(2)
    m1.metric("Predicted category", label)
    m2.metric("Recorded next-day AQI", f"{actual:.0f}")
    st.markdown(f"<div class='advice' style='background:{color}'><b>{label}:</b> {ADVICE[label]}</div>",
                unsafe_allow_html=True)

    hist = cd[cd.Date <= pd.Timestamp(picked)].tail(30)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.Date, y=hist.AQI, mode="lines+markers",
                             name="Recorded AQI", line=dict(color="#4FC3D9")))
    fig.add_trace(go.Scatter(x=[pd.Timestamp(picked) + pd.Timedelta(days=1)], y=[pred],
                             mode="markers", name="Predicted next day",
                             marker=dict(size=13, color=color, line=dict(width=2, color="white"))))
    fig.update_layout(title=f"{city}: last 30 days and prediction")
    st.plotly_chart(style(fig, 340))

    st.markdown("<div class='note'>Uses the XGBoost model. Yesterday's values and the 7-day average stay "
                "as recorded; only the what-if inputs change. The data ends in July 2020, so this is not "
                "a live forecast.</div>", unsafe_allow_html=True)