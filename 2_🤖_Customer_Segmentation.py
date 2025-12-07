import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# ----------------------------------------------------
# 🔧 Streamlit Page Config
# ----------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation (RFM + KMeans)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------
# 🎨 Simple High-Contrast CSS
# ----------------------------------------------------
CUSTOM_CSS = """
<style>
/* Main background */
[data-testid="stAppViewContainer"] {
    background: #f3f4f6;
}

/* Sidebar dark theme */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #0f172a);
}
[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}

/* Titles */
h1, h2, h3 {
    font-family: "Segoe UI", system-ui, sans-serif;
    color: #0f172a;
}

/* KPI cards */
.kpi-card {
    padding: 18px 20px;
    border-radius: 18px;
    text-align: center;
    background: linear-gradient(135deg, #1d4ed8 0%, #6366f1 40%, #ec4899 100%);
    color: #ffffff !important;
    font-weight: 600;
    box-shadow: 0 12px 30px rgba(15, 23, 42, 0.5);
    border: 1px solid rgba(248, 250, 252, 0.6);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.kpi-card:hover {
    transform: translateY(-4px) scale(1.02);
    box-shadow: 0 20px 45px rgba(15, 23, 42, 0.75);
}
.kpi-card h2, .kpi-card h3 {
    color: #ffffff !important;
    margin: 0.1rem 0;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background: #ffffff;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------------------------------
# 📂 Load & Clean Data
# ----------------------------------------------------
@st.cache_data
def load_data():
    # Look for file in ./data/ or current folder
    candidate_paths = [
        Path("data/Online Retail.xlsx"),
        Path("Online Retail.xlsx"),
    ]
    file_path = None
    for p in candidate_paths:
        if p.exists():
            file_path = p
            break

    if file_path is None:
        st.error(
            "❌ Could not find `Online Retail.xlsx`. "
            "Place it in the same folder as this script or inside a `data/` folder."
        )
        st.stop()

    df = pd.read_excel(file_path)

    # Basic column check
    expected_cols = [
        "InvoiceNo", "StockCode", "Description",
        "Quantity", "InvoiceDate", "UnitPrice",
        "CustomerID", "Country"
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        st.error(f"❌ Missing columns in Excel file: {missing}")
        st.stop()

    # Date & CustomerID cleaning
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate", "CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype(int)

    # Keep only positive sales
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    # Monetary value per row
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    return df


df = load_data()

# ----------------------------------------------------
# 🏠 Header
# ----------------------------------------------------
st.markdown(
    "<h1 style='text-align:center;'>🤖 Customer Segmentation using RFM + KMeans</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:#4b5563;'>Build RFM features • Cluster customers • Visualize segments • Download results</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ----------------------------------------------------
# 🧭 Sidebar – Controls
# ----------------------------------------------------
st.sidebar.title("🧭 Segmentation Settings")

# Use all data or filter by country/date only for viewing (clustering will always use all data to avoid "not enough customers")
countries = ["All"] + sorted(df["Country"].unique().tolist())
selected_country = st.sidebar.selectbox("🌍 Filter View by Country", countries, index=0)

min_date = df["InvoiceDate"].min().date()
max_date = df["InvoiceDate"].max().date()
date_range = st.sidebar.date_input(
    "📅 Filter View Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(date_range, tuple):
    view_start, view_end = date_range
else:
    view_start, view_end = min_date, max_date

view_start_ts = pd.to_datetime(view_start)
view_end_ts = pd.to_datetime(view_end) + pd.Timedelta(days=1)

st.sidebar.markdown("---")

n_clusters = st.sidebar.slider("Number of Clusters (k)", 2, 8, 4)
scale_log = st.sidebar.checkbox("Log-transform Monetary before scaling", value=True)

# ----------------------------------------------------
# 👀 Data for VIEW only (NOT for clustering)
# ----------------------------------------------------
view_df = df.copy()
if selected_country != "All":
    view_df = view_df[view_df["Country"] == selected_country]

view_df = view_df[
    (view_df["InvoiceDate"] >= view_start_ts) &
    (view_df["InvoiceDate"] < view_end_ts)
]

if view_df.empty:
    st.warning("⚠ No transactions match the selected view filters. Showing full data instead.")
    view_df = df.copy()

# ----------------------------------------------------
# 🧮 Build RFM on FULL data (for stable clustering)
# ----------------------------------------------------
max_date = df["InvoiceDate"].max()

rfm = df.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (max_date - x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("TotalPrice", "sum"),
).reset_index()

# Remove weird values
rfm = rfm[(rfm["Recency"] >= 0) & (rfm["Monetary"] > 0)]

# ----------------------------------------------------
# 🔢 RFM KPIs
# ----------------------------------------------------
total_customers = len(rfm)
avg_recency = rfm["Recency"].mean() if total_customers > 0 else 0
avg_frequency = rfm["Frequency"].mean() if total_customers > 0 else 0
avg_monetary = rfm["Monetary"].mean() if total_customers > 0 else 0

k1, k2, k3 = st.columns(3)
with k1:
    st.markdown(
        f"<div class='kpi-card'><h3>👥 Customers Segmented</h3><h2>{total_customers:,}</h2></div>",
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        f"<div class='kpi-card'><h3>📅 Avg Recency (days)</h3><h2>{avg_recency:,.1f}</h2></div>",
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        f"<div class='kpi-card'><h3>💰 Avg Monetary</h3><h2>${avg_monetary:,.1f}</h2></div>",
        unsafe_allow_html=True,
    )

st.markdown("")

# ----------------------------------------------------
# 🧩 Tabs
# ----------------------------------------------------
tab_seg, tab_rfm, tab_raw = st.tabs(
    ["🤖 Segmentation", "📊 RFM Distributions", "📂 Raw Data (View)"]
)

# ----------------------------------------------------
# 🤖 TAB 1 – Segmentation
# ----------------------------------------------------
with tab_seg:
    st.subheader("🤖 RFM + KMeans Clustering")

    if len(rfm) < n_clusters:
        st.warning(
            f"⚠ Not enough customers ({len(rfm)}) for {n_clusters} clusters. "
            f"Reduce k or use more data."
        )
    else:
        # Prepare for scaling
        rfm_scaled = rfm.copy()
        if scale_log:
            rfm_scaled["Monetary"] = np.log1p(rfm_scaled["Monetary"])

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(
            rfm_scaled[["Recency", "Frequency", "Monetary"]]
        )

        # KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        rfm["Cluster"] = kmeans.fit_predict(scaled_features)

        # ---- Label clusters by Monetary rank (FIXED) ----
        summary = (
            rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]]
            .mean()
            .sort_values("Monetary", ascending=False)
            .reset_index()
        )

        label_candidates = [
            "🏆 Champions",
            "🟢 Loyal",
            "🟡 Potential",
            "🟠 At Risk",
            "🔵 New",
            "⚫ Hibernating",
            "🧊 Lost",
            "💤 Others",
        ]

        # Map cluster -> label by order in summary
        cluster_label_map = {}
        for i, row in summary.iterrows():
            cluster_id = int(row["Cluster"])
            if i < len(label_candidates):
                cluster_label_map[cluster_id] = label_candidates[i]
            else:
                cluster_label_map[cluster_id] = "Segment " + str(cluster_id)

        rfm["Segment"] = rfm["Cluster"].map(cluster_label_map)

        st.markdown("##### 🎯 3D View of Customer Segments (RFM Space)")
        fig3d = px.scatter_3d(
            rfm,
            x="Recency",
            y="Frequency",
            z="Monetary",
            color="Segment",
            hover_data=["CustomerID", "Cluster"],
        )
        fig3d.update_layout(
            scene=dict(
                xaxis_title="Recency (days)",
                yaxis_title="Frequency",
                zaxis_title="Monetary",
            )
        )
        st.plotly_chart(fig3d, use_container_width=True)

        st.markdown("##### 📊 Cluster Summary (Average RFM)")
        cluster_summary = (
            rfm.groupby(["Cluster", "Segment"])[["Recency", "Frequency", "Monetary"]]
            .mean()
            .round(2)
            .reset_index()
        )
        st.dataframe(cluster_summary, use_container_width=True)

        st.markdown("##### 👀 Sample Segmented Customers")
        st.dataframe(rfm.head(50), use_container_width=True)

        # Download segmented data
        csv = rfm.to_csv(index=False)
        st.download_button(
            "⬇ Download Segmented Customers (CSV)",
            data=csv,
            file_name="customer_segments_rfm_kmeans.csv",
            mime="text/csv",
        )

# ----------------------------------------------------
# 📊 TAB 2 – RFM Distributions
# ----------------------------------------------------
with tab_rfm:
    st.subheader("📊 RFM Feature Distributions")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### Recency Distribution")
        fig_r = px.histogram(rfm, x="Recency", nbins=30, marginal="box")
        st.plotly_chart(fig_r, use_container_width=True)

    with colB:
        st.markdown("#### Frequency Distribution")
        fig_f = px.histogram(rfm, x="Frequency", nbins=30, marginal="box")
        st.plotly_chart(fig_f, use_container_width=True)

    st.markdown("#### Monetary Distribution")
    fig_m = px.histogram(rfm, x="Monetary", nbins=30, marginal="box")
    st.plotly_chart(fig_m, use_container_width=True)

# ----------------------------------------------------
# 📂 TAB 3 – Raw Data (View)
# ----------------------------------------------------
with tab_raw:
    st.subheader("📂 Raw Transactions (View Filters Applied)")
    st.markdown("Showing top 50 rows from filtered view data:")
    st.dataframe(view_df.head(50), use_container_width=True)
