import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# ----------------------------------------------------
# 🔧 Page Config
# ----------------------------------------------------
st.set_page_config(
    page_title="Retail Analytics & Customer Segmentation",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------
# 🎨 Custom CSS (high-contrast, visible headings)
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

/* Global headings – strong and dark */
h1, h2, h3, h4 {
    font-family: "Segoe UI", system-ui, sans-serif !important;
    color: #0a0a23 !important;
    font-weight: 700 !important;
}

/* Section headings (like '📈 Detailed Sales Analysis', '🏷️ Top Products') */
.stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
    font-size: 1.3rem !important;
    margin-top: 0.8rem !important;
    margin-bottom: 0.3rem !important;
}

/* Small accent bar under h2/h3 in markdown */
.stMarkdown h2::after, .stMarkdown h3::after {
    content: "";
    display: block;
    width: 70px;
    height: 4px;
    background: linear-gradient(90deg, #4F46E5, #EC4899);
    margin-top: 6px;
    border-radius: 10px;
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

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    padding-top: 10px;
    padding-bottom: 10px;
    border-radius: 999px;
    background-color: rgba(148, 163, 184, 0.25);
    color: #111827;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #6366F1, #EC4899);
    color: white !important;
}

/* Dataframe background */
[data-testid="stDataFrame"] {
    background: #ffffff;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------------------------------
# 📂 Data Loading & Cleaning
# ----------------------------------------------------
@st.cache_data
def load_data():
    candidate_paths = [
        Path("data/Online Retail.xlsx"),
        Path("Online Retail.xlsx")
    ]
    file_path = None
    for p in candidate_paths:
        if p.exists():
            file_path = p
            break

    if file_path is None:
        st.error(
            "❌ Could not find `Online Retail.xlsx`. "
            "Place it in the same folder as `app.py` or inside a `data/` folder."
        )
        st.stop()

    df = pd.read_excel(file_path)

    expected_cols = [
        "InvoiceNo", "StockCode", "Description",
        "Quantity", "InvoiceDate", "UnitPrice",
        "CustomerID", "Country"
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        st.error(f"❌ Missing columns in Excel file: {missing}")
        st.stop()

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])
    df = df.dropna(subset=["CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype(int)

    # keep only positive transactions
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    # time-based features
    df["InvoiceMonth"] = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["DayName"] = df["InvoiceDate"].dt.day_name()
    df["Hour"] = df["InvoiceDate"].dt.hour

    return df


df = load_data()

# ----------------------------------------------------
# 🧭 Sidebar Filters
# ----------------------------------------------------
st.sidebar.title("🧭 Controls & Filters")

countries = ["All"] + sorted(df["Country"].unique().tolist())
selected_country = st.sidebar.selectbox("🌍 Country", countries, index=0)

min_date = df["InvoiceDate"].min().date()
max_date = df["InvoiceDate"].max().date()
date_range = st.sidebar.date_input(
    "📅 Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(date_range, tuple):
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

start_ts = pd.to_datetime(start_date)
end_ts = pd.to_datetime(end_date) + pd.Timedelta(days=1)

product_search = st.sidebar.text_input("🔎 Search Product Description", "").strip()

min_invoice_value = st.sidebar.slider(
    "💸 Minimum Invoice Total (for analysis)",
    min_value=0,
    max_value=int(df["TotalPrice"].max()),
    value=0,
    step=100,
)

# Filter data
filtered_df = df.copy()

if selected_country != "All":
    filtered_df = filtered_df[filtered_df["Country"] == selected_country]

filtered_df = filtered_df[
    (filtered_df["InvoiceDate"] >= start_ts) &
    (filtered_df["InvoiceDate"] < end_ts)
]

if product_search:
    filtered_df = filtered_df[
        filtered_df["Description"].str.contains(product_search, case=False, na=False)
    ]

invoice_totals = filtered_df.groupby("InvoiceNo")["TotalPrice"].transform("sum")
filtered_df = filtered_df[invoice_totals >= min_invoice_value]

if filtered_df.empty:
    st.warning("⚠ No data for current filters. Try relaxing filters.")
    st.stop()

# ----------------------------------------------------
# 🏠 Header
# ----------------------------------------------------
st.markdown(
    "<h1 style='text-align:center;'>🛒 Retail Analytics & Customer Segmentation</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:#4b5563;'>Data Cleaning • EDA • Feature Engineering • RFM Clustering</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ----------------------------------------------------
# 🔢 KPI Section
# ----------------------------------------------------
total_revenue = filtered_df["TotalPrice"].sum()
total_orders = filtered_df["InvoiceNo"].nunique()
total_customers = filtered_df["CustomerID"].nunique()
avg_order_value = total_revenue / total_orders if total_orders > 0 else 0

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(
        f"<div class='kpi-card'><h3>💰 Revenue</h3><h2>${total_revenue:,.0f}</h2></div>",
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        f"<div class='kpi-card'><h3>🧾 Orders</h3><h2>{total_orders:,}</h2></div>",
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        f"<div class='kpi-card'><h3>👥 Customers</h3><h2>{total_customers:,}</h2></div>",
        unsafe_allow_html=True,
    )
with k4:
    st.markdown(
        f"<div class='kpi-card'><h3>📦 Avg Order Value</h3><h2>${avg_order_value:,.0f}</h2></div>",
        unsafe_allow_html=True,
    )

st.markdown("")

# ----------------------------------------------------
# 🧩 Tabs
# ----------------------------------------------------
tab_overview, tab_sales, tab_cluster, tab_data = st.tabs(
    ["📌 Overview", "📈 Sales Analysis", "🤖 Customer Segmentation", "📂 Data Explorer"]
)

# ----------------------------------------------------
# 📌 TAB 1: Overview
# ----------------------------------------------------
with tab_overview:
    st.subheader("📌 High-Level Overview")

    monthly = (
        filtered_df.groupby("InvoiceMonth")["TotalPrice"]
        .sum()
        .reset_index()
        .sort_values("InvoiceMonth")
    )

    fig_rev = px.area(
        monthly,
        x="InvoiceMonth",
        y="TotalPrice",
        title="Revenue Over Time",
        markers=True,
    )
    fig_rev.update_layout(xaxis_title="Month", yaxis_title="Revenue")
    st.plotly_chart(fig_rev, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### 🌍 Top 10 Countries (by Revenue)")
        country_rev = (
            filtered_df.groupby("Country")["TotalPrice"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        fig_c = px.bar(
            country_rev,
            x="Country",
            y="TotalPrice",
            title="Top Countries by Revenue",
        )
        st.plotly_chart(fig_c, use_container_width=True)

    with c2:
        st.markdown("#### 📅 Revenue by Day of Week")
        dow = (
            filtered_df.groupby("DayName")["TotalPrice"]
            .sum()
            .reindex(
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            )
            .reset_index()
        )
        fig_dow = px.bar(
            dow,
            x="DayName",
            y="TotalPrice",
            title="Revenue by Day of Week",
        )
        st.plotly_chart(fig_dow, use_container_width=True)

# ----------------------------------------------------
# 📈 TAB 2: Sales Analysis
# ----------------------------------------------------
with tab_sales:
    st.subheader("📈 Detailed Sales Analysis")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 🏷️ Top Products")
        top_n = st.slider("Show Top N Products", min_value=5, max_value=30, value=10, step=5)
        prod = (
            filtered_df.groupby("Description")["TotalPrice"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
            .reset_index()
        )
        fig_prod = px.bar(
            prod,
            x="TotalPrice",
            y="Description",
            orientation="h",
            title=f"Top {top_n} Products by Revenue",
        )
        fig_prod.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_prod, use_container_width=True)

    with col_right:
        st.markdown("#### ⌚ Hourly Sales Pattern")
        hour_sales = (
            filtered_df.groupby("Hour")["TotalPrice"]
            .sum()
            .reset_index()
            .sort_values("Hour")
        )
        fig_hr = px.line(
            hour_sales,
            x="Hour",
            y="TotalPrice",
            markers=True,
            title="Revenue by Hour of Day",
        )
        st.plotly_chart(fig_hr, use_container_width=True)

    st.markdown("#### 🔥 Day vs Hour Intensity (Heatmap)")
    pivot = (
        filtered_df.pivot_table(
            index="DayName",
            columns="Hour",
            values="TotalPrice",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(index=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    )

    fig_heat = px.imshow(
        pivot,
        aspect="auto",
        title="Revenue Heatmap (Day vs Hour)",
        labels=dict(x="Hour of Day", y="Day of Week", color="Revenue"),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ----------------------------------------------------
# 🤖 TAB 3: Customer Segmentation (RFM + KMeans)
# ----------------------------------------------------
with tab_cluster:
    st.subheader("🤖 Customer Segmentation using RFM + KMeans")
    st.info("Tip: For stronger clusters, prefer using full data (uncheck 'Use filtered' option).")

    use_filtered = st.checkbox("Use filtered data for segmentation", value=False)
    cluster_df = filtered_df if use_filtered else df

    max_date = cluster_df["InvoiceDate"].max()
    rfm = cluster_df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (max_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    ).reset_index()

    rfm = rfm[(rfm["Recency"] >= 0) & (rfm["Monetary"] > 0)]

    if len(rfm) < 5:
        st.warning("⚠ Not enough customers for clustering. Try using full data or relaxing filters.")
    else:
        st.markdown("##### ⚙️ Clustering Settings")
        colc1, colc2 = st.columns(2)
        with colc1:
            n_clusters = st.slider("Number of Clusters (k)", 2, 8, 4)
        with colc2:
            scale_log = st.checkbox("Apply log transform to Monetary", value=True)

        rfm_for_scale = rfm.copy()
        if scale_log:
            rfm_for_scale["Monetary"] = np.log1p(rfm_for_scale["Monetary"])

        scaler = StandardScaler()
        scaled = scaler.fit_transform(rfm_for_scale[["Recency", "Frequency", "Monetary"]])

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        rfm["Cluster"] = kmeans.fit_predict(scaled)

        # Cluster summary sorted by Monetary (for labels)
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

        # Map cluster -> label by order
        cluster_label_map = {}
        for i, row in summary.iterrows():
            cluster_id = int(row["Cluster"])
            if i < len(label_candidates):
                cluster_label_map[cluster_id] = label_candidates[i]
            else:
                cluster_label_map[cluster_id] = f"Segment {cluster_id}"

        rfm["Segment"] = rfm["Cluster"].map(cluster_label_map)

        fig3d = px.scatter_3d(
            rfm,
            x="Recency",
            y="Frequency",
            z="Monetary",
            color="Segment",
            hover_data=["CustomerID", "Cluster"],
            title="Customer Segments in RFM Space",
        )
        st.plotly_chart(fig3d, use_container_width=True)

        st.markdown("##### 📊 Cluster Summary (Average RFM per Segment)")
        cluster_summary = (
            rfm.groupby(["Cluster", "Segment"])[["Recency", "Frequency", "Monetary"]]
            .mean()
            .round(2)
            .reset_index()
        )
        st.dataframe(cluster_summary, use_container_width=True)

        csv = rfm.to_csv(index=False)
        st.download_button(
            "⬇ Download Segmented Customers (CSV)",
            data=csv,
            file_name="customer_segments.csv",
            mime="text/csv",
        )

# ----------------------------------------------------
# 📂 TAB 4: Data Explorer
# ----------------------------------------------------
with tab_data:
    st.subheader("📂 Data Explorer & Summary")

    st.markdown("#### 🔍 Sample of Cleaned & Filtered Data")
    st.dataframe(filtered_df.head(50), use_container_width=True)

    st.markdown("#### 📈 Summary Statistics (Numeric Columns)")
    num_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        st.dataframe(
            filtered_df[num_cols].describe().T.round(2),
            use_container_width=True,
        )

    st.markdown("#### 📊 Correlation Heatmap (Numeric Features)")
    if len(num_cols) > 1:
        corr = filtered_df[num_cols].corr()
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            title="Correlation Heatmap",
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Not enough numeric columns for correlation heatmap.")
