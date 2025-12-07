import streamlit as st
import pandas as pd
import plotly.express as px

@st.cache_data
def load_data():
    return pd.read_excel("data/Online Retail.xlsx")

df = load_data()
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
df['InvoiceMonth'] = df['InvoiceDate'].dt.to_period('M').dt.to_timestamp()

st.header("📊 Sales Analytics Dashboard")

col1, col2, col3 = st.columns(3)
col1.metric("Revenue", f"${df['TotalPrice'].sum():,.0f}")
col2.metric("Orders", f"{df['InvoiceNo'].nunique():,}")
col3.metric("Customers", f"{df['CustomerID'].nunique():,}")

st.markdown("### 📈 Monthly Revenue Growth")
monthly = df.groupby("InvoiceMonth")['TotalPrice'].sum().reset_index()
st.plotly_chart(px.area(monthly, x="InvoiceMonth", y="TotalPrice", markers=True,
                        title="Revenue Trend"), use_container_width=True)

st.markdown("### 🛍️ Top Product Categories")
top_products = df.groupby("Description")['TotalPrice'].sum().nlargest(10).reset_index()
st.plotly_chart(px.bar(top_products, x='TotalPrice', y='Description',
                       orientation='h', title="Best Sellers"),
                use_container_width=True)
