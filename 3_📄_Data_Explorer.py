import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    return pd.read_excel("data/Online Retail.xlsx")

df = load_data()

st.header("📄 Explore Raw / Cleaned Data")
st.dataframe(df.sample(50))
