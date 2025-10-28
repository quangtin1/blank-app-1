import streamlit as st
from backtest import run_backtest

st.set_page_config(page_title="BTC AI Trader Dashboard", layout="wide")
st.title("📈 BTC AI Trading Dashboard")

if st.button("Run Backtest"):
    with st.spinner("Running backtest..."):
        portfolio = run_backtest()
        stats = portfolio.stats()
        st.subheader("Performance Stats")
        st.dataframe(stats.astype(str))