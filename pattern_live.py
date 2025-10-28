# streamlit_app.py

import streamlit as st
import pandas as pd
import matplotlib.dates as mdates
from datetime import timedelta
import plotly.graph_objects as go

# 1) Full-width layout
st.set_page_config(page_title="NIFTY50 Breakout Detector", layout="wide")
st.title("🕵️‍♂️ NIFTY50 Option-Chain Breakout Detector")

# 2) File uploader
uploaded = st.file_uploader("Upload NIFTY50 option-chain CSV", type="csv")
if not uploaded:
    st.info("Please upload your CSV file.")
    st.stop()

# 3) Load DataFrame
df = pd.read_csv(uploaded, parse_dates=["DATETIME"])
df["index"] = df.index

# 4) Valid ATM strikes (where STR==ATM)
valid_atm = sorted(df.loc[df["STR"] == df["ATM"], "STR"].unique())
if not valid_atm:
    st.error("No rows where STR == ATM found. Check your CSV.")
    st.stop()

# 5) Sidebar controls (detection only)
with st.sidebar:
    st.header("Detection Parameters")
    atm_options      = ["All"] + valid_atm
    atm_value        = st.selectbox("ATM Strike", atm_options)
    window_minutes   = st.slider("Detection Window (minutes)", 1, 60, value=5)
    threshold_points = st.slider("Threshold (points)", 1, 100, value=15)
    run_button       = st.button("Run Detection")

# helper for coloring negatives in the summary
def red_if_negative(val):
    try:
        return "color: red" if val < 0 else ""
    except:
        return ""

# 6) Run detection on button click
if run_button:
    combined_summaries = []
    combined_segments  = []
    atm_list = valid_atm if atm_value == "All" else [atm_value]

    for atm in atm_list:
        df_atm = (
            df[(df["STR"] == df["ATM"]) & (df["ATM"] == atm)]
              [["index","DATETIME","CE","PE","SPO"]]
              .sort_values("DATETIME")
              .set_index("DATETIME")
        )
        if df_atm.empty:
            continue

        times     = list(df_atm.index)
        window    = timedelta(minutes=window_minutes)
        threshold = threshold_points
        i         = 0

        while i < len(times):
            start = times[i]
            r0    = df_atm.loc[start]
            pe0, ce0 = r0["PE"], r0["CE"]
            win_end  = start + window

            win      = df_atm.loc[start:win_end]
            pe_moves = win["PE"] - pe0
            ce_moves = win["CE"] - ce0

            hits_pe = pe_moves[pe_moves >= threshold].index
            hits_ce = ce_moves[ce_moves >= threshold].index
            hits    = hits_pe.union(hits_ce).sort_values()

            if not hits.empty:
                hit = hits[0]
                rh  = df_atm.loc[hit]
                combined_summaries.append({
                    "ATM":        atm,
                    "start_time": start,
                    "end_time":   hit,
                    "start_pos":  r0["index"],
                    "end_pos":    rh["index"],
                    "start_PE":   pe0,
                    "end_PE":     rh["PE"],
                    "PE_move":    rh["PE"] - pe0,
                    "start_CE":   ce0,
                    "end_CE":     rh["CE"],
                    "CE_move":    rh["CE"] - ce0,
                    "window_min": window_minutes,
                    "threshold":  threshold_points
                })
                combined_segments.append(df_atm.loc[start:hit])
                i = times.index(hit) + 1
            else:
                i += 1

    # 7) Assemble and stash results
    if not combined_segments:
        st.warning("No breakouts detected with these settings.")
        st.stop()

    df_summary        = pd.DataFrame(combined_summaries).sort_values("start_time")
    df_seg            = pd.concat(combined_segments).drop_duplicates().sort_index()
    df_full_breakouts = df[df["index"].isin(df_seg["index"])].sort_values("DATETIME")

    st.session_state["df_summary"] = df_summary
    st.session_state["df_full"]    = df_full_breakouts

# 8) If results exist, display summary + segments + candlesticks
if "df_summary" in st.session_state:
    df_summary = st.session_state["df_summary"]
    df_full    = st.session_state["df_full"]

    # Breakout Summary
    st.subheader("🔎 Breakout Summary")
    styled = (
        df_summary.style
        .format({
            "start_pos": "{:.0f}", "end_pos": "{:.0f}",
            "start_PE": "{:.2f}",  "end_PE": "{:.2f}",
            "PE_move": "{:.2f}",   "start_CE":"{:.2f}",
            "end_CE": "{:.2f}",    "CE_move":"{:.2f}",
            "window_min":"{:.0f}", "threshold":"{:.0f}"
        })
        .applymap(red_if_negative, subset=["PE_move","CE_move"])
    )
    st.dataframe(styled, use_container_width=True)

    # Full Breakout Rows
    st.subheader("📈 Breakout Segments (full original data)")
    st.dataframe(df_full, use_container_width=True)

   
    # 10) Timeframe selector below header
    tf_map = {
        "5 sec":  "5S",
        "30 sec": "30S",
        "1 min":  "1T",
        "5 min":  "5T",
        "15 min": "15T"
    }
    # center the selectbox in three columns
    c1, c2, c3 = st.columns([1,3,1])
    with c1:
         # 9) Candlesticks Header
        st.header("📊 Candlesticks")
    with c3:
        timeframe_label = st.selectbox("Timeframe", list(tf_map.keys()), index=2)

    # 11) Prepare OHLC for the chosen timeframe
    freq    = tf_map[timeframe_label]
    df_full = df_full.set_index("DATETIME")
    def make_ohlc(col):
        ohlc = (
            df_full[col]
            .resample(freq)
            .agg(["first","max","min","last"])
            .dropna()
        )
        ohlc.columns = ["open","high","low","close"]
        return ohlc

    ohlc_spo = make_ohlc("SPO")
    ohlc_pe  = make_ohlc("PE")
    ohlc_ce  = make_ohlc("CE")

    # 12) Display each candlestick in its own row
    st.subheader("SPO")
    fig_spo = go.Figure(data=[go.Candlestick(
        x=ohlc_spo.index,
        open=ohlc_spo["open"], high=ohlc_spo["high"],
        low=ohlc_spo["low"],   close=ohlc_spo["close"],
        increasing_line_color="green",
        decreasing_line_color="red",
        showlegend=False
    )])
    fig_spo.update_layout(xaxis_rangeslider_visible=False, margin=dict(l=20,r=20,t=30,b=20))
    st.plotly_chart(fig_spo, use_container_width=True)

    st.subheader("PE")
    fig_pe = go.Figure(data=[go.Candlestick(
        x=ohlc_pe.index,
        open=ohlc_pe["open"], high=ohlc_pe["high"],
        low=ohlc_pe["low"],   close=ohlc_pe["close"],
        increasing_line_color="green",
        decreasing_line_color="red",
        showlegend=False
    )])
    fig_pe.update_layout(xaxis_rangeslider_visible=False, margin=dict(l=20,r=20,t=30,b=20))
    st.plotly_chart(fig_pe, use_container_width=True)

    st.subheader("CE")
    fig_ce = go.Figure(data=[go.Candlestick(
        x=ohlc_ce.index,
        open=ohlc_ce["open"], high=ohlc_ce["high"],
        low=ohlc_ce["low"],   close=ohlc_ce["close"],
        increasing_line_color="green",
        decreasing_line_color="red",
        showlegend=False
    )])
    fig_ce.update_layout(xaxis_rangeslider_visible=False, margin=dict(l=20,r=20,t=30,b=20))
    st.plotly_chart(fig_ce, use_container_width=True)
