import streamlit as st
import pandas as pd
from datetime import timedelta
import plotly.graph_objects as go

# ---------------------------------
# 1) Page layout / title
# ---------------------------------
st.set_page_config(page_title="NIFTY50 Breakout Detector", layout="wide")
st.title("🕵️‍♂️ NIFTY50 Option-Chain Breakout Detector")

# ---------------------------------
# 2) File uploader
# ---------------------------------
uploaded = st.file_uploader("Upload NIFTY50 option-chain CSV", type="csv")
if not uploaded:
    st.info("Please upload your CSV file.")
    st.stop()

# ---------------------------------
# 3) Load CSV into DataFrame
# ---------------------------------
df = pd.read_csv(uploaded, parse_dates=["DATETIME"])
df["index"] = df.index

# Safety: check core columns exist
required_cols = {"DATETIME", "STR", "ATM", "CE", "PE", "SPO"}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"Your CSV is missing columns: {missing}")
    st.stop()

# Precompute helpful derived columns if missing
if "STRADDLE" not in df.columns and {"CE", "PE"}.issubset(df.columns):
    df["STRADDLE"] = df["CE"] + df["PE"]

if {"CE_OI", "PE_OI"}.issubset(df.columns) and "TOTAL_OI" not in df.columns:
    df["TOTAL_OI"] = df["CE_OI"] + df["PE_OI"]

if {"CE", "PE"}.issubset(df.columns) and "CE_PE_DIFF" not in df.columns:
    df["CE_PE_DIFF"] = df["CE"] - df["PE"]

# ---------------------------------
# 4) Get valid ATM strikes (where STR == ATM at that moment)
# ---------------------------------
valid_atm = sorted(df.loc[df["STR"] == df["ATM"], "STR"].unique())
if not valid_atm:
    st.error("No rows where STR == ATM found. Check your CSV.")
    st.stop()

# ---------------------------------
# 5) Sidebar controls for burst detection
# ---------------------------------
with st.sidebar:
    st.header("Burst Detection Parameters")
    atm_options      = ["All"] + valid_atm
    atm_value        = st.selectbox("ATM Strike", atm_options)
    window_minutes   = st.slider("Detection Window (minutes)", 1, 60, value=5)
    threshold_points = st.slider("Threshold (points)", 1, 100, value=15)
    # run_button is removed; we always run with these values

def red_if_negative(val):
    try:
        return "color: red" if val < 0 else ""
    except:
        return ""

# ---------------------------------
# 6) Burst Detection Logic
#    (now ALWAYS runs using the sidebar values)
# ---------------------------------
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

if combined_segments:
    df_summary        = pd.DataFrame(combined_summaries).sort_values("start_time")
    df_seg            = pd.concat(combined_segments).drop_duplicates().sort_index()
    df_full_breakouts = df[df["index"].isin(df_seg["index"])].sort_values("DATETIME")
else:
    df_summary        = pd.DataFrame()  # empty
    df_full_breakouts = pd.DataFrame()

# ---------------------------------
# 7) Show Burst Detection Results (always tries to render)
# ---------------------------------
if df_summary.empty:
    st.subheader("🔎 Breakout Summary")
    st.info("No breakouts detected with current settings.")
else:
    # Summary table of detected bursts
    st.subheader("🔎 Breakout Summary")
    styled = (
        df_summary.style
        .format({
            "start_pos": "{:.0f}",
            "end_pos": "{:.0f}",
            "start_PE": "{:.2f}",
            "end_PE": "{:.2f}",
            "PE_move": "{:.2f}",
            "start_CE":"{:.2f}",
            "end_CE": "{:.2f}",
            "CE_move":"{:.2f}",
            "window_min":"{:.0f}",
            "threshold":"{:.0f}"
        })
        .applymap(red_if_negative, subset=["PE_move","CE_move"])
    )
    st.dataframe(styled, use_container_width=True)

    # Full raw rows contributing to those bursts
    st.subheader("📈 Breakout Segments (full original data)")
    st.dataframe(df_full_breakouts, use_container_width=True)

    # ---------------------------------
    # 8) Candlesticks from breakout rows
    # ---------------------------------
    tf_map = {
        "5 sec":  "5S",
        "30 sec": "30S",
        "1 min":  "1T",
        "5 min":  "5T",
        "15 min": "15T"
    }

    c1, c2, c3 = st.columns([1,3,1])
    with c1:
        st.header("📊 Candlesticks")
    with c3:
        timeframe_label = st.selectbox("Timeframe", list(tf_map.keys()), index=2)

    freq    = tf_map[timeframe_label]

    # Only build candles if we have rows
    if not df_full_breakouts.empty:
        df_full_idxed = df_full_breakouts.set_index("DATETIME")

        def make_ohlc(col):
            ohlc = (
                df_full_idxed[col]
                .resample(freq)
                .agg(["first","max","min","last"])
                .dropna()
            )
            ohlc.columns = ["open","high","low","close"]
            return ohlc

        ohlc_spo = make_ohlc("SPO")
        ohlc_pe  = make_ohlc("PE")
        ohlc_ce  = make_ohlc("CE")

        st.subheader("SPO")
        fig_spo = go.Figure(data=[go.Candlestick(
            x=ohlc_spo.index,
            open=ohlc_spo["open"], high=ohlc_spo["high"],
            low=ohlc_spo["low"],   close=ohlc_spo["close"],
            increasing_line_color="green",
            decreasing_line_color="red",
            showlegend=False
        )])
        fig_spo.update_layout(
            xaxis_rangeslider_visible=False,
            margin=dict(l=20,r=20,t=30,b=20)
        )
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
        fig_pe.update_layout(
            xaxis_rangeslider_visible=False,
            margin=dict(l=20,r=20,t=30,b=20)
        )
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
        fig_ce.update_layout(
            xaxis_rangeslider_visible=False,
            margin=dict(l=20,r=20,t=30,b=20)
        )
        st.plotly_chart(fig_ce, use_container_width=True)

# ---------------------------------
# 8) ATC-style Deviation Bands (Rolling mean ± σ)
#    - ATM selector
#    - metric selector
#    - rolling lookback
#    - horizontal zoom via Plotly rangeslider
#    - POINT COLORING by zscore (red/blue/black)
# ---------------------------------

st.header("📡 Deviation Bands (Mean ± Std Dev)")

# build candidate metrics list dynamically, prioritizing useful ones
candidate_metrics = [
    col for col in [
        "ACT",                 # your "actual move" / activity metric
        "CE", "PE", "SPO", "SFUT",
        "SDIF", "SPO_DELTA",
        "STRADDLE", "STRADDLE_DELTA",
        "CE_OI", "PE_OI",
        "CE_IV", "PE_IV",
        "TOTAL_OI", "CE_PE_DIFF"
    ]
    if col in df.columns
]

# --- UI controls for ATC chart ---
col_left, col_mid, col_right = st.columns(3)

with col_left:
    atm_for_band = st.selectbox(
        "ATM Strike for Deviation Chart (ATM)",
        sorted(df["ATM"].unique()),
        key="band_atm"
    )

with col_mid:
    price_col_choice = st.selectbox(
        "Metric to Analyze (rolling mean/σ)",
        candidate_metrics,
        index=0,
        key="band_price_col"
    )

with col_right:
    lookback = st.slider(
        "Rolling Window (bars)",
        min_value=5,
        max_value=200,
        value=20,
        key="band_lookback"
    )

# pull only rows where this ATM was actually ATM (STR == ATM and ATM == selected value)
df_band = (
    df[(df["STR"] == df["ATM"]) & (df["ATM"] == atm_for_band)]
    .sort_values("DATETIME")
    .reset_index(drop=True)
    .copy()
)

if df_band.empty:
    st.warning("No rows found for the selected ATM in the CSV.")
else:
    # rolling stats on the chosen metric (e.g. ACT)
    df_band["mean"] = df_band[price_col_choice].rolling(lookback).mean()
    df_band["std"]  = df_band[price_col_choice].rolling(lookback).std()

    df_band["upper1"] = df_band["mean"] + df_band["std"] * 1
    df_band["lower1"] = df_band["mean"] - df_band["std"] * 1
    df_band["upper2"] = df_band["mean"] + df_band["std"] * 2
    df_band["lower2"] = df_band["mean"] - df_band["std"] * 2

    # mark statistical breakouts
    df_band["zscore"] = (df_band[price_col_choice] - df_band["mean"]) / df_band["std"]
    df_band["breakout_flag"] = df_band["zscore"].abs() >= 2

    # --- COLOR LOGIC FOR POINTS ---
    # red  = |z| >= 2  (outside ±2σ / orange)
    # blue = 1 <= |z| < 2 (outside ±1σ / yellow but inside ±2σ)
    # black = |z| < 1 (normal)
    def classify_color(z):
        if abs(z) >= 2:
            return "red"
        elif abs(z) >= 1:
            return "blue"
        else:
            return "black"

    df_band["point_color"] = df_band["zscore"].apply(classify_color)

    # Build the Plotly figure
    fig_band = go.Figure()

    # ±2σ band (outer, light orange)
    fig_band.add_trace(go.Scatter(
        x=df_band["DATETIME"], y=df_band["upper2"],
        mode="lines",
        line=dict(width=0, color="rgba(255,165,0,0.2)"),
        name="+2σ"
    ))
    fig_band.add_trace(go.Scatter(
        x=df_band["DATETIME"], y=df_band["lower2"],
        mode="lines",
        line=dict(width=0, color="rgba(255,165,0,0.2)"),
        fill="tonexty",
        fillcolor="rgba(255,165,0,0.15)",
        name="-2σ"
    ))

    # ±1σ band (inner, yellow)
    fig_band.add_trace(go.Scatter(
        x=df_band["DATETIME"], y=df_band["upper1"],
        mode="lines",
        line=dict(width=0, color="rgba(255,255,0,0.3)"),
        name="+1σ"
    ))
    fig_band.add_trace(go.Scatter(
        x=df_band["DATETIME"], y=df_band["lower1"],
        mode="lines",
        line=dict(width=0, color="rgba(255,255,0,0.3)"),
        fill="tonexty",
        fillcolor="rgba(255,255,0,0.2)",
        name="-1σ"
    ))

    # rolling mean (gray dashed)
    fig_band.add_trace(go.Scatter(
        x=df_band["DATETIME"], y=df_band["mean"],
        mode="lines",
        line=dict(color="gray", width=2, dash="dash"),
        name="Rolling mean"
    ))

    # actual values with colored markers by zscore
    fig_band.add_trace(go.Scatter(
        x=df_band["DATETIME"],
        y=df_band[price_col_choice],
        mode="markers+lines",
        line=dict(color="rgba(50,50,50,0.3)", width=1),
        marker=dict(
            color=df_band["point_color"],
            size=6,
            line=dict(width=0.5, color="white")
        ),
        name=f"{price_col_choice} actual"
    ))

    # Initial visible window for x-axis
    start_time = df_band["DATETIME"].min()
    end_time   = start_time + pd.Timedelta(minutes=15)

    fig_band.update_layout(
        title=f"ATM {atm_for_band} — {price_col_choice} Deviation Bands (rolling {lookback})",
        xaxis_title="Time",
        yaxis_title=price_col_choice,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            range=[start_time, end_time],
            rangeslider=dict(visible=True),
            type="date"
        ),
        legend=dict(
            orientation="v",
            x=1.01,
            xanchor="left",
            y=1.0
        )
    )

    st.plotly_chart(fig_band, use_container_width=True)

    # Table of "statistical breakout" rows (|z| >= 2σ)
    st.subheader("🚨 Statistical Breakouts (|zscore| ≥ 2σ)")
    breakout_rows = df_band.loc[
        df_band["breakout_flag"],
        ["DATETIME", price_col_choice, "zscore"]
    ]
    if breakout_rows.empty:
        st.write("No ±2σ breakouts detected for this ATM / metric.")
    else:
        st.dataframe(breakout_rows, use_container_width=True)

# --- NEW SECTION: Delta Chart (ACT / SDIF / SPO_DELTA) ---
st.header("📈 Delta Chart (ACT / SDIF / SPO_DELTA)")

needed_cols = ["ACT", "SDIF", "SPO_DELTA"]
have_all = all(col in df.columns for col in needed_cols)

if not have_all:
    st.info("ACT / SDIF / SPO_DELTA not all found in this CSV, skipping delta chart.")
else:
    # use the same ATM as chosen for deviation bands
    df_delta = (
        df[(df["STR"] == df["ATM"]) & (df["ATM"] == atm_for_band)]
        .sort_values("DATETIME")
        .reset_index(drop=True)
        .copy()
    )

    if df_delta.empty:
        st.warning("No rows for this ATM to plot delta chart.")
    else:
        delta_fig = go.Figure()

        # ACT Δ → Red
        delta_fig.add_trace(go.Scatter(
            x=df_delta["DATETIME"],
            y=df_delta["ACT"],
            mode="lines",
            line=dict(width=2, color="red"),
            name="ACT"
        ))

        # SDIF Δ → Blue
        delta_fig.add_trace(go.Scatter(
            x=df_delta["DATETIME"],
            y=df_delta["SDIF"],
            mode="lines",
            line=dict(width=2, color="blue"),
            name="SDIF"
        ))

        # SPO Δ → Green
        delta_fig.add_trace(go.Scatter(
            x=df_delta["DATETIME"],
            y=df_delta["SPO_DELTA"],
            mode="lines",
            line=dict(width=2, color="green"),
            name="SPO Δ"
        ))

        # same initial zoom style: first 15 minutes of this ATM's data
        delta_start = df_delta["DATETIME"].min()
        delta_end   = delta_start + pd.Timedelta(minutes=15)

        delta_fig.update_layout(
            title=f"ATM {atm_for_band} — ACT / SDIF / SPO_DELTA",
            xaxis_title="Time",
            yaxis_title="Delta Values",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(
                range=[delta_start, delta_end],
                rangeslider=dict(visible=True),
                type="date"
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0
            )
        )

        st.plotly_chart(delta_fig, use_container_width=True)
