# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.graph_objects as go
from io import StringIO

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="NIFTY50 Breakout Detector", layout="wide")
st.title("🕵️‍♂️ NIFTY50 Option-Chain Breakout Detector v2")

# =========================
# FILE UPLOAD
# =========================
uploaded = st.file_uploader("Upload NIFTY50 option-chain CSV", type="csv")

if not uploaded:
    st.info("Please upload your CSV file.")
    st.stop()

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(uploaded, parse_dates=["DATETIME"])
df["index"] = df.index  # preserve original index ref

# sanity checks
required_cols = [
    "DATETIME", "ATM", "STR", "SPO",
    "CE", "PE",
    "CE_OI", "PE_OI",
    "CE_PREV_OI", "PE_PREV_OI",
    "STRADDLE", "STRADDLE_DELTA"
]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    st.error(f"Missing required columns in CSV: {missing_cols}")
    st.stop()

# =========================
# FIND VALID ATMs
# =========================
valid_atm = sorted(df.loc[df["STR"] == df["ATM"], "STR"].unique())
if not valid_atm:
    st.error("No rows where STR == ATM found. Check your CSV.")
    st.stop()

# =========================
# SIDEBAR CONTROLS
# =========================
with st.sidebar:
    st.header("Detection Parameters")

    # Which ATM strikes to analyze?
    atm_options = ["All"] + [str(x) for x in valid_atm]
    atm_value = st.selectbox("ATM Strike (for detection)", atm_options, index=0)

    # breakout rules
    window_minutes = st.slider("Detection Window (minutes)", 1, 60, value=5)
    threshold_points = st.slider("Directional Threshold (points ↑ in CE or PE)", 1, 100, value=15)

    # straddle breakout rules (IV / vol spike)
    straddle_threshold = st.slider("Straddle Breakout Threshold (points ↑ in CE+PE/STRADDLE)", 1, 150, value=25)

    st.markdown("You can still view any ATM after detection in the main panel.")


# =========================
# HELPER FUNCTIONS
# =========================

def classify_move(ce_move, pe_move, ce_change_oi, pe_change_oi):
    """
    Return human-readable breakout label.
    We consider who is attacking and how.
    """

    ce_hit = ce_move >= threshold_points
    pe_hit = pe_move >= threshold_points

    if ce_hit and pe_hit:
        base = "VOLATILITY SPIKE (CE & PE)"
    elif ce_hit:
        base = "CALL SIDE SPIKE"
    elif pe_hit:
        base = "PUT SIDE SPIKE"
    else:
        base = "NONE"

    # interpret OI context
    # CE side
    ce_context = ""
    if ce_hit:
        if ce_change_oi is not None:
            if ce_change_oi > 0:
                ce_context = "CE Long Build / Aggressive Call Buying"
            elif ce_change_oi < 0:
                ce_context = "CE Short Cover / Call Writers Exiting"
    # PE side
    pe_context = ""
    if pe_hit:
        if pe_change_oi is not None:
            if pe_change_oi > 0:
                pe_context = "PE Long Build / Aggressive Put Buying (Bearish)"
            elif pe_change_oi < 0:
                pe_context = "PE Short Cover / Put Writers Exiting (Bullish)"

    context_bits = [x for x in [ce_context, pe_context] if x]
    context_txt = " | ".join(context_bits)

    if context_txt:
        return f"{base} → {context_txt}"
    return base


def detect_breakouts_for_atm(df, atm_strike, window_minutes, threshold_points, straddle_threshold):
    """
    Core detection:
    - directional breakouts (CE or PE jumps >= threshold)
    - straddle breakouts (STRADDLE jump >= straddle_threshold)
    Returns summaries (list[dict]) and union of all breakout rows (DataFrame).
    """

    # rows where STR==ATM and ATM==atm_strike
    df_atm = (
        df[(df["STR"] == df["ATM"]) & (df["ATM"] == atm_strike)]
        .loc[:, [
            "index","DATETIME","SPO",
            "CE","PE",
            "CE_OI","PE_OI",
            "CE_PREV_OI","PE_PREV_OI",
            "STRADDLE","STRADDLE_DELTA"
        ]]
        .sort_values("DATETIME")
        .set_index("DATETIME")
    )

    if df_atm.empty:
        return [], pd.DataFrame([])

    times = list(df_atm.index)
    w = timedelta(minutes=window_minutes)

    summaries = []
    segs = []

    i = 0
    while i < len(times):
        start_t = times[i]
        start_row = df_atm.loc[start_t]

        ce0 = start_row["CE"]
        pe0 = start_row["PE"]
        spo0 = start_row["SPO"]
        ce_oi0 = start_row["CE_OI"]
        pe_oi0 = start_row["PE_OI"]

        straddle0 = start_row["STRADDLE"]

        end_t_limit = start_t + w
        win = df_atm.loc[start_t:end_t_limit]

        # price moves
        ce_moves = win["CE"] - ce0
        pe_moves = win["PE"] - pe0

        # straddle move
        straddle_moves = win["STRADDLE"] - straddle0

        # hits
        hits_ce = ce_moves[ce_moves >= threshold_points].index
        hits_pe = pe_moves[pe_moves >= threshold_points].index
        hits_straddle = straddle_moves[straddle_moves >= straddle_threshold].index

        all_hits = hits_ce.union(hits_pe).union(hits_straddle).sort_values()

        if len(all_hits) > 0:
            hit_t = all_hits[0]
            hit_row = df_atm.loc[hit_t]

            ce1 = hit_row["CE"]
            pe1 = hit_row["PE"]
            spo1 = hit_row["SPO"]
            ce_oi1 = hit_row["CE_OI"]
            pe_oi1 = hit_row["PE_OI"]

            straddle1 = hit_row["STRADDLE"]

            # compute deltas
            ce_move = ce1 - ce0
            pe_move = pe1 - pe0
            spot_move = spo1 - spo0
            ce_oi_delta = (ce_oi1 - ce_oi0) if (pd.notnull(ce_oi1) and pd.notnull(ce_oi0)) else None
            pe_oi_delta = (pe_oi1 - pe_oi0) if (pd.notnull(pe_oi1) and pd.notnull(pe_oi0)) else None
            straddle_move = straddle1 - straddle0

            # classify
            move_type = classify_move(ce_move, pe_move, ce_oi_delta, pe_oi_delta)

            # mark whether this was straddle-vol breakout
            vol_spike = straddle_move >= straddle_threshold

            summaries.append({
                "ATM": atm_strike,
                "start_time": start_t,
                "end_time": hit_t,

                "start_pos": start_row["index"],
                "end_pos": hit_row["index"],

                "start_CE": ce0,
                "end_CE": ce1,
                "CE_move": ce_move,

                "start_PE": pe0,
                "end_PE": pe1,
                "PE_move": pe_move,

                "start_SPO": spo0,
                "end_SPO": spo1,
                "SPO_move": spot_move,

                "CE_OI_start": ce_oi0,
                "CE_OI_end": ce_oi1,
                "CE_OI_delta": ce_oi_delta,

                "PE_OI_start": pe_oi0,
                "PE_OI_end": pe_oi1,
                "PE_OI_delta": pe_oi_delta,

                "STRADDLE_start": straddle0,
                "STRADDLE_end": straddle1,
                "STRADDLE_move": straddle_move,
                "VolatilitySpike": vol_spike,

                "Signal": move_type,

                "window_min": window_minutes,
                "threshold_points": threshold_points,
                "straddle_threshold": straddle_threshold,
            })

            segs.append(df_atm.loc[start_t:hit_t])
            i = times.index(hit_t) + 1
        else:
            i += 1

    if not segs:
        return [], pd.DataFrame([])

    df_seg = pd.concat(segs).drop_duplicates().sort_index()
    return summaries, df_seg


def red_if_negative(val):
    try:
        return "color: red" if val < 0 else ""
    except:
        return ""


def build_ohlc(df_timeindexed, col, freq):
    """
    Build OHLC resample for a given column.
    """
    ohlc = (
        df_timeindexed[col]
        .resample(freq)
        .agg(["first","max","min","last"])
        .dropna()
    )
    ohlc.columns = ["open","high","low","close"]
    return ohlc


def add_breakout_vrects(fig, breakout_windows, freq_index=False):
    """
    breakout_windows: list of (start_time, end_time)
    We add shaded regions to the plot so you can visually see where breakout fired.
    If freq_index=True, we assume the x-axis is already resampled,
    so we clip to that index range by nearest timestamps.
    """
    for (s,e) in breakout_windows:
        fig.add_vrect(
            x0=s, x1=e,
            fillcolor="rgba(255,165,0,0.15)",  # light orange translucent
            line_width=0,
            layer="below"
        )
    return fig


# =========================
# DETECTION PIPELINE
# =========================

# figure out which ATMs to detect on
if atm_value == "All":
    atm_list_for_detection = valid_atm
else:
    # atm_value comes as string from selectbox
    # convert to same dtype as df["ATM"] if possible
    try:
        chosen_val = float(atm_value)
        # attempt int-like if safe
        if chosen_val.is_integer():
            chosen_val = int(chosen_val)
        atm_list_for_detection = [chosen_val]
    except:
        atm_list_for_detection = [atm_value]

all_summaries = []
all_segments = []

for atm_strike in atm_list_for_detection:
    summaries, seg = detect_breakouts_for_atm(
        df,
        atm_strike=atm_strike,
        window_minutes=window_minutes,
        threshold_points=threshold_points,
        straddle_threshold=straddle_threshold
    )
    if summaries:
        all_summaries.extend(summaries)
    if not seg.empty:
        all_segments.append(seg)

if not all_segments:
    st.warning("No breakouts detected with these settings.")
    st.stop()

df_summary = pd.DataFrame(all_summaries).sort_values("start_time")
df_break_union = pd.concat(all_segments).drop_duplicates()
# map back to full original rows by index
df_full_breakouts = df[df["index"].isin(df_break_union["index"])].sort_values("DATETIME")

# store in session (optional, but nice for later)
st.session_state["df_summary"] = df_summary
st.session_state["df_full_breakouts"] = df_full_breakouts


# =========================
# POST-DETECTION FILTERS / DOWNLOADS
# =========================

st.subheader("🔎 Breakout Summary")

# Let user focus visualization on a specific ATM after detection
atm_choices_after = ["All"] + [str(x) for x in sorted(df_summary["ATM"].unique())]
atm_view = st.selectbox("Filter view to ATM", atm_choices_after, index=0)

if atm_view != "All":
    try:
        atm_view_val = float(atm_view)
        if atm_view_val.is_integer():
            atm_view_val = int(atm_view_val)
    except:
        atm_view_val = atm_view
    df_summary_view = df_summary[df_summary["ATM"] == atm_view_val].copy()
    df_full_view = df_full_breakouts[df_full_breakouts["ATM"] == atm_view_val].copy()
else:
    df_summary_view = df_summary.copy()
    df_full_view = df_full_breakouts.copy()

# Style + show summary
styled = (
    df_summary_view.style
    .format({
        "start_CE":"{:.2f}", "end_CE":"{:.2f}", "CE_move":"{:.2f}",
        "start_PE":"{:.2f}", "end_PE":"{:.2f}", "PE_move":"{:.2f}",
        "start_SPO":"{:.2f}", "end_SPO":"{:.2f}", "SPO_move":"{:.2f}",
        "CE_OI_start":"{:.0f}", "CE_OI_end":"{:.0f}", "CE_OI_delta":"{:.0f}",
        "PE_OI_start":"{:.0f}", "PE_OI_end":"{:.0f}", "PE_OI_delta":"{:.0f}",
        "STRADDLE_start":"{:.2f}", "STRADDLE_end":"{:.2f}", "STRADDLE_move":"{:.2f}",
        "window_min":"{:.0f}", "threshold_points":"{:.0f}", "straddle_threshold":"{:.0f}"
    })
    .applymap(red_if_negative, subset=["CE_move","PE_move","SPO_move","CE_OI_delta","PE_OI_delta"])
)
st.dataframe(styled, use_container_width=True)

col_dl1, col_dl2 = st.columns(2)
with col_dl1:
    csv_summary = df_summary_view.to_csv(index=False)
    st.download_button(
        label="⬇️ Download Breakout Summary CSV",
        data=csv_summary,
        file_name="breakout_summary.csv",
        mime="text/csv"
    )
with col_dl2:
    csv_full = df_full_view.to_csv(index=False)
    st.download_button(
        label="⬇️ Download Breakout Rows CSV",
        data=csv_full,
        file_name="breakout_rows.csv",
        mime="text/csv"
    )

# =========================
# SHOW RAW BREAKOUT ROWS
# =========================
st.subheader("📈 Breakout Segments (full original data)")
st.dataframe(df_full_view, use_container_width=True)

# =========================
# CANDLESTICK VISUALS
# =========================

# timeframe selector row
st.header("📊 Candlesticks")

tf_map = {
    "5 sec":  "5S",
    "30 sec": "30S",
    "1 min":  "1T",
    "5 min":  "5T",
    "15 min": "15T"
}
timeframe_label = st.selectbox("Timeframe", list(tf_map.keys()), index=2)
freq = tf_map[timeframe_label]

# We will plot using ONLY df_full_view (the ATM filter result),
# because otherwise mixing multiple ATM ranges can be confusing.
df_plot = df_full_view.copy().set_index("DATETIME").sort_index()

# Build OHLC sets
ohlc_spo = build_ohlc(df_plot, "SPO", freq)
ohlc_pe  = build_ohlc(df_plot, "PE", freq)
ohlc_ce  = build_ohlc(df_plot, "CE", freq)

# Build breakout shading windows for chart overlay
# We'll take each row of df_summary_view and create (start_time, end_time)
break_windows = [
    (row["start_time"], row["end_time"])
    for _, row in df_summary_view.iterrows()
]

# SPO chart
st.subheader("SPO (Spot Index)")
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
    margin=dict(l=20,r=20,t=30,b=20),
    title_text=None
)
fig_spo = add_breakout_vrects(fig_spo, break_windows)
st.plotly_chart(fig_spo, use_container_width=True)

# PE chart
st.subheader("PE (Put Premium)")
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
    margin=dict(l=20,r=20,t=30,b=20),
    title_text=None
)
fig_pe = add_breakout_vrects(fig_pe, break_windows)
st.plotly_chart(fig_pe, use_container_width=True)

# CE chart
st.subheader("CE (Call Premium)")
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
    margin=dict(l=20,r=20,t=30,b=20),
    title_text=None
)
fig_ce = add_breakout_vrects(fig_ce, break_windows)
st.plotly_chart(fig_ce, use_container_width=True)

# =========================
# NOTES / RECAP
# =========================
st.markdown(
"""
**How to read this dashboard:**

- **Signal**  
  - CALL SIDE SPIKE = CE jumped fast  
  - PUT SIDE SPIKE = PE jumped fast  
  - VOLATILITY SPIKE (CE & PE) = both jumped / straddle expansion  

- **OI_delta meaning**  
  - CE_OI_delta > 0 + CE_move up → new call longs (bullish aggression)  
  - CE_OI_delta < 0 + CE_move up → call writers covering (bullish squeeze)  
  - PE_OI_delta > 0 + PE_move up → new put longs (bearish aggression)  
  - PE_OI_delta < 0 + PE_move up → put writers covering (bullish relief)

- **SPO_move**  
  Confirms if the underlying actually moved with the option spike
  (real momentum) vs just IV jump.

- **Orange shaded areas on charts**  
  Those are detected breakout windows.
"""
)
