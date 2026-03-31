"""
dashboard/app.py — Smart Count Tramway
=======================================
Streamlit dashboard for visualising passenger count statistics
stored in the SQLite database produced by main.py.

Run with:
    streamlit run dashboard/app.py

Features:
  • Real-time auto-refresh (configurable interval)
  • KPI cards: total entries, exits, net occupancy
  • Time-series line chart per stop
  • Hourly traffic bar chart (peak hours analysis)
  • Per-stop breakdown table

Author : Smart Count Tramway Team
"""

import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Allow importing src/ from the dashboard sub-folder ───────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.database import Database

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Smart Count Tramway",
    page_icon="🚋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — Industrial Control Room Aesthetic
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Barlow+Condensed:wght@300;400;600;700;800&family=Rajdhani:wght@300;400;500;600;700&display=swap');

/* ── Root tokens ── */
:root {
    --bg-base:        #07090f;
    --bg-panel:       #0d1117;
    --bg-card:        #111820;
    --bg-card-hover:  #162030;
    --border-dim:     #1e2d3d;
    --border-bright:  #1e4060;
    --amber:          #f5a623;
    --amber-dim:      #7a5310;
    --teal:           #00c9a7;
    --teal-dim:       #004d3f;
    --red:            #ff4757;
    --red-dim:        #5a0f17;
    --blue:           #2196f3;
    --blue-dim:       #0a2a4a;
    --text-primary:   #cdd9e5;
    --text-secondary: #6b8098;
    --text-mono:      #7ec8e3;
    --scan-line:      rgba(0,201,167,0.03);
    --glow-teal:      0 0 20px rgba(0,201,167,0.25);
    --glow-amber:     0 0 20px rgba(245,166,35,0.25);
}

/* ── Global reset ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-base) !important;
    font-family: 'Rajdhani', sans-serif;
}

/* Scanline overlay */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg,
        var(--scan-line) 0px,
        transparent 1px,
        transparent 3px
    );
    pointer-events: none;
    z-index: 9999;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-panel) !important;
    border-right: 1px solid var(--border-dim) !important;
}

[data-testid="stSidebar"] > div {
    padding-top: 1.5rem !important;
}

/* Sidebar title */
[data-testid="stSidebar"] h1 {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 800 !important;
    font-size: 1.8rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: var(--teal) !important;
    line-height: 1.1 !important;
}

[data-testid="stSidebar"] .stCaption {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.62rem !important;
    color: var(--text-secondary) !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* Sidebar widgets */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stTextInput label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.68rem !important;
    color: var(--text-secondary) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}

[data-testid="stSidebar"] input,
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-bright) !important;
    color: var(--text-primary) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
}

/* ── Main area ── */
.main .block-container {
    padding: 1.5rem 2rem 2rem !important;
    max-width: 100% !important;
}

/* ── Page title ── */
h1 {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 800 !important;
    font-size: 2.6rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--text-primary) !important;
    margin-bottom: 0 !important;
    line-height: 1 !important;
}

h1 span { color: var(--teal); }

/* ── Subheaders ── */
h2, h3 {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--text-secondary) !important;
    font-size: 0.9rem !important;
    margin-bottom: 0.5rem !important;
    border-bottom: 1px solid var(--border-dim);
    padding-bottom: 0.4rem;
}

/* ── Caption / status bar ── */
.stCaption {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.65rem !important;
    color: var(--text-secondary) !important;
    letter-spacing: 0.06em !important;
}

/* ── Divider ── */
hr {
    border-color: var(--border-dim) !important;
    margin: 1rem 0 !important;
}

/* ── KPI / Metric cards ── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-dim) !important;
    border-radius: 4px !important;
    padding: 1.1rem 1.2rem 0.9rem !important;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, box-shadow 0.2s;
}

[data-testid="stMetric"]:hover {
    border-color: var(--teal) !important;
    box-shadow: var(--glow-teal) !important;
}

/* Corner accent */
[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px;
    height: 100%;
    background: linear-gradient(to bottom, var(--teal), transparent);
}

[data-testid="stMetricLabel"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.62rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    color: var(--text-secondary) !important;
}

[data-testid="stMetricValue"] {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 700 !important;
    font-size: 2.4rem !important;
    color: var(--text-primary) !important;
    line-height: 1.1 !important;
    letter-spacing: 0.04em;
}

[data-testid="stMetricDelta"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.6rem !important;
}

/* ── Charts container ── */
[data-testid="stPlotlyChart"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-dim) !important;
    border-radius: 4px !important;
    padding: 0.5rem !important;
}

/* ── Info boxes ── */
.stInfo {
    background: var(--blue-dim) !important;
    border-left: 3px solid var(--blue) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    color: var(--text-primary) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-dim) !important;
    border-radius: 4px !important;
}

[data-testid="stExpander"] summary {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: var(--text-secondary) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.68rem !important;
}

/* ── Toggle ── */
[data-testid="stToggle"] label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.68rem !important;
    color: var(--text-secondary) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}

/* ── Status indicator (live pulse) ── */
.live-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--teal);
    padding: 3px 10px;
    border: 1px solid var(--teal-dim);
    border-radius: 2px;
    background: var(--teal-dim);
}

.pulse-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--teal);
    animation: pulse 1.4s ease-in-out infinite;
    flex-shrink: 0;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); box-shadow: 0 0 0 0 rgba(0,201,167,0.7); }
    50% { opacity: 0.6; transform: scale(0.85); box-shadow: 0 0 0 4px rgba(0,201,167,0); }
}

/* ── Section label ── */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: var(--text-secondary);
    border-bottom: 1px solid var(--border-dim);
    padding-bottom: 4px;
    margin-bottom: 8px;
}

/* ── Metric accent colors ── */
.metric-entry [data-testid="stMetric"]::before  { background: linear-gradient(to bottom, var(--teal), transparent); }
.metric-exit [data-testid="stMetric"]::before   { background: linear-gradient(to bottom, var(--red), transparent); }
.metric-net [data-testid="stMetric"]::before    { background: linear-gradient(to bottom, var(--amber), transparent); }
.metric-log [data-testid="stMetric"]::before    { background: linear-gradient(to bottom, var(--blue), transparent); }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-bright); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--teal-dim); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — controls
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:0.5rem;">
        <span style="font-size:2.2rem;line-height:1;">🚋</span>
        <div>
            <div style="font-family:'Barlow Condensed',sans-serif;font-weight:800;font-size:1.5rem;
                        letter-spacing:0.15em;text-transform:uppercase;color:#00c9a7;line-height:1.05;">
                Smart Count
            </div>
            <div style="font-family:'Barlow Condensed',sans-serif;font-weight:600;font-size:1.1rem;
                        letter-spacing:0.15em;text-transform:uppercase;color:#6b8098;line-height:1;">
                Tramway
            </div>
        </div>
    </div>
    <div style="font-family:'Space Mono',monospace;font-size:0.6rem;color:#4a6070;
                text-transform:uppercase;letter-spacing:0.08em;margin-bottom:1rem;">
        SETRAM Mostaganem · Edge AI System
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    db_path = st.text_input("Database path", value="data/tramway_counts.db")
    refresh_interval = st.slider("Auto-refresh (seconds)", 5, 120, 30)

    st.divider()
    stop_filter = st.selectbox(
        "Filter by stop",
        options=["All Stops", "Kharouba", "Salamandre", "Gare SNTF",
                 "Nouvelle Gare Routière", "Centre Ville"],
    )

    auto_refresh = st.toggle("Auto-refresh", value=True)

    st.divider()
    # System status block
    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:0.6rem;text-transform:uppercase;
                letter-spacing:0.1em;color:#4a6070;margin-bottom:0.6rem;">System Status</div>
    <div style="display:flex;flex-direction:column;gap:6px;">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="font-family:'Space Mono',monospace;font-size:0.62rem;color:#6b8098;">Camera Feed</span>
            <span style="font-family:'Space Mono',monospace;font-size:0.62rem;color:#00c9a7;">● ACTIVE</span>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="font-family:'Space Mono',monospace;font-size:0.62rem;color:#6b8098;">AI Model</span>
            <span style="font-family:'Space Mono',monospace;font-size:0.62rem;color:#00c9a7;">● RUNNING</span>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="font-family:'Space Mono',monospace;font-size:0.62rem;color:#6b8098;">Database</span>
            <span style="font-family:'Space Mono',monospace;font-size:0.62rem;color:#f5a623;">● SYNCING</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_db(path: str) -> Database:
    """Cache the DB connection so Streamlit doesn't reconnect every rerun."""
    return Database(db_path=path)


def load_data(db: Database, stop: str):
    """Load all relevant data from SQLite."""
    stop_arg = None if stop == "All Stops" else stop
    records = db.get_all_counts(stop_name=stop_arg)
    totals  = db.get_totals_by_stop()
    hourly  = db.get_hourly_traffic(stop_name=stop_arg)
    return records, totals, hourly


db = get_db(db_path)
records, totals, hourly = load_data(db, stop_filter)

# Convert to DataFrames
df_records = pd.DataFrame(
    [
        dict(
            stop=r.stop_name,
            timestamp=pd.to_datetime(r.timestamp),
            entries=r.entries,
            exits=r.exits,
            net=r.net,
        )
        for r in records
    ]
)

df_totals = pd.DataFrame(totals, columns=["Stop", "Total Entries", "Total Exits"])
df_hourly = pd.DataFrame(hourly)

# ─────────────────────────────────────────────────────────────────────────────
# Shared Plotly layout theme
# ─────────────────────────────────────────────────────────────────────────────

PLOT_LAYOUT = dict(
    plot_bgcolor="#111820",
    paper_bgcolor="#111820",
    font=dict(family="Rajdhani, sans-serif", color="#6b8098", size=11),
    margin=dict(l=12, r=12, t=8, b=8),
    xaxis=dict(
        gridcolor="#1a2535",
        linecolor="#1e2d3d",
        tickfont=dict(family="Space Mono", size=9, color="#4a6070"),
        zerolinecolor="#1a2535",
    ),
    yaxis=dict(
        gridcolor="#1a2535",
        linecolor="#1e2d3d",
        tickfont=dict(family="Space Mono", size=9, color="#4a6070"),
        zerolinecolor="#1a2535",
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="#1e2d3d",
        font=dict(family="Space Mono", size=9, color="#6b8098"),
    ),
)

COLOR_ENTRIES = "#00c9a7"
COLOR_EXITS   = "#ff4757"

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

current_time = time.strftime('%H:%M:%S')
current_date = time.strftime('%Y-%m-%d')

header_col, badge_col = st.columns([4, 1])
with header_col:
    st.markdown(f"""
    <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:2px;">
        <h1 style="margin:0;font-family:'Barlow Condensed',sans-serif;font-weight:800;
                   font-size:2.4rem;letter-spacing:0.1em;text-transform:uppercase;color:#cdd9e5;">
            🚋 Smart Count <span style="color:#00c9a7;">Tramway</span>
        </h1>
    </div>
    <div style="font-family:'Space Mono',monospace;font-size:0.62rem;color:#4a6070;
                text-transform:uppercase;letter-spacing:0.1em;margin-top:2px;">
        Mostaganem Network &nbsp;/&nbsp; Showing: <span style="color:#7ec8e3;">{stop_filter}</span>
        &nbsp;/&nbsp; {current_date} &nbsp;
        <span style="color:#f5a623;">{current_time}</span>
    </div>
    """, unsafe_allow_html=True)
with badge_col:
    st.markdown("""
    <div style="display:flex;justify-content:flex-end;padding-top:0.5rem;">
        <div class="live-badge">
            <div class="pulse-dot"></div>
            Live Monitoring
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# KPI Cards
# ─────────────────────────────────────────────────────────────────────────────

total_entries = df_records["entries"].sum() if not df_records.empty else 0
total_exits   = df_records["exits"].sum()   if not df_records.empty else 0
net_occupancy = total_entries - total_exits

st.markdown('<div class="section-label">Key Performance Indicators</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-entry">', unsafe_allow_html=True)
    st.metric(
        label="▲  Total Entries",
        value=f"{total_entries:,}",
        delta=f"+{int(df_records.tail(5)['entries'].sum())} last 5 records" if not df_records.empty else None,
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-exit">', unsafe_allow_html=True)
    st.metric(
        label="▼  Total Exits",
        value=f"{total_exits:,}",
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-net">', unsafe_allow_html=True)
    st.metric(
        label="◈  Net Occupancy",
        value=f"{net_occupancy:,}",
        delta=f"{'⚠ Crowded' if net_occupancy > 50 else '✓ Normal'}",
        delta_color="inverse",
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-log">', unsafe_allow_html=True)
    st.metric(
        label="⊞  Records Logged",
        value=f"{len(df_records):,}",
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Charts — Row 1
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-label">Temporal Analysis</div>', unsafe_allow_html=True)
chart_col1, chart_col2 = st.columns([2, 1])

with chart_col1:
    st.markdown("##### Passenger Flow Over Time")
    if not df_records.empty:
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=df_records["timestamp"], y=df_records["entries"],
            name="Entries", mode="lines",
            line=dict(color=COLOR_ENTRIES, width=2),
            fill="tozeroy",
            fillcolor="rgba(0,201,167,0.07)",
        ))
        fig_line.add_trace(go.Scatter(
            x=df_records["timestamp"], y=df_records["exits"],
            name="Exits", mode="lines",
            line=dict(color=COLOR_EXITS, width=2),
            fill="tozeroy",
            fillcolor="rgba(255,71,87,0.07)",
        ))
        fig_line.update_layout(
            **PLOT_LAYOUT,
            xaxis_title="",
            yaxis_title="",
            height=300,
            legend_title_text="",
        )
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("No data recorded yet. Start main.py to begin counting.")

with chart_col2:
    st.markdown("##### Entry / Exit Split")
    if total_entries + total_exits > 0:
        fig_pie = go.Figure(data=[go.Pie(
            labels=["Entries", "Exits"],
            values=[total_entries, total_exits],
            hole=0.62,
            marker=dict(
                colors=[COLOR_ENTRIES, COLOR_EXITS],
                line=dict(color="#07090f", width=3),
            ),
            textfont=dict(family="Space Mono", size=9),
        )])
        fig_pie.add_annotation(
            text=f"<b>{int(total_entries/(total_entries+total_exits)*100) if (total_entries+total_exits)>0 else 0}%</b><br><span style='font-size:9px'>entries</span>",
            x=0.5, y=0.5,
            font=dict(family="Barlow Condensed", size=20, color="#cdd9e5"),
            showarrow=False,
        )
        fig_pie.update_layout(
            **PLOT_LAYOUT,
            height=300,
            showlegend=True,
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Awaiting data…")

# ─────────────────────────────────────────────────────────────────────────────
# Charts — Row 2
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.markdown('<div class="section-label">Network & Peak Analysis</div>', unsafe_allow_html=True)
chart_col3, chart_col4 = st.columns([1, 1])

with chart_col3:
    st.markdown("##### Hourly Traffic — Peak Hours")
    if not df_hourly.empty:
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=df_hourly["hour"], y=df_hourly["entries"],
            name="Entries",
            marker=dict(color=COLOR_ENTRIES, opacity=0.85, line=dict(color=COLOR_ENTRIES, width=0)),
        ))
        fig_bar.add_trace(go.Bar(
            x=df_hourly["hour"], y=df_hourly["exits"],
            name="Exits",
            marker=dict(color=COLOR_EXITS, opacity=0.85, line=dict(color=COLOR_EXITS, width=0)),
        ))
        fig_bar.update_layout(
            **PLOT_LAYOUT,
            barmode="group",
            bargap=0.25,
            bargroupgap=0.05,
            height=300,
            xaxis_title="Hour of Day",
            yaxis_title="",
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No hourly data yet.")

with chart_col4:
    st.markdown("##### Totals by Stop")
    if not df_totals.empty:
        fig_stops = go.Figure()
        fig_stops.add_trace(go.Bar(
            x=df_totals["Stop"], y=df_totals["Total Entries"],
            name="Entries",
            marker=dict(color=COLOR_ENTRIES, opacity=0.85),
        ))
        fig_stops.add_trace(go.Bar(
            x=df_totals["Stop"], y=df_totals["Total Exits"],
            name="Exits",
            marker=dict(color=COLOR_EXITS, opacity=0.85),
        ))
        stops_layout = {
            **PLOT_LAYOUT,
            "barmode": "group",
            "bargap": 0.25,
            "bargroupgap": 0.05,
            "height": 300,
            "xaxis_title": "",
            "yaxis_title": "",
            "xaxis": {
                **PLOT_LAYOUT["xaxis"],
                "tickangle": -30,
                "tickfont": dict(family="Space Mono", size=8, color="#4a6070"),
            },
        }
        fig_stops.update_layout(**stops_layout)
        st.plotly_chart(fig_stops, use_container_width=True)
    else:
        st.info("No stop data yet.")

# ─────────────────────────────────────────────────────────────────────────────
# Raw Data Table
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
with st.expander("⊞  Raw Count Records — Expand to inspect", expanded=False):
    if not df_records.empty:
        display_df = df_records.sort_values("timestamp", ascending=False).copy()
        display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "stop":      st.column_config.TextColumn("Stop"),
                "timestamp": st.column_config.TextColumn("Timestamp"),
                "entries":   st.column_config.NumberColumn("Entries",  format="%d"),
                "exits":     st.column_config.NumberColumn("Exits",    format="%d"),
                "net":       st.column_config.NumberColumn("Net",      format="%d"),
            },
        )
    else:
        st.info("No records found in the database.")

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="margin-top:2rem;padding-top:1rem;border-top:1px solid #1e2d3d;
            display:flex;justify-content:space-between;align-items:center;">
    <div style="font-family:'Space Mono',monospace;font-size:0.58rem;color:#2a3a4a;
                text-transform:uppercase;letter-spacing:0.1em;">
        Smart Count Tramway · SETRAM Mostaganem · Edge AI Passenger Counting
    </div>
    <div style="font-family:'Space Mono',monospace;font-size:0.58rem;color:#2a3a4a;
                text-transform:uppercase;letter-spacing:0.1em;">
        System v1.0 · Anthropic Edge Pipeline
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Auto-refresh
# ─────────────────────────────────────────────────────────────────────────────

if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()