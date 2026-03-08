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
# Sidebar — controls
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/color/96/tram.png", width=80)
    st.title("Smart Count\nTramway")
    st.caption("SETRAM Mostaganem — Edge AI System")

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
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.title("🚋 Smart Count Tramway — Live Dashboard")
st.caption(f"Mostaganem Tramway Network  •  Showing: **{stop_filter}**  •  Last update: {time.strftime('%H:%M:%S')}")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# KPI Cards
# ─────────────────────────────────────────────────────────────────────────────

total_entries = df_records["entries"].sum() if not df_records.empty else 0
total_exits   = df_records["exits"].sum()   if not df_records.empty else 0
net_occupancy = total_entries - total_exits

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    label="🟢 Total Entries",
    value=f"{total_entries:,}",
    delta=f"+{int(df_records.tail(5)['entries'].sum())} (last 5 records)" if not df_records.empty else None,
)
col2.metric(
    label="🔴 Total Exits",
    value=f"{total_exits:,}",
)
col3.metric(
    label="📊 Net Occupancy",
    value=f"{net_occupancy:,}",
    delta=f"{'Crowded' if net_occupancy > 50 else 'Normal'}",
    delta_color="inverse",
)
col4.metric(
    label="📍 Records Logged",
    value=f"{len(df_records):,}",
)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Charts — Row 1
# ─────────────────────────────────────────────────────────────────────────────

chart_col1, chart_col2 = st.columns([2, 1])

with chart_col1:
    st.subheader("📈 Passenger Flow Over Time")
    if not df_records.empty:
        fig_line = px.line(
            df_records,
            x="timestamp",
            y=["entries", "exits"],
            color_discrete_map={"entries": "#00CC96", "exits": "#EF553B"},
            labels={"value": "Passengers", "timestamp": "Time", "variable": "Direction"},
            title="",
        )
        fig_line.update_layout(
            plot_bgcolor="#0E1117",
            paper_bgcolor="#0E1117",
            font_color="white",
            legend_title_text="",
        )
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("No data recorded yet. Start main.py to begin counting.")

with chart_col2:
    st.subheader("🍩 Entry vs Exit Split")
    if total_entries + total_exits > 0:
        fig_pie = go.Figure(data=[go.Pie(
            labels=["Entries", "Exits"],
            values=[total_entries, total_exits],
            hole=0.5,
            marker_colors=["#00CC96", "#EF553B"],
        )])
        fig_pie.update_layout(
            plot_bgcolor="#0E1117",
            paper_bgcolor="#0E1117",
            font_color="white",
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Awaiting data…")

# ─────────────────────────────────────────────────────────────────────────────
# Charts — Row 2
# ─────────────────────────────────────────────────────────────────────────────

chart_col3, chart_col4 = st.columns([1, 1])

with chart_col3:
    st.subheader("⏰ Hourly Traffic (Peak Hours)")
    if not df_hourly.empty:
        fig_bar = px.bar(
            df_hourly,
            x="hour",
            y=["entries", "exits"],
            barmode="group",
            color_discrete_map={"entries": "#00CC96", "exits": "#EF553B"},
            labels={"hour": "Hour of Day", "value": "Passengers"},
            title="",
        )
        fig_bar.update_layout(
            plot_bgcolor="#0E1117",
            paper_bgcolor="#0E1117",
            font_color="white",
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No hourly data yet.")

with chart_col4:
    st.subheader("📍 Totals by Stop")
    if not df_totals.empty:
        fig_stops = px.bar(
            df_totals,
            x="Stop",
            y=["Total Entries", "Total Exits"],
            barmode="group",
            color_discrete_map={"Total Entries": "#00CC96", "Total Exits": "#EF553B"},
        )
        fig_stops.update_layout(
            plot_bgcolor="#0E1117",
            paper_bgcolor="#0E1117",
            font_color="white",
        )
        st.plotly_chart(fig_stops, use_container_width=True)
    else:
        st.info("No stop data yet.")

# ─────────────────────────────────────────────────────────────────────────────
# Raw Data Table
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
with st.expander("🗃️  Raw Count Records", expanded=False):
    if not df_records.empty:
        st.dataframe(
            df_records.sort_values("timestamp", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No records found in the database.")

# ─────────────────────────────────────────────────────────────────────────────
# Auto-refresh
# ─────────────────────────────────────────────────────────────────────────────

if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
