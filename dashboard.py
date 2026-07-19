# ============================================
# MULTIMODAL URBAN MOBILITY KPI DASHBOARD
# Real-time Anomaly Detection System
# Deeksha Renukaprasad — MSc Data Science
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from sklearn.ensemble import IsolationForest
import requests
import datetime
import time

# ============================================
# PAGE CONFIG — MUST BE FIRST
# ============================================

st.set_page_config(
    page_title="NYC Multimodal Mobility Dashboard",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# FULL LIGHT MODE CSS
# ============================================

st.markdown("""
<style>
    /* Full app background */
    .stApp {
        background-color: #F0F2F6;
        color: #1a1a2e;
    }

    /* Main content */
    .main .block-container {
        background-color: #F0F2F6;
        padding: 1.5rem 2rem;
        max-width: 100%;
    }

    /* Sidebar dark navy */
    section[data-testid="stSidebar"] {
        background-color: #1a1a2e !important;
    }
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #ccc !important;
        font-size: 13px;
    }
    section[data-testid="stSidebar"] h2 {
        color: white !important;
        font-size: 16px !important;
    }

    /* KPI metric cards */
    div[data-testid="metric-container"] {
        background: white;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        border-top: 4px solid #2980b9;
    }
    div[data-testid="metric-container"] label {
        color: #888 !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="metric-container"] 
    div[data-testid="stMetricValue"] > div {
        color: #1a1a2e !important;
        font-size: 28px !important;
        font-weight: 700 !important;
    }
    div[data-testid="metric-container"] 
    div[data-testid="stMetricDelta"] {
        font-size: 12px !important;
    }

    /* Section headers */
    .section-header {
        font-size: 15px;
        font-weight: 600;
        color: #1a1a2e;
        margin: 8px 0 12px 0;
        padding-bottom: 6px;
        border-bottom: 2px solid #2980b9;
        display: inline-block;
    }

    /* Chart card wrapper */
    .chart-card {
        background: white;
        border-radius: 14px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 16px;
    }

    /* Anomaly alert cards */
    .anomaly-high {
        background: white;
        border-left: 5px solid #e74c3c;
        border-radius: 10px;
        padding: 12px 15px;
        margin-bottom: 10px;
        box-shadow: 0 2px 6px rgba(231,76,60,0.12);
    }
    .anomaly-medium {
        background: white;
        border-left: 5px solid #f39c12;
        border-radius: 10px;
        padding: 12px 15px;
        margin-bottom: 10px;
        box-shadow: 0 2px 6px rgba(243,156,18,0.12);
    }
    .anomaly-title {
        font-weight: 700;
        font-size: 13px;
        color: #1a1a2e;
        margin-bottom: 4px;
    }
    .anomaly-detail {
        font-size: 12px;
        color: #555;
        line-height: 1.7;
    }
    .anomaly-cause {
        font-size: 12px;
        color: #2980b9;
        font-style: italic;
        margin-top: 4px;
    }

    /* Info summary cards */
    .info-card {
        background: white;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        font-size: 13px;
        color: #333;
        line-height: 1.9;
    }
    .info-card h4 {
        color: #1a1a2e;
        font-size: 14px;
        margin-bottom: 8px;
        font-weight: 600;
    }

    /* Status badge */
    .status-ok {
        background: #e8f8f0;
        color: #27ae60;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 13px;
        display: inline-block;
    }
    .status-alert {
        background: #fdf0f0;
        color: #e74c3c;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 13px;
        display: inline-block;
    }

    /* Main title */
    .main-title {
        font-size: 30px;
        font-weight: 800;
        color: #1a1a2e;
        margin-bottom: 2px;
    }
    .main-subtitle {
        font-size: 13px;
        color: #888;
        margin-bottom: 20px;
    }

    /* Divider */
    .divider {
        height: 1px;
        background: #dde1e7;
        margin: 18px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# DATA LOADING
# ============================================

@st.cache_data(ttl=3600)
def load_taxi_data():
    try:
        df = pd.read_csv(
            '/Users/deeksharprasad/thesis_project/outputs/taxi_full_year_anomalies.csv'
        )
        df['pickup_date'] = pd.to_datetime(df['pickup_date'])
        return df
    except Exception as e:
        st.error(f"Error loading taxi data: {e}")
        return None

@st.cache_data(ttl=3600)
def load_weather_data():
    try:
        df = pd.read_csv(
            '/Users/deeksharprasad/thesis_project/data/nyc_weather_2023.csv'
        )
        df['pickup_date'] = pd.to_datetime(df['pickup_date'])
        return df
    except Exception as e:
        return None

taxi_data = load_taxi_data()
weather_data = load_weather_data()

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.markdown("## 🏙️ Dashboard Controls")
    st.markdown("---")

    selected_modes = st.multiselect(
        "Transport Modes",
        ["🚕 Taxi", "🚲 Citi Bike", "🚇 MTA Subway"],
        default=["🚕 Taxi", "🚲 Citi Bike", "🚇 MTA Subway"]
    )

    st.markdown("### 📅 Date Range")
    date_from = st.date_input(
        "From",
        value=datetime.date(2023, 1, 1),
        min_value=datetime.date(2023, 1, 1),
        max_value=datetime.date(2023, 12, 31)
    )
    date_to = st.date_input(
        "To",
        value=datetime.date(2023, 12, 31),
        min_value=datetime.date(2023, 1, 1),
        max_value=datetime.date(2023, 12, 31)
    )

    st.markdown("### ⚙️ Detection Settings")
    threshold = st.slider(
        "Anomaly Sensitivity (Z-score)",
        min_value=1.5,
        max_value=3.5,
        value=2.5,
        step=0.1,
        help="Lower = more sensitive"
    )

    auto_refresh = st.toggle("⚡ Auto-refresh (hourly)", value=False)

    st.markdown("---")
    st.markdown("**Thesis:** MSc Data Science")
    st.markdown("**Author:** Deeksha Renukaprasad")
    st.markdown("**Supervisor:** Talha Ali Khan")
    st.markdown(
        f"**Updated:** {datetime.datetime.now().strftime('%d %b %Y, %H:%M')}"
    )

# ============================================
# MAIN HEADER
# ============================================

st.markdown(
    '<p class="main-title">🚕 NYC Multimodal Urban Mobility — KPI Dashboard</p>',
    unsafe_allow_html=True
)
st.markdown(
    '<p class="main-subtitle">Real-time anomaly detection and causal attribution '
    'across road, bike-share, and transit modes · Full Year 2023</p>',
    unsafe_allow_html=True
)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ============================================
# FILTER DATA
# ============================================

if taxi_data is not None:
    filtered = taxi_data[
        (taxi_data['pickup_date'] >= pd.Timestamp(date_from)) &
        (taxi_data['pickup_date'] <= pd.Timestamp(date_to))
    ].copy()

    filtered['zscore'] = stats.zscore(filtered['trip_count'])
    filtered['is_anomaly'] = filtered['zscore'].abs() > threshold

    total_trips = filtered['trip_count'].sum()
    avg_daily = filtered['trip_count'].mean()
    anomaly_count = int(filtered['is_anomaly'].sum())
    anomaly_rate = (anomaly_count / len(filtered) * 100) if len(filtered) > 0 else 0
    avg_temp = weather_data['temp_mean'].mean() if weather_data is not None else 0

# ============================================
# ROW 1 — KPI METRICS
# ============================================

st.markdown('<p class="section-header">📊 Key Performance Indicators</p>',
            unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "🚕 Total Taxi Trips",
        f"{total_trips/1e6:.1f}M",
        delta=f"Full year 2023"
    )
with col2:
    st.metric(
        "📅 Avg Daily Trips",
        f"{avg_daily:,.0f}",
        delta=f"per day"
    )
with col3:
    st.metric(
        "🚨 Anomalies",
        f"{anomaly_count}",
        delta=f"{anomaly_rate:.1f}% of days",
        delta_color="inverse"
    )
with col4:
    st.metric(
        "🌡️ Avg Temperature",
        f"{avg_temp:.1f}°C",
        delta="NYC 2023"
    )
with col5:
    status_html = (
        '<span class="status-alert">🔴 Active Alerts</span>'
        if anomaly_count > 0
        else '<span class="status-ok">🟢 All Normal</span>'
    )
    st.markdown(f"**System Status**<br>{status_html}",
                unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ============================================
# ROW 2 — DEMAND CHART + ANOMALY ALERTS
# ============================================

col_chart, col_alerts = st.columns([2.2, 1])

with col_chart:
    st.markdown('<p class="section-header">📈 Multimodal Demand — Full Year 2023</p>',
                unsafe_allow_html=True)

    if taxi_data is not None:
        normal = filtered[~filtered['is_anomaly']]
        anomalies = filtered[filtered['is_anomaly']]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=filtered['pickup_date'],
            y=filtered['trip_count'],
            mode='lines',
            name='Daily Trips',
            line=dict(color='#2980b9', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(41,128,185,0.08)',
            hovertemplate='<b>%{x|%b %d}</b><br>Trips: %{y:,.0f}<extra></extra>'
        ))

        if len(anomalies) > 0:
            fig.add_trace(go.Scatter(
                x=anomalies['pickup_date'],
                y=anomalies['trip_count'],
                mode='markers',
                name='Anomaly',
                marker=dict(
                    color='#e74c3c',
                    size=11,
                    symbol='circle',
                    line=dict(color='white', width=1.5)
                ),
                hovertemplate=(
                    '<b>%{x|%b %d}</b><br>'
                    'Trips: %{y:,.0f}<br>'
                    '<b style="color:red">⚠ ANOMALY</b><extra></extra>'
                )
            ))

        mean_val = filtered['trip_count'].mean()
        fig.add_hline(
            y=mean_val,
            line_dash="dot",
            line_color="#95a5a6",
            line_width=1.5,
            annotation_text=f"Mean: {mean_val:,.0f}",
            annotation_font_size=11,
            annotation_font_color="#666"
        )

        fig.update_layout(
            height=380,
            paper_bgcolor='white',
            plot_bgcolor='white',
            xaxis=dict(
                showgrid=True,
                gridcolor='#f0f0f0',
                title='',
                tickfont=dict(size=11, color='#666')
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='#f0f0f0',
                title='Daily Trips',
                tickfont=dict(size=11, color='#666'),
                tickformat=',.0f'
            ),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1,
                font=dict(size=12)
            ),
            hovermode='x unified',
            margin=dict(l=10, r=10, t=10, b=10)
        )

        st.plotly_chart(fig, use_container_width=True)

with col_alerts:
    st.markdown('<p class="section-header">🚨 Anomaly Alert Panel</p>',
                unsafe_allow_html=True)

    if taxi_data is not None:
        confirmed = filtered[filtered['is_anomaly']].sort_values(
            'zscore', key=abs, ascending=False
        )

        if len(confirmed) == 0:
            st.markdown("""
            <div style="background:white;border-radius:12px;padding:20px;
                        text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.06);">
                <div style="font-size:32px;margin-bottom:8px">✅</div>
                <div style="color:#27ae60;font-weight:600;font-size:14px">
                    No anomalies detected
                </div>
                <div style="color:#888;font-size:12px;margin-top:4px">
                    All modes operating normally
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for _, row in confirmed.iterrows():
                causal = row.get('causal_label', 'Under investigation')
                zscore_val = row['zscore']
                severity = "🔴 High" if abs(zscore_val) > 3.5 else "🟡 Medium"
                card_class = "anomaly-high" if abs(zscore_val) > 3.5 else "anomaly-medium"
                border_col = "#e74c3c" if abs(zscore_val) > 3.5 else "#f39c12"

                st.markdown(f"""
                <div style="background:white;border-left:5px solid {border_col};
                            border-radius:10px;padding:13px 15px;margin-bottom:10px;
                            box-shadow:0 2px 6px rgba(0,0,0,0.07);">
                    <div style="font-weight:700;font-size:13px;color:#1a1a2e;
                                margin-bottom:5px;">{severity}</div>
                    <div style="font-size:12px;color:#444;line-height:1.8;">
                        📅 {row['pickup_date'].strftime('%b %d, %Y')}<br>
                        🚕 {int(row['trip_count']):,} trips<br>
                        📊 Z-score: {zscore_val:.2f}<br>
                    </div>
                    <div style="font-size:12px;color:#2980b9;
                                font-style:italic;margin-top:5px;">
                        🔍 {causal}
                    </div>
                </div>
                """, unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ============================================
# ROW 3 — WEATHER + SHAP
# ============================================

col_weather, col_shap = st.columns(2)

with col_weather:
    st.markdown('<p class="section-header">🌦️ Weather Impact on Demand</p>',
                unsafe_allow_html=True)

    if taxi_data is not None and weather_data is not None:
        merged = filtered.merge(weather_data, on='pickup_date', how='inner')

        fig_w = px.scatter(
            merged,
            x='temp_mean',
            y='trip_count',
            color='precip_mm',
            color_continuous_scale='Blues',
            labels={
                'temp_mean': 'Temperature (°C)',
                'trip_count': 'Daily Trips',
                'precip_mm': 'Precipitation (mm)'
            },
            hover_data={
                'pickup_date': '|%b %d',
                'temp_mean': ':.1f',
                'trip_count': ':,.0f',
                'precip_mm': ':.1f'
            }
        )
        fig_w.update_traces(
            marker=dict(size=7, opacity=0.7, line=dict(width=0.5, color='white'))
        )
        fig_w.update_layout(
            height=320,
            paper_bgcolor='white',
            plot_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='#f0f0f0',
                       tickfont=dict(size=11, color='#666')),
            yaxis=dict(showgrid=True, gridcolor='#f0f0f0',
                       tickfont=dict(size=11, color='#666'),
                       tickformat=',.0f'),
            coloraxis_colorbar=dict(
                title='Precip<br>(mm)',
                tickfont=dict(size=10)
            ),
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_w, use_container_width=True)

with col_shap:
    st.markdown('<p class="section-header">🧠 SHAP Causal Attribution</p>',
                unsafe_allow_html=True)

    features = [
        'Snowfall (mm)', 'Temperature (°C)', 'Month',
        'Day of week', 'Week number', 'Trip count',
        'Precipitation (mm)', 'Is weekend'
    ]
    importance = [
        0.1783, 0.1790, 0.1971,
        0.2069, 0.2395, 0.2427,
        0.4444, 0.4507
    ]
    colors = [
        '#27ae60' if i < 0.25
        else '#f39c12' if i < 0.35
        else '#e74c3c'
        for i in importance
    ]

    fig_shap = go.Figure(go.Bar(
        x=importance,
        y=features,
        orientation='h',
        marker=dict(color=colors),
        hovertemplate='%{y}<br>SHAP: %{x:.4f}<extra></extra>'
    ))
    fig_shap.update_layout(
        xaxis_title='Mean |SHAP value|',
        height=320,
        paper_bgcolor='white',
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='#f0f0f0',
                   tickfont=dict(size=11, color='#666')),
        yaxis=dict(tickfont=dict(size=11, color='#333')),
        margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig_shap, use_container_width=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ============================================
# ROW 4 — MONTHLY TRENDS + CORRELATION
# ============================================

col_monthly, col_corr = st.columns(2)

with col_monthly:
    st.markdown('<p class="section-header">📅 Monthly Demand Trends</p>',
                unsafe_allow_html=True)

    if taxi_data is not None:
        monthly = taxi_data.copy()
        monthly['month_num'] = monthly['pickup_date'].dt.month
        monthly['month'] = monthly['pickup_date'].dt.strftime('%b')
        monthly_agg = monthly.groupby(
            ['month', 'month_num']
        )['trip_count'].mean().reset_index()
        monthly_agg = monthly_agg.sort_values('month_num')

        fig_m = go.Figure(go.Bar(
            x=monthly_agg['month'],
            y=monthly_agg['trip_count'],
            marker=dict(
                color=monthly_agg['trip_count'],
                colorscale=[[0, '#aed6f1'], [1, '#1a5276']],
                showscale=False
            ),
            hovertemplate='%{x}<br>Avg: %{y:,.0f} trips<extra></extra>'
        ))
        fig_m.update_layout(
            height=300,
            paper_bgcolor='white',
            plot_bgcolor='white',
            yaxis=dict(
                title='Avg Daily Trips',
                showgrid=True,
                gridcolor='#f0f0f0',
                tickformat=',.0f',
                tickfont=dict(size=11, color='#666')
            ),
            xaxis=dict(tickfont=dict(size=11, color='#666')),
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_m, use_container_width=True)

with col_corr:
    st.markdown('<p class="section-header">🔄 Cross-Modal Correlation</p>',
                unsafe_allow_html=True)

    modes = ['Taxi', 'Citi Bike', 'MTA']
    corr_matrix = [
        [1.000, -0.118,  0.067],
        [-0.118, 1.000, -0.104],
        [0.067, -0.104,  1.000]
    ]

    fig_c = go.Figure(go.Heatmap(
        z=corr_matrix,
        x=modes,
        y=modes,
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=[[f'{v:.3f}' for v in row] for row in corr_matrix],
        texttemplate='<b>%{text}</b>',
        textfont=dict(size=14),
        showscale=True,
        colorbar=dict(tickfont=dict(size=10))
    ))
    fig_c.update_layout(
        height=300,
        paper_bgcolor='white',
        plot_bgcolor='white',
        xaxis=dict(tickfont=dict(size=12, color='#333')),
        yaxis=dict(tickfont=dict(size=12, color='#333')),
        margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig_c, use_container_width=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ============================================
# ROW 5 — SUMMARY CARDS
# ============================================

st.markdown('<p class="section-header">📋 Analysis Summary</p>',
            unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card">
        <h4>📦 Dataset</h4>
        🚕 Taxi: 37.2M trips (2023)<br>
        🚲 Citi Bike: 988k trips (2023)<br>
        🚇 MTA: 235M riders (Mar–Dec)<br>
        🌦️ Weather: 365 days (NOAA)<br>
        📅 Date range: Jan–Dec 2023
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h4>🤖 Models</h4>
        📊 Z-score baseline (|z| > 2.5)<br>
        🌲 Isolation Forest (5% contamination)<br>
        🧠 SHAP explainability (8 features)<br>
        🔗 Cross-modal correlation analysis<br>
        ✅ Precision / Recall / F1 evaluated
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="info-card">
        <h4>🔍 Key Findings</h4>
        🌧️ Precipitation = 2nd top SHAP feature<br>
        📉 Taxi–Bike correlation: r = −0.118<br>
        🌀 Tropical Storm Ophelia: Sep 22–24<br>
        🎄 Christmas Day anomaly: Dec 25<br>
        🚨 {anomaly_count} confirmed anomalies detected
    </div>
    """, unsafe_allow_html=True)

# ============================================
# AUTO REFRESH
# ============================================

if auto_refresh:
    time.sleep(3600)
    st.rerun()