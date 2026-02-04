import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium import plugins
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time
from sklearn.linear_model import Ridge
import branca.colormap as cm

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="ManKaaval: Illegal Sand Mining Detection",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        font-family: Inter, sans-serif;
    }
    h1, h2, h3 {
        color: #ffffff;
    }
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        color: #00d4ff;
    }
    [data-testid="stMetricLabel"] {
        color: #aaaaaa;
    }
    [data-testid="stMetricDelta"] {
        display: none;
    }
    table thead tr th:first-child {
        display: none;
    }
    table tbody th {
        display: none;
    }
    .risk-card {
        background: linear-gradient(135deg, #1e1e1e 0%, #2a2a2a 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #ff4444;
        margin-bottom: 10px;
    }
    .info-card {
        background-color: #1a1a1a;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #333;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'selected_site' not in st.session_state:
    st.session_state.selected_site = None
if 'last_click_coords' not in st.session_state:
    st.session_state.last_click_coords = None

# ============================================================================
# DATA LOADING WITH CACHING
# ============================================================================
@st.cache_data
def load_data():
    """Load predictions and time-series data from CSV files"""
    try:
        # Load baseline features with predictions
        baseline_path = "dashboardData/baseline_features_predictions.csv"
        if not os.path.exists(baseline_path):
            st.error(f"❌ Baseline features file not found at: {baseline_path}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        df = pd.read_csv(baseline_path)

        # Validate required columns
        required_cols = ['id', 'lat', 'lon', 'probability']
        if not all(col in df.columns for col in required_cols):
            st.error(f"❌ Dataset missing required columns: {required_cols}")
            st.stop()

        # Clean data - remove NaN values
        df = df.dropna(subset=['lat', 'lon', 'probability'])

        # Load time-series data
        ts_path = "dashboardData/timeseries_features.csv"
        ts_df = pd.DataFrame()

        if os.path.exists(ts_path):
            ts_df = pd.read_csv(ts_path)
            ts_df['date'] = pd.to_datetime(ts_df['date'])
        else:
            st.warning(f"⚠️ Time-series file not found at: {ts_path}")

        # Load SHAP values
        shap_path = "dashboardData/shap_values_baseline_features.csv"
        shap_df = pd.DataFrame()
        if os.path.exists(shap_path):
            shap_df = pd.read_csv(shap_path)
        else:
            st.warning(f"⚠️ SHAP values file not found at: {shap_path}")

        # Load SCA-ready sites CSV
        sca_ready_path = "dashboardData/sca_ready_sites.csv"
        sca_ready_df = pd.DataFrame()
        if os.path.exists(sca_ready_path):
            sca_ready_df = pd.read_csv(sca_ready_path)
            st.success(f"✅ Loaded {len(sca_ready_df):,} SCA-ready sites")
        else:
            st.warning(f"⚠️ SCA-ready sites file not found.")

        return df, ts_df, shap_df, sca_ready_df

    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        import traceback
        st.code(traceback.format_exc())
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


# ============================================================================
# FOLIUM MAP CREATION WITH 1KM GRID
# ============================================================================

def get_risk_color(probability):
    """Map probability to color gradient (green -> yellow -> red)"""
    if probability < 0.33:
        # Green to Yellow
        r = int(255 * (probability / 0.33))
        g = 255
        b = 0
    elif probability < 0.67:
        # Yellow to Orange
        r = 255
        g = int(255 * (1 - (probability - 0.33) / 0.34))
        b = 0
    else:
        # Orange to Red
        r = 255
        g = int(255 * (1 - (probability - 0.67) / 0.33) * 0.5)
        b = 0
    
    return f'#{r:02x}{g:02x}{b:02x}'


def create_folium_grid_map(df, sca_ready_df=None):
    """Create a Folium map with 1km grid cells showing mining risk"""
    
    if df.empty:
        # Default center (Bihar, India)
        m = folium.Map(
            location=[25.5, 85.5],
            zoom_start=7,
            tiles='OpenStreetMap'
        )
        return m
    
    # Calculate map center
    center_lat = df['lat'].mean()
    center_lon = df['lon'].mean()
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles='OpenStreetMap',
        control_scale=True
    )
    
    # Add satellite imagery layer
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite Imagery',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Create feature groups for layer control
    grid_layer = folium.FeatureGroup(name='Mining Risk Grid (1km)', show=True)
    sca_layer = folium.FeatureGroup(name=f'SCA-Ready Sites ({len(sca_ready_df) if sca_ready_df is not None and not sca_ready_df.empty else 0})', show=True)
    
    # Create color scale legend
    colormap = cm.LinearColormap(
        colors=['#00ff00', '#ffff00', '#ff0000'],
        vmin=0,
        vmax=1,
        caption='Mining Risk Probability'
    )
    colormap.add_to(m)
    
    # Add 1km grid cells for each site
    offset = 0.0045  # Approximate 1km in degrees
    
    for idx, row in df.iterrows():
        lat = row['lat']
        lon = row['lon']
        prob = row['probability']
        site_id = row['id']
        
        # Create 1km x 1km square
        bounds = [
                [lat - offset, lon - offset],  # SW corner
                [lat + offset, lon + offset],  # NE corner
            ]
        
        # Determine color based on probability
        fill_color = get_risk_color(prob)
        
        # Determine risk level
        if prob > 0.7:
            risk_level = "🔴 HIGH RISK"
        elif prob > 0.4:
            risk_level = "🟡 MODERATE RISK"
        else:
            risk_level = "🟢 LOW RISK"
        
        # Create popup with site information
        popup_html = f"""
        <div style="font-family: Arial; width: 200px;">
            <h4 style="margin: 0 0 10px 0; color: #333;">Site {site_id}</h4>
            <p style="margin: 5px 0;"><b>Risk Level:</b> {risk_level}</p>
            <p style="margin: 5px 0;"><b>Probability:</b> {prob*100:.1f}%</p>
            <p style="margin: 5px 0;"><b>Coordinates:</b></p>
            <p style="margin: 5px 0; font-size: 11px;">Lat: {lat:.4f}, Lon: {lon:.4f}</p>
            <p style="margin: 10px 0 0 0; font-size: 11px; color: #666;">Click for detailed analysis</p>
        </div>
        """
        
        # Add rectangle to grid layer
        folium.Rectangle(
            bounds=bounds,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"Site {site_id}: {prob*100:.1f}%",
            color='#333333',
            weight=1,
            fill=True,
            fillColor=fill_color,
            fillOpacity=0.6,
            # Store site data in the rectangle for click detection
        ).add_to(grid_layer)
        
        # Add marker at center for easier clicking
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"Site {site_id}",
            color='white',
            fillColor='white',
            fillOpacity=0.8,
            weight=1
        ).add_to(grid_layer)
    
    # Add SCA-ready sites with green outlines
    if sca_ready_df is not None and not sca_ready_df.empty:
        for idx, row in sca_ready_df.iterrows():
            lat = row['lat']
            lon = row['lon']
            site_id = row['id']
            timepoints = row.get('timepoint_count', 0)
            
            # Create 1km x 1km square outline
            bounds = [
                [lat - offset, lon - offset],  # SW corner
                [lat + offset, lon + offset],  # NE corner
            ]
            
            popup_html = f"""
            <div style="font-family: Arial; width: 200px;">
                <h4 style="margin: 0 0 10px 0; color: #00ff00;">🔬 SCA-Ready Site</h4>
                <p style="margin: 5px 0;"><b>Site ID:</b> {site_id}</p>
                <p style="margin: 5px 0;"><b>Time Points:</b> {timepoints}</p>
                <p style="margin: 10px 0 0 0; font-size: 11px; color: #666;">
                    Sufficient data for Synthetic Control Analysis
                </p>
            </div>
            """
            
            # Add green outline
            folium.Rectangle(
                bounds=bounds,
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"SCA-Ready: {site_id}",
                color='#00ff00',
                weight=3,
                fill=False,
                opacity=1.0
            ).add_to(sca_layer)
    
    # Add layers to map
    grid_layer.add_to(m)
    sca_layer.add_to(m)
    
    # Add layer control
    folium.LayerControl(position='topright', collapsed=False).add_to(m)
    
    # Add fullscreen button
    plugins.Fullscreen(
        position='topleft',
        title='Fullscreen',
        title_cancel='Exit Fullscreen',
        force_separate_button=True
    ).add_to(m)
    
    # Add mouse position display
    plugins.MousePosition().add_to(m)
    
    return m


# ============================================================================
# SITE SELECTION LOGIC
# ============================================================================
def find_nearest_site(click_lat, click_lon, df, threshold=0.006):
    """Find nearest site to clicked location"""
    if df.empty:
        return None

    coords = df[['lat', 'lon']].values
    click_point = np.array([click_lat, click_lon])
    distances = np.linalg.norm(coords - click_point, axis=1)
    nearest_idx = np.argmin(distances)
    min_distance = distances[nearest_idx]

    nearest_site = df.iloc[nearest_idx]
    print(f"Click: ({click_lat:.4f}, {click_lon:.4f})")
    print(f"Nearest: {nearest_site['id']} at ({nearest_site['lat']:.4f}, {nearest_site['lon']:.4f})")
    print(f"Distance: {min_distance:.6f}° ({min_distance*111:.1f}km)")
    print(f"Probability: {nearest_site['probability']:.3f}")

    if min_distance < threshold:
        return nearest_site

    return None


# ============================================================================
# SHAP VISUALIZATION
# ============================================================================
def create_shap_plot(site_id, shap_df):
    """Create a bar chart showing feature importance/attribution for a specific site"""
    site_shap = shap_df[shap_df['id'] == site_id].copy()
    if site_shap.empty:
        return None

    cols_to_drop = ['id', 'lat', 'lon', 'probability', 'prediction']
    features_shap = site_shap.drop(columns=[c for c in cols_to_drop if c in site_shap.columns])

    plot_data = features_shap.melt(var_name='Feature', value_name='SHAP Value')
    plot_data = plot_data.sort_values(by='SHAP Value', ascending=True)

    plot_data['color'] = plot_data['SHAP Value'].apply(lambda x: '#ff4444' if x > 0 else '#00d4ff')

    fig = go.Figure(go.Bar(
        x=plot_data['SHAP Value'],
        y=plot_data['Feature'],
        orientation='h',
        marker_color=plot_data['color'],
        hovertemplate='<b>%{y}</b><br>Contribution: %{x:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title='Feature Attribution (SHAP)',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30,30,30,0.5)',
        font=dict(color='white', size=10),
        margin=dict(l=10, r=10, t=30, b=10),
        height=300,
        xaxis_title='Impact on Prediction',
        yaxis_gridcolor='rgba(128,128,128,0.2)'
    )

    return fig


# ============================================================================
# SYNTHETIC CONTROL ANALYSIS
# ============================================================================
def perform_sca_analysis(site_id, baseline_df, ts_df):
    """Perform SITE-SPECIFIC Synthetic Control Analysis"""
    
    if "sca_debug_counter" not in st.session_state:
        st.session_state.sca_debug_counter = 0
        st.session_state.sca_last_site = None
        st.session_state.sca_last_time = None

    st.session_state.sca_debug_counter += 1
    st.session_state.sca_last_site = site_id
    st.session_state.sca_last_time = time.strftime("%H:%M:%S")

    print(f"[SCA DEBUG] Run #{st.session_state.sca_debug_counter} for site {site_id}")
    
    if ts_df.empty or baseline_df.empty:
        return None, None
    
    ts_df = ts_df.copy()
    ts_df['date'] = pd.to_datetime(ts_df['date'])
    
    # Get time series for treated site
    ts_treated = ts_df[ts_df['id'] == site_id].copy()
    
    if ts_treated.empty:
        print(f"[SCA] No time series data for site {site_id}")
        return None, None
    
    print(f"[SCA] Treated site {site_id}: {len(ts_treated)} time points")
    
    # Find control sites
    control_site_ids = baseline_df[
        (baseline_df['probability'] < 0.4) & 
        (baseline_df['id'] != site_id)
    ]['id'].unique()
    
    if len(control_site_ids) == 0:
        print("[SCA] No control sites found")
        return None, None
    
    print(f"[SCA] Found {len(control_site_ids)} control sites")
    
    # Get control time series
    ts_control = ts_df[ts_df['id'].isin(control_site_ids)].copy()
    
    if ts_control.empty:
        print("[SCA] No control time series")
        return None, None
    
    # Clean data
    ts_treated_clean = ts_treated[['date', 'NDVI', 'MNDWI', 'BSI']].dropna()
    ts_control_clean = ts_control[['date', 'NDVI', 'MNDWI', 'BSI']].dropna()
    
    # Aggregate controls
    control_agg = ts_control_clean.groupby('date')[['NDVI', 'MNDWI', 'BSI']].mean().reset_index()
    control_agg = control_agg.rename(columns={
        'NDVI': 'control_NDVI', 
        'MNDWI': 'control_MNDWI', 
        'BSI': 'control_BSI'
    })
    
    # Rename treated columns
    treated_agg = ts_treated_clean[['date', 'NDVI', 'MNDWI', 'BSI']].rename(columns={
        'NDVI': 'treated_NDVI',
        'MNDWI': 'treated_MNDWI',
        'BSI': 'treated_BSI'
    })
    
    # Merge
    df_sca = treated_agg.merge(control_agg, on='date', how='inner')
    
    if len(df_sca) < 5:
        print(f"[SCA] Insufficient data points: {len(df_sca)}")
        return None, None
    
    print(f"[SCA] Merged data: {len(df_sca)} dates")
    
    # Define pre/post treatment
    median_date = df_sca['date'].median()
    df_sca['period'] = df_sca['date'].apply(lambda x: 'PRE' if x < median_date else 'POST')
    
    df_pre = df_sca[df_sca['period'] == 'PRE'].copy()
    df_post = df_sca[df_sca['period'] == 'POST'].copy()
    
    if len(df_pre) < 3 or len(df_post) < 3:
        print(f"[SCA] Insufficient pre/post data: {len(df_pre)}/{len(df_post)}")
        return None, None
    
    # Fit synthetic control
    metrics = ['NDVI', 'MNDWI', 'BSI']
    weights = {}
    
    for metric in metrics:
        try:
            X = df_pre[[f'control_{metric}']].values
            y = df_pre[[f'treated_{metric}']].values
            
            model = Ridge(alpha=0.01, fit_intercept=True)
            model.fit(X, y)
            
            weight = model.coef_.flatten()[0]
            weights[metric] = weight
            
            print(f"[SCA] {metric}: weight={weight:.4f}")
            
        except Exception as e:
            print(f"[SCA] Error fitting {metric}: {e}")
            weights[metric] = 1.0
    
    # Create synthetic counterfactual
    for metric in metrics:
        df_sca[f'synthetic_{metric}'] = weights[metric] * df_sca[f'control_{metric}']
        df_sca[f'effect_{metric}'] = df_sca[f'treated_{metric}'] - df_sca[f'synthetic_{metric}']
    
    # Prepare results
    results_dict = {}
    for metric in metrics:
        metric_res = pd.DataFrame({
            'date': df_sca['date'],
            'actual': df_sca[f'treated_{metric}'],
            'synthetic': df_sca[f'synthetic_{metric}']
        })
        results_dict[metric] = metric_res
    
    control_ids = list(control_site_ids)[:5]
    
    return control_ids, results_dict


def create_sca_plots(sca_results_dict):
    """Create three subplots for SCA (NDVI, MNDWI, BSI)"""
    if not sca_results_dict:
        return None

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            'NDVI (Vegetation) - Actual vs Synthetic',
            'MNDWI (Water) - Actual vs Synthetic',
            'BSI (Soil) - Actual vs Synthetic'
        )
    )

    metrics = ['NDVI', 'MNDWI', 'BSI']
    colors = {'NDVI': '#44ff44', 'MNDWI': '#00d4ff', 'BSI': '#ffaa00'}

    for i, metric in enumerate(metrics, 1):
        if metric not in sca_results_dict:
            continue
        
        res = sca_results_dict[metric]
        
        # Actual line
        fig.add_trace(
            go.Scatter(
                x=res['date'],
                y=res['actual'],
                name=f'{metric} (Actual)',
                line=dict(color=colors[metric], width=2),
                hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Actual:</b> %{y:.3f}<extra></extra>'
            ),
            row=i, col=1
        )
        
        # Synthetic line
        fig.add_trace(
            go.Scatter(
                x=res['date'],
                y=res['synthetic'],
                name=f'{metric} (Synthetic)',
                line=dict(color='white', dash='dash', width=1.5),
                hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>Synthetic:</b> %{y:.3f}<extra></extra>'
            ),
            row=i, col=1
        )

    fig.update_layout(
        height=700,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30,30,30,0.5)',
        font=dict(color='white', size=11),
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode='x unified'
    )

    fig.update_xaxes(gridcolor='rgba(128,128,128,0.2)', showgrid=True, title_text='Date', row=3, col=1)
    fig.update_yaxes(gridcolor='rgba(128,128,128,0.2)', showgrid=True)

    return fig


# ============================================================================
# MAIN APP
# ============================================================================

# Title and header
st.markdown("# 🛰️ ManKaaval: Illegal Sand Mining Detection")
st.markdown("Real-Time Satellite Surveillance System | IRIS 2025")

# Load data
df, ts_df, shap_df, sca_ready_df = load_data()

# Create layout
col_left, col_right = st.columns([3, 2])

# Display statistics in sidebar
if not df.empty:
    st.sidebar.markdown("### 📊 Dataset Statistics")
    st.sidebar.metric("Total Sites", f"{len(df):,}")
    st.sidebar.metric("High Risk Sites", f"{len(df[df['probability'] > 0.7]):,}")
    st.sidebar.metric("Average Risk", f"{df['probability'].mean()*100:.1f}%")
    
    if not sca_ready_df.empty:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔬 SCA Analysis Status")
        st.sidebar.metric("SCA-Ready Sites", f"{len(sca_ready_df):,}")
        if len(df) > 0:
            st.sidebar.caption(f"{(len(sca_ready_df)/len(df)*100):.1f}% have sufficient time-series data")

# ============================================================================
# LEFT COLUMN: INTERACTIVE MAP
# ============================================================================

with col_left:
    st.markdown("## 🗺️ Interactive Risk Map (1km Grid)")
    st.markdown("*Click on any grid cell to view detailed analysis and trends*")
    
    if not sca_ready_df.empty:
        st.caption("🟢 **Green outlined sites** have sufficient time-series data for SCA analysis")

    # Create map (cached)
    if 'folium_map' not in st.session_state or st.session_state.get('force_map_refresh', False):
        with st.spinner("🗺️ Loading map and predictions..."):
            st.session_state.folium_map = create_folium_grid_map(
                df, 
                sca_ready_df if not sca_ready_df.empty else None
            )
        st.session_state.force_map_refresh = False

    # Render map
    map_output = st_folium(
        st.session_state.folium_map,
        width="100%",
        height=600,
        returned_objects=["last_clicked"],
        key="mankaaval_folium_map"
    )

    # Handle click events
    if map_output and map_output.get("last_clicked"):
        click_lat = map_output["last_clicked"]["lat"]
        click_lon = map_output["last_clicked"]["lng"]

        last_click = st.session_state.get('last_click_coords')
        current_click = (click_lat, click_lon)

        if last_click != current_click:
            st.session_state.last_click_coords = current_click

            selected = find_nearest_site(click_lat, click_lon, df)

            if selected is not None:
                st.session_state.selected_site = selected
            else:
                st.session_state.selected_site = None
                st.warning("⚠️ No mining site found at this location. Click closer to a grid cell.")


# ============================================================================
# RIGHT COLUMN: SITE ANALYSIS
# ============================================================================
with col_right:
    
    if st.session_state.selected_site is None:
        # Global Overview
        st.markdown("## 📊 Global Overview")
        if df.empty:
            st.info("No data loaded. Please check file paths.")
        else:
            total_sites = len(df)
            high_risk = df[df['probability'] > 0.7]
            high_risk_count = len(high_risk)
            avg_risk = df['probability'].mean()

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Sites", f"{total_sites:,}")
            with col2:
                st.metric("High Risk", f"{high_risk_count:,}", 
                         delta=f"{(high_risk_count/total_sites)*100:.1f}%", 
                         delta_color="inverse")

            st.metric("Average Risk", f"{avg_risk*100:.1f}%")

            st.markdown("---")
            st.markdown("### 📈 Risk Distribution")
            fig_dist = px.histogram(df, x='probability', nbins=30, 
                                   color_discrete_sequence=['#00d4ff'])
            fig_dist.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(30,30,30,0.5)',
                font_color='white',
                height=250,
                margin=dict(l=0, r=0, t=10, b=0),
                showlegend=False
            )
            st.plotly_chart(fig_dist, use_container_width=True)

            st.markdown("---")
            st.markdown("### 🎯 Top Risk Sites")
            top_5 = df.nlargest(5, 'probability')[['id', 'probability']]
            for idx, row in top_5.iterrows():
                risk_pct = row['probability'] * 100
                color = '#ff4444' if risk_pct > 80 else '#ffaa00' if risk_pct > 60 else '#ffff00'
                st.markdown(
                    f'<div class="risk-card"><b>{row["id"]}</b>: <span style="color:{color}">{risk_pct:.1f}%</span></div>',
                    unsafe_allow_html=True
                )

    else:
        # Selected Site Analysis
        site = st.session_state.selected_site
        site_id = site['id']
        site_lat = site['lat']
        site_lon = site['lon']
        site_prob = site['probability']
        risk_pct = site_prob * 100
        risk_color = '#ff4444' if risk_pct > 70 else '#ffaa00' if risk_pct > 40 else '#44ff44'

        # Risk Score Card
        st.markdown(
            f'''
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #1a1a1a, #2a2a2a); border-radius: 15px; border: 2px solid {risk_color};">
                <p style="color: #aaa; margin: 0; font-size: 0.9rem;">RISK PROBABILITY</p>
                <h1 style="color: {risk_color}; margin: 10px 0; font-size: 3rem;">{risk_pct:.1f}%</h1>
                <p style="color: {risk_color}; margin: 0; font-weight: bold;">
                    {"🔴 HIGH RISK" if risk_pct > 70 else "🟡 MODERATE" if risk_pct > 40 else "🟢 LOW RISK"}
                </p>
            </div>
            ''',
            unsafe_allow_html=True
        )

        st.markdown("---")

        # SHAP Attribution
        st.markdown("### 🎯 Why This Risk?")
        if not shap_df.empty:
            shap_fig = create_shap_plot(site_id, shap_df)
            if shap_fig:
                st.plotly_chart(shap_fig, use_container_width=True, 
                               key=f"shap_plot_{site_id}")
                st.caption("Positive values (red) increase mining detection likelihood. Negative values (blue) decrease it.")
            else:
                st.info("No SHAP attribution data for this site.")
        else:
            st.info("SHAP attribution data not loaded.")

        st.markdown("---")

        # Synthetic Control Analysis
        st.markdown("### 🔬 Causal Verification (Synthetic Control Analysis)")
        
        with st.spinner("Running Synthetic Control Analysis..."):
            control_ids, sca_results = perform_sca_analysis(site_id, df, ts_df)
            
            if "sca_debug_counter" in st.session_state:
                st.caption(
                    f"SCA recomputed **{st.session_state.sca_debug_counter}** times. "
                    f"Last run for site **{st.session_state.sca_last_site}** "
                    f"at **{st.session_state.sca_last_time}**."
                )
            
            if control_ids and sca_results:
                st.success(f"✅ Comparing treated site against {len(control_ids)} control sites")
                
                sca_fig = create_sca_plots(sca_results)
                if sca_fig:
                    st.plotly_chart(sca_fig, use_container_width=True, 
                                   key=f"sca_plot_{site_id}")
                    st.caption("White dashed line shows the synthetic counterfactual. Divergence indicates mining-related changes.")
                else:
                    st.warning("Could not generate SCA visualization.")
            else:
                st.warning("⚠️ Insufficient data for synthetic control analysis.")
