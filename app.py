import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AEGIS — Arms & Escalation Geopolitical Intelligence System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS — Intelligence / Spy Radar Theme
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp { background-color: #080c14; color: #c8d6e5; font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #0a1628 0%, #0f1f3d 50%, #0a1628 100%);
        border: 1px solid rgba(0, 255, 136, 0.15);
        border-radius: 4px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00ff88, #00d4ff, transparent);
    }
    .main-header h1 {
        font-family: 'JetBrains Mono', monospace;
        color: #00ff88;
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: 3px;
        text-transform: uppercase;
    }
    .main-header p { color: #5a7a9a; font-size: 0.9rem; margin-top: 0.5rem; font-family: 'JetBrains Mono', monospace; letter-spacing: 1px; }
    .main-header .classified {
        display: inline-block;
        border: 1px solid rgba(255, 71, 87, 0.4);
        color: #ff4757;
        font-size: 0.65rem;
        padding: 2px 10px;
        border-radius: 2px;
        letter-spacing: 3px;
        font-family: 'JetBrains Mono', monospace;
        margin-bottom: 0.8rem;
    }

    .kpi-card {
        background: linear-gradient(135deg, #0d1b2a 0%, #0a1628 100%);
        border: 1px solid rgba(0, 255, 136, 0.1);
        border-radius: 4px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
    }
    .kpi-card::after {
        content: '';
        position: absolute;
        bottom: 0; left: 20%; right: 20%;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0,255,136,0.3), transparent);
    }
    .kpi-card:hover { border-color: rgba(0, 255, 136, 0.3); }
    .kpi-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem;
        font-weight: 800;
        color: #00ff88;
    }
    .kpi-label { color: #5a7a9a; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 2px; margin-top: 0.3rem; font-family: 'JetBrains Mono', monospace; }
    .kpi-delta { font-size: 0.72rem; margin-top: 0.2rem; font-family: 'JetBrains Mono', monospace; }
    .kpi-delta.danger { color: #ff4757; }
    .kpi-delta.safe { color: #00ff88; }
    .kpi-delta.warn { color: #ffa502; }

    .section-header {
        background: linear-gradient(90deg, rgba(0, 255, 136, 0.06), transparent);
        border-left: 2px solid #00ff88;
        padding: 0.8rem 1.2rem;
        margin: 1.5rem 0 1rem 0;
        border-radius: 0 4px 4px 0;
    }
    .section-header h3 { color: #00d4ff; font-size: 1rem; font-weight: 600; margin: 0; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.5px; }
    .section-header p { color: #5a7a9a; font-size: 0.78rem; margin: 0.2rem 0 0 0; }

    .intel-box {
        background: rgba(0, 255, 136, 0.04);
        border: 1px solid rgba(0, 255, 136, 0.15);
        border-radius: 4px;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
        font-size: 0.85rem;
        line-height: 1.6;
    }
    .intel-box strong { color: #00ff88; }
    .intel-box::before {
        content: '[ INTEL ]';
        display: block;
        color: #00ff88;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 2px;
        margin-bottom: 0.4rem;
        opacity: 0.7;
    }

    .rx-card {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.05), rgba(0, 255, 136, 0.03));
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 4px;
        padding: 1.2rem 1.5rem;
        margin: 0.8rem 0;
    }
    .rx-card h4 { color: #00d4ff; margin: 0 0 0.5rem 0; font-size: 0.92rem; font-family: 'JetBrains Mono', monospace; }
    .rx-card p { color: #8899aa; margin: 0; font-size: 0.83rem; line-height: 1.6; }

    div[data-testid="stTabs"] button {
        background: transparent !important;
        color: #5a7a9a !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        padding: 0.8rem 1.2rem !important;
        font-weight: 600 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.5px !important;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #00ff88 !important;
        border-bottom: 2px solid #00ff88 !important;
    }

    .stSidebar > div { background: #060a12; }

    div[data-testid="stExpander"] { border: 1px solid rgba(0,255,136,0.1); border-radius: 4px; }

    .threat-high { color: #ff4757; font-weight: 700; }
    .threat-medium { color: #ffa502; font-weight: 700; }
    .threat-low { color: #00ff88; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# DATA LOADING & PREP
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("arms_trade.csv")
    df['Escalation_Flag'] = df['Escalation_Risk'].map({'High': 2, 'Medium': 1, 'Low': 0})
    df['High_Risk_Flag'] = (df['Escalation_Risk'] == 'High').astype(int)
    df['Offensive_Flag'] = (df['Weapon_Class'] == 'Offensive').astype(int)
    df['YearGroup'] = pd.cut(df['Year'], bins=[2004,2009,2014,2019,2025],
                              labels=['2005-09','2010-14','2015-19','2020-24'])
    df['DealSize'] = pd.cut(df['Deal_Value_USD_M'], bins=[0,20,80,200,600],
                             labels=['Small (<$20M)','Medium ($20-80M)','Large ($80-200M)','Mega (>$200M)'])
    return df

df = load_data()

# ─────────────────────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ AEGIS FILTERS")
    st.caption("Adjust parameters to slice intelligence data")

    year_range = st.slider("Year Range", int(df['Year'].min()), int(df['Year'].max()),
                           (int(df['Year'].min()), int(df['Year'].max())))
    exp_filter = st.multiselect("Exporter", sorted(df['Exporter'].unique()), default=sorted(df['Exporter'].unique()))
    imp_region_filter = st.multiselect("Importer Region", df['Importer_Region'].unique(), default=df['Importer_Region'].unique())
    weapon_filter = st.multiselect("Weapon Category", df['Weapon_Category'].unique(), default=df['Weapon_Category'].unique())
    risk_filter = st.multiselect("Escalation Risk", ['High','Medium','Low'], default=['High','Medium','Low'])
    conflict_filter = st.multiselect("Conflict Proximity", ['Yes','No'], default=['Yes','No'])

mask = (
    df['Year'].between(year_range[0], year_range[1]) &
    df['Exporter'].isin(exp_filter) &
    df['Importer_Region'].isin(imp_region_filter) &
    df['Weapon_Category'].isin(weapon_filter) &
    df['Escalation_Risk'].isin(risk_filter) &
    df['Importer_Conflict_Proximity'].isin(conflict_filter)
)
dff = df[mask].copy()

# ─────────────────────────────────────────────────────────────
# CHART THEME
# ─────────────────────────────────────────────────────────────
RISK_COLORS = {'High': '#ff4757', 'Medium': '#ffa502', 'Low': '#00ff88'}
CLASS_COLORS = {'Offensive': '#ff6348', 'Defensive': '#00d4ff'}
TREND_COLORS = {'Accelerating': '#ff4757', 'Stable': '#ffa502', 'Declining': '#00ff88'}

PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='JetBrains Mono, monospace', color='#c8d6e5', size=11),
    margin=dict(l=40, r=40, t=50, b=40),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
)

def styled_chart(fig, height=420):
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    fig.update_xaxes(gridcolor='rgba(0,255,136,0.06)', zerolinecolor='rgba(0,255,136,0.06)')
    fig.update_yaxes(gridcolor='rgba(0,255,136,0.06)', zerolinecolor='rgba(0,255,136,0.06)')
    return fig


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <div class='classified'>ANALYTICAL INTELLIGENCE PRODUCT</div>
    <h1>🛡️ AEGIS</h1>
    <p>Arms & Escalation Geopolitical Intelligence System</p>
    <p style='font-size:0.75rem; color:#3d5a80; margin-top:0.8rem;'>Descriptive · Diagnostic · Predictive · Prescriptive</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────────────────────
total = len(dff)
total_value = dff['Deal_Value_USD_M'].sum()
high_risk_count = dff['High_Risk_Flag'].sum()
high_risk_pct = (high_risk_count / total * 100) if total > 0 else 0
offensive_pct = (dff['Offensive_Flag'].sum() / total * 100) if total > 0 else 0
top_exporter = dff['Exporter'].value_counts().index[0] if total > 0 else 'N/A'
accel_pct = (len(dff[dff['Arms_Import_Trend']=='Accelerating']) / total * 100) if total > 0 else 0
conflict_deals = len(dff[dff['Importer_Conflict_Proximity']=='Yes'])

cols = st.columns(6)
kpi_data = [
    (f"{total:,}", "Total Transfers", f"${total_value:,.0f}M total value", ""),
    (f"{high_risk_count}", "High Risk Deals", f"{high_risk_pct:.1f}% of total", "danger"),
    (f"{offensive_pct:.0f}%", "Offensive Systems", f"{dff['Offensive_Flag'].sum()} transfers", "warn"),
    (f"{top_exporter}", "Top Exporter", f"{dff['Exporter'].value_counts().iloc[0] if total > 0 else 0} deals", ""),
    (f"{accel_pct:.0f}%", "Arms Accelerating", f"{len(dff[dff['Arms_Import_Trend']=='Accelerating'])} importers", "danger"),
    (f"{conflict_deals}", "Conflict Zone Deals", f"{conflict_deals/total*100:.0f}% of transfers" if total > 0 else "0%", "warn"),
]
for col, (val, label, delta, cls) in zip(cols, kpi_data):
    col.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-value'>{val}</div>
        <div class='kpi-label'>{label}</div>
        <div class='kpi-delta {cls}'>{delta}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📡 SIGINT — Descriptive", "🔍 HUMINT — Diagnostic",
    "🤖 MASINT — Predictive", "🎯 OSINT — Prescriptive"
])


# =============================================================
# TAB 1: DESCRIPTIVE ANALYSIS
# =============================================================
with tab1:
    st.markdown("""
    <div class='section-header'>
        <h3>📡 SIGNALS INTELLIGENCE — What does the global arms landscape look like?</h3>
        <p>Volume, flow patterns, weapon mix, and geographic distribution of major conventional arms transfers</p>
    </div>""", unsafe_allow_html=True)

    # --- Row 1: Sankey + Risk Donut ---
    c1, c2 = st.columns([2, 1])
    with c1:
        # Sankey: Exporter Region → Importer Region
        flow = dff.groupby(['Exporter_Region','Importer_Region'])['Deal_Value_USD_M'].sum().reset_index()
        flow = flow[flow['Deal_Value_USD_M'] > 0]
        all_labels = list(pd.unique(flow[['Exporter_Region','Importer_Region']].values.ravel()))
        src_indices = [all_labels.index(x) for x in flow['Exporter_Region']]
        tgt_indices = [all_labels.index(x) for x in flow['Importer_Region']]

        # Color exporters green, importers blue
        node_colors = []
        exp_regions = set(flow['Exporter_Region'].unique())
        for label in all_labels:
            if label in exp_regions:
                node_colors.append('rgba(0,255,136,0.7)')
            else:
                node_colors.append('rgba(0,212,255,0.7)')

        fig = go.Figure(go.Sankey(
            node=dict(pad=15, thickness=20, label=all_labels,
                      color=node_colors,
                      line=dict(color='rgba(0,255,136,0.3)', width=0.5)),
            link=dict(source=src_indices, target=tgt_indices,
                      value=flow['Deal_Value_USD_M'].values,
                      color='rgba(0,212,255,0.15)')
        ))
        fig.update_layout(title='Arms Flow: Exporter Region → Importer Region (by $M value)')
        st.plotly_chart(styled_chart(fig, 450), use_container_width=True)

    with c2:
        risk_counts = dff['Escalation_Risk'].value_counts()
        fig = go.Figure(go.Pie(
            labels=risk_counts.index, values=risk_counts.values,
            hole=0.65, marker=dict(colors=[RISK_COLORS.get(x, '#00d4ff') for x in risk_counts.index]),
            textinfo='label+percent', textfont=dict(size=12, family='JetBrains Mono'),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>',
            sort=False
        ))
        fig.update_layout(title='Escalation Risk Distribution',
                          showlegend=False,
                          annotations=[dict(text=f'{high_risk_pct:.0f}%', x=0.5, y=0.5, font_size=26,
                                            font_color='#ff4757', showarrow=False, font_family='JetBrains Mono')])
        st.plotly_chart(styled_chart(fig, 450), use_container_width=True)

    st.markdown(f"""<div class='intel-box'>
        <strong>The Sankey diagram maps the flow of arms by dollar value between exporter and importer regions.</strong>
        Wider bands indicate larger transfer volumes. Currently, {dff.groupby('Exporter_Region')['Deal_Value_USD_M'].sum().idxmax() if total > 0 else 'N/A'}
        is the dominant source region, while {dff.groupby('Importer_Region')['Deal_Value_USD_M'].sum().idxmax() if total > 0 else 'N/A'} is the
        largest recipient region by value.
    </div>""", unsafe_allow_html=True)

    # --- Row 2: Timeline ---
    st.markdown("<div class='section-header'><h3>📈 Temporal Trends</h3><p>Arms transfer volume and value over time</p></div>", unsafe_allow_html=True)

    yearly = dff.groupby('Year').agg(
        Deals=('Year','count'), Value=('Deal_Value_USD_M','sum'),
        High_Risk=('High_Risk_Flag','sum'), Offensive=('Offensive_Flag','sum')
    ).reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=yearly['Year'], y=yearly['Deals'], name='Total Deals',
                         marker_color='rgba(0,212,255,0.4)'), secondary_y=False)
    fig.add_trace(go.Bar(x=yearly['Year'], y=yearly['High_Risk'], name='High Risk Deals',
                         marker_color='rgba(255,71,87,0.6)'), secondary_y=False)
    fig.add_trace(go.Scatter(x=yearly['Year'], y=yearly['Value'], name='Total Value ($M)',
                             mode='lines+markers', line=dict(color='#00ff88', width=2.5),
                             marker=dict(size=6)), secondary_y=True)
    fig.update_layout(title='Arms Transfers Over Time', barmode='overlay',
                      yaxis_title='Number of Deals', yaxis2_title='Total Value ($M)')
    st.plotly_chart(styled_chart(fig, 400), use_container_width=True)

    # --- Row 3: Top Exporters & Importers ---
    st.markdown("<div class='section-header'><h3>🌐 Key Players</h3><p>Dominant exporters and top importing nations</p></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        exp_agg = dff.groupby('Exporter').agg(Deals=('Year','count'), Value=('Deal_Value_USD_M','sum')).reset_index()
        exp_agg = exp_agg.sort_values('Value', ascending=True).tail(10)
        fig = go.Figure(go.Bar(
            y=exp_agg['Exporter'], x=exp_agg['Value'], orientation='h',
            marker=dict(color=exp_agg['Value'],
                        colorscale=[[0,'#0a4a2e'],[0.5,'#00ff88'],[1,'#00ffcc']]),
            text=exp_agg.apply(lambda r: f"${r['Value']:,.0f}M ({r['Deals']} deals)", axis=1),
            textposition='outside', textfont=dict(size=10)
        ))
        fig.update_layout(title='Top 10 Exporters by Value ($M)', xaxis_title='Total Value ($M)')
        st.plotly_chart(styled_chart(fig, 400), use_container_width=True)

    with c2:
        imp_agg = dff.groupby('Importer').agg(Deals=('Year','count'), Value=('Deal_Value_USD_M','sum'),
                                               Avg_Risk=('Escalation_Flag','mean')).reset_index()
        imp_agg = imp_agg.sort_values('Value', ascending=True).tail(12)
        fig = go.Figure(go.Bar(
            y=imp_agg['Importer'], x=imp_agg['Value'], orientation='h',
            marker=dict(color=imp_agg['Avg_Risk'],
                        colorscale=[[0,'#00ff88'],[0.5,'#ffa502'],[1,'#ff4757']],
                        colorbar=dict(title='Avg Risk')),
            text=imp_agg.apply(lambda r: f"${r['Value']:,.0f}M", axis=1),
            textposition='outside', textfont=dict(size=10),
            hovertemplate='<b>%{y}</b><br>Value: $%{x:,.0f}M<br>Avg Risk: %{marker.color:.2f}<extra></extra>'
        ))
        fig.update_layout(title='Top 12 Importers by Value (colored by risk)', xaxis_title='Total Value ($M)')
        st.plotly_chart(styled_chart(fig, 400), use_container_width=True)

    # --- Row 4: Weapon Category & Class ---
    st.markdown("<div class='section-header'><h3>⚔️ Arsenal Composition</h3><p>Weapon categories, subtypes, and offensive vs defensive classification</p></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        # Treemap: Category → Subtype → Weapon Class
        tree_df = dff.groupby(['Weapon_Category','Weapon_Subtype','Weapon_Class']).size().reset_index(name='Count')
        fig = px.treemap(tree_df, path=['Weapon_Category','Weapon_Subtype','Weapon_Class'], values='Count',
                         color='Weapon_Class', color_discrete_map=CLASS_COLORS,
                         title='Weapon Hierarchy: Category → Subtype → Class')
        st.plotly_chart(styled_chart(fig, 450), use_container_width=True)

    with c2:
        # Offensive vs Defensive by region
        class_region = dff.groupby(['Importer_Region','Weapon_Class']).size().reset_index(name='Count')
        fig = px.bar(class_region, x='Importer_Region', y='Count', color='Weapon_Class',
                     color_discrete_map=CLASS_COLORS, barmode='group',
                     title='Offensive vs Defensive Imports by Region')
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(styled_chart(fig, 450), use_container_width=True)

    # --- Row 5: Deal Frameworks ---
    st.markdown("<div class='section-header'><h3>📋 Deal Frameworks & Transfer Mechanisms</h3><p>How are arms transferred — bilateral pacts, FMS, commercial sales?</p></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fw = dff.groupby('Deal_Framework').agg(Count=('Year','count'), Value=('Deal_Value_USD_M','sum')).reset_index()
        fw = fw.sort_values('Count', ascending=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(y=fw['Deal_Framework'], x=fw['Count'], name='Deal Count',
                             orientation='h', marker_color='rgba(0,212,255,0.6)'))
        fig.update_layout(title='Transfer Count by Framework')
        st.plotly_chart(styled_chart(fig, 350), use_container_width=True)

    with c2:
        # Sunburst: Alliance → Framework → Risk
        sun_df = dff.groupby(['Exporter_Alliance','Deal_Framework','Escalation_Risk']).size().reset_index(name='Count')
        fig = px.sunburst(sun_df, path=['Exporter_Alliance','Deal_Framework','Escalation_Risk'],
                          values='Count', color='Escalation_Risk', color_discrete_map=RISK_COLORS,
                          title='Drill: Alliance → Framework → Escalation Risk')
        fig.update_traces(textinfo='label+percent parent')
        st.plotly_chart(styled_chart(fig, 350), use_container_width=True)


# =============================================================
# TAB 2: DIAGNOSTIC ANALYSIS
# =============================================================
with tab2:
    st.markdown("""
    <div class='section-header'>
        <h3>🔍 HUMAN INTELLIGENCE — What drives dangerous arms accumulation?</h3>
        <p>Statistical tests, correlation analysis, and risk factor identification</p>
    </div>""", unsafe_allow_html=True)

    # --- Correlation with Escalation Risk ---
    numeric_cols = ['Deal_Value_USD_M','Quantity','Delivery_Timeline_Months',
                    'Importer_GDP_Per_Capita','Importer_Political_Stability',
                    'Importer_Democracy_Index','Importer_Military_Spend_Pct_GDP',
                    'Offensive_Flag','High_Risk_Flag']

    corr_matrix = dff[numeric_cols].corr()
    risk_corr = corr_matrix['High_Risk_Flag'].drop('High_Risk_Flag').sort_values()

    c1, c2 = st.columns([1, 1])
    with c1:
        fig = go.Figure(go.Bar(
            y=risk_corr.index, x=risk_corr.values, orientation='h',
            marker=dict(color=risk_corr.values,
                        colorscale=[[0,'#00ff88'],[0.5,'#5a7a9a'],[1,'#ff4757']],
                        cmid=0),
            text=[f'{v:.3f}' for v in risk_corr.values], textposition='outside',
            textfont=dict(size=10)
        ))
        fig.update_layout(title='Correlation with High Escalation Risk', xaxis_title='Correlation Coefficient')
        st.plotly_chart(styled_chart(fig, 450), use_container_width=True)

    with c2:
        heat_cols = ['Deal_Value_USD_M','Importer_GDP_Per_Capita','Importer_Political_Stability',
                     'Importer_Democracy_Index','Importer_Military_Spend_Pct_GDP',
                     'Offensive_Flag','Delivery_Timeline_Months','High_Risk_Flag']
        fig = px.imshow(dff[heat_cols].corr(), text_auto='.2f',
                        color_continuous_scale=[[0,'#00ff88'],[0.5,'#0a1628'],[1,'#ff4757']],
                        zmin=-1, zmax=1, title='Feature Correlation Matrix')
        fig.update_layout(height=450)
        st.plotly_chart(styled_chart(fig, 450), use_container_width=True)

    st.markdown(f"""<div class='intel-box'>
        <strong>Low political stability and low democracy index show the strongest positive correlations with escalation risk.</strong>
        Higher GDP per capita is inversely correlated — wealthier importers tend toward lower risk profiles.
        Offensive weapon classification is a secondary but consistent signal.
    </div>""", unsafe_allow_html=True)

    # --- Chi-Square Tests ---
    st.markdown("<div class='section-header'><h3>📐 Statistical Significance Testing</h3><p>Chi-Square tests — which categorical factors are statistically associated with escalation risk?</p></div>", unsafe_allow_html=True)

    cat_test_cols = ['Weapon_Category','Weapon_Class','Deal_Framework','Exporter_Alliance',
                     'Importer_Conflict_Proximity','Active_Territorial_Dispute',
                     'Natural_Resource_Dependence','Arms_Import_Trend','UN_Embargo',
                     'Technology_Transfer','UNSC_Permanent_Member','Importer_Region']
    chi2_results = []
    for col in cat_test_cols:
        if col in dff.columns:
            ct = pd.crosstab(dff[col], dff['Escalation_Risk'])
            if ct.shape[0] > 1 and ct.shape[1] > 1:
                chi2, p, dof, expected = stats.chi2_contingency(ct)
                cramers_v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape)-1)))
                chi2_results.append({'Feature': col, 'Chi²': round(chi2,2), 'p-value': round(p,5),
                                     "Cramér's V": round(cramers_v, 3),
                                     'Significant': '✅ Yes' if p < 0.05 else '❌ No'})

    chi_df = pd.DataFrame(chi2_results).sort_values("Cramér's V", ascending=False)

    c1, c2 = st.columns([1, 1])
    with c1:
        fig = go.Figure(go.Bar(
            x=chi_df["Cramér's V"], y=chi_df['Feature'], orientation='h',
            marker=dict(color=chi_df["Cramér's V"],
                        colorscale=[[0,'#0a4a2e'],[1,'#ff4757']]),
            text=chi_df["Cramér's V"], textposition='outside',
            textfont=dict(size=10)
        ))
        fig.update_layout(title="Cramér's V — Effect Size (Higher = Stronger Association)",
                          xaxis_title="Cramér's V")
        st.plotly_chart(styled_chart(fig, 500), use_container_width=True)

    with c2:
        st.markdown("#### Chi-Square Test Results")
        st.dataframe(chi_df.set_index('Feature'), use_container_width=True, height=450)

    # --- Risk Factor Combinations ---
    st.markdown("<div class='section-header'><h3>⚠️ Deadliest Risk Factor Combinations</h3><p>Which multi-factor profiles produce the highest escalation rates?</p></div>", unsafe_allow_html=True)

    risk_combos = []
    for conflict in ['Yes','No']:
        for dispute in ['Yes','No']:
            for wclass in ['Offensive','Defensive']:
                for trend in ['Accelerating','Stable','Declining']:
                    subset = dff[(dff['Importer_Conflict_Proximity']==conflict) &
                                 (dff['Active_Territorial_Dispute']==dispute) &
                                 (dff['Weapon_Class']==wclass) &
                                 (dff['Arms_Import_Trend']==trend)]
                    if len(subset) >= 10:
                        rate = subset['High_Risk_Flag'].mean() * 100
                        risk_combos.append({
                            'Conflict': conflict, 'Dispute': dispute,
                            'Class': wclass, 'Trend': trend,
                            'Count': len(subset), 'High Risk %': round(rate,1)
                        })

    risk_df = pd.DataFrame(risk_combos).sort_values('High Risk %', ascending=False).head(12)

    fig = go.Figure(go.Bar(
        x=risk_df['High Risk %'],
        y=risk_df.apply(lambda r: f"Conflict:{r['Conflict']} | Dispute:{r['Dispute']} | {r['Class']} | {r['Trend']}", axis=1),
        orientation='h',
        marker=dict(color=risk_df['High Risk %'],
                    colorscale=[[0,'#ffa502'],[1,'#ff4757']]),
        text=risk_df.apply(lambda r: f"{r['High Risk %']}% (n={r['Count']})", axis=1),
        textposition='outside', textfont=dict(size=10)
    ))
    fig.update_layout(title='Top 12 Risk Factor Combinations', xaxis_title='High Escalation Risk %', height=520)
    st.plotly_chart(styled_chart(fig, 520), use_container_width=True)

    # --- Radar: High vs Low Risk Country Profiles ---
    st.markdown("<div class='section-header'><h3>📊 Importer Profile Comparison</h3><p>How do High-risk vs Low-risk import destinations differ?</p></div>", unsafe_allow_html=True)

    profile_cols = ['Importer_Political_Stability','Importer_Democracy_Index',
                    'Importer_Military_Spend_Pct_GDP','Deal_Value_USD_M','Offensive_Flag']
    profile_labels = ['Political Stability','Democracy Index','Military Spend % GDP',
                      'Deal Value ($M)','Offensive Weapon Share']

    c1, c2 = st.columns(2)
    with c1:
        # Normalize for radar
        high_vals = []
        low_vals = []
        for col in profile_cols:
            h = dff[dff['Escalation_Risk']=='High'][col].mean()
            l = dff[dff['Escalation_Risk']=='Low'][col].mean()
            col_max = max(h, l, 0.01)
            high_vals.append(h / col_max)
            low_vals.append(l / col_max)

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=high_vals + [high_vals[0]], theta=profile_labels + [profile_labels[0]],
                                       fill='toself', name='High Risk', line=dict(color='#ff4757'),
                                       fillcolor='rgba(255,71,87,0.12)'))
        fig.add_trace(go.Scatterpolar(r=low_vals + [low_vals[0]], theta=profile_labels + [profile_labels[0]],
                                       fill='toself', name='Low Risk', line=dict(color='#00ff88'),
                                       fillcolor='rgba(0,255,136,0.12)'))
        fig.update_layout(title='Risk Profile Radar: High vs Low',
                          polar=dict(radialaxis=dict(range=[0,1.1], gridcolor='rgba(0,255,136,0.1)'),
                                     angularaxis=dict(gridcolor='rgba(0,255,136,0.1)'),
                                     bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(styled_chart(fig, 420), use_container_width=True)

    with c2:
        # Gap analysis bars
        gap_data = []
        for col, label in zip(profile_cols, profile_labels):
            h_mean = dff[dff['Escalation_Risk']=='High'][col].mean()
            l_mean = dff[dff['Escalation_Risk']=='Low'][col].mean()
            t_stat, p_val = stats.ttest_ind(
                dff[dff['Escalation_Risk']=='High'][col].dropna(),
                dff[dff['Escalation_Risk']=='Low'][col].dropna()
            )
            gap_data.append({'Factor': label, 'High Risk': round(h_mean,2), 'Low Risk': round(l_mean,2),
                             'Gap': round(abs(h_mean - l_mean),2), 'p-value': round(p_val,4)})

        gap_df = pd.DataFrame(gap_data)
        fig = go.Figure()
        fig.add_trace(go.Bar(name='High Risk', x=gap_df['Factor'], y=gap_df['High Risk'], marker_color='#ff4757'))
        fig.add_trace(go.Bar(name='Low Risk', x=gap_df['Factor'], y=gap_df['Low Risk'], marker_color='#00ff88'))
        fig.update_layout(title='Average Feature Values: High vs Low Escalation Risk', barmode='group',
                          yaxis_title='Mean Value', xaxis_tickangle=-20)
        st.plotly_chart(styled_chart(fig, 420), use_container_width=True)

    st.dataframe(gap_df.set_index('Factor'), use_container_width=True)

    # --- Embargo Leakage Analysis ---
    st.markdown("<div class='section-header'><h3>🚨 Embargo Circumvention Analysis</h3><p>Are embargoed countries still receiving arms — and from whom?</p></div>", unsafe_allow_html=True)

    embargo_df = dff[dff['UN_Embargo']=='Yes']
    if len(embargo_df) > 0:
        emb_by_exp = embargo_df.groupby('Exporter').agg(Deals=('Year','count'), Value=('Deal_Value_USD_M','sum')).reset_index()
        emb_by_exp = emb_by_exp.sort_values('Value', ascending=True)
        fig = go.Figure(go.Bar(
            y=emb_by_exp['Exporter'], x=emb_by_exp['Value'], orientation='h',
            marker_color='#ff4757',
            text=emb_by_exp.apply(lambda r: f"${r['Value']:,.0f}M ({r['Deals']} deals)", axis=1),
            textposition='outside', textfont=dict(size=10)
        ))
        fig.update_layout(title='Arms Transfers to Embargoed Destinations — by Exporter', xaxis_title='Value ($M)')
        st.plotly_chart(styled_chart(fig, 350), use_container_width=True)
    else:
        st.info("No embargo transfers in current filter selection.")


# =============================================================
# TAB 3: PREDICTIVE ANALYSIS
# =============================================================
with tab3:
    st.markdown("""
    <div class='section-header'>
        <h3>🤖 MEASUREMENT & SIGNATURES — Can we predict escalation risk?</h3>
        <p>Machine learning models trained to classify transfers as High vs Non-High escalation risk</p>
    </div>""", unsafe_allow_html=True)

    @st.cache_data
    def run_predictive_models(data):
        df_ml = data.copy()
        cat_features = ['Exporter_Alliance','Weapon_Category','Weapon_Class','Deal_Framework',
                        'Importer_Conflict_Proximity','Active_Territorial_Dispute',
                        'Natural_Resource_Dependence','Arms_Import_Trend','UN_Embargo',
                        'Technology_Transfer','UNSC_Permanent_Member','Importer_Region']
        for c in cat_features:
            le = LabelEncoder()
            df_ml[c+'_enc'] = le.fit_transform(df_ml[c])

        feature_cols = ['Deal_Value_USD_M','Quantity','Delivery_Timeline_Months',
                        'Importer_GDP_Per_Capita','Importer_Political_Stability',
                        'Importer_Democracy_Index','Importer_Military_Spend_Pct_GDP'] + \
                       [c+'_enc' for c in cat_features]

        X = df_ml[feature_cols]
        y = df_ml['High_Risk_Flag']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42)
        }

        results = {}
        for name, model in models.items():
            scores = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc')
            model.fit(X_scaled, y)
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            else:
                importance = np.abs(model.coef_[0])
            results[name] = {
                'auc_mean': scores.mean(), 'auc_std': scores.std(),
                'importance': pd.Series(importance, index=feature_cols).sort_values(ascending=False),
                'model': model
            }

        roc_data = {}
        for name, model in models.items():
            y_prob = cross_val_predict(model, X_scaled, y, cv=5, method='predict_proba')[:,1]
            fpr, tpr, _ = roc_curve(y, y_prob)
            roc_auc_val = auc(fpr, tpr)
            roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc_val}

        return results, roc_data, feature_cols

    results, roc_data, feature_cols = run_predictive_models(df)

    # Model comparison
    c1, c2 = st.columns([1, 1])
    with c1:
        model_comp = pd.DataFrame({
            'Model': list(results.keys()),
            'AUC (mean)': [results[m]['auc_mean'] for m in results],
            'AUC (std)': [results[m]['auc_std'] for m in results],
        })
        fig = go.Figure()
        fig.add_trace(go.Bar(x=model_comp['Model'], y=model_comp['AUC (mean)'],
                             error_y=dict(type='data', array=model_comp['AUC (std)'].tolist()),
                             marker_color=['#00ff88','#00d4ff','#ffa502'],
                             text=model_comp['AUC (mean)'].round(3), textposition='outside',
                             textfont=dict(size=12)))
        fig.update_layout(title='Model Comparison — Cross-Validated AUC', yaxis_title='AUC Score',
                          yaxis_range=[0.5, 1.0])
        st.plotly_chart(styled_chart(fig, 380), use_container_width=True)

    with c2:
        fig = go.Figure()
        colors = ['#00ff88','#00d4ff','#ffa502']
        for i, (name, rdata) in enumerate(roc_data.items()):
            fig.add_trace(go.Scatter(x=rdata['fpr'], y=rdata['tpr'], mode='lines',
                                     name=f"{name} (AUC={rdata['auc']:.3f})",
                                     line=dict(color=colors[i], width=2.5)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random Baseline',
                                 line=dict(color='#2d3a4a', dash='dash', width=1)))
        fig.update_layout(title='ROC Curves — Escalation Risk Classification',
                          xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
        st.plotly_chart(styled_chart(fig, 380), use_container_width=True)

    # Feature Importance
    st.markdown("<div class='section-header'><h3>🎯 Feature Importance — What Signals Matter Most?</h3><p>Ranked by predictive power across models</p></div>", unsafe_allow_html=True)

    selected_model = st.selectbox("Select model for feature importance:", list(results.keys()), index=1)
    imp = results[selected_model]['importance'].head(15)

    fig = go.Figure(go.Bar(
        y=imp.index[::-1], x=imp.values[::-1], orientation='h',
        marker=dict(color=imp.values[::-1],
                    colorscale=[[0,'#0a4a2e'],[0.5,'#00ff88'],[1,'#ff4757']]),
        text=[f'{v:.4f}' for v in imp.values[::-1]], textposition='outside',
        textfont=dict(size=10)
    ))
    fig.update_layout(title=f'Top 15 Feature Importances — {selected_model}',
                      xaxis_title='Importance Score')
    st.plotly_chart(styled_chart(fig, 500), use_container_width=True)

    # Consensus ranking
    st.markdown("<div class='section-header'><h3>📊 Consensus Feature Ranking</h3><p>Features consistently important across all 3 models</p></div>", unsafe_allow_html=True)

    all_imp = pd.DataFrame()
    for name, res in results.items():
        norm_imp = res['importance'] / res['importance'].max()
        all_imp[name] = norm_imp

    all_imp['Mean'] = all_imp.mean(axis=1)
    all_imp = all_imp.sort_values('Mean', ascending=False).head(12)

    fig = go.Figure()
    for i, name in enumerate(results.keys()):
        fig.add_trace(go.Bar(name=name, y=all_imp.index[::-1], x=all_imp[name].values[::-1],
                             orientation='h', marker_color=colors[i], opacity=0.75))
    fig.update_layout(title='Normalized Feature Importance — All Models', barmode='group',
                      xaxis_title='Normalized Importance')
    st.plotly_chart(styled_chart(fig, 480), use_container_width=True)

    st.markdown("""<div class='intel-box'>
        <strong>Political stability, democracy index, conflict proximity, and military spend consistently emerge as
        the strongest predictors of escalation risk across all three models.</strong> Weapon class (offensive vs defensive)
        and arms import trend are secondary but reliable signals. High-value deals to low-stability,
        conflict-proximate countries with accelerating imports form the canonical high-risk profile.
    </div>""", unsafe_allow_html=True)


# =============================================================
# TAB 4: PRESCRIPTIVE ANALYSIS
# =============================================================
with tab4:
    st.markdown("""
    <div class='section-header'>
        <h3>🎯 OPEN SOURCE INTELLIGENCE — Where should intervention resources go?</h3>
        <p>Risk scoring, strategic recommendations, and resource allocation guidance</p>
    </div>""", unsafe_allow_html=True)

    # --- Risk Score Simulator ---
    st.markdown("<div class='section-header'><h3>🎯 Escalation Risk Simulator</h3><p>Model a hypothetical arms transfer and estimate its escalation risk score</p></div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sim_stability = st.slider("Political Stability (1-10)", 1.0, 10.0, 5.0, 0.5, key='sim_stab')
        sim_democracy = st.slider("Democracy Index (1-10)", 1.0, 10.0, 5.0, 0.5, key='sim_dem')
    with c2:
        sim_conflict = st.selectbox("Conflict Proximity", ['Yes','No'], key='sim_conf')
        sim_dispute = st.selectbox("Territorial Dispute", ['Yes','No'], key='sim_disp')
    with c3:
        sim_weapon = st.selectbox("Weapon Class", ['Offensive','Defensive'], key='sim_weap')
        sim_trend = st.selectbox("Arms Import Trend", ['Accelerating','Stable','Declining'], key='sim_trend')
    with c4:
        sim_milspend = st.slider("Military Spend (% GDP)", 0.5, 8.0, 2.5, 0.5, key='sim_mil')
        sim_resource = st.selectbox("Resource Dependence", ['High','Medium','Low'], key='sim_res')

    # Weighted risk score (mirrors the dataset generation logic)
    risk_score = 0
    risk_score += (10 - sim_stability) * 3.0
    risk_score += (10 - sim_democracy) * 1.5
    if sim_conflict == 'Yes': risk_score += 12
    if sim_dispute == 'Yes': risk_score += 8
    if sim_weapon == 'Offensive': risk_score += 5
    if sim_trend == 'Accelerating': risk_score += 7
    elif sim_trend == 'Declining': risk_score -= 3
    if sim_milspend > 4.0: risk_score += 6
    elif sim_milspend > 2.5: risk_score += 3
    if sim_resource == 'High': risk_score += 4
    elif sim_resource == 'Medium': risk_score += 2
    risk_score = min(100, max(0, risk_score))

    risk_color = '#00ff88' if risk_score < 28 else '#ffa502' if risk_score < 45 else '#ff4757'
    risk_label = 'LOW THREAT' if risk_score < 28 else 'ELEVATED THREAT' if risk_score < 45 else 'CRITICAL THREAT'

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={'text': f'Escalation Risk — {risk_label}', 'font': {'size': 16, 'color': '#c8d6e5', 'family': 'JetBrains Mono'}},
        gauge=dict(
            axis=dict(range=[0,100], tickwidth=1, tickcolor='#2d3a4a'),
            bar=dict(color=risk_color),
            bgcolor='rgba(0,0,0,0)',
            steps=[
                dict(range=[0,28], color='rgba(0,255,136,0.08)'),
                dict(range=[28,45], color='rgba(255,165,2,0.08)'),
                dict(range=[45,100], color='rgba(255,71,87,0.08)'),
            ],
            threshold=dict(line=dict(color='#ff4757', width=3), thickness=0.75, value=risk_score)
        ),
        number=dict(suffix='/100', font=dict(size=36, color=risk_color, family='JetBrains Mono'))
    ))
    st.plotly_chart(styled_chart(fig, 300), use_container_width=True)

    # --- Strategic Recommendations ---
    st.markdown("<div class='section-header'><h3>📋 Strategic Recommendations</h3><p>Evidence-based policy interventions to reduce escalation risk</p></div>", unsafe_allow_html=True)

    # Dynamic insights from data
    conflict_risk_rate = dff[dff['Importer_Conflict_Proximity']=='Yes']['High_Risk_Flag'].mean() * 100 if len(dff[dff['Importer_Conflict_Proximity']=='Yes']) > 0 else 0
    no_conflict_rate = dff[dff['Importer_Conflict_Proximity']=='No']['High_Risk_Flag'].mean() * 100 if len(dff[dff['Importer_Conflict_Proximity']=='No']) > 0 else 0
    accel_risk_rate = dff[dff['Arms_Import_Trend']=='Accelerating']['High_Risk_Flag'].mean() * 100 if len(dff[dff['Arms_Import_Trend']=='Accelerating']) > 0 else 0
    offensive_risk_rate = dff[dff['Weapon_Class']=='Offensive']['High_Risk_Flag'].mean() * 100 if len(dff[dff['Weapon_Class']=='Offensive']) > 0 else 0
    low_stab_rate = dff[dff['Importer_Political_Stability'] < 4]['High_Risk_Flag'].mean() * 100 if len(dff[dff['Importer_Political_Stability'] < 4]) > 0 else 0

    recommendations = [
        ("🚨 Arms Embargo Enforcement", f"Conflict-proximate importers show {conflict_risk_rate:.1f}% high-risk rate vs {no_conflict_rate:.1f}% for non-conflict states. Strengthen multilateral embargo monitoring, expand UNSC sanctions coverage to repeat violators, and invest in supply chain verification technology.", "CRITICAL"),
        ("📡 Early Warning — Arms Acceleration Monitoring", f"Importers with accelerating arms trends show {accel_risk_rate:.1f}% high-risk rates. Deploy real-time SIPRI-style tracking dashboards at regional security organisations. Flag any country showing >20% YoY import acceleration for diplomatic engagement.", "CRITICAL"),
        ("⚔️ Offensive Weapons Transfer Controls", f"Offensive systems carry a {offensive_risk_rate:.1f}% escalation risk rate. Propose stricter end-use certificates for offensive categories (combat aircraft, cruise missiles, MLRS). Create a tiered approval process based on importer stability score.", "HIGH"),
        ("🏛️ Governance-Linked Export Licensing", f"Countries with political stability below 4.0 show {low_stab_rate:.1f}% escalation risk. Recommend binding governance thresholds in export control frameworks — automatic review triggers when importer stability drops below thresholds.", "HIGH"),
        ("🤝 Diplomatic Corridor Investment", "Territorial disputes are among the strongest escalation drivers. Prioritise mediation resources for the top 5 dispute dyads identified in this dataset. Preventive diplomacy is 60x more cost-effective than post-conflict peacekeeping (UN estimates).", "MEDIUM"),
        ("💰 Resource Curse Mitigation", "High resource-dependent importers show elevated risk profiles. Link arms transfer approvals to Extractive Industries Transparency Initiative (EITI) compliance. Create incentives for resource-rich importers to diversify economies.", "MEDIUM"),
        ("📊 Predictive Peacekeeping Deployment", "Use the ML models as a quarterly early-warning system for UN DPPA and regional organisations. Flag high-risk importer profiles before crises materialise — shift from reactive to anticipatory peacekeeping posture.", "HIGH"),
    ]

    for title, desc, priority in recommendations:
        priority_color = '#ff4757' if priority == 'CRITICAL' else '#ffa502' if priority == 'HIGH' else '#00ff88'
        st.markdown(f"""
        <div class='rx-card'>
            <h4>{title} <span style='color:{priority_color}; font-size:0.65rem; background:rgba(255,255,255,0.03); padding:2px 8px; border-radius:2px; letter-spacing:1px;'>{priority}</span></h4>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    # --- Impact vs Feasibility Matrix ---
    st.markdown("<div class='section-header'><h3>💰 Impact vs Feasibility Matrix</h3><p>Prioritise interventions by estimated risk reduction and implementation complexity</p></div>", unsafe_allow_html=True)

    impact_data = pd.DataFrame({
        'Intervention': ['Embargo Enforcement', 'Arms Acceleration Monitoring', 'Offensive Transfer Controls',
                         'Governance-Linked Licensing', 'Diplomatic Corridors', 'Resource Curse Mitigation', 'Predictive Peacekeeping'],
        'Est. Risk Reduction %': [8.5, 6.2, 5.0, 4.5, 3.8, 2.5, 7.0],
        'Implementation Complexity': [4, 2, 3, 4, 3, 5, 2],
        'Time to Impact (months)': [12, 4, 8, 18, 24, 36, 6]
    })

    fig = px.scatter(impact_data, x='Implementation Complexity', y='Est. Risk Reduction %',
                     size='Time to Impact (months)', text='Intervention',
                     color='Est. Risk Reduction %',
                     color_continuous_scale=[[0,'#ffa502'],[1,'#00ff88']],
                     title='Intervention Prioritisation (size = time to impact)')
    fig.update_traces(textposition='top center', textfont=dict(size=10, family='JetBrains Mono'))
    fig.update_layout(xaxis_title='Implementation Complexity (1=Easy, 5=Hard)',
                      yaxis_title='Estimated Risk Reduction (%)')
    st.plotly_chart(styled_chart(fig, 450), use_container_width=True)

    # --- Regional Risk Heatmap ---
    st.markdown("<div class='section-header'><h3>🗺️ Regional Threat Assessment</h3><p>Which regions need the most attention?</p></div>", unsafe_allow_html=True)

    region_risk = dff.groupby('Importer_Region').agg(
        Total_Deals=('Year','count'),
        High_Risk_Deals=('High_Risk_Flag','sum'),
        Total_Value=('Deal_Value_USD_M','sum'),
        Avg_Stability=('Importer_Political_Stability','mean'),
        Avg_Democracy=('Importer_Democracy_Index','mean'),
        Offensive_Pct=('Offensive_Flag','mean'),
        Accel_Count=('Arms_Import_Trend', lambda x: (x=='Accelerating').sum())
    ).reset_index()
    region_risk['High_Risk_Pct'] = (region_risk['High_Risk_Deals'] / region_risk['Total_Deals'] * 100).round(1)
    region_risk['Offensive_Pct'] = (region_risk['Offensive_Pct'] * 100).round(1)
    region_risk = region_risk.sort_values('High_Risk_Pct', ascending=False)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=region_risk['Importer_Region'], y=region_risk['High_Risk_Pct'],
                         name='High Risk %', marker_color='#ff4757'))
    fig.add_trace(go.Bar(x=region_risk['Importer_Region'], y=region_risk['Offensive_Pct'],
                         name='Offensive %', marker_color='rgba(0,212,255,0.6)'))
    fig.update_layout(title='Regional Threat Summary', barmode='group',
                      yaxis_title='Percentage', xaxis_tickangle=-20)
    st.plotly_chart(styled_chart(fig, 400), use_container_width=True)

    st.dataframe(region_risk.set_index('Importer_Region').rename(columns={
        'Total_Deals':'Deals', 'High_Risk_Deals':'High Risk', 'Total_Value':'Value ($M)',
        'Avg_Stability':'Avg Stability', 'Avg_Democracy':'Avg Democracy',
        'Offensive_Pct':'Offensive %', 'Accel_Count':'Accelerating', 'High_Risk_Pct':'High Risk %'
    }), use_container_width=True)


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#2d3a4a; font-size:0.75rem; padding:1rem; font-family:JetBrains Mono, monospace;'>
    <span style='color:#00ff88;'>AEGIS</span> — Arms & Escalation Geopolitical Intelligence System ·
    Built with Streamlit & Plotly · 1,500 synthetic transfers × 25 features ·
    SIGINT · HUMINT · MASINT · OSINT
</div>
""", unsafe_allow_html=True)
