"""
Customer Segmentation Dashboard (BA11 Storytelling Compliant)
==============================================================
A comprehensive Streamlit dashboard following BA11 guidelines:
- 5-Second Rule: Instantly scannable KPIs
- F-Pattern Layout: KPIs top-left, Filters top-right, Trends center, Details bottom
- Color Semantics: Green=Good, Red=Bad, Blue=Primary
- 60/40 Lagging/Leading Indicators

Run from project root:
    streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS (BA11: Color Semantics)
# =============================================================================
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 1rem;
        max-width: 100%;
    }
    
    /* KPI Cards - BA11 Style */
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .kpi-card.good {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .kpi-card.bad {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
    }
    .kpi-card.neutral {
        background: linear-gradient(135deg, #4776E6 0%, #8E54E9 100%);
    }
    
    .kpi-value {
        font-size: 32px;
        font-weight: bold;
    }
    .kpi-label {
        font-size: 14px;
        opacity: 0.9;
    }
    .kpi-delta {
        font-size: 12px;
        margin-top: 5px;
    }
    
    /* Section headers */
    .section-header {
        border-left: 4px solid #667eea;
        padding-left: 10px;
        margin: 20px 0 10px 0;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data
def load_data():
    """Load all required datasets."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    dataset_path = os.path.join(project_root, 'dataset')
    
    data = {}
    
    # Load customer data (priority order)
    for filename in ['customers_segmented_remediated.csv', 'customers_segmented.csv', 
                     'user_clustered_gmm.csv', 'customers.csv']:
        try:
            data['customers'] = pd.read_csv(os.path.join(dataset_path, filename))
            data['segment_source'] = filename
            break
        except FileNotFoundError:
            continue
    else:
        data['customers'] = None
        data['segment_source'] = 'None'
    
    # Load events
    try:
        events = pd.read_csv(os.path.join(dataset_path, 'events.csv'))
        events['datetime'] = pd.to_datetime(events['timestamp'], unit='ms', utc=True)
        events['date'] = events['datetime'].dt.date
        data['events'] = events
    except FileNotFoundError:
        data['events'] = None
    
    # Load CLV data
    try:
        data['clv'] = pd.read_csv(os.path.join(dataset_path, 'clv_customer.csv'))
    except FileNotFoundError:
        data['clv'] = None
    
    # Load Propensity data
    try:
        data['propensity'] = pd.read_csv(os.path.join(dataset_path, 'customer_propensity_scores.csv'))
    except FileNotFoundError:
        data['propensity'] = None
    
    return data

# =============================================================================
# KPI FUNCTIONS (BA11: 5-Second Rule)
# =============================================================================
def render_kpi_card(label, value, delta=None, delta_good=True, card_type="neutral"):
    """Render a single KPI card with BA11 color semantics."""
    delta_html = ""
    if delta is not None:
        arrow = "↑" if delta >= 0 else "↓"
        color = "#27ae60" if (delta >= 0) == delta_good else "#e74c3c"
        delta_html = f'<div class="kpi-delta" style="color:{color}">{arrow} {abs(delta):.1f}%</div>'
    
    return f"""
    <div class="kpi-card {card_type}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """

def create_kpi_section(data):
    """Create the top KPI section following F-Pattern (top-left priority)."""
    events = data['events']
    customers = data['customers']
    
    if events is None or customers is None:
        st.warning("Data not loaded.")
        return
    
    # Calculate metrics
    total_visitors = events['visitorid'].nunique()
    paying_customers = events[events['event'] == 'transaction']['visitorid'].nunique()
    total_revenue = events[events['event'] == 'transaction']['transactionid'].sum()
    conversion_rate = (paying_customers / total_visitors) * 100
    
    # CLV metrics (Leading)
    avg_clv = 0
    if data['clv'] is not None and 'CLV_1_month' in data['clv'].columns:
        avg_clv = data['clv']['CLV_1_month'].mean()
    
    # Propensity metrics (Leading)
    avg_propensity = 0
    if data['propensity'] is not None and 'propensity_score' in data['propensity'].columns:
        avg_propensity = data['propensity']['propensity_score'].mean() * 100
    
    # F-Pattern: KPIs on the left (most important first)
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.markdown(render_kpi_card("Total Visitors", f"{total_visitors:,}", card_type="neutral"), unsafe_allow_html=True)
    with col2:
        st.markdown(render_kpi_card("Paying Customers", f"{paying_customers:,}", card_type="neutral"), unsafe_allow_html=True)
    with col3:
        # Lagging indicator
        st.markdown(render_kpi_card("💰 Total Revenue", f"{total_revenue:,.0f}", card_type="good"), unsafe_allow_html=True)
    with col4:
        # Leading indicator
        color = "good" if conversion_rate > 1 else "bad"
        st.markdown(render_kpi_card("📈 Conversion Rate", f"{conversion_rate:.2f}%", card_type=color), unsafe_allow_html=True)
    with col5:
        # Leading indicator
        st.markdown(render_kpi_card("📊 Avg CLV", f"{avg_clv:.0f}", card_type="neutral"), unsafe_allow_html=True)
    with col6:
        # Leading indicator
        color = "good" if avg_propensity > 20 else "bad"
        st.markdown(render_kpi_card("🎯 Avg Propensity", f"{avg_propensity:.1f}%", card_type=color), unsafe_allow_html=True)

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
def create_funnel_chart(data):
    """Create conversion funnel (BA11: Funnel Framework)."""
    events = data['events']
    if events is None:
        return None
    
    viewers = events[events['event']=='view']['visitorid'].nunique()
    cart_adders = events[events['event']=='addtocart']['visitorid'].nunique()
    purchasers = events[events['event']=='transaction']['visitorid'].nunique()
    
    fig = go.Figure(go.Funnel(
        y = ["View", "Add to Cart", "Purchase"],
        x = [viewers, cart_adders, purchasers],
        textposition = "inside",
        textinfo = "value+percent previous",
        marker = {"color": ["#3498db", "#f39c12", "#27ae60"]},
        connector = {"line": {"color": "#4a4a4a", "dash": "dot", "width": 2}}
    ))
    
    fig.update_layout(
        title="Customer Conversion Funnel",
        height=350,
        margin=dict(t=50, b=20, l=20, r=20)
    )
    
    return fig

def create_segment_bar_chart(data):
    """Create segment distribution as bar chart (BA11: No pie charts >5 categories)."""
    customers = data['customers']
    if customers is None:
        return None
    
    segment_col = 'Segment_Name' if 'Segment_Name' in customers.columns else 'Cluster'
    if segment_col not in customers.columns:
        return None
    
    segment_counts = customers[segment_col].value_counts().reset_index()
    segment_counts.columns = ['Segment', 'Count']
    
    fig = px.bar(
        segment_counts,
        x='Segment',
        y='Count',
        color='Count',
        color_continuous_scale=['#e74c3c', '#f39c12', '#27ae60'],
        title='Customer Segments Distribution'
    )
    
    fig.update_layout(height=350, showlegend=False, margin=dict(t=50, b=20))
    fig.update_xaxes(tickangle=45)
    
    return fig

def create_clv_chart(data):
    """Create CLV by segment chart."""
    customers = data['customers']
    clv = data['clv']
    
    if customers is None or clv is None:
        return None
    
    segment_col = 'Segment_Name' if 'Segment_Name' in customers.columns else 'Cluster'
    if segment_col not in customers.columns:
        return None
    
    # Merge
    merged = customers.merge(clv[['visitorid', 'CLV_1_month']], on='visitorid', how='left')
    clv_by_segment = merged.groupby(segment_col)['CLV_1_month'].mean().sort_values(ascending=True).reset_index()
    clv_by_segment.columns = ['Segment', 'Avg_CLV']
    
    # Color code: Green for high, Red for low
    median_clv = clv_by_segment['Avg_CLV'].median()
    clv_by_segment['color'] = clv_by_segment['Avg_CLV'].apply(
        lambda x: '#27ae60' if x > median_clv else '#e74c3c'
    )
    
    fig = px.bar(
        clv_by_segment,
        y='Segment',
        x='Avg_CLV',
        orientation='h',
        color='Avg_CLV',
        color_continuous_scale=['#e74c3c', '#f39c12', '#27ae60'],
        title='Customer Lifetime Value by Segment'
    )
    
    fig.add_vline(x=median_clv, line_dash="dash", line_color="orange", 
                  annotation_text=f"Median: {median_clv:.0f}")
    fig.update_layout(height=350, showlegend=False, margin=dict(t=50, b=20, l=120))
    
    return fig

def create_propensity_histogram(data):
    """Create propensity score distribution."""
    propensity = data['propensity']
    if propensity is None or 'propensity_score' not in propensity.columns:
        return None
    
    fig = px.histogram(
        propensity,
        x='propensity_score',
        nbins=50,
        title='Propensity Score Distribution',
        color_discrete_sequence=['#3498db']
    )
    
    fig.add_vline(x=0.5, line_dash="dash", line_color="red", 
                  annotation_text="50% Threshold")
    fig.update_layout(height=350, margin=dict(t=50, b=20))
    fig.update_xaxes(title="Propensity Score (0-1)")
    fig.update_yaxes(title="Count")
    
    return fig

def create_time_series(data):
    """Create daily activity time series."""
    events = data['events']
    if events is None:
        return None
    
    daily = events.groupby('date').agg({
        'visitorid': 'nunique',
        'event': 'count'
    }).reset_index()
    daily.columns = ['date', 'unique_visitors', 'total_events']
    daily['date'] = pd.to_datetime(daily['date'])
    
    # Transactions per day
    transactions_daily = events[events['event'] == 'transaction'].groupby('date').size().reset_index()
    transactions_daily.columns = ['date', 'transactions']
    transactions_daily['date'] = pd.to_datetime(transactions_daily['date'])
    daily = daily.merge(transactions_daily, on='date', how='left').fillna(0)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=daily['date'], y=daily['unique_visitors'], name="Unique Visitors",
                   line=dict(color='#3498db', width=2)),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=daily['date'], y=daily['transactions'], name="Transactions",
                   line=dict(color='#27ae60', width=2)),
        secondary_y=True
    )
    
    fig.update_layout(
        title="Daily Activity Trends",
        height=300,
        margin=dict(t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    fig.update_yaxes(title_text="Unique Visitors", secondary_y=False)
    fig.update_yaxes(title_text="Transactions", secondary_y=True)
    
    return fig

def create_rfm_heatmap(data):
    """Create RFM heatmap by segment."""
    customers = data['customers']
    if customers is None:
        return None
    
    segment_col = 'Segment_Name' if 'Segment_Name' in customers.columns else 'Cluster'
    if segment_col not in customers.columns:
        return None
    
    features = ['Recency', 'Frequency', 'Monetary', 'Conversion_Rate']
    available = [f for f in features if f in customers.columns]
    if not available:
        return None
    
    profiles = customers.groupby(segment_col)[available].mean()
    profiles_norm = (profiles - profiles.min()) / (profiles.max() - profiles.min())
    
    fig = px.imshow(
        profiles_norm.T,
        labels=dict(x="Segment", y="Feature", color="Normalized"),
        x=profiles_norm.index.astype(str),
        y=profiles_norm.columns,
        color_continuous_scale="RdYlGn",
        title="RFM Profiles by Segment"
    )
    
    # Add actual values as text
    for i, feature in enumerate(profiles_norm.columns):
        for j, segment in enumerate(profiles_norm.index):
            fig.add_annotation(
                x=j, y=i,
                text=f"{profiles.loc[segment, feature]:.1f}",
                showarrow=False,
                font=dict(color="black", size=10)
            )
    
    fig.update_layout(height=250, margin=dict(t=50, b=20, l=100))
    
    return fig

def create_action_matrix():
    """Create action recommendation table."""
    actions = pd.DataFrame({
        'Segment': ['VIP Champions', 'Big Spenders', 'Loyal Regulars', 'At Risk', 'Window Shoppers'],
        'Priority': ['🔴 High', '🔴 High', '🟡 Medium', '🟠 High', '🟢 Low'],
        'Strategy': ['Retain & Reward', 'Upsell Premium', 'Cross-sell', 'Win-Back', 'Convert'],
        'Actions': [
            'VIP access, exclusive offers',
            'Premium bundles, tier upgrades',
            'Personalized recommendations',
            '"We miss you" campaigns',
            'First-purchase incentives'
        ],
        'KPIs': ['Retention Rate', 'AOV, CLTV', 'Frequency', 'Reactivation Rate', 'Conversion Rate']
    })
    return actions

# =============================================================================
# PAGE FUNCTIONS
# =============================================================================
def page_overview(data):
    """Executive Overview Page (5-Second Scannable)."""
    st.header("📊 Executive Overview")
    
    # KPI Cards (Top-Left priority per F-Pattern)
    create_kpi_section(data)
    
    st.markdown("---")
    
    # Row 1: Funnel + Segments (Center)
    col1, col2 = st.columns(2)
    
    with col1:
        funnel_fig = create_funnel_chart(data)
        if funnel_fig:
            st.plotly_chart(funnel_fig, use_container_width=True)
    
    with col2:
        segment_fig = create_segment_bar_chart(data)
        if segment_fig:
            st.plotly_chart(segment_fig, use_container_width=True)
    
    # Row 2: Time Series (Full width)
    ts_fig = create_time_series(data)
    if ts_fig:
        st.plotly_chart(ts_fig, use_container_width=True)

def page_segment_deep_dive(data):
    """Segment Deep Dive Page."""
    st.header("🔍 Segment Deep Dive")
    
    customers = data['customers']
    if customers is None:
        st.warning("No customer data available.")
        return
    
    segment_col = 'Segment_Name' if 'Segment_Name' in customers.columns else 'Cluster'
    if segment_col not in customers.columns:
        st.warning("No segment column found.")
        return
    
    # Sidebar filter
    segments = customers[segment_col].unique().tolist()
    selected_segment = st.selectbox("Select Segment", ['All'] + segments)
    
    if selected_segment != 'All':
        filtered = customers[customers[segment_col] == selected_segment]
    else:
        filtered = customers
    
    # Metrics for selected segment
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Customers", f"{len(filtered):,}")
    with col2:
        if 'Recency' in filtered.columns:
            st.metric("Avg Recency", f"{filtered['Recency'].mean():.1f}")
    with col3:
        if 'Frequency' in filtered.columns:
            st.metric("Avg Frequency", f"{filtered['Frequency'].mean():.1f}")
    with col4:
        if 'Monetary' in filtered.columns:
            st.metric("Avg Monetary", f"{filtered['Monetary'].mean():.0f}")
    
    st.markdown("---")
    
    # RFM Heatmap
    rfm_fig = create_rfm_heatmap(data)
    if rfm_fig:
        st.plotly_chart(rfm_fig, use_container_width=True)
    
    # CLV Chart
    clv_fig = create_clv_chart(data)
    if clv_fig:
        st.plotly_chart(clv_fig, use_container_width=True)

def page_advanced_analytics(data):
    """Advanced Analytics Page (CLV, Propensity, PCA)."""
    st.header("📈 Advanced Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Lifetime Value")
        clv_fig = create_clv_chart(data)
        if clv_fig:
            st.plotly_chart(clv_fig, use_container_width=True)
        else:
            st.info("CLV data not available. Run CLV notebook first.")
    
    with col2:
        st.subheader("Purchase Propensity")
        prop_fig = create_propensity_histogram(data)
        if prop_fig:
            st.plotly_chart(prop_fig, use_container_width=True)
        else:
            st.info("Propensity data not available. Run propensity model first.")
    
    st.markdown("---")
    
    # Leading vs Lagging Summary
    st.subheader("📊 Lagging vs Leading Indicators (60/40 Rule)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔴 Lagging Indicators (60%)")
        st.markdown("**What happened** - Historical performance")
        if data['events'] is not None:
            total_revenue = data['events'][data['events']['event'] == 'transaction']['transactionid'].sum()
            total_transactions = len(data['events'][data['events']['event'] == 'transaction'])
            st.write(f"- Total Revenue: **{total_revenue:,.0f}**")
            st.write(f"- Total Transactions: **{total_transactions:,}**")
    
    with col2:
        st.markdown("### 🟢 Leading Indicators (40%)")
        st.markdown("**What will happen** - Predictive metrics")
        if data['clv'] is not None and 'CLV_1_month' in data['clv'].columns:
            avg_clv = data['clv']['CLV_1_month'].mean()
            st.write(f"- Avg CLV (1 Month): **{avg_clv:.2f}**")
        if data['propensity'] is not None and 'propensity_score' in data['propensity'].columns:
            avg_prop = data['propensity']['propensity_score'].mean() * 100
            st.write(f"- Avg Propensity: **{avg_prop:.1f}%**")

def page_recommendations(data):
    """Recommendations Page."""
    st.header("📋 Action Recommendations")
    
    # Action Matrix Table
    st.subheader("Segment Action Matrix")
    actions = create_action_matrix()
    st.dataframe(actions, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Recommendations
    st.subheader("🎯 Recommended Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### ⚡ Immediate (0-30 days)")
        st.markdown("""
        1. Deploy segment labels to CRM
        2. Launch VIP retention program
        3. Trigger At Risk win-back campaign
        """)
    
    with col2:
        st.markdown("### 📅 Short-Term (1-3 months)")
        st.markdown("""
        1. A/B test Window Shopper conversion
        2. Build recommendation engine
        3. Implement segment email sequences
        """)
    
    with col3:
        st.markdown("### 🔄 Ongoing")
        st.markdown("""
        1. Monitor segment migration monthly
        2. Re-cluster quarterly
        3. Track segment KPIs in this dashboard
        """)

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    # Sidebar Navigation
    st.sidebar.title("👥 Customer Segmentation")
    st.sidebar.markdown("---")
    
    # Page Selection
    page = st.sidebar.radio(
        "Navigation",
        ["📊 Overview", "🔍 Segment Deep Dive", "📈 Advanced Analytics", "📋 Recommendations"]
    )
    
    # Load data
    data = load_data()
    
    # Data Status in Sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Data Status")
    st.sidebar.write(f"✓ Customers: {'✅' if data['customers'] is not None else '❌'}")
    st.sidebar.write(f"✓ Events: {'✅' if data['events'] is not None else '❌'}")
    st.sidebar.write(f"✓ CLV: {'✅' if data['clv'] is not None else '❌'}")
    st.sidebar.write(f"✓ Propensity: {'✅' if data['propensity'] is not None else '❌'}")
    st.sidebar.write(f"Source: {data.get('segment_source', 'None')}")
    
    # Render selected page
    if page == "📊 Overview":
        page_overview(data)
    elif page == "🔍 Segment Deep Dive":
        page_segment_deep_dive(data)
    elif page == "📈 Advanced Analytics":
        page_advanced_analytics(data)
    elif page == "📋 Recommendations":
        page_recommendations(data)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("*BA11 Storytelling Compliant*")

if __name__ == "__main__":
    main()
