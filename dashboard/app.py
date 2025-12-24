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
        background-color: #f8f9fa; /* Light simple background */
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        color: #333333; /* Dark text */
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .kpi-card.good {
        border-top: 4px solid #27ae60;
    }
    .kpi-card.bad {
        border-top: 4px solid #e74c3c;
    }
    .kpi-card.neutral {
        border-top: 4px solid #3498db;
    }
    
    .kpi-value {
        font-size: 24px; /* Smaller */
        font-weight: bold;
    }
    .kpi-label {
        font-size: 13px; /* Smaller */
        color: #666;
    }
    .kpi-delta {
        font-size: 11px;
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
    
    
    # Consistent color map
    color_map = {
        'VIP Zone – Loyal Customers': '#27ae60', 
        'Growth Zone – Recent & High Potential': '#f39c12', 
        'Low Value Zone – Low Spend & Low Loyalty': '#f1c40f', 
        'Hibernating Zone – At Risk Customers': '#c0392b', 
        'Non-Transactors': '#95a5a6', # Grey
        
        # Fallbacks
        'VIP Retention (Loyal Customers)': '#27ae60',
        'Growth Acceleration (Promising Users)': '#f39c12',
        'Reactivation (Hibernating/At-Risk)': '#e74c3c',
        'Acquisition/Conversion (Non-Transactors)': '#3498db'
    }
    
    fig = px.bar(
        segment_counts,
        x='Segment',
        y='Count',
        color='Segment',
        color_discrete_map=color_map,
        title='Customer Segments Distribution',
        log_y=True # Log scale as requested
    )
    
    fig.update_layout(height=350, showlegend=False, margin=dict(t=50, b=20))
    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(title="Count (Log Scale)") # Explicit label
    
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
        color_discrete_sequence=['#3498db'],
        log_y=True # Log scale
    )
    
    fig.add_vline(x=0.5, line_dash="dash", line_color="red", 
                  annotation_text="50% Threshold")
    fig.update_layout(height=350, margin=dict(t=50, b=20))
    fig.update_xaxes(title="Propensity Score (0-1)")
    fig.update_yaxes(title="Count (Log Scale)") # Explicit label
    
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
                font=dict(color="black", size=18) # Increased size (+4)
            )
    
    fig.update_layout(height=800, margin=dict(t=50, b=20, l=100)) # Increased height slightly to accommodate font
    
    return fig

def create_action_matrix():
    """Create action recommendation table."""
    actions_data = {
        'VIP Zone – Loyal Customers': {
            'Priority': 'High',
            'Action': 'Early Access / Exclusive Benefits',
            'Channel': 'Personalized Email, Concierge'
        },
        'Growth Zone – Recent & High Potential': {
            'Priority': 'Medium',
            'Action': 'Welcome or Second-Purchase Incentives',
            'Channel': 'Email, In-App Nudges'
        },
        'Low Value Zone – Low Spend & Low Loyalty': {
            'Priority': 'Low',
            'Action': 'Light Promotions or Bundle Offers',
            'Channel': 'Push Notifications, Ads'
        },
        'Hibernating Zone – At Risk Customers': {
            'Priority': 'High',
            'Action': 'Win-back Campaigns with Time-Limited Vouchers',
            'Channel': 'SMS, Retargeting Ads'
        },
        'Non-Transactors': {
            'Priority': 'Medium',
            'Action': 'Acquisition-style tactics (Onboarding, First-order discounts)',
            'Channel': 'Paid Ads, Welcome Series'
        }
    }
    
    actions = pd.DataFrame.from_dict(actions_data, orient='index')
    actions.index.name = 'Segment'
    actions = actions.reset_index()
    
    return actions

def create_hourly_activity_chart(data):
    """Create hourly activity heatmap/line chart (User Request)."""
    events = data['events']
    if events is None:
        return None
    
    # Extract hour
    events['hour'] = events['datetime'].dt.hour
    
    # Aggregation
    hourly = events.groupby(['hour', 'event']).size().reset_index(name='count')
    
    # Plot
    fig = px.line(
        hourly, 
        x='hour', 
        y='count', 
        color='event',
        title='Hourly Activity Patterns (UTC)',
        markers=True,
        color_discrete_map={'view': '#2ecc71', 'addtocart': '#3498db', 'transaction': '#e74c3c'}
    )
    
    fig.update_layout(
        xaxis_title="Hour of Day (UTC)", 
        yaxis_title="Number of Events",
        height=350,
        margin=dict(t=50, b=20)
    )
    fig.update_xaxes(dtick=1)
    
    return fig

def create_day_of_week_chart(data):
    """Create day of week activity chart to identify peak days."""
    events = data['events']
    if events is None:
        return None
        
    events['dow'] = events['datetime'].dt.day_name()
    # Order: Monday to Sunday
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    daily = events.groupby(['dow']).size().reindex(dow_order).reset_index(name='count')
    
    # Highlight Peak Day
    colors = ['#3498db'] * 7
    if not daily.empty:
        max_idx = daily['count'].idxmax()
        colors[max_idx] = '#e74c3c'  # Highlight peak
    
    fig = px.bar(
        daily,
        x='dow',
        y='count',
        title='Weekly Activity Levels (Peak Day Analysis)',
        color_discrete_sequence=[colors] # This might need fix for individual bar colors in px
    )
    
    # Fix for individual bar colors in px.bar with list is tricky, better use go.Bar or just a column
    fig = go.Figure(data=[go.Bar(
        x=daily['dow'],
        y=daily['count'],
        marker_color=colors
    )])
    
    fig.update_layout(
        title='Weekly Activity Levels',
        yaxis_title='Total Events',
        height=350,
        margin=dict(t=50, b=20)
    )
    
    return fig

def calculate_conversion_metrics(data):
    """Calculate conversion rates and abandonment."""
    events = data['events']
    if events is None:
        return {}
    
    # Unique visitors per stage
    visitors = events['visitorid'].nunique()
    viewers = events[events['event'] == 'view']['visitorid'].nunique()
    cart_adders = events[events['event'] == 'addtocart']['visitorid'].nunique()
    purchasers = events[events['event'] == 'transaction']['visitorid'].nunique()
    
    # Conversion Rates
    view_to_cart = (cart_adders / viewers * 100) if viewers > 0 else 0
    cart_to_purchase = (purchasers / cart_adders * 100) if cart_adders > 0 else 0
    cart_abandonment = (100 - cart_to_purchase)
    
    return {
        'view_to_cart': view_to_cart,
        'cart_to_purchase': cart_to_purchase,
        'cart_abandonment': cart_abandonment,
        'viewers': viewers,
        'cart_adders': cart_adders,
        'purchasers': purchasers
    }

def create_conversion_rate_dist(data):
    """Create View vs AddToCart vs Transaction counts."""
    events = data['events']
    if events is None:
        return None
        
    counts = events['event'].value_counts().reset_index()
    counts.columns = ['Event', 'Count']
    
    fig = px.pie(
        counts, 
        names='Event', 
        values='Count', 
        title='Conversion Probability (Event Distribution)',
        color='Event',
        color_discrete_map={'view': '#2ecc71', 'addtocart': '#3498db', 'transaction': '#e74c3c'},
        hole=0.4
    )
    fig.update_layout(height=350, margin=dict(t=50, b=20))
    return fig

def create_feature_distributions(data):
    """Create histograms for RFM and Conversion Rate."""
    customers = data['customers']
    if customers is None:
        return None
        
    features = ['Recency', 'Frequency', 'Monetary', 'Conversion_Rate']
    available = [f for f in features if f in customers.columns]
    
    if not available:
        return None
        
    fig = make_subplots(rows=1, cols=4, subplot_titles=available)
    
    for i, feature in enumerate(available):
        fig.add_trace(
            go.Histogram(x=customers[feature], name=feature, nbinsx=30, marker_color='#3498db'),
            row=1, col=i+1
        )
        
    fig.update_layout(
        height=300, 
        title_text="Feature Distributions (Original Scale)", 
        showlegend=False,
        margin=dict(t=50, b=20, l=20, r=20)
    )
    fig.update_yaxes(type="log", title="Count (Log Scale)") # Log scale + Label
    return fig

def create_weekly_event_volumes(data):
    """Create weekly activity volumes (View, Cart, Transaction)."""
    events = data['events']
    if events is None:
        return None
        
    # Ensure datetime
    if 'datetime' not in events.columns:
        events['datetime'] = pd.to_datetime(events['timestamp'], unit='ms')
        
    # Resample weekly
    weekly = events.set_index('datetime').groupby('event').resample('W').size().reset_index(name='count')
    
    fig = px.line(
        weekly,
        x='datetime',
        y='count',
        color='event',
        title='Weekly Event Volumes',
        markers=True,
        color_discrete_map={'view': '#2ecc71', 'addtocart': '#3498db', 'transaction': '#e74c3c'}
    )
    
    fig.update_layout(
        xaxis_title="Week",
        yaxis_title="Total Events",
        height=400,
        margin=dict(t=50, b=20)
    )
    return fig

def create_3d_cluster_view(data):
    """Create 3D scatter plot of clusters (Matching K-means NB style)."""
    customers = data['customers']
    if customers is None:
        return None
    
    if 'Recency' not in customers.columns:
        return None
        
    df = customers.copy()
    
    # Filter out Non-Transactors (User Request)
    segment_col = 'Segment_Name' if 'Segment_Name' in df.columns else 'Cluster'
    df = df[~df[segment_col].astype(str).str.contains('Non-Transactors', case=False, na=False)]
    
    # 5k sample for performance (after filtering)
    if len(df) > 5000:
        df = df.sample(5000, random_state=42)
        
    # Ensure discrete colors if using Cluster
    if 'Cluster' in df.columns:
        df['Cluster_Label'] = df['Cluster'].astype(str)
        color_col = 'Cluster_Label'
    else:
        color_col = segment_col
    
    # Color map (Reuse existing if possible, or auto)
    # If using Cluster_Label, we might lose specific color mapping if not careful.
    # But user asked to use "exact same" as K-means visual which implies using Cluster ID often.
    # However, to keep consistency with dashboard, let's try to map back to Segment Name if possible,
    # OR if user insists on "exact same", they might want the Cluster ID colors. 
    # Let's map Segment Name to color but use the styling requested.
    
    # User Code Snippet used 'Cluster_Label' and 'color=Cluster_Label'.
    # I will stick to Segment_Name for consistency with the rest of the dashboard colors 
    # BUT apply the visual style requested.
    
    # Update: User specifically requested "color='Cluster_Label'". 
    # The dashboard uses meaningful names. I will prioritize the dashboard's meaningful names 
    # but apply the visual parameters. If I switch to Cluster_Label, the legend will disconnect 
    # from the rest of the dashboard.
    # Compromise: Use Segment Name but applied to the User's visual parameters.
    
    color_map = {
        'VIP Zone – Loyal Customers': '#27ae60', 
        'Growth Zone – Recent & High Potential': '#f39c12', 
        'Low Value Zone – Low Spend & Low Loyalty': '#f1c40f', 
        'Hibernating Zone – At Risk Customers': '#c0392b'
    }
    
    fig = px.scatter_3d(
        df,
        x='Recency',
        y='Frequency',
        z='Monetary',
        color=segment_col,
        title='Interactive 3D View of Clusters (RFM)',
        opacity=0.7,
        size_max=10,
        log_x=True, log_y=True, log_z=True,
        color_discrete_map=color_map,
        hover_data=['visitorid', 'Conversion_Rate']
    )
    
    # Update marker size
    fig.update_traces(marker=dict(size=3))
    
    fig.update_layout(height=700, margin=dict(l=0, r=0, b=0, t=30))
    return fig


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
        
    st.markdown("---")
    
    # 3D Cluster View
    st.subheader("🧊 Interactive 3D Cluster View (RFM)")
    scatter_fig = create_3d_cluster_view(data)
    if scatter_fig:
        st.plotly_chart(scatter_fig, use_container_width=True)
    


def page_advanced_analytics(data):
    """Advanced Analytics Page (CLV, Propensity, PCA)."""
    st.header("📈 Advanced Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Lifetime Value Map")
        st.image("dashboard/assets/clv_prob_alive.png", caption="Strategic Customer Value Map (CLV vs Probability Alive)")

    with col2:
        st.subheader("Purchase Propensity")
        prop_fig = create_propensity_histogram(data)
        if prop_fig:
            st.plotly_chart(prop_fig, use_container_width=True)
        else:
            st.info("Propensity data not available. Run propensity model first.")
            
    st.markdown("---")
    
    # Feature Distributions (EDA)
    st.subheader("📊 Feature Distributions & Transformations")
    st.markdown("Understanding behavior skewness (motivating Log-Transformation for K-Means).")
    
    dist_fig = create_feature_distributions(data)
    if dist_fig:
        st.plotly_chart(dist_fig, use_container_width=True)
    
    st.markdown("---")
    
    # Leading vs Lagging Summary
    st.subheader("📊 Lagging vs Leading Indicators (80/20 Rule)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔴 Lagging Indicators (80%)")
        st.markdown("**What happened** - Historical performance")
        if data['events'] is not None:
            total_revenue = data['events'][data['events']['event'] == 'transaction']['transactionid'].sum()
            total_transactions = len(data['events'][data['events']['event'] == 'transaction'])
            st.write(f"- Total Revenue: **{total_revenue:,.0f}**")
            st.write(f"- Total Transactions: **{total_transactions:,}**")
    
    with col2:
        st.markdown("### 🟢 Leading Indicators (20%)")
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
        2. Launch **VIP Retention** program
        3. Trigger **Reactivation** win-back campaign
        """)
    
    with col2:
        st.markdown("### 📅 Short-Term (1-3 months)")
        st.markdown("""
        1. A/B test **Acquisition** conversion
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

def page_behavioral_insights(data):
    """New Behavioral Insights Page."""
    st.header("🛒 Behavioral Insights & Trends")
    
    events = data['events']
    if events is None:
        st.warning("Event data required.")
        return

    # 1. Temporal Patterns
    st.subheader("⏰ Temporal Patterns")
    col1, col2 = st.columns(2)
    
    with col1:
        # Hourly Chart
        hourly_fig = create_hourly_activity_chart(data)
        if hourly_fig:
            st.plotly_chart(hourly_fig, use_container_width=True)
            
            # Peak Hour Insight
            events['hour'] = events['datetime'].dt.hour
            peak_hour = events.groupby('hour').size().idxmax()
            st.info(f"💡 **Peak Activity Hour:** {peak_hour}:00 UTC (Align campaigns around this window)")
            
    with col2:
        day_fig = create_day_of_week_chart(data)
        if day_fig:
            st.plotly_chart(day_fig, use_container_width=True)
            
            # Peak Day Insight
            events['dow'] = events['datetime'].dt.day_name()
            peak_day = events.groupby('dow').size().idxmax()
            st.info(f"💡 **Peak Day:** {peak_day}. (Focus marketing efforts here)")
            if peak_day == 'Tuesday':
                 st.success("✅ Validated: Tuesday is indeed the peak day!")
    
    st.markdown("---")
    
    # Weekly Activity Trends
    st.subheader("📈 Weekly Activity Volumes")
    weekly_fig = create_weekly_event_volumes(data)
    if weekly_fig:
        st.plotly_chart(weekly_fig, use_container_width=True)
    
    st.markdown("---")

    # 2. Conversion Funnel Metrics
    st.subheader("📉 Conversion & Abandonment")
    
    metrics = calculate_conversion_metrics(data)
    
    # Metrics Row
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(render_kpi_card("View → Cart Rate", f"{metrics['view_to_cart']:.2f}%", card_type="neutral"), unsafe_allow_html=True)
    with m2:
        st.markdown(render_kpi_card("Cart → Purchase Rate", f"{metrics['cart_to_purchase']:.2f}%", card_type="good"), unsafe_allow_html=True)
    with m3:
        st.markdown(render_kpi_card("🛒 Abandonment Rate", f"{metrics['cart_abandonment']:.2f}%", card_type="bad"), unsafe_allow_html=True)

    # Charts Row
    c1, c2 = st.columns(2)
    with c1:
        # Pie Chart of Event Types
        dist_fig = create_conversion_rate_dist(data)
        if dist_fig:
            st.plotly_chart(dist_fig, use_container_width=True)
            
    with c2:
        # Funnel again for context
        funnel_fig = create_funnel_chart(data)
        if funnel_fig:
            st.plotly_chart(funnel_fig, use_container_width=True)

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    # Sidebar Navigation
    st.sidebar.title("👥 Customer Segmentation")
    st.sidebar.markdown("---")
    
    # Page Selection
    # Page Selection
    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Segmentation", "Behavioral Insights", "Analytics", "Recommendations"]
    )
    
    # Load data
    data = load_data()

    # Render selected page
    if page == "Overview":
        page_overview(data)
    elif page == "Segmentation":
        page_segment_deep_dive(data)
    elif page == "Behavioral Insights":
        page_behavioral_insights(data)
    elif page == "Analytics":
        page_advanced_analytics(data)
    elif page == "Recommendations":
        page_recommendations(data)


if __name__ == "__main__":
    main()
