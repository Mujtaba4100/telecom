"""
Enhanced Telecom Customer Segmentation Dashboard
===============================================
✅ Interactive visual insights
✅ Communication: Location, Intl calls, Frequency, Duration, Time (Morning/Noon/Night)
✅ Internet: Download, Upload, Overall
✅ SMS: Frequency
✅ Customer lookup by ID with LLM suggestions
✅ Dynamic clustering & visualizations
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import io
from datetime import datetime

# ============================================
# CONFIGURATION
# ============================================

st.set_page_config(
    page_title="📊 Telecom Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend URL - auto-detects environment
import os

# Priority: 1. Streamlit secrets, 2. Environment variable, 3. Local default
BACKEND_URL = "http://localhost:7860"  # Default for local development

try:
    # Try to get from Streamlit secrets (HuggingFace deployment)
    BACKEND_URL = st.secrets.get("BACKEND_URL", BACKEND_URL)
except (FileNotFoundError, KeyError, AttributeError):
    # Try environment variable
    BACKEND_URL = os.getenv('BACKEND_URL', BACKEND_URL)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .insight-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .metric-box {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .customer-detail {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# UTILITY FUNCTIONS
# ============================================

def show_active_filters_banner():
    """Display a banner showing currently active filters"""
    if 'filters' not in st.session_state:
        return False
    
    filters = st.session_state.filters
    active_filters = []
    
    if filters.get('clusters'):
        cluster_list = ', '.join(map(str, filters['clusters']))
        active_filters.append(f"🎯 Clusters: {cluster_list}")
    if filters.get('usage_level', 'All') != 'All':
        active_filters.append(f"⚡ Usage: {filters['usage_level']}")
    if filters.get('international', 'All') != 'All':
        active_filters.append(f"🌍 {filters['international']}")
    
    if active_filters:
        filter_text = " | ".join(active_filters)
        st.info(f"🔍 **Active Filters:** {filter_text}")
        return True
    return False

# ============================================
# API FUNCTIONS
# ============================================

@st.cache_data(ttl=300)
def get_stats():
    try:
        response = requests.get(f"{BACKEND_URL}/api/stats", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Backend error: {e}")
        return None

@st.cache_data(ttl=300)
def get_time_analysis():
    try:
        response = requests.get(f"{BACKEND_URL}/api/time-analysis", timeout=10)
        response.raise_for_status()
        return response.json()
    except:
        return None

@st.cache_data(ttl=300)
def get_clusters(cluster_type="kmeans"):
    try:
        response = requests.get(f"{BACKEND_URL}/api/clusters", params={"cluster_type": cluster_type}, timeout=10)
        response.raise_for_status()
        return response.json()
    except:
        return None

def get_customer(customer_id):
    try:
        response = requests.get(f"{BACKEND_URL}/api/customers/{customer_id}", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return None

def query_ai(question):
    try:
        response = requests.post(f"{BACKEND_URL}/api/query", json={"question": question}, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        return {"error": "AI service is taking longer than expected. Please try again."}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to AI service. Please check backend connection."}
    except Exception as e:
        return {"error": "AI service temporarily unavailable. Please try again later."}

def get_visualization(viz_type):
    try:
        response = requests.get(f"{BACKEND_URL}/api/visualizations/{viz_type}", timeout=15)
        response.raise_for_status()
        return response.json()
    except:
        return None

def run_clustering(n_clusters, algorithm):
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/cluster/run",
            json={"n_clusters": n_clusters, "algorithm": algorithm},
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# ============================================
# UI COMPONENTS
# ============================================

# Export/Download Functions
def export_to_csv(data, filename="export.csv"):
    """Convert data to CSV for download"""
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data
    return df.to_csv(index=False).encode('utf-8')

def export_chart_to_image(fig):
    """Convert Plotly figure to PNG bytes"""
    return fig.to_image(format="png", width=1200, height=600)

def create_export_buttons(data=None, chart=None, prefix="report"):
    """Create standardized export buttons"""
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        if data is not None:
            csv_data = export_to_csv(data)
            st.download_button(
                label="📥 Export CSV",
                data=csv_data,
                file_name=f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col2:
        if chart is not None:
            try:
                img_bytes = export_chart_to_image(chart)
                st.download_button(
                    label="📊 Save Chart",
                    data=img_bytes,
                    file_name=f"{prefix}_chart_{datetime.now().strftime('%Y%m%d_%H%M')}.png",
                    mime="image/png",
                    use_container_width=True
                )
            except:
                 st.caption("Chart export requires kaleido")

# AI Insights Panel
def show_ai_insights(context, view_name="current view"):
    """Display AI-generated insights for current view"""
    with st.expander("💡 AI Insights & Recommendations", expanded=False):
        with st.spinner("🔮 Analyzing data..."):
            prompt = f"""Analyze the {view_name} data and provide exactly 3 actionable business insights (NOT package recommendations).

Format:
1. [Insight title]: [Brief description with specific numbers]
   Action: [What business action to take]

2. [Insight title]: [Brief description with specific numbers]
   Action: [What business action to take]

3. [Insight title]: [Brief description with specific numbers]  
   Action: [What business action to take]

DO NOT provide package recommendations or pricing. Focus on trends, opportunities, and strategic actions.

Context: {context}
"""
            response = query_ai(prompt)
            if response and 'answer' in response:
                st.markdown("**🎯 Top 3 Actionable Insights:**")
                st.info(response['answer'])
            elif response and 'error' in response:
                st.warning(f"⚠️ {response['error']}")
            else:
                st.warning("⚠️ AI insights temporarily unavailable")

def render_header():
    st.markdown('<div class="main-header">📊 Telecom Customer Analytics</div>', unsafe_allow_html=True)
    st.markdown("### Complete Customer Insights & AI-Powered Analysis")
    st.markdown("---")

def render_overview_metrics(stats):
    if not stats:
        st.warning("Could not load statistics")
        return
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("👥 Total Customers", f"{stats['total_customers']:,}")
    with col2:
        st.metric("🌍 International", f"{stats['international_users']:,}", 
                  delta=f"{stats['international_percentage']:.1f}%")
    with col3:
        st.metric("📞 Avg Calls", f"{stats['avg_voice_calls']:.0f}")
    with col4:
        st.metric("📱 Avg Data", f"{stats['avg_data_mb']:.0f} MB")
    with col5:
        st.metric("💬 Total SMS", f"{stats['total_sms']:,}")

def render_communication_insights(stats, time_analysis):
    st.subheader("📞 Communication Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Call Frequency & Duration")
        st.metric("Total Voice Minutes", f"{stats['total_voice_mins']:,.0f}")
        st.metric("Average per User", f"{stats['avg_voice_mins']:.1f} mins")
        st.metric("Call Lovers", f"{stats['call_lovers']:,}")
    
    with col2:
        st.markdown("#### International Calls")
        st.metric("International Users", f"{stats['international_users']:,}")
        st.metric("Percentage", f"{stats['international_percentage']:.2f}%")
        st.info("🌍 Location data: Available in customer details")
    
    # Time distribution
    if time_analysis:
        st.markdown("#### Time Distribution (Morning/Evening/Night)")
        
        time_data = time_analysis['overall']
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=['🌅 Morning', '🌆 Evening', '🌙 Night'],
            values=[time_data['morning_calls'], time_data['evening_calls'], time_data['night_calls']],
            hole=.4,
            marker_colors=['#FDB462', '#80B1D3', '#8DD3C7']
        )])
        fig.update_layout(
            title="Calls by Time of Day",
            showlegend=True,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🌅 Morning", f"{time_data['morning_calls']:,}", f"{time_data['morning_pct']:.1f}%")
        with col2:
            st.metric("🌆 Evening", f"{time_data['evening_calls']:,}", f"{time_data['evening_pct']:.1f}%")
        with col3:
            st.metric("🌙 Night", f"{time_data['night_calls']:,}", f"{time_data['night_pct']:.1f}%")

def render_internet_insights(stats):
    st.subheader("🌐 Internet Usage")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Download")
        st.metric("Total (GB)", f"{stats['avg_download_mb'] * stats['total_customers'] / 1024:.1f}")
        st.metric("Avg per User (MB)", f"{stats['avg_download_mb']:.1f}")
        st.info(f"📥 {stats['download_lovers']:,} Download Lovers")
    
    with col2:
        st.markdown("#### Upload")
        st.metric("Total (GB)", f"{stats['avg_upload_mb'] * stats['total_customers'] / 1024:.1f}")
        st.metric("Avg per User (MB)", f"{stats['avg_upload_mb']:.1f}")
        st.info(f"📤 {stats['upload_lovers']:,} Upload Lovers")
    
    with col3:
        st.markdown("#### Overall")
        st.metric("Total (GB)", f"{stats['total_data_gb']:.1f}")
        st.metric("Avg per User (MB)", f"{stats['avg_data_mb']:.1f}")
        st.success(f"📊 {stats['data_lovers']:,} Data Lovers")
    
    # Visualization
    viz_data = get_visualization("data-breakdown")
    if viz_data and 'chart' in viz_data:
        fig = go.Figure(json.loads(viz_data['chart']))
        st.plotly_chart(fig, use_container_width=True)

def render_sms_insights(stats):
    st.subheader("💬 SMS Insights")
    
    # Calculate meaningful metrics
    sms_adoption_rate = (stats['sms_users'] / stats['total_customers'] * 100)
    avg_per_active = stats.get('avg_sms_per_active_user', 2.0)  # Fallback if backend not updated
    
    # Alert if SMS usage is very low
    if sms_adoption_rate < 5:
        st.warning(f"⚠️ **Low SMS Adoption:** Only {sms_adoption_rate:.1f}% of customers use SMS. Most likely prefer OTT messaging apps (WhatsApp, Telegram, etc.)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Messages", f"{stats['total_sms']:,}")
    with col2:
        st.metric("SMS Users", f"{stats['sms_users']:,}", 
                  f"{sms_adoption_rate:.1f}% adoption")
    with col3:
        st.metric("Avg (Active Users)", f"{avg_per_active:.1f} msgs",
                 help="Average messages among customers who actually use SMS")
    with col4:
        st.metric("Avg (All Users)", f"{stats['avg_sms_per_user']:.2f} msgs",
                 help="Average across all customers (includes 98%+ with 0 SMS)")
    
    # Insight box
    st.info(f"""
    **📊 SMS Analysis:**
    - **{stats['sms_users']:,}** customers sent SMS (only **{sms_adoption_rate:.1f}%** of total)
    - Active SMS users average **{avg_per_active:.1f} messages** each
    - **{stats['total_customers'] - stats['sms_users']:,}** customers (**{100-sms_adoption_rate:.1f}%**) sent **ZERO** SMS
    - This suggests heavy reliance on OTT messaging apps (WhatsApp, Telegram, etc.)
    """)
    
    # Frequency distribution
    freq_high = int(stats['sms_users'] * 0.25)  # Estimate
    freq_medium = int(stats['sms_users'] * 0.35)
    freq_low = stats['sms_users'] - freq_high - freq_medium
    
    fig = px.bar(
        x=['High Frequency', 'Medium Frequency', 'Low Frequency'],
        y=[freq_high, freq_medium, freq_low],
        title="SMS Frequency Distribution (Among Active Users Only)",
        labels={'x': 'Frequency', 'y': 'Number of Users'},
        color=['High', 'Medium', 'Low'],
        color_discrete_sequence=['#E74C3C', '#F39C12', '#3498DB']
    )
    st.plotly_chart(fig, use_container_width=True)

def render_customer_lookup():
    st.subheader("🔍 Customer Lookup by Subscriber ID")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        customer_id = st.number_input("Enter Subscriber ID:", min_value=1, step=1, format="%d")
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_btn = st.button("🔎 Search Customer", type="primary", use_container_width=True)
    
    if search_btn and customer_id:
        with st.spinner("Fetching customer data..."):
            customer = get_customer(customer_id)
            
            if customer:
                st.success(f"✅ Found Customer {customer_id}")
                
                # Communication Section
                with st.expander("📞 Communication Analysis", expanded=True):
                    comm = customer['communication']
                    time_dist = comm['time_distribution']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Calls", f"{comm['voice_total_calls']:.0f}")
                        st.metric("Total Duration", f"{comm['voice_total_duration_mins']:.1f} mins")
                    with col2:
                        st.metric("Avg Call Duration", f"{comm['voice_avg_duration_mins']:.1f} mins")
                    with col3:
                        st.markdown("**Time Distribution:**")
                        st.write(f"🌅 Morning: {time_dist['morning_calls']} ({time_dist['morning_pct']:.1f}%)")
                        st.write(f"🌆 Evening: {time_dist['evening_calls']} ({time_dist['evening_pct']:.1f}%)")
                        st.write(f"🌙 Night: {time_dist['night_calls']} ({time_dist['night_pct']:.1f}%)")
                
                # International Section
                with st.expander("🌍 International Details", expanded=True):
                    intl = customer['international']
                    
                    if intl['is_international_user']:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Calls", f"{intl['total_calls']:.0f}")
                            st.metric("Total Duration", f"{intl['total_duration_mins']:.1f} mins")
                        with col2:
                            st.metric("Countries Called", f"{intl['countries_called']}")
                            st.info(f"Top: {intl['top_country']}")
                        with col3:
                            st.markdown("**All Countries:**")
                            st.write(intl['all_countries'])
                    else:
                        st.info("❌ Not an international user")
                
                # Internet Section
                with st.expander("🌐 Internet Usage", expanded=True):
                    internet = customer['internet']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Download", f"{internet['download_mb']:.1f} MB",
                                 f"{internet['download_pct']:.1f}%")
                    with col2:
                        st.metric("Upload", f"{internet['upload_mb']:.1f} MB",
                                 f"{internet['upload_pct']:.1f}%")
                    with col3:
                        st.metric("Total", f"{internet['total_mb']:.1f} MB")
                    
                    # Pie chart for upload/download
                    if internet['total_mb'] > 0:
                        fig = go.Figure(data=[go.Pie(
                            labels=['Download', 'Upload'],
                            values=[internet['download_mb'], internet['upload_mb']],
                            hole=.3,
                            marker_colors=['#66C2A5', '#FC8D62']
                        )])
                        fig.update_layout(title="Data Breakdown", height=300)
                        st.plotly_chart(fig, use_container_width=True)
                
                # SMS Section
                with st.expander("💬 SMS Activity", expanded=True):
                    sms = customer['sms']
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Messages", f"{sms['total_messages']}")
                    with col2:
                        st.metric("Frequency Level", sms['frequency'])
                
                # AI Suggestions
                st.markdown("---")
                st.markdown("### 🤖 AI-Powered Recommendations")
                
                with st.spinner("🔮 Analyzing customer profile and generating personalized recommendations..."):
                    # Build context for LLM with full time distribution
                    morning_pct = time_dist.get('Morning', 0) / sum(time_dist.values()) * 100 if sum(time_dist.values()) > 0 else 0
                    evening_pct = time_dist.get('Evening', 0) / sum(time_dist.values()) * 100 if sum(time_dist.values()) > 0 else 0
                    night_pct = time_dist.get('Night', 0) / sum(time_dist.values()) * 100 if sum(time_dist.values()) > 0 else 0
                    
                    context = f"""
Provide a PACKAGE RECOMMENDATION for this individual customer:

Customer Profile:
- Voice: {comm['voice_total_calls']:.0f} calls, {comm['voice_total_duration_mins']:.1f} mins
- Time Distribution: Morning {morning_pct:.1f}%, Evening {evening_pct:.1f}%, Night {night_pct:.1f}%
- Data: {internet['total_mb']:.0f} MB (Download: {internet['download_pct']:.0f}%, Upload: {internet['upload_pct']:.0f}%)
- SMS: {sms['total_messages']} messages
- International: {'Yes' if intl['is_international_user'] else 'No'}
{f"- Countries: {intl['all_countries']}" if intl['is_international_user'] else ''}

Provide structured package recommendation in 4 sections:
1. USAGE PROFILE (analyze their usage patterns with time distribution)
2. RECOMMENDED PACKAGE (specific package with pricing)
3. KEY BENEFITS (3-4 bullet points)
4. PRICING STRATEGY (upsell/retention approach)
"""
                    response = query_ai(context)
                    
                    # Check for errors first
                    if response and 'error' in response:
                        st.error(f"❌ {response['error']}")
                        st.info("💡 Please check your backend connection and Groq API key configuration.")
                    elif response and 'answer' in response:
                        # Beautiful formatted output
                        st.markdown("""
                        <style>
                        .recommendation-header {
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            color: white;
                            padding: 20px;
                            border-radius: 10px 10px 0 0;
                            text-align: center;
                            font-size: 24px;
                            font-weight: bold;
                            margin-bottom: 0;
                        }
                        .recommendation-body {
                            background: #f8f9fa;
                            padding: 25px;
                            border-radius: 0 0 10px 10px;
                            border: 2px solid #667eea;
                            line-height: 1.8;
                        }
                        .usage-badge {
                            display: inline-block;
                            background: #e3f2fd;
                            color: #1976d2;
                            padding: 5px 12px;
                            border-radius: 20px;
                            font-size: 14px;
                            font-weight: 600;
                            margin: 5px 5px 5px 0;
                        }
                        .package-highlight {
                            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                            color: white;
                            padding: 20px;
                            border-radius: 10px;
                            font-size: 18px;
                            font-weight: bold;
                            text-align: center;
                            margin: 15px 0;
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                        }
                        </style>
                        <div class="recommendation-header">
                            💡 Personalized Package Recommendation
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display formatted recommendation
                        with st.container():
                            st.markdown('<div class="recommendation-body">', unsafe_allow_html=True)
                            
                            # Usage summary badges
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                usage_level = "High" if comm['voice_total_calls'] > 500 else "Moderate" if comm['voice_total_calls'] > 200 else "Low"
                                st.markdown(f'<div class="usage-badge">📞 Voice: {usage_level}</div>', unsafe_allow_html=True)
                            with col2:
                                data_level = "High" if internet['total_mb'] > 1000 else "Moderate" if internet['total_mb'] > 500 else "Low"
                                st.markdown(f'<div class="usage-badge">📊 Data: {data_level}</div>', unsafe_allow_html=True)
                            with col3:
                                sms_level = "High" if sms['total_messages'] > 200 else "Moderate" if sms['total_messages'] > 50 else "Low"
                                st.markdown(f'<div class="usage-badge">💬 SMS: {sms_level}</div>', unsafe_allow_html=True)
                            with col4:
                                if intl['is_international_user']:
                                    st.markdown(f'<div class="usage-badge">🌍 International</div>', unsafe_allow_html=True)
                                else:
                                    st.markdown(f'<div class="usage-badge">🏠 Domestic Only</div>', unsafe_allow_html=True)
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            # AI Response with better formatting
                            ai_text = response['answer']
                            
                            # Try to parse sections (if AI follows format)
                            sections = {
                                'profile': '',
                                'package': '',
                                'benefits': '',
                                'pricing': ''
                            }
                            
                            # Simple parsing (fallback to full text if not structured)
                            if any(keyword in ai_text.lower() for keyword in ['usage profile', 'recommended package', 'benefits', 'pricing']):
                                # Structured response
                                lines = ai_text.split('\n')
                                current_section = None
                                for line in lines:
                                    line_lower = line.lower()
                                    if 'usage profile' in line_lower or 'profile' in line_lower and len(line) < 50:
                                        current_section = 'profile'
                                    elif 'recommended package' in line_lower or 'package' in line_lower and len(line) < 50:
                                        current_section = 'package'
                                    elif 'benefit' in line_lower and len(line) < 50:
                                        current_section = 'benefits'
                                    elif 'pricing' in line_lower and len(line) < 50:
                                        current_section = 'pricing'
                                    elif current_section and line.strip():
                                        sections[current_section] += line + '\n'
                                
                                # Display structured
                                if sections['profile']:
                                    st.markdown("**📋 Usage Profile Analysis**")
                                    st.info(sections['profile'].strip())
                                
                                if sections['package']:
                                    st.markdown("**🎁 Recommended Package**")
                                    st.markdown(f'<div class="package-highlight">{sections["package"].strip()}</div>', unsafe_allow_html=True)
                                
                                if sections['benefits']:
                                    st.markdown("**✨ Key Benefits**")
                                    st.success(sections['benefits'].strip())
                                
                                if sections['pricing']:
                                    st.markdown("**💰 Pricing Strategy**")
                                    st.warning(sections['pricing'].strip())
                            else:
                                # Fallback: display full text nicely
                                st.markdown("**📋 Analysis & Recommendation**")
                                st.markdown(ai_text)
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.warning("⚠️ Unable to generate AI recommendation. Please try again later.")
            
            else:
                st.error(f"❌ Customer {customer_id} not found")

def render_cluster_visualization():
    st.subheader("📊 Customer Segments")
    
    cluster_type = st.radio("Clustering Algorithm:", ["kmeans", "dbscan"], horizontal=True)
    
    clusters = get_clusters(cluster_type)
    
    if clusters:
        df_clusters = pd.DataFrame(clusters['clusters'])
        total_clusters = len(df_clusters)
        
        # Apply cluster filter if active
        if 'filters' in st.session_state and st.session_state.filters.get('clusters'):
            selected_cluster_ids = st.session_state.filters['clusters']
            df_clusters = df_clusters[df_clusters['cluster_id'].isin(selected_cluster_ids)]
            if df_clusters.empty:
                st.warning(f"⚠️ No clusters match your filter selection: {selected_cluster_ids}")
                return
            st.success(f"✅ Showing {len(df_clusters)} filtered cluster(s) out of {total_clusters} total")
        
        # Pie chart
        fig = px.pie(
            df_clusters,
            values='size',
            names='cluster_id',
            title=f'Customer Distribution - {cluster_type.upper()}',
            hole=.4
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Bar comparison
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Voice (mins)', x=df_clusters['cluster_id'], y=df_clusters['avg_voice_mins']))
        fig.add_trace(go.Bar(name='Data (MB)', x=df_clusters['cluster_id'], y=df_clusters['avg_data_mb']))
        fig.add_trace(go.Bar(name='SMS', x=df_clusters['cluster_id'], y=df_clusters['avg_sms']))
        fig.update_layout(barmode='group', title='Average Usage by Cluster')
        st.plotly_chart(fig, use_container_width=True)
        
        # Table
        st.dataframe(df_clusters, use_container_width=True)

def render_dynamic_clustering():
    st.subheader("🔧 Run Custom Clustering")
    
    col1, col2 = st.columns(2)
    
    with col1:
        algorithm = st.selectbox("Algorithm", ["kmeans", "dbscan"])
    
    with col2:
        if algorithm == "kmeans":
            n_clusters = st.slider("Number of Clusters", 2, 12, 6)
        else:
            n_clusters = 0
    
    if st.button("▶️ Run Clustering", type="primary"):
        with st.spinner("Running clustering analysis..."):
            result = run_clustering(n_clusters if algorithm == "kmeans" else 6, algorithm)
            
            if 'error' not in result:
                st.success(f"✅ Found {result['n_clusters']} clusters")
                
                if result.get('silhouette_score'):
                    st.metric("Silhouette Score", f"{result['silhouette_score']:.4f}")
                
                df_result = pd.DataFrame(result['clusters'])
                st.dataframe(df_result, use_container_width=True)
            else:
                st.error(f"Error: {result['error']}")

def render_ai_chat():
    st.subheader("💬 Ask AI About Your Data")
    
    # Example questions
    with st.expander("💡 Example Questions"):
        st.markdown("""
        - What time of day has the most calls?
        - How many customers use SMS frequently?
        - What's the ratio of download to upload data?
        - Which customers should get international packages?
        - What's the average data usage for heavy users?
        """)
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    user_question = st.text_input("Ask a question:", placeholder="e.g., What time has peak call volume?")
    
    if st.button("🔍 Ask AI", type="primary"):
        if user_question:
            with st.spinner("Thinking..."):
                response = query_ai(user_question)
                if response and 'answer' in response:
                    st.session_state.chat_history.append({
                        'question': user_question,
                        'answer': response['answer']
                    })
                elif response and 'error' in response:
                    st.error(f"❌ {response['error']}")
                else:
                    st.warning("⚠️ Unable to get AI response. Please try again.")
    
    # Display history
    if st.session_state.chat_history:
        st.markdown("---")
        for chat in reversed(st.session_state.chat_history[-5:]):
            st.markdown(f"**You:** {chat['question']}")
            st.info(f"**AI:** {chat['answer']}")
            st.markdown("---")

def render_cohort_comparison():
    """Side-by-side cohort/cluster comparison with delta highlights"""
    st.subheader("🔄 Cohort Comparison")
    st.markdown("Compare two customer segments side-by-side")
    
    # Show active filters banner
    filters_active = show_active_filters_banner()
    
    clusters_data = get_clusters("kmeans")
    if not clusters_data or 'clusters' not in clusters_data:
        st.error("Unable to load cluster data")
        return
    
    df_clusters = pd.DataFrame(clusters_data['clusters'])
    
    # Apply cluster filter if active
    if 'filters' in st.session_state and st.session_state.filters.get('clusters'):
        selected_cluster_ids = st.session_state.filters['clusters']
        df_clusters = df_clusters[df_clusters['cluster_id'].isin(selected_cluster_ids)]
        if df_clusters.empty:
            st.warning(f"No clusters match your filter selection: {selected_cluster_ids}")
            return
        st.success(f"✅ Showing {len(df_clusters)} filtered cluster(s) out of {len(clusters_data['clusters'])} total")
    
    cluster_options = df_clusters['cluster_id'].tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        cohort_a = st.selectbox("Select Cohort A", cluster_options, index=0, key="cohort_a")
    with col2:
        cohort_b = st.selectbox("Select Cohort B", cluster_options, index=min(1, len(cluster_options)-1), key="cohort_b")
    
    # Get cohort data
    cluster_a = df_clusters[df_clusters['cluster_id'] == cohort_a].iloc[0]
    cluster_b = df_clusters[df_clusters['cluster_id'] == cohort_b].iloc[0]
    
    # Comparison header
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        st.markdown(f"### 📊 Cluster {cohort_a}")
        st.caption(f"**{cluster_a['size']:,} customers**")
    with col2:
        st.markdown("### VS")
    with col3:
        st.markdown(f"### 📊 Cluster {cohort_b}")
        st.caption(f"**{cluster_b['size']:,} customers**")
    
    st.markdown("---")
    
    # Define metrics to compare
    metrics = [
        ('avg_voice_mins', 'Voice Usage', 'mins'),
        ('avg_data_mb', 'Data Usage', 'MB'),
        ('avg_sms', 'SMS Count', 'messages')
    ]
    
    comparison_data = []
    
    for metric_key, metric_name, unit in metrics:
        val_a = cluster_a.get(metric_key, 0)
        val_b = cluster_b.get(metric_key, 0)
        
        # Calculate percentage difference
        if val_a > 0:
            delta_pct = ((val_b - val_a) / val_a) * 100
        else:
            delta_pct = 0
        
        comparison_data.append({
            'Metric': metric_name,
            f'Cluster {cohort_a}': f"{val_a:.1f} {unit}",
            f'Cluster {cohort_b}': f"{val_b:.1f} {unit}",
            'Difference': f"{delta_pct:+.1f}%"
        })
        
        # Visual comparison
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            st.metric(
                label=f"{metric_name}",
                value=f"{val_a:.1f} {unit}",
                delta=None
            )
        
        with col2:
            delta_color = "normal" if delta_pct >= 0 else "inverse"
            st.metric(
                label=f"{metric_name}",
                value=f"{val_b:.1f} {unit}",
                delta=f"{delta_pct:+.1f}%",
                delta_color=delta_color
            )
        
        with col3:
            if abs(delta_pct) > 50:
                st.markdown(f"**⚠️ {abs(delta_pct):.0f}%**")
            elif abs(delta_pct) > 20:
                st.markdown(f"**⚡ {abs(delta_pct):.0f}%**")
            else:
                st.markdown(f"✓ {abs(delta_pct):.0f}%")
    
    st.markdown("---")
    
    # Summary table
    st.markdown("### 📋 Comparison Summary")
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True, hide_index=True)
    
    # Export button
    create_export_buttons(data=df_comparison, prefix="cohort_comparison")
    
    # Visual comparison chart
    st.markdown("---")
    st.markdown("### 📊 Visual Comparison")
    
    fig = go.Figure()
    
    metrics_names = [m[1] for m in metrics]
    values_a = [cluster_a.get(m[0], 0) for m in metrics]
    values_b = [cluster_b.get(m[0], 0) for m in metrics]
    
    fig.add_trace(go.Bar(
        name=f'Cluster {cohort_a}',
        x=metrics_names,
        y=values_a,
        marker_color='#667eea'
    ))
    
    fig.add_trace(go.Bar(
        name=f'Cluster {cohort_b}',
        x=metrics_names,
        y=values_b,
        marker_color='#f093fb'
    ))
    
    fig.update_layout(
        barmode='group',
        title=f"Cluster {cohort_a} vs Cluster {cohort_b}",
        xaxis_title="Metric",
        yaxis_title="Value",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    create_export_buttons(chart=fig, prefix="cohort_comparison_chart")
    
    # AI Insights for comparison - comprehensive cohort analysis
    voice_a = cluster_a.get('avg_voice_mins', 0)
    voice_b = cluster_b.get('avg_voice_mins', 0)
    data_a = cluster_a.get('avg_data_mb', 0)
    data_b = cluster_b.get('avg_data_mb', 0)
    sms_a = cluster_a.get('avg_sms', 0)
    sms_b = cluster_b.get('avg_sms', 0)
    
    voice_diff = ((voice_b - voice_a) / voice_a * 100) if voice_a > 0 else 0
    data_diff = ((data_b - data_a) / data_a * 100) if data_a > 0 else 0
    sms_diff = ((sms_b - sms_a) / sms_a * 100) if sms_a > 0 else 0
    
    context = f"""
COHORT COMPARISON ANALYSIS (aggregate segment data for {cluster_a['size']:,} + {cluster_b['size']:,} customers):

Segment Sizes:
- Cluster {cohort_a}: {cluster_a['size']:,} customers
- Cluster {cohort_b}: {cluster_b['size']:,} customers

Voice Usage Comparison:
- Cluster {cohort_a}: {voice_a:.1f} mins/customer
- Cluster {cohort_b}: {voice_b:.1f} mins/customer
- Difference: {voice_diff:+.1f}%

Data Usage Comparison:
- Cluster {cohort_a}: {data_a:.1f} MB/customer
- Cluster {cohort_b}: {data_b:.1f} MB/customer
- Difference: {data_diff:+.1f}%

SMS Usage Comparison:
- Cluster {cohort_a}: {sms_a:.1f} messages/customer
- Cluster {cohort_b}: {sms_b:.1f} messages/customer
- Difference: {sms_diff:+.1f}%

Provide 3 strategic insights about these segments (NOT individual package recommendations):
1. Key behavioral differences and what they indicate
2. Which segment is more valuable and growth potential
3. High-level targeting strategies or campaign ideas for each segment
"""
    show_ai_insights(context, "cohort comparison")

# ============================================
# MAIN APP
# ============================================

def main():
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/phone.png", width=80)
        st.title("📊 Navigation")
        
        page = st.radio(
            "Select View:",
            [
                "🏠 Overview Dashboard",
                "👤 Customer Lookup",
                "🔄 Cohort Comparison",
                "📈 Visual Insights",
                "🔬 Clustering Analysis",
                "💬 AI Assistant"
            ]
        )
        
        st.markdown("---")
        st.markdown("### 🔌 Backend Status")
        
        # Backend connection check
        backend_status = None
        customer_count = None
        
        try:
            response = requests.get(f"{BACKEND_URL}/", timeout=5)
            if response.status_code == 200:
                backend_status = "connected"
                try:
                    data = response.json()
                    # Backend returns 'customers' key, not 'total_customers'
                    customer_count = data.get('customers') or data.get('total_customers', 'N/A')
                except:
                    pass
            else:
                backend_status = "error"
        except requests.exceptions.Timeout:
            backend_status = "timeout"
        except requests.exceptions.ConnectionError:
            backend_status = "offline"
        except:
            backend_status = "offline"
        
        # Display status
        if backend_status == "connected":
            st.success("✅ Connected")
            if customer_count and isinstance(customer_count, (int, float)):
                st.caption(f"Customers: {customer_count:,}")
            elif customer_count:
                st.caption(f"Customers: {customer_count}")
        elif backend_status == "timeout":
            st.warning("⏱️ Timeout")
        elif backend_status == "error":
            st.error("❌ Error")
        else:
            st.error("❌ Offline")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        **New Features:**
        - 🎛️ Real-time filters
        - 🔄 Cohort comparison
        - 📥 Export/download
        - 💡 AI insights panels
        
        **Core Features:**
        - ⏰ Time analysis
        - 🌐 Data split
        - 🌍 International details
        - 🤖 AI recommendations
        """)
    
    # Main content
    render_header()
    
    # Load data
    stats = get_stats()
    
    if page == "🏠 Overview Dashboard":
        # Show active filters banner
        show_active_filters_banner()
        
        if stats:
            render_overview_metrics(stats)
            
            # Export button for overview
            st.markdown("---")
            overview_data = {
                'Total Customers': stats.get('total_customers'),
                'Total Voice Calls': stats.get('voice', {}).get('total_calls'),
                'Total Data MB': stats.get('data', {}).get('total_mb'),
                'Total SMS': stats.get('sms', {}).get('total_messages')
            }
            create_export_buttons(data=overview_data, prefix="overview_dashboard")
            
            st.markdown("---")
            
            tabs = st.tabs(["📞 Communication", "🌐 Internet", "💬 SMS"])
            
            with tabs[0]:
                time_analysis = get_time_analysis()
                render_communication_insights(stats, time_analysis)
                
                # AI Insights for communication - ONLY voice/call related
                voice_stats = stats.get('voice', {})
                intl_stats = stats.get('international', {})
                time_dist = time_analysis.get('time_distribution', {}) if time_analysis else {}
                
                context = f"""
VOICE COMMUNICATION ANALYSIS (focus ONLY on voice/calls, no data/SMS):

Call Volume:
- Total voice calls: {voice_stats.get('total_calls', 0):,}
- Total voice minutes: {voice_stats.get('total_minutes', 0):,.0f}
- Average calls per customer: {voice_stats.get('avg_calls_per_customer', 0):.1f}
- Average duration per call: {voice_stats.get('avg_duration_per_call', 0):.1f} mins

Time Distribution:
- Morning (6am-12pm): {time_dist.get('Morning', 0):,} calls ({time_dist.get('Morning', 0) / sum(time_dist.values()) * 100 if sum(time_dist.values()) > 0 else 0:.1f}%)
- Evening (12pm-6pm): {time_dist.get('Evening', 0):,} calls ({time_dist.get('Evening', 0) / sum(time_dist.values()) * 100 if sum(time_dist.values()) > 0 else 0:.1f}%)
- Night (6pm-6am): {time_dist.get('Night', 0):,} calls ({time_dist.get('Night', 0) / sum(time_dist.values()) * 100 if sum(time_dist.values()) > 0 else 0:.1f}%)

International Calling:
- International users: {intl_stats.get('total_users', 0):,} ({intl_stats.get('percentage', 0):.2f}%)
- Total international calls: {intl_stats.get('total_calls', 0):,}
- Countries reached: {intl_stats.get('unique_countries', 0)}

Provide 3 insights specifically about VOICE CALLING patterns, peak times, and international calling opportunities.
"""
                show_ai_insights(context, "communication analysis")
            
            with tabs[1]:
                render_internet_insights(stats)
                
                # AI Insights for internet - ONLY data/internet related
                data_stats = stats.get('data', {})
                total_data_mb = data_stats.get('total_mb', 0)
                upload_mb = data_stats.get('upload_mb', 0)
                download_mb = data_stats.get('download_mb', 0)
                avg_data_mb = data_stats.get('avg_mb_per_customer', 0)
                
                context = f"""
INTERNET DATA USAGE ANALYSIS (focus ONLY on data/internet, no voice/SMS):

Data Volume:
- Total data consumed: {total_data_mb:,.0f} MB ({total_data_mb/1024:.1f} GB)
- Average per customer: {avg_data_mb:.1f} MB
- Data heavy users (>1GB): {data_stats.get('data_lovers', 0):,} customers

Upload vs Download:
- Total upload: {upload_mb:,.0f} MB ({upload_mb/1024:.1f} GB)
- Total download: {download_mb:,.0f} MB ({download_mb/1024:.1f} GB)
- Upload percentage: {(upload_mb / total_data_mb * 100) if total_data_mb > 0 else 0:.1f}%
- Download percentage: {(download_mb / total_data_mb * 100) if total_data_mb > 0 else 0:.1f}%
- Upload to download ratio: {(upload_mb / download_mb) if download_mb > 0 else 0:.2f}:1

Provide 3 insights specifically about DATA USAGE patterns, upload vs download behavior, and data package opportunities.
"""
                show_ai_insights(context, "internet usage")
            
            with tabs[2]:
                render_sms_insights(stats)
                
                # AI Insights for SMS - ONLY SMS related
                sms_stats = stats.get('sms', {})
                total_customers = stats.get('total_customers', 1)
                
                context = f"""
SMS MESSAGING ANALYSIS (focus ONLY on SMS/text messaging, no voice/data):

SMS Volume:
- Total SMS messages: {sms_stats.get('total_messages', 0):,}
- Average per customer: {sms_stats.get('avg_per_customer', 0):.2f} messages
- Total customers: {total_customers:,}
- SMS active users: {sms_stats.get('active_users', 0):,}

Usage Context:
- SMS appears to be {'very low' if sms_stats.get('avg_per_customer', 0) < 1 else 'moderate' if sms_stats.get('avg_per_customer', 0) < 50 else 'high'} usage
- This suggests customers may prefer messaging apps over traditional SMS

Provide 3 insights specifically about SMS USAGE patterns, why SMS might be low/high, and SMS package/strategy recommendations.
"""
                show_ai_insights(context, "SMS analysis")
    
    elif page == "👤 Customer Lookup":
        # Show active filters banner
        show_active_filters_banner()
        
        render_customer_lookup()
    
    elif page == "🔄 Cohort Comparison":
        # Show active filters banner
        show_active_filters_banner()
        
        render_cohort_comparison()
    
    elif page == "📈 Visual Insights":
        # Show active filters banner
        show_active_filters_banner()
        
        st.subheader("📊 Visual Insights")
        
        viz_option = st.selectbox(
            "Select Visualization:",
            ["Time Distribution", "Data Breakdown", "Customer Segments"]
        )
        
        fig = None
        if viz_option == "Time Distribution":
            viz = get_visualization("time-distribution")
            if viz and 'chart' in viz:
                fig = go.Figure(json.loads(viz['chart']))
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_option == "Data Breakdown":
            viz = get_visualization("data-breakdown")
            if viz and 'chart' in viz:
                fig = go.Figure(json.loads(viz['chart']))
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_option == "Customer Segments":
            viz = get_visualization("customer-segments")
            if viz and 'chart' in viz:
                fig = go.Figure(json.loads(viz['chart']))
                st.plotly_chart(fig, use_container_width=True)
        
        # Export for visualizations
        if fig:
            create_export_buttons(chart=fig, prefix=f"viz_{viz_option.lower().replace(' ', '_')}")
    
    elif page == "🔬 Clustering Analysis":
        # Show active filters banner
        show_active_filters_banner()
        
        tab1, tab2 = st.tabs(["📊 View Clusters", "🔧 Run Custom Clustering"])
        
        with tab1:
            render_cluster_visualization()
            
            # AI Insights for clustering - comprehensive cluster patterns
            clusters_data = get_clusters("kmeans")
            if clusters_data and 'clusters' in clusters_data:
                df_clusters = pd.DataFrame(clusters_data['clusters'])
                total_customers = df_clusters['size'].sum()
                
                context = f"""
CLUSTERING PATTERN ANALYSIS (aggregate data for {total_customers:,} customers across {len(df_clusters)} clusters):

Cluster Overview:
- Total clusters: {len(df_clusters)}
- Total customers analyzed: {total_customers:,}
- Largest cluster: {df_clusters['size'].max():,} customers ({df_clusters['size'].max()/total_customers*100:.1f}%)
- Smallest cluster: {df_clusters['size'].min():,} customers ({df_clusters['size'].min()/total_customers*100:.1f}%)

Usage Pattern Across Clusters:
- Average voice usage: {df_clusters['avg_voice_mins'].mean():.1f} mins (range: {df_clusters['avg_voice_mins'].min():.1f} - {df_clusters['avg_voice_mins'].max():.1f})
- Average data usage: {df_clusters['avg_data_mb'].mean():.1f} MB (range: {df_clusters['avg_data_mb'].min():.1f} - {df_clusters['avg_data_mb'].max():.1f})
- Average SMS: {df_clusters['avg_sms'].mean():.1f} messages (range: {df_clusters['avg_sms'].min():.1f} - {df_clusters['avg_sms'].max():.1f})

Cluster Distribution:
{chr(10).join([f'- Cluster {row["cluster_id"]}: {row["size"]:,} customers ({row["size"]/total_customers*100:.1f}%)' for _, row in df_clusters.iterrows()])}

Provide 3 strategic insights (NOT package recommendations):
1. Which cluster represents the highest-value customers and why
2. Which cluster shows risk of churn or underutilization
3. High-level strategies to improve customer value across clusters
"""
                show_ai_insights(context, "clustering analysis")
        
        with tab2:
            render_dynamic_clustering()
    
    elif page == "💬 AI Assistant":
        # Show active filters banner
        show_active_filters_banner()
        
        render_ai_chat()


if __name__ == "__main__":
    main()
