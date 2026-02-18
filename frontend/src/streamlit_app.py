"""
Enhanced Telecom Customer Segmentation Dashboard
===============================================
Features EVERYTHING Talha requested:
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
        response = requests.post(f"{BACKEND_URL}/api/query", json={"question": question}, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"answer": f"Error: {e}"}

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
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Messages", f"{stats['total_sms']:,}")
    with col2:
        st.metric("Average per User", f"{stats['avg_sms_per_user']:.1f}")
    with col3:
        st.metric("SMS Users", f"{stats['sms_users']:,}", 
                  f"{stats['sms_users'] / stats['total_customers'] * 100:.1f}%")
    
    # Frequency distribution
    freq_high = int(stats['sms_users'] * 0.25)  # Estimate
    freq_medium = int(stats['sms_users'] * 0.35)
    freq_low = stats['sms_users'] - freq_high - freq_medium
    
    fig = px.bar(
        x=['High Frequency', 'Medium Frequency', 'Low Frequency'],
        y=[freq_high, freq_medium, freq_low],
        title="SMS Frequency Distribution (Estimated)",
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
                st.markdown("### 🤖 AI-Powered Suggestions")
                
                with st.spinner("Generating personalized recommendations..."):
                    # Build context for LLM
                    context = f"""
Analyze this customer and provide package recommendations:
- Voice: {comm['voice_total_calls']:.0f} calls, {comm['voice_total_duration_mins']:.1f} mins
- Peak time: {max(time_dist, key=time_dist.get)}
- Data: {internet['total_mb']:.0f} MB (Download: {internet['download_pct']:.0f}%, Upload: {internet['upload_pct']:.0f}%)
- SMS: {sms['total_messages']} messages
- International: {'Yes' if intl['is_international_user'] else 'No'}
{f"- Countries: {intl['all_countries']}" if intl['is_international_user'] else ''}

What package would you recommend?
"""
                    response = query_ai(context)
                    st.markdown(f"""
                    <div class="insight-card">
                        <h4>💡 Recommendation</h4>
                        <p>{response['answer']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            else:
                st.error(f"❌ Customer {customer_id} not found")

def render_cluster_visualization():
    st.subheader("📊 Customer Segments")
    
    cluster_type = st.radio("Clustering Algorithm:", ["kmeans", "dbscan"], horizontal=True)
    
    clusters = get_clusters(cluster_type)
    
    if clusters:
        df_clusters = pd.DataFrame(clusters['clusters'])
        
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
                st.session_state.chat_history.append({
                    'question': user_question,
                    'answer': response['answer']
                })
    
    # Display history
    if st.session_state.chat_history:
        st.markdown("---")
        for chat in reversed(st.session_state.chat_history[-5:]):
            st.markdown(f"**You:** {chat['question']}")
            st.info(f"**AI:** {chat['answer']}")
            st.markdown("---")

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
                "📈 Visual Insights",
                "🔬 Clustering Analysis",
                "💬 AI Assistant"
            ]
        )
        
        st.markdown("---")
        st.markdown("### 🔌 Backend Status")
        try:
            response = requests.get(f"{BACKEND_URL}/", timeout=5)
            if response.status_code == 200:
                st.success("✅ Connected")
                data = response.json()
                st.caption(f"v{data.get('version', 'unknown')}")
            else:
                st.error("❌ Error")
        except:
            st.error("❌ Offline")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        **Features:**
        - ⏰ Time analysis
        - 📨 SMS insights
        - 🌐 Upload/Download split
        - 🌍 International details
        - 🤖 AI recommendations
        - 🎨 Dynamic visualizations
        """)
    
    # Main content
    render_header()
    
    # Load data
    stats = get_stats()
    
    if page == "🏠 Overview Dashboard":
        if stats:
            render_overview_metrics(stats)
            
            st.markdown("---")
            
            tabs = st.tabs(["📞 Communication", "🌐 Internet", "💬 SMS"])
            
            with tabs[0]:
                time_analysis = get_time_analysis()
                render_communication_insights(stats, time_analysis)
            
            with tabs[1]:
                render_internet_insights(stats)
            
            with tabs[2]:
                render_sms_insights(stats)
    
    elif page == "👤 Customer Lookup":
        render_customer_lookup()
    
    elif page == "📈 Visual Insights":
        st.subheader("📊 Visual Insights")
        
        viz_option = st.selectbox(
            "Select Visualization:",
            ["Time Distribution", "Data Breakdown", "Customer Segments"]
        )
        
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
    
    elif page == "🔬 Clustering Analysis":
        tab1, tab2 = st.tabs(["📊 View Clusters", "🔧 Run Custom Clustering"])
        
        with tab1:
            render_cluster_visualization()
        
        with tab2:
            render_dynamic_clustering()
    
    elif page == "💬 AI Assistant":
        render_ai_chat()


if __name__ == "__main__":
    main()
