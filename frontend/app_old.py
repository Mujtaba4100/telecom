"""
Telecom Customer Segmentation Dashboard
=======================================
Streamlit frontend that connects to FastAPI backend.
Features:
- Interactive dashboard
- Chat with AI (Gemini)
- Cluster visualizations
- Package recommendations
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional
import json
import time

# ============================================
# CONFIGURATION
# ============================================

# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="Telecom Customer Segmentation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend URL - change this to your deployed backend
try:
    BACKEND_URL = st.secrets["BACKEND_URL"]
except:
    BACKEND_URL = "http://localhost:7860"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .cluster-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background: #e3f2fd;
        text-align: right;
    }
    .ai-message {
        background: #f5f5f5;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# API FUNCTIONS
# ============================================

@st.cache_data(ttl=300)
def get_stats():
    """Fetch overall statistics from backend"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/stats", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error connecting to backend: {e}")
        return None

@st.cache_data(ttl=300)
def get_clusters(cluster_type: str = "kmeans"):
    """Fetch cluster information from backend"""
    try:
        response = requests.get(
            f"{BACKEND_URL}/api/clusters", 
            params={"cluster_type": cluster_type},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching clusters: {e}")
        return None

@st.cache_data(ttl=300)
def get_packages():
    """Fetch package recommendations from backend"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/packages", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching packages: {e}")
        return None

def query_ai(question: str):
    """Send question to AI via backend"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/query",
            json={"question": question},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"answer": f"Error: {e}", "data": None}

def search_customers(query: str, limit: int = 10):
    """Semantic search for customers"""
    try:
        response = requests.get(
            f"{BACKEND_URL}/api/search",
            params={"query": query, "limit": limit},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"results": [], "error": str(e)}

def get_customer(customer_id: int):
    """Get specific customer details"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/customers/{customer_id}", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return None

# ============================================
# UI COMPONENTS
# ============================================

def render_header():
    """Render main header"""
    st.markdown('<div class="main-header">📊 Telecom Customer Segmentation</div>', unsafe_allow_html=True)
    st.markdown("---")

def render_metrics(stats):
    """Render key metrics"""
    if stats is None:
        st.warning("Could not load statistics")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="👥 Total Customers",
            value=f"{stats['total_customers']:,}"
        )
    
    with col2:
        st.metric(
            label="🌍 International Users",
            value=f"{stats['international_users']:,}",
            delta=f"{stats['international_percentage']:.1f}%"
        )
    
    with col3:
        st.metric(
            label="📞 Avg Voice (mins)",
            value=f"{stats['avg_voice_mins']:.1f}"
        )
    
    with col4:
        st.metric(
            label="📱 Avg Data (MB)",
            value=f"{stats['avg_data_mb']:.1f}"
        )

def render_cluster_chart(clusters_data):
    """Render cluster distribution chart"""
    if clusters_data is None:
        return
    
    df = pd.DataFrame(clusters_data['clusters'])
    
    # Pie chart
    fig = px.pie(
        df, 
        values='size', 
        names='user_type',
        title='Customer Segments Distribution',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

def render_cluster_comparison(clusters_data):
    """Render cluster comparison bar chart"""
    if clusters_data is None:
        return
    
    df = pd.DataFrame(clusters_data['clusters'])
    
    # Bar chart comparing metrics
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Avg Voice (mins)',
        x=df['user_type'],
        y=df['avg_voice_mins'],
        marker_color='#636EFA'
    ))
    
    fig.add_trace(go.Bar(
        name='Avg Data (GB)',
        x=df['user_type'],
        y=df['avg_data_gb'],
        marker_color='#EF553B'
    ))
    
    fig.add_trace(go.Bar(
        name='Avg Intl Calls',
        x=df['user_type'],
        y=df['avg_intl_calls'],
        marker_color='#00CC96'
    ))
    
    fig.update_layout(
        barmode='group',
        title='Usage Comparison Across Segments',
        xaxis_title='Customer Segment',
        yaxis_title='Average Value'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_packages(packages_data):
    """Render package recommendations"""
    if packages_data is None:
        return
    
    st.subheader("📦 Recommended Packages by Segment")
    
    cols = st.columns(3)
    
    for i, pkg in enumerate(packages_data['packages']):
        with cols[i % 3]:
            with st.container():
                st.markdown(f"""
                <div class="cluster-card">
                    <h4>🎯 {pkg['user_type']}</h4>
                    <p><strong>Customers:</strong> {pkg['size']:,} ({pkg['percentage']}%)</p>
                    <hr>
                    <p>📱 <strong>Data:</strong> {pkg['package']['data_gb']} GB</p>
                    <p>📞 <strong>Calls:</strong> {pkg['package']['call_minutes']} mins</p>
                    {f"<p>✈️ <strong>International:</strong> {pkg['package']['intl_minutes']} mins</p>" if pkg['package']['includes_international'] else ""}
                </div>
                """, unsafe_allow_html=True)

def render_chat_interface():
    """Render AI chat interface"""
    st.subheader("💬 Ask AI About Your Data")
    
    # Example questions
    with st.expander("💡 Example Questions"):
        st.markdown("""
        - How many international users are there?
        - Which cluster has the highest data usage?
        - What package would you recommend for heavy data users?
        - Compare cluster 0 and cluster 1
        - What are the characteristics of each segment?
        """)
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat input
    user_question = st.text_input(
        "Ask a question about your customer data:",
        placeholder="e.g., Which cluster has the most international users?"
    )
    
    if st.button("🔍 Ask", type="primary"):
        if user_question:
            with st.spinner("Thinking..."):
                response = query_ai(user_question)
                
                # Add to history
                st.session_state.chat_history.append({
                    'question': user_question,
                    'answer': response['answer']
                })
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        for chat in reversed(st.session_state.chat_history[-5:]):  # Last 5 conversations
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong> {chat['question']}
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="chat-message ai-message">
                <strong>AI:</strong> {chat['answer']}
            </div>
            """, unsafe_allow_html=True)

def render_customer_lookup():
    """Render customer lookup interface"""
    st.subheader("🔎 Customer Lookup")
    
    customer_id = st.number_input("Enter Customer ID:", min_value=1, step=1)
    
    if st.button("Search Customer"):
        with st.spinner("Searching..."):
            customer = get_customer(customer_id)
            
            if customer:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Customer Details")
                    st.json({
                        "ID": customer['subscriberid'],
                        "Voice (mins)": customer['voice_total_duration_mins'],
                        "Calls": customer['voice_total_calls'],
                        "Data (MB)": customer['data_total_mb'],
                        "International Calls": customer['intl_total_calls'],
                        "Is International": "Yes" if customer['is_international_user'] else "No"
                    })
                
                with col2:
                    st.markdown("### Segment Info")
                    if customer['cluster_info']:
                        st.info(f"**Segment:** {customer['cluster_info']['user_type']}")
                        st.markdown(f"""
                        **Cluster:** {customer['kmeans_cluster']}  
                        **Cluster Size:** {customer['cluster_info']['size']:,} customers  
                        **Recommended Package:**
                        - Data: {customer['cluster_info']['package_recommendation']['data_gb']} GB
                        - Calls: {customer['cluster_info']['package_recommendation']['call_minutes']} mins
                        """)
            else:
                st.error(f"Customer {customer_id} not found")

def render_semantic_search():
    """Render semantic search interface"""
    st.subheader("🔍 Semantic Search")
    
    search_query = st.text_input(
        "Search customers by description:",
        placeholder="e.g., high data usage international caller"
    )
    
    if st.button("Search", key="semantic_search"):
        if search_query:
            with st.spinner("Searching..."):
                results = search_customers(search_query)
                
                if results.get('results'):
                    st.markdown(f"**Found {len(results['results'])} matching customers:**")
                    
                    for r in results['results']:
                        with st.expander(f"Customer {r['customer_id']} (Score: {r['similarity_score']:.2%})"):
                            st.write(r['description'])
                else:
                    st.warning("No results found")


# ============================================
# MAIN APP
# ============================================

def main():
    """Main app function"""
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/phone.png", width=80)
        st.title("Navigation")
        
        page = st.radio(
            "Select Page:",
            ["📊 Dashboard", "💬 AI Chat", "🔎 Customer Lookup", "📦 Packages"]
        )
        
        st.markdown("---")
        st.markdown("### Backend Status")
        try:
            response = requests.get(f"{BACKEND_URL}/", timeout=5)
            if response.status_code == 200:
                st.success("✅ Connected")
            else:
                st.error("❌ Error")
        except:
            st.error("❌ Offline")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        Customer segmentation dashboard 
        powered by:
        - 🤖 Gemini AI
        - 📊 KMeans & DBSCAN
        - 🔍 HuggingFace Embeddings
        """)
    
    # Main content
    render_header()
    
    if page == "📊 Dashboard":
        # Load data
        stats = get_stats()
        clusters = get_clusters()
        
        # Metrics
        render_metrics(stats)
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            render_cluster_chart(clusters)
        
        with col2:
            render_cluster_comparison(clusters)
        
        # Cluster details table
        if clusters:
            st.markdown("### 📋 Cluster Details")
            df = pd.DataFrame(clusters['clusters'])
            df = df[['cluster_id', 'user_type', 'size', 'percentage', 
                     'avg_voice_mins', 'avg_data_gb', 'avg_intl_calls']]
            df.columns = ['ID', 'Segment', 'Customers', '%', 
                         'Avg Voice (mins)', 'Avg Data (GB)', 'Avg Intl Calls']
            st.dataframe(df, use_container_width=True)
    
    elif page == "💬 AI Chat":
        render_chat_interface()
        
        st.markdown("---")
        render_semantic_search()
    
    elif page == "🔎 Customer Lookup":
        render_customer_lookup()
    
    elif page == "📦 Packages":
        packages = get_packages()
        render_packages(packages)
        
        st.markdown("---")
        st.info("💡 Packages are automatically calculated as 20% above the average usage for each segment, rounded to business-friendly values.")


if __name__ == "__main__":
    main()
