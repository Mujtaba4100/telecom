"""
Enhanced Telecom Customer Segmentation Backend API
=================================================
FastAPI backend with:
- Enhanced cluster analysis with ALL data fields
- Time-based analysis (morning/evening/night)
- SMS insights
- Upload/Download breakdown
- Dynamic visualization generation
- On-demand clustering
- Groq LLM integration
- HuggingFace embeddings for semantic search
"""

import os
import json
import sqlite3
import pickle
import io
import base64
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from datetime import datetime

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss

# ML imports
from sklearn.cluster import MiniBatchKMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Global flag for FAISS initialization
faiss_building = False

# Visualization imports
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# ============================================
# CONFIGURATION
# ============================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Data paths
MERGED_DATA_PATH = "merged_subscriber_data.csv"
INTL_DATA_PATH = "international_calls.csv"
CLUSTERED_DATA_PATH = "golden_table_clustered.csv"
DB_PATH = "data/database.db"
FAISS_INDEX_PATH = "data/faiss_index.bin"
EMBEDDINGS_PATH = "data/embeddings.pkl"

# Global variables
df = None
df_full = None  # Full data with all fields
conn = None
embedding_model = None
faiss_index = None
groq_client = None


# ============================================
# STARTUP / SHUTDOWN
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup"""
    global df, df_full, conn, embedding_model, faiss_index, groq_client
    
    print("🚀 Starting Enhanced Telecom API...")
    
    # Load full data with all fields
    if os.path.exists(MERGED_DATA_PATH):
        df_merged = pd.read_csv(MERGED_DATA_PATH)
        if os.path.exists(INTL_DATA_PATH):
            df_intl = pd.read_csv(INTL_DATA_PATH)
            df_full = pd.merge(df_merged, df_intl, on='subscriberid', how='left')
        else:
            df_full = df_merged
        
        # Fill NaN values
        df_full = df_full.fillna(0)
        print(f"✓ Loaded {len(df_full):,} customers with enhanced data")
        
        # Load clustered results if available
        if os.path.exists(CLUSTERED_DATA_PATH):
            df_clustered = pd.read_csv(CLUSTERED_DATA_PATH)
            # Merge cluster labels into full data
            df_full = pd.merge(
                df_full, 
                df_clustered[['subscriberid', 'kmeans_cluster', 'dbscan_cluster']], 
                on='subscriberid', 
                how='left'
            )
        
        df = df_full.copy()
    else:
        print("⚠ Data files not found")
        df = df_full = create_sample_data()
    
    # Initialize database
    init_database()
    
    # Load models
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ Loaded embedding model")
    except Exception as e:
        print(f"⚠ Embedding model error: {e}")
    
    if GROQ_API_KEY:
        try:
            groq_client = Groq(api_key=GROQ_API_KEY)
            print("✓ Initialized Groq")
        except Exception as e:
            print(f"⚠ Groq error: {e}")
    
    # FAISS index will build on first search request (lazy loading)
    print("ℹ FAISS index will build on first search request")
    print("✅ API ready!")
    
    yield
    
    if conn:
        conn.close()
    print("👋 Shutdown complete")


# ============================================
# INITIALIZE APP
# ============================================

app = FastAPI(
    title="Enhanced Telecom Segmentation API",
    description="Advanced telecom customer analytics with time-based insights",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# PYDANTIC MODELS
# ============================================

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    data: Optional[Dict[str, Any]] = None

class EnhancedCustomerInfo(BaseModel):
    subscriberid: int
    
    # Voice communication
    voice_total_duration_mins: float
    voice_total_calls: float
    voice_morning_calls: float
    voice_evening_calls: float
    voice_night_calls: float
    
    # SMS
    sms_total_messages: float
    
    # Data
    data_total_mb: float
    data_downlink_mb: float
    data_uplink_mb: float
    
    # International
    intl_total_calls: float
    intl_total_duration_mins: float
    intl_countries_called: float
    intl_top_country: Optional[str]
    
    # User types
    call_lover: int
    download_lover: int
    upload_lover: int
    data_lover: int
    
    # Clustering
    kmeans_cluster: Optional[int]
    dbscan_cluster: Optional[int]

class ClusterRequest(BaseModel):
    n_clusters: int = 6
    algorithm: str = "kmeans"  # kmeans or dbscan


# ============================================
# HELPER FUNCTIONS
# ============================================

def create_sample_data():
    """Create sample data"""
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        'subscriberid': range(1, n+1),
        'voice_total_duration_mins': np.random.exponential(10, n),
        'voice_total_calls': np.random.poisson(10, n),
        'voice_morning_calls': np.random.poisson(3, n),
        'voice_evening_calls': np.random.poisson(4, n),
        'voice_night_calls': np.random.poisson(3, n),
        'sms_total_messages': np.random.poisson(5, n),
        'data_total_mb': np.random.exponential(400, n),
        'data_downlink_mb': np.random.exponential(300, n),
        'data_uplink_mb': np.random.exponential(100, n),
        'intl_total_calls': np.random.poisson(0.5, n),
        'intl_total_duration_mins': np.random.exponential(0.5, n),
        'intl_countries_called': np.random.poisson(0.3, n),
        'call_lover': np.random.choice([0, 1], n, p=[0.75, 0.25]),
        'data_lover': np.random.choice([0, 1], n, p=[0.75, 0.25]),
        'kmeans_cluster': np.random.choice(range(6), n),
        'dbscan_cluster': np.random.choice(range(12), n),
    })


def init_database():
    """Initialize SQLite database"""
    global conn, df_full
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    df_full.to_sql('customers', conn, if_exists='replace', index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_subscriberid ON customers(subscriberid)")
    print("✓ Database initialized")


def init_faiss_index():
    """Build FAISS index for semantic search"""
    global faiss_index, embedding_model, df
    
    if embedding_model is None:
        return
    
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            faiss_index = faiss.read_index(FAISS_INDEX_PATH)
            print("✓ Loaded FAISS index")
            return
        except:
            pass
    
    # Build index
    print("Building FAISS index...")
    descriptions = []
    for _, row in df.iterrows():
        desc = f"Customer {row['subscriberid']}: "
        desc += f"{row.get('voice_total_calls', 0):.0f} voice calls, "
        desc += f"{row.get('data_total_mb', 0):.0f} MB data, "
        desc += f"{row.get('sms_total_messages', 0):.0f} SMS, "
        if row.get('intl_total_calls', 0) > 0:
            desc += f"{row.get('intl_total_calls', 0):.0f} international calls"
        descriptions.append(desc)
    
    embeddings = embedding_model.encode(descriptions, show_progress_bar=True, batch_size=32)
    
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    faiss_index.add(embeddings)
    
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    print("✓ Built FAISS index")


def get_cluster_label(row):
    """Get human-readable cluster label"""
    if row['intl_total_calls'] > 0:
        if row['data_total_mb'] > row['data_total_mb'].median():
            return "International Data Users"
        else:
            return "International Callers"
    elif row['voice_total_calls'] > row['voice_total_calls'].quantile(0.75):
        return "Heavy Voice Users"
    elif row['data_total_mb'] > row['data_total_mb'].quantile(0.75):
        return "Heavy Data Users"
    elif row['sms_total_messages'] > row['sms_total_messages'].quantile(0.75):
        return "SMS Enthusiasts"
    else:
        return "Light Users"


# ============================================
# ENDPOINTS
# ============================================

@app.get("/")
def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "version": "2.0",
        "customers": len(df) if df is not None else 0,
        "columns": list(df.columns) if df is not None else [],
        "features": [
            "time_analysis",
            "sms_insights", 
            "upload_download_split",
            "international_details",
            "dynamic_clustering",
            "dynamic_visualizations"
        ]
    }


@app.get("/api/stats")
def get_stats():
    """Get overall statistics with enhanced metrics"""
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    return {
        "total_customers": int(len(df)),
        "international_users": int(df[df['intl_total_calls'] > 0]['subscriberid'].nunique()),
        "international_percentage": float((df['intl_total_calls'] > 0).sum() / len(df) * 100),
        
        # Voice stats
        "avg_voice_mins": float(df['voice_total_duration_mins'].mean()),
        "avg_voice_calls": float(df['voice_total_calls'].mean()),
        "total_voice_mins": float(df['voice_total_duration_mins'].sum()),
        
        # Time breakdown
        "morning_calls": int(df['voice_morning_calls'].sum()),
        "evening_calls": int(df['voice_evening_calls'].sum()),
        "night_calls": int(df['voice_night_calls'].sum()),
        
        # SMS stats
        "total_sms": int(df['sms_total_messages'].sum()),
        "avg_sms_per_user": float(df['sms_total_messages'].mean()),
        "avg_sms_per_active_user": float(df[df['sms_total_messages'] > 0]['sms_total_messages'].mean()) if (df['sms_total_messages'] > 0).sum() > 0 else 0,
        "sms_users": int((df['sms_total_messages'] > 0).sum()),
        
        # Data stats
        "avg_data_mb": float(df['data_total_mb'].mean()),
        "avg_download_mb": float(df['data_downlink_mb'].mean()),
        "avg_upload_mb": float(df['data_uplink_mb'].mean()),
        "total_data_gb": float(df['data_total_mb'].sum() / 1024),
        
        # User types
        "call_lovers": int(df['call_lover'].sum()),
        "data_lovers": int(df['data_lover'].sum()),
        "download_lovers": int(df.get('download_lover', pd.Series([0])).sum()),
        "upload_lovers": int(df.get('upload_lover', pd.Series([0])).sum()),
    }


@app.get("/api/customers/{customer_id}")
def get_customer(customer_id: int):
    """Get detailed customer information"""
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    customer = df[df['subscriberid'] == customer_id]
    
    if customer.empty:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    
    row = customer.iloc[0]
    
    # Calculate time distribution
    total_calls_by_time = (
        row.get('voice_morning_calls', 0) + 
        row.get('voice_evening_calls', 0) + 
        row.get('voice_night_calls', 0)
    )
    
    return {
        "subscriberid": int(row['subscriberid']),
        
        # Communication
        "communication": {
            "voice_total_duration_mins": float(row['voice_total_duration_mins']),
            "voice_total_calls": float(row['voice_total_calls']),
            "voice_avg_duration_mins": float(row.get('voice_avg_duration_mins', 0)),
            "time_distribution": {
                "morning_calls": int(row.get('voice_morning_calls', 0)),
                "evening_calls": int(row.get('voice_evening_calls', 0)),
                "night_calls": int(row.get('voice_night_calls', 0)),
                "morning_pct": float(row.get('voice_morning_calls', 0) / total_calls_by_time * 100 if total_calls_by_time > 0 else 0),
                "evening_pct": float(row.get('voice_evening_calls', 0) / total_calls_by_time * 100 if total_calls_by_time > 0 else 0),
                "night_pct": float(row.get('voice_night_calls', 0) / total_calls_by_time * 100 if total_calls_by_time > 0 else 0),
            }
        },
        
        # International
        "international": {
            "total_calls": float(row.get('intl_total_calls', 0)),
            "total_duration_mins": float(row.get('intl_total_duration_mins', 0)),
            "countries_called": int(row.get('intl_countries_called', 0)),
            "top_country": str(row.get('intl_top_country', 'N/A')) if pd.notna(row.get('intl_top_country')) else 'N/A',
            "all_countries": str(row.get('intl_all_countries', 'N/A')) if pd.notna(row.get('intl_all_countries')) else 'N/A',
            "is_international_user": bool(row.get('intl_total_calls', 0) > 0)
        },
        
        # Internet
        "internet": {
            "total_mb": float(row['data_total_mb']),
            "download_mb": float(row.get('data_downlink_mb', 0)),
            "upload_mb": float(row.get('data_uplink_mb', 0)),
            "download_pct": float(row.get('data_downlink_mb', 0) / row['data_total_mb'] * 100 if row['data_total_mb'] > 0 else 0),
            "upload_pct": float(row.get('data_uplink_mb', 0) / row['data_total_mb'] * 100 if row['data_total_mb'] > 0 else 0),
        },
        
        # SMS
        "sms": {
            "total_messages": int(row.get('sms_total_messages', 0)),
            "frequency": "High" if row.get('sms_total_messages', 0) > df['sms_total_messages'].quantile(0.75) else 
                        "Medium" if row.get('sms_total_messages', 0) > df['sms_total_messages'].quantile(0.25) else "Low"
        },
        
        # User profile
        "profile": {
            "call_lover": bool(row.get('call_lover', 0)),
            "data_lover": bool(row.get('data_lover', 0)),
            "download_lover": bool(row.get('download_lover', 0)),
            "upload_lover": bool(row.get('upload_lover', 0)),
            "kmeans_cluster": int(row.get('kmeans_cluster', -1)) if pd.notna(row.get('kmeans_cluster')) else None,
            "dbscan_cluster": int(row.get('dbscan_cluster', -1)) if pd.notna(row.get('dbscan_cluster')) else None,
        }
    }


@app.get("/api/time-analysis")
def get_time_analysis():
    """Get time-based analysis of voice calls"""
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    total_morning = df['voice_morning_calls'].sum()
    total_evening = df['voice_evening_calls'].sum()
    total_night = df['voice_night_calls'].sum()
    total_all = total_morning + total_evening + total_night
    
    return {
        "overall": {
            "morning_calls": int(total_morning),
            "evening_calls": int(total_evening),
            "night_calls": int(total_night),
            "morning_pct": float(total_morning / total_all * 100 if total_all > 0 else 0),
            "evening_pct": float(total_evening / total_all * 100 if total_all > 0 else 0),
            "night_pct": float(total_night / total_all * 100 if total_all > 0 else 0),
        },
        "peak_time": "Morning" if total_morning == max(total_morning, total_evening, total_night) else
                     "Evening" if total_evening == max(total_morning, total_evening, total_night) else "Night",
        "by_user_type": {
            "call_lovers": {
                "morning": int(df[df['call_lover'] == 1]['voice_morning_calls'].sum()),
                "evening": int(df[df['call_lover'] == 1]['voice_evening_calls'].sum()),
                "night": int(df[df['call_lover'] == 1]['voice_night_calls'].sum()),
            },
            "others": {
                "morning": int(df[df['call_lover'] == 0]['voice_morning_calls'].sum()),
                "evening": int(df[df['call_lover'] == 0]['voice_evening_calls'].sum()),
                "night": int(df[df['call_lover'] == 0]['voice_night_calls'].sum()),
            }
        }
    }


@app.get("/api/visualizations/time-distribution")
def viz_time_distribution():
    """Generate time distribution chart"""
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    time_data = {
        'Time Period': ['Morning', 'Evening', 'Night'],
        'Total Calls': [
            df['voice_morning_calls'].sum(),
            df['voice_evening_calls'].sum(),
            df['voice_night_calls'].sum()
        ]
    }
    
    fig = px.bar(
        time_data,
        x='Time Period',
        y='Total Calls',
        title='Call Distribution by Time of Day',
        color='Time Period',
        color_discrete_map={'Morning': '#FDB462', 'Evening': '#80B1D3', 'Night': '#8DD3C7'}
    )
    
    return JSONResponse(content={"chart": fig.to_json()})


@app.get("/api/visualizations/data-breakdown")
def viz_data_breakdown():
    """Generate upload/download breakdown chart"""
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    data_summary = {
        'Type': ['Download', 'Upload'],
        'Total (GB)': [
            df['data_downlink_mb'].sum() / 1024,
            df['data_uplink_mb'].sum() / 1024
        ]
    }
    
    fig = px.pie(
        data_summary,
        values='Total (GB)',
        names='Type',
        title='Data Usage: Download vs Upload',
        color_discrete_sequence=['#66C2A5', '#FC8D62']
    )
    
    return JSONResponse(content={"chart": fig.to_json()})


@app.get("/api/visualizations/customer-segments")
def viz_customer_segments():
    """Generate customer segments visualization"""
    if df is None or 'kmeans_cluster' not in df.columns:
        raise HTTPException(status_code=500, detail="Clustering data not available")
    
    # Get cluster statistics
    cluster_stats = df.groupby('kmeans_cluster').agg({
        'subscriberid': 'count',
        'voice_total_calls': 'mean',
        'data_total_mb': 'mean',
        'sms_total_messages': 'mean'
    }).reset_index()
    
    cluster_stats.columns = ['Cluster', 'Customers', 'Avg Calls', 'Avg Data (MB)', 'Avg SMS']
    
    fig = px.bar(
        cluster_stats,
        x='Cluster',
        y='Customers',
        title='Customer Distribution Across Segments',
        color='Customers',
        color_continuous_scale='viridis'
    )
    
    return JSONResponse(content={"chart": fig.to_json()})


@app.post("/api/cluster/run")
def run_clustering(request: ClusterRequest):
    """Run clustering on-demand"""
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    # Select features
    feature_cols = [
        'voice_total_duration_mins', 'voice_total_calls',
        'data_total_mb', 'sms_total_messages'
    ]
    
    # Add international if exists
    if 'intl_total_calls' in df.columns:
        feature_cols.append('intl_total_calls')
    
    X = df[feature_cols].fillna(0)
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Cluster
    if request.algorithm == "kmeans":
        model = MiniBatchKMeans(n_clusters=request.n_clusters, random_state=42, batch_size=1000)
        labels = model.fit_predict(X_scaled)
        
        # Calculate silhouette score
        if len(df) > 10000:
            sample_idx = np.random.choice(len(df), 10000, replace=False)
            score = silhouette_score(X_scaled[sample_idx], labels[sample_idx])
        else:
            score = silhouette_score(X_scaled, labels)
        
    elif request.algorithm == "dbscan":
        model = DBSCAN(eps=0.3, min_samples=10)
        labels = model.fit_predict(X_scaled)
        score = None
    else:
        raise HTTPException(status_code=400, detail="Invalid algorithm")
    
    # Get cluster stats
    df_temp = df.copy()
    df_temp['cluster'] = labels
    
    cluster_info = []
    for cluster_id in sorted(df_temp['cluster'].unique()):
        cluster_data = df_temp[df_temp['cluster'] == cluster_id]
        cluster_info.append({
            "cluster_id": int(cluster_id),
            "size": int(len(cluster_data)),
            "percentage": float(len(cluster_data) / len(df) * 100),
            "avg_voice_calls": float(cluster_data['voice_total_calls'].mean()),
            "avg_data_mb": float(cluster_data['data_total_mb'].mean()),
            "avg_sms": float(cluster_data.get('sms_total_messages', pd.Series([0])).mean()),
        })
    
    return {
        "algorithm": request.algorithm,
        "n_clusters": int(labels.max() + 1),
        "silhouette_score": float(score) if score else None,
        "clusters": cluster_info
    }


@app.post("/api/query")
def query_with_llm(request: QueryRequest):
    """Query data using Groq LLM"""
    if groq_client is None:
        raise HTTPException(status_code=503, detail="Groq API not configured")
    
    # Build context with safe column access
    def safe_col_sum(col_name, default=0):
        """Safely get column sum or return default"""
        return df[col_name].sum() if col_name in df.columns else default
    
    def safe_col_mean(col_name, default=0):
        """Safely get column mean or return default"""
        return df[col_name].mean() if col_name in df.columns else default
    
    def safe_col_count(col_name, condition_value=0):
        """Safely count rows where column > condition_value"""
        if col_name in df.columns:
            return (df[col_name] > condition_value).sum()
        return 0
    
    context = f"""
You are a telecom analytics AI assistant analyzing Pakistani telecom customer data. Provide clear, actionable insights.

IMPORTANT: This is Pakistani telecom data. Use PKR (Pakistani Rupees) for all pricing. Market context: Pakistan has competitive telecom pricing with packages ranging PKR 500-2500/month.

CUSTOMER DATABASE STATISTICS:

📊 Overview:
- Total Customers: {len(df):,}
- International Users: {int(safe_col_count('intl_total_calls', 0)):,} ({safe_col_count('intl_total_calls', 0)/len(df)*100:.1f}%)

📞 Voice Communication:
- Total Calls: {safe_col_sum('voice_total_calls'):,.0f}
- Total Duration: {safe_col_sum('voice_total_duration_mins'):,.0f} mins
- Average per User: {safe_col_mean('voice_total_calls'):.1f} calls, {safe_col_mean('voice_total_duration_mins'):.1f} mins

{'📅 Time Distribution:' if 'voice_morning_calls' in df.columns else ''}
{f"- Morning (6am-12pm): {safe_col_sum('voice_morning_calls'):,.0f} calls" if 'voice_morning_calls' in df.columns else ''}
{f"- Evening (12pm-6pm): {safe_col_sum('voice_evening_calls'):,.0f} calls" if 'voice_evening_calls' in df.columns else ''}
{f"- Night (6pm-6am): {safe_col_sum('voice_night_calls'):,.0f} calls" if 'voice_night_calls' in df.columns else ''}

{'💬 SMS:' if 'sms_total_messages' in df.columns else ''}
{f"- Total Messages: {safe_col_sum('sms_total_messages'):,.0f}" if 'sms_total_messages' in df.columns else ''}
{f"- Average per User: {safe_col_mean('sms_total_messages'):.1f} messages" if 'sms_total_messages' in df.columns else ''}

📊 Data Usage:
- Total Data: {safe_col_sum('data_total_mb'):,.0f} MB ({safe_col_sum('data_total_mb')/1024:.1f} GB)
- Average per User: {safe_col_mean('data_total_mb'):.1f} MB
{f"- Total Download: {safe_col_sum('data_downlink_mb') / 1024:.1f} GB" if 'data_downlink_mb' in df.columns else ''}
{f"- Total Upload: {safe_col_sum('data_uplink_mb') / 1024:.1f} GB" if 'data_uplink_mb' in df.columns else ''}

---

USER QUESTION: {request.question}

RESPONSE INSTRUCTIONS:

📌 **ONLY use the 4-section package format below if:**
   - The question explicitly contains "package", "recommend", "plan", "pricing", or "offer"
   - AND it's about an INDIVIDUAL customer (mentions specific usage numbers for one person)

📌 **For all other questions** (insights, trends, analysis, comparisons):
   - Provide 3 concise, actionable insights
   - Focus on business opportunities, patterns, and strategies
   - DO NOT format as package recommendations
   - Keep it brief and data-driven

---

IF PACKAGE RECOMMENDATION (Individual Customer Only):

**1. USAGE PROFILE**
- Intelligently identify usage patterns from time distribution percentages
- Mention ALL significant time periods (>25% is significant)
- Recognize patterns: bimodal (2 peaks), uniform (balanced), concentrated (1 dominant)
- Consider work patterns: morning+night = commuter, night-heavy = night owl, etc.

**2. RECOMMENDED PACKAGE**
- Size to cover 120-150% of actual usage for growth headroom
- EXCLUDE services with 0 usage (if data=0 MB, don't include data)
- Name should reflect the dominant pattern intelligently
- Realistic pricing in PKR (Pakistani Rupees): PKR 500-2500/month typical range
  * Basic packages: PKR 500-900/month
  * Mid-tier packages: PKR 900-1600/month
  * Premium packages: PKR 1600-2500/month

**3. KEY BENEFITS**
- Focus on: cost savings, usage coverage, flexibility, value match
- Quantify benefits where possible ("save 20%", "covers 150% of usage")

**4. PRICING STRATEGY**
- Specific discounts with business justification (in PKR)
- Upsell opportunities for underutilized services
- Quantify expected impact (ARPU increase, churn reduction)
- Consider Pakistani market competition and customer affordability
"""
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": context}],
            temperature=0.7,
            max_tokens=1024
        )
        return QueryResponse(answer=response.choices[0].message.content, data=None)
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Query error: {error_details}")
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")


@app.get("/api/search")
def semantic_search(query: str = Query(..., description="Search query"), limit: int = 10):
    """Semantic search for customers"""
    global faiss_index, faiss_building
    
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Embedding model not available")
    
    # Lazy load FAISS index on first request
    if faiss_index is None and not faiss_building:
        faiss_building = True
        try:
            init_faiss_index()
        finally:
            faiss_building = False
    
    if faiss_index is None:
        raise HTTPException(status_code=503, detail="FAISS index building, please try again in a moment")
    
    # Embed query
    query_embedding = embedding_model.encode([query])
    faiss.normalize_L2(query_embedding)
    
    # Search
    scores, indices = faiss_index.search(query_embedding, limit)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(df):
            customer = df.iloc[idx]
            results.append({
                "customer_id": int(customer['subscriberid']),
                "similarity_score": float(score),
                "voice_calls": float(customer['voice_total_calls']),
                "data_mb": float(customer['data_total_mb']),
                "sms": int(customer.get('sms_total_messages', 0)),
                "is_international": bool(customer.get('intl_total_calls', 0) > 0)
            })
    
    return {"results": results}


@app.get("/api/clusters")
def get_clusters(cluster_type: str = "kmeans"):
    """Get cluster information"""
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    cluster_col = f"{cluster_type}_cluster"
    if cluster_col not in df.columns:
        raise HTTPException(status_code=404, detail=f"{cluster_type} clusters not found")
    
    cluster_info = []
    for cluster_id in sorted(df[cluster_col].unique()):
        if pd.isna(cluster_id):
            continue
        
        cluster_data = df[df[cluster_col] == cluster_id]
        cluster_info.append({
            "cluster_id": int(cluster_id),
            "size": int(len(cluster_data)),
            "percentage": float(len(cluster_data) / len(df) * 100),
            "avg_voice_mins": float(cluster_data['voice_total_duration_mins'].mean()),
            "avg_data_mb": float(cluster_data['data_total_mb'].mean()),
            "avg_sms": float(cluster_data.get('sms_total_messages', pd.Series([0])).mean()),
            "avg_intl_calls": float(cluster_data.get('intl_total_calls', pd.Series([0])).mean()),
        })
    
    return {"cluster_type": cluster_type, "clusters": cluster_info}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
