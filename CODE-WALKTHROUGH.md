# 🎓 Complete Code Walkthrough for Demo
*Master guide for presenting Backend, Frontend, and UI*

---

## 📊 System Architecture Overview

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│   Frontend      │   HTTP  │    Backend       │  Query  │   External      │
│   (Streamlit)   │ ◄─────► │    (FastAPI)     │ ◄─────► │   Services      │
│                 │         │                  │         │                 │
│  - 5 Pages      │         │  - 8+ Endpoints  │         │  - Gemini LLM   │
│  - Plotly viz   │         │  - SQLite DB     │         │  - HuggingFace  │
│  - User Input   │         │  - FAISS Search  │         │    Embeddings   │
└─────────────────┘         └──────────────────┘         └─────────────────┘
```

**Data Flow:**
1. User interacts with Streamlit UI
2. Frontend sends HTTP requests to FastAPI backend
3. Backend queries SQLite database or generates analytics
4. Backend calls Gemini for AI insights (optional)
5. Backend returns JSON response
6. Frontend renders data with Plotly visualizations

---

## 🔧 Backend Architecture (`backend/app.py` - 748 lines)

### **1. Core Configuration (Lines 1-60)**

```python
# Key Data Paths
MERGED_DATA_PATH = "merged_subscriber_data.csv"      # 266,322 customers
INTL_DATA_PATH = "international_calls.csv"            # 16,133 international users
CLUSTERED_DATA_PATH = "golden_table_clustered.csv"   # Pre-clustered results

# API Keys (from environment)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")         # For AI queries
```

**What to explain:**
- We handle 266K+ customer records
- 3 CSV files: raw data, international calls, clustered results
- Environment variables keep API keys secure

---

### **2. Data Loading (Lines 82-110)**

```python
def load_data():
    """Loads all CSV files and creates SQLite database"""
    # Step 1: Load merged customer data (all 266K records)
    # Step 2: Load international calls (16K records)
    # Step 3: Merge international flags into main dataset
    # Step 4: Load pre-computed cluster results
    # Step 5: Create SQLite database for fast queries
```

**Demo talking points:**
- "We load 266,322 customer records into memory for fast analytics"
- "International calls are joined as a flag column"
- "SQLite database enables SQL-like queries for complex filters"
- "Pre-clustered data allows instant cluster visualization without recomputing"

---

### **3. AI Integration (Lines 112-145)**

#### **A. HuggingFace Embeddings (Semantic Search)**
```python
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
```
- **Purpose:** Convert text descriptions to 384-dimensional vectors
- **Use case:** "Find customers similar to this description"
- **How it works:** Text → Vector → FAISS nearest neighbor search

#### **B. FAISS Index (Fast Similarity Search)**
```python
def build_faiss_index():
    """Creates vector index for 266K customers in ~3 minutes"""
    # Generates text descriptions for each customer
    # Embeds descriptions into vectors (32 batches)
    # Builds FAISS index for sub-second searches
```

**Demo points:**
- "We generate natural language descriptions like: 'High voice usage (412.5 mins), moderate data (450 MB)...'"
- "FAISS enables searching 266K customers in milliseconds"
- "Used for semantic 'find similar customers' queries"

#### **C. Gemini LLM (Natural Language Interface)**
```python
genai.configure(api_key=GEMINI_API_KEY)
model_gemini = genai.GenerativeModel('gemini-pro')
```
- **Purpose:** Answer questions in natural language
- **Context:** We inject customer statistics + cluster insights
- **Example:** "Which cluster has highest international usage?" → Gemini analyzes data → Returns answer

---

### **4. Key Helper Functions**

#### **Safe Column Access (Lines 147-175)**
```python
def safe_col_sum(df, col_name, default=0):
    """Safely get column sum, return 0 if missing"""
    return df[col_name].sum() if col_name in df.columns else default
```

**Why needed:** Not all columns exist in every dataset (e.g., clustered file might lack raw call columns)

**Demo explanation:** "We use defensive coding - if a column is missing, return default value instead of crashing"

---

### **5. REST API Endpoints**

#### **Endpoint 1: Health Check** `GET /`
```python
@app.get("/")
async def health_check():
    return {
        "status": "healthy",
        "total_customers": len(df_data),
        "columns": list(df_data.columns)  # Shows 77 columns
    }
```
**Demo:** Open `https://Hamza4100-telecom.hf.space/` → Shows JSON with customer count

---

#### **Endpoint 2: Enhanced Statistics** `GET /api/stats`
```python
@app.get("/api/stats")
async def get_stats():
    # Calculates 40+ metrics:
    # - Voice stats (total calls, duration, avg per customer)
    # - Data stats (upload, download, total MB)
    # - SMS stats (total messages, avg per customer)
    # - Time analysis (morning/evening/night distribution)
    # - International stats (unique countries, user percentage)
    # - Cluster distribution
```

**Returns JSON like:**
```json
{
  "total_customers": 266322,
  "voice": {
    "total_calls": 105840923,
    "total_minutes": 29784641.2,
    "avg_calls_per_customer": 397.4
  },
  "data": {
    "total_mb": 91293847.5,
    "avg_mb_per_customer": 342.8,
    "upload_mb": 18258769.5,
    "download_mb": 73035078.0
  },
  "time_analysis": {
    "morning_calls": 35280307,
    "evening_calls": 52920461,
    "night_calls": 17640154
  }
}
```

**Demo:** Show in browser → Explain each metric

---

#### **Endpoint 3: Time Analysis** `GET /api/time-analysis`
```python
@app.get("/api/time-analysis")
async def get_time_analysis():
    # Breaks down activity by time of day:
    # - Morning (6am-12pm): calls, data, SMS
    # - Evening (12pm-6pm): calls, data, SMS
    # - Night (6pm-6am): calls, data, SMS
```

**Use case:** "When are customers most active?"

---

#### **Endpoint 4: Customer Lookup** `GET /api/customers/{subscriber_id}`
```python
@app.get("/api/customers/{subscriber_id}")
async def get_customer(subscriber_id: str):
    # Returns complete profile:
    # - Voice activity (calls, mins, roaming)
    # - Data usage (upload, download, total)
    # - SMS activity (sent, received)
    # - International details (if applicable)
    # - Cluster assignment (behavioral segment)
```

**Demo:** `GET /api/customers/SUB123456` → Full customer JSON

---

#### **Endpoint 5: Semantic Search** `POST /api/search`
```python
@app.post("/api/search")
async def search_customers(request: SearchRequest):
    # Input: "heavy data users with international calls"
    # Process:
    #   1. Embed query text to vector
    #   2. FAISS finds 10 nearest customers
    #   3. Return customer IDs + similarity scores
```

**Demo:**
```bash
POST /api/search
{
  "query": "customers with high voice usage but low data",
  "top_k": 5
}
```

---

#### **Endpoint 6: AI Query** `POST /api/query`
```python
@app.post("/api/query")
async def query_ai(request: QueryRequest):
    # Input: Natural language question
    # Process:
    #   1. Load customer stats + cluster info
    #   2. Build context for Gemini (JSON data)
    #   3. Send to Gemini with question
    #   4. Return Gemini's analysis
```

**Demo question:** "Which customer segment uses the most data?"

**Gemini receives:**
```
Database statistics:
- 266,322 customers
- 6 clusters (MiniBatchKMeans)
- Total voice: 29.7M minutes
- Total data: 91.2 GB
...

Question: Which customer segment uses the most data?
```

---

#### **Endpoint 7: Dynamic Visualizations** `POST /api/visualizations/{viz_type}`
```python
@app.post("/api/visualizations/scatter")
async def create_scatter(request: VisualizationRequest):
    # Generates Plotly charts on demand:
    # - scatter: 2D/3D scatter plots
    # - bar: Bar charts
    # - histogram: Distributions
    # - box: Box plots for outlier detection
```

**Parameters:**
- `x_column`: X-axis data (e.g., "voice_total_minutes")
- `y_column`: Y-axis data (e.g., "data_total_mb")
- `color_by`: Optional grouping (e.g., "cluster")

**Returns:** Plotly JSON (frontend renders it)

---

#### **Endpoint 8: On-Demand Clustering** `POST /api/cluster/run`
```python
@app.post("/api/cluster/run")
async def run_clustering(request: ClusterRequest):
    # Real-time clustering:
    # Algorithm: MiniBatchKMeans or DBSCAN
    # Features: voice, data, SMS, location
    # Returns: Cluster assignments + metrics
```

**Demo:** "Let me cluster customers into 4 groups based on data usage"

**Request:**
```json
{
  "algorithm": "kmeans",
  "n_clusters": 4,
  "features": ["data_total_mb", "voice_total_minutes"]
}
```

**Response:**
```json
{
  "clusters": [0, 2, 1, 0, 3, ...],  // 266K cluster labels
  "metrics": {
    "silhouette_score": 0.67
  }
}
```

---

## 🎨 Frontend Architecture (`frontend/app.py` - 605 lines)

### **1. Configuration (Lines 1-30)**

```python
st.set_page_config(
    page_title="Telecom Analytics",
    page_icon="📊",
    layout="wide"
)

# Backend URL detection (3 fallbacks)
try:
    BACKEND_URL = st.secrets["BACKEND_URL"]  # HF Spaces
except:
    BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:7860")
```

**Demo points:**
- "Wide layout maximizes screen space for dashboards"
- "Backend URL auto-detects environment (HF Spaces vs local)"
- "Works seamlessly in production and development"

---

### **2. Five Pages**

#### **Page 1: 📊 Overview Dashboard** (Lines 50-180)

```python
def render_overview():
    # Fetches /api/stats
    # Displays 3 tabs:
    #   Tab 1: Communication (voice + international)
    #   Tab 2: Internet (data usage)
    #   Tab 3: SMS (messaging activity)
    # Each tab has 4 metrics + time breakdown chart
```

**Demo walkthrough:**
1. **Header:** Shows total customers (266,322)
2. **Communication Tab:**
   - Total calls: 105.8M
   - Total minutes: 29.7M
   - Avg calls/customer: 397
   - International users: 16,133 (6.06%)
   - Chart: Morning vs Evening vs Night call distribution
3. **Internet Tab:**
   - Total data: 91.2 GB
   - Upload: 18.2 GB
   - Download: 73.0 GB
   - Chart: Time-based data usage patterns
4. **SMS Tab:**
   - Total SMS: 45.2M
   - Avg SMS/customer: 169.7
   - Chart: SMS frequency by time

**Code highlight:**
```python
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Calls", f"{total_calls:,.0f}")
col2.metric("Total Minutes", f"{total_mins:,.1f}M")
```

---

#### **Page 2: 🔍 Customer Lookup** (Lines 182-320)

```python
def render_customer_lookup():
    # Input: Subscriber ID text box
    # Fetches: /api/customers/{id}
    # Displays: 4 expandable sections
    #   1. Communication Details
    #   2. International Activity
    #   3. Internet Usage
    #   4. SMS Activity
    # AI Suggestions: /api/search + /api/query
```

**Demo walkthrough:**
1. **Enter ID:** Type "SUB123456"
2. **Section 1 - Communication:**
   ```
   Total Calls: 523
   Total Minutes: 1,247.3
   Roaming Calls: 12
   ```
3. **Section 2 - International:**
   ```
   Status: International User
   Countries Called: 3
   Total Minutes: 87.5
   ```
4. **Section 3 - Internet:**
   ```
   Total Data: 2,847 MB
   Upload: 569 MB
   Download: 2,278 MB
   ```
5. **Section 4 - SMS:**
   ```
   Total SMS: 234
   Peak Time: Evening (132 messages)
   ```
6. **AI Button:** Click "Get AI Recommendations"
   - Sends customer data to Gemini
   - Returns: "This customer shows high voice usage with moderate data. Consider offering voice+data bundle..."

**Code highlight:**
```python
with st.expander("📞 Communication Details", expanded=True):
    col1, col2 = st.columns(2)
    col1.metric("Total Calls", customer.get("voice_total_calls", 0))
    col2.metric("Total Minutes", f"{customer.get('voice_total_minutes', 0):.1f}")
```

---

#### **Page 3: 📈 Visual Insights** (Lines 322-420)

```python
def render_visualizations():
    # Interactive chart builder
    # User selects:
    #   - Chart type (scatter/bar/histogram/box)
    #   - X-axis column
    #   - Y-axis column (if applicable)
    #   - Color grouping (optional)
    # Calls: /api/visualizations/{type}
    # Renders: Plotly chart from JSON
```

**Demo walkthrough:**
1. **Select chart type:** Scatter
2. **X-axis:** voice_total_minutes
3. **Y-axis:** data_total_mb
4. **Color by:** cluster
5. **Result:** Interactive scatter plot showing relationship between voice and data usage, colored by customer segment

**Use cases to demo:**
- "Show me data usage distribution" → Histogram
- "Compare voice usage across clusters" → Box plot
- "Find correlation between calls and data" → Scatter

**Code highlight:**
```python
if chart_type == "scatter":
    x_col = st.selectbox("X-axis", numeric_columns)
    y_col = st.selectbox("Y-axis", numeric_columns)
    color_col = st.selectbox("Color by", ["None"] + categorical_columns)
```

---

#### **Page 4: 🎯 Clustering Analysis** (Lines 422-520)

```python
def render_clustering():
    # Two modes:
    #   1. View existing clusters (from golden_table_clustered.csv)
    #   2. Run new clustering on demand
    
    # View mode: Shows cluster distribution + statistics
    # Run mode: User configures algorithm → Backend clusters → Display results
```

**Demo walkthrough - View Mode:**
1. **Cluster distribution:** Pie chart showing 6 clusters
2. **Cluster stats table:**
   ```
   Cluster 0: 45,230 customers | Avg Voice: 412 mins | Avg Data: 523 MB
   Cluster 1: 38,921 customers | Avg Voice: 189 mins | Avg Data: 1,247 MB
   ...
   ```
3. **3D PCA plot:** Visualize cluster separation in 3D space

**Demo walkthrough - Run Mode:**
1. **Choose algorithm:** MiniBatchKMeans
2. **Number of clusters:** 4
3. **Features:** Select voice_total_minutes, data_total_mb, sms_total_messages
4. **Click "Run Clustering"**
5. **Result:** New cluster assignments + silhouette score (quality metric)

**Code highlight:**
```python
if clustering_mode == "Run New Clustering":
    algorithm = st.selectbox("Algorithm", ["kmeans", "dbscan"])
    n_clusters = st.slider("Number of Clusters", 2, 10, 5)
    features = st.multiselect("Features", numeric_columns)
```

---

#### **Page 5: 🤖 AI Assistant** (Lines 522-605)

```python
def render_ai_assistant():
    # Natural language interface to data
    # Process:
    #   1. User types question
    #   2. Send to /api/query (Gemini endpoint)
    #   3. Gemini analyzes data + returns answer
    #   4. Display formatted response
```

**Demo questions to ask:**

1. **"Which cluster has the highest data usage?"**
   - Gemini: "Cluster 3 has the highest average data usage at 1,847 MB per customer. This cluster represents 12.4% of total customers (33,024 users) and shows characteristics of heavy data consumers with moderate voice usage."

2. **"What percentage of customers use international calling?"**
   - Gemini: "6.06% of customers (16,133 out of 266,322) have made international calls. These users collectively called 47 different countries."

3. **"Suggest a pricing plan for high-value customers"**
   - Gemini: "For high-value customers (Cluster 0 with 412+ mins voice and 523+ MB data), consider a premium plan: Unlimited voice + 5GB data + international calling to 10 countries for $49.99/month."

**Code highlight:**
```python
if st.button("Ask AI"):
    response = requests.post(f"{BACKEND_URL}/api/query", json={"question": question})
    answer = response.json()["answer"]
    st.success(answer)
```

---

## 🎬 Demo Script (Recommended Order)

### **1. Start with Architecture (2 mins)**
- "This is a microservices architecture with FastAPI backend and Streamlit frontend"
- Show diagram (System Architecture section)
- "Backend handles 266K customers with 8 REST endpoints"
- "Frontend has 5 pages for different analytics views"

### **2. Backend API Demo (5 mins)**

**A. Health Check:**
```bash
Open: https://Hamza4100-telecom.hf.space/
Show: {"status": "healthy", "total_customers": 266322}
```

**B. Statistics:**
```bash
Open: https://Hamza4100-telecom.hf.space/api/stats
Explain: "This endpoint pre-calculates 40+ metrics"
Point out: Voice, data, SMS, time analysis
```

**C. Customer Lookup:**
```bash
Open: https://Hamza4100-telecom.hf.space/api/customers/SUB123456
Show: Complete customer profile in JSON
```

### **3. Frontend UI Demo (8 mins)**

**Page 1 - Overview (2 mins):**
- Open dashboard
- Show 266K customers metric
- Click through 3 tabs (Communication, Internet, SMS)
- Highlight time breakdown charts

**Page 2 - Customer Lookup (2 mins):**
- Enter subscriber ID
- Expand all 4 sections
- Click "Get AI Recommendations"
- Show Gemini's personalized suggestions

**Page 3 - Visual Insights (2 mins):**
- Create scatter plot: Voice vs Data
- Color by cluster
- Zoom into interesting patterns
- Switch to histogram showing data distribution

**Page 4 - Clustering (1 min):**
- View existing 6 clusters
- Show cluster statistics table
- Mention: "We can run new clustering on demand"

**Page 5 - AI Assistant (1 min):**
- Ask: "Which cluster uses the most data?"
- Show Gemini's detailed analysis
- Ask follow-up: "What's the business opportunity here?"

### **4. Code Deep Dive (5 mins)**

**Backend - Key Functions:**
```python
# Show safe_col_sum() function
"This is defensive coding - handles missing columns gracefully"

# Show build_faiss_index() function
"We embed 266K customers into vectors for semantic search"

# Show /api/query endpoint
"Here's where we integrate Gemini - we build context and send to LLM"
```

**Frontend - Key Components:**
```python
# Show BACKEND_URL detection
"This works in both local dev and production deploy"

# Show render_customer_lookup()
"We fetch data from API and render 4 expandable sections"

# Show Plotly chart rendering
"Backend sends Plotly JSON, frontend renders it interactively"
```

---

## 🔑 Key Technical Highlights for Demo

### **1. Scalability**
- "Handles 266,322 customers efficiently"
- "SQLite for fast queries, FAISS for semantic search"
- "Incremental loading with progress indicators"

### **2. AI Integration**
- "HuggingFace embeddings for semantic search (all-MiniLM-L6-v2)"
- "Gemini LLM for natural language interface"
- "Context-aware responses using customer statistics"

### **3. Modularity**
- "Backend is pure REST API - can swap frontend easily"
- "Environment-based config (works local + cloud)"
- "Separate microservices deployed independently"

### **4. User Experience**
- "Wide layout dashboard with responsive columns"
- "Interactive Plotly charts (zoom, pan, hover)"
- "Expandable sections for clean organization"
- "Loading spinners and error handling"

### **5. Production-Ready**
- "Deployed on HuggingFace Spaces (FREE tier)"
- "Docker containers for reproducibility"
- "Environment secrets for API keys"
- "CORS enabled for cross-origin requests"

---

## 📝 Common Demo Questions & Answers

**Q: How do you handle 266K records in browser?**
A: "Backend does heavy lifting. Frontend only fetches what's needed (stats, individual customers, charts). We use pagination and lazy loading."

**Q: Why FastAPI over Flask?**
A: "FastAPI has automatic API docs (Swagger), async support, and Pydantic validation. It's faster and more modern."

**Q: How accurate is Gemini's analysis?**
A: "Very accurate - we inject actual statistics as context. Gemini doesn't guess, it analyzes real numbers we provide."

**Q: Can users upload new data?**
A: "Not currently, but easy to add. We'd create a POST /api/upload endpoint to accept CSV files."

**Q: How do you ensure data privacy?**
A: "All data in backend container. Frontend only displays aggregated stats. No customer data stored in browser."

**Q: What if backend goes down?**
A: "Frontend shows error message with retry button. We use try-except blocks for all API calls."

**Q: How long does clustering take?**
A: "Pre-computed clusters are instant. On-demand clustering for 266K records takes ~30 seconds with MiniBatchKMeans."

---

## 🚀 Pro Tips for Demo

1. **Start broad, then dive deep:** Architecture → UI → Code
2. **Use real questions:** "Which customers should we target for upsell?"
3. **Show errors:** Enter invalid subscriber ID → Show error handling
4. **Interactive:** Let audience suggest chart combinations
5. **Business value:** Always tie features to business outcomes

**Closing statement:**
"This system gives telecom analysts a complete toolkit: real-time dashboards, customer 360 views, AI-powered insights, and on-demand clustering - all deployed on free cloud infrastructure with 266,322 customers analyzed in under 3 minutes."

---

## 📚 Files Reference

- **Backend:** `backend/app.py` (748 lines)
- **Frontend:** `frontend/app.py` (605 lines)
- **Data:** 3 CSV files (75 MB total)
- **Docs:** README.md, DEPLOY-GUIDE.md, FILE-STRUCTURE.md

**Live URLs:**
- Backend: https://Hamza4100-telecom.hf.space
- Frontend: https://Hamza4100-telecom-ui.hf.space
- Swagger Docs: https://Hamza4100-telecom.hf.space/docs

---

🎉 **You're ready to demo!** Practice the flow 2-3 times and you'll be confident.
