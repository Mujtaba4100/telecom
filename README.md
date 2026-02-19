# 📊 Telecom Customer Segmentation System

Complete AI-powered customer analytics platform with clustering, visualizations, and LLM insights.

## 🚀 Features

### ✅ Communication Analysis
- 📞 Call frequency, duration, and patterns
- ⏰ Time-based analysis (Morning/Evening/Night)
- 🌍 International call tracking with countries

### ✅ Internet Usage
- 📥 Download tracking
- 📤 Upload tracking
- 📊 Total data consumption

### ✅ SMS Insights
- 💬 Message frequency
- 📈 Usage patterns

### ✅ AI-Powered Features
- 🤖 Groq LLM for intelligent insights
- 🔍 Semantic search with HuggingFace embeddings
- 💡 Personalized package recommendations

### ✅ Advanced Analytics
- 📊 KMeans & DBSCAN clustering (6 clusters)
- 🎨 Interactive visualizations (Plotly)
- 🔬 On-demand clustering analysis
- 🔄 Cohort comparison with delta highlighting
- 📥 Export CSV reports and PNG charts
- 💡 AI insights panels on every view

---

## 🏃 Quick Start (Local)

### 1. Install Dependencies
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

### 2. Set API Key
```bash
$env:GROQ_API_KEY="your_key_here"  # PowerShell
# OR
export GROQ_API_KEY="your_key_here"  # Bash
```

### 3. Run Backend
```bash
cd backend
uvicorn app:app --reload --port 7860
```

### 4. Run Frontend (New Terminal)
```bash
cd frontend
streamlit run app.py
```

✅ Open: http://localhost:8501

---

## ☁️ Deploy to HuggingFace (FREE)

See [QUICK-DEPLOY.md](QUICK-DEPLOY.md) for 5-minute deployment guide.

**TL;DR:**
1. Create 2 HF Spaces (one Docker for backend, one Streamlit for frontend)
2. Upload files
3. Set `GROQ_API_KEY` secret on backend
4. Set `BACKEND_URL` variable on frontend
5. Done! 🎉

---

## 📁 Project Structure

```
talhabhai/
├── backend/                # FastAPI backend
│   ├── app.py             # Main API (all endpoints)
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/              # Streamlit dashboard
│   ├── app.py            # UI with all features
│   └── requirements.txt
│
├── customer_segmentation.py   # Data processing pipeline
├── cluster_visualization.py   # Report generation
│
├── merged_subscriber_data.csv      # Raw data
├── international_calls.csv         # International data
└── golden_table_clustered.csv     # Processed data with clusters
```

---

## 🎯 Complete UI Guide

### 📑 Sidebar Navigation

The dashboard has **6 main pages** (select from sidebar):

---

### 1️⃣ 🏠 Overview Dashboard

**Purpose:** See statistics for ALL customers

**What you see:**
- **Top metrics:** Total customers, international users, avg calls/data/SMS
- **Three tabs:**
  - **📞 Communication:** Voice stats, time-of-day breakdown (Morning/Evening/Night), pie chart + AI insights
  - **🌐 Internet:** Download/Upload breakdown, usage stats, pie chart + AI insights
  - **💬 SMS:** Total messages, frequency distribution, adoption rate (1.9%), active user avg (2.0 msgs) + AI insights
- **Export buttons:** Download CSV reports and save charts as PNG

**How to use:** Just scroll and explore - everything updates automatically! Expand AI Insights for recommendations.

---

### 2️⃣ 👤 Customer Lookup

**Purpose:** Search for ONE specific customer by ID

**How to use:**
1. Enter **Subscriber ID** (e.g., 1, 100, 1000)
2. Click **"🔎 Search Customer"**

**What you get (4 expandable sections):**
- **📞 Communication Analysis:** Calls, duration, time breakdown (Morning/Evening/Night)
- **🌍 International Details:** Countries called, durations, call history
- **🌐 Internet Usage:** Download/Upload with pie chart
- **💬 SMS Activity:** Message count and frequency level
- **🤖 AI-Powered Recommendations:** Personalized package suggestions from Groq with usage badges

**Example:** Enter ID 100 → See their complete profile + structured AI recommendation

---

### 3️⃣ � Cohort Comparison

**Purpose:** Compare two customer segments side-by-side

**How to use:**
1. Select **Cohort A** (e.g., Cluster 0)
2. Select **Cohort B** (e.g., Cluster 2)
3. View side-by-side comparison with percentage deltas

**What you get:**
- **Comparison metrics:** Voice usage, Data usage, SMS count
- **Visual indicators:** ⚠️ >50% difference, ⚡ 20-50% difference, ✓ <20% difference
- **Summary table:** All metrics with percentage differences
- **Visual chart:** Bar chart comparing both cohorts
- **AI Insights:** Strategic recommendations for each segment
- **Export buttons:** Download comparison CSV and chart PNG

**Example:** Compare Cluster 0 (light users) vs Cluster 2 (data-heavy users) → See 1330% data usage difference!

---

### 4️⃣ �📈 Visual Insights

**Purpose:** Generate interactive charts on-demand

**How to use:**
1. Select from dropdown:
   - **Time Distribution** → Calls by time of day
   - **Data Breakdown** → Download vs Upload
   - **Customer Segments** → Cluster distribution
2. Chart appears instantly (interactive - hover, zoom, pan)

---

### 5️⃣ 🔬 Clustering Analysis

**Purpose:** Explore and create customer segments

**Two tabs:**

**📊 View Clusters:**
- Select algorithm (KMeans/DBSCAN)
- See pie chart, bar comparison, detailed table
- Understand how customers are grouped

**🔧 Run Custom Clustering:**
- Choose algorithm and parameters
- Click "Run Clustering"
- Get new segmentation with quality scores

**What is a cluster?** Group of similar customers (e.g., "Heavy data users", "Voice callers")

---

### 6️⃣ 💬 AI Assistant

**Purpose:** Ask questions in natural language

**How to use:**
1. Type your question
2. Click **"🔍 Ask AI"**
3. Get intelligent answer from Groq

**Example questions:**
- "What time of day has the most calls?"
- "How many customers use SMS frequently?"
- "What's the download vs upload ratio?"
- "Which customers need international packages?"

**Shows:** Last 5 conversations

---

## 💡 Common Use Cases

| Scenario | Go To |
|----------|-------|
| See general trends | 🏠 Overview Dashboard |
| Check specific customer | 👤 Customer Lookup (enter ID) |
| Compare customer segments | 🔄 Cohort Comparison |
| Visualize patterns | 📈 Visual Insights |
| Group similar customers | 🔬 Clustering Analysis |
| Ask data questions | 💬 AI Assistant |
| Export reports/charts | Any page with 📥 buttons |

---

## 🛠️ Tech Stack

**Backend:**
- FastAPI
- scikit-learn (KMeans, DBSCAN)
- Groq AI (llama-3.3-70b-versatile)
- HuggingFace Transformers
- FAISS (semantic search)
- SQLite

**Frontend:**
- Streamlit
- Plotly
- Pandas

**Deployment:**
- HuggingFace Spaces
- Docker

---

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/api/stats` | GET | Overall statistics |
| `/api/customers/{id}` | GET | Customer details |
| `/api/time-analysis` | GET | Time-based call analysis |
| `/api/clusters` | GET | Cluster information |
| `/api/query` | POST | AI query (Groq) |
| `/api/search` | GET | Semantic search |
| `/api/cluster/run` | POST | On-demand clustering |
| `/api/visualizations/*` | GET | Dynamic charts |

---

## 📄 License

MIT

## 👥 Contributors

Built for advanced telecom customer analytics.

---

## 🆘 Troubleshooting

**Backend not responding?**
- Check if port 7860 is free
- Ensure data files are in backend directory
- Verify GROQ_API_KEY is set

**Frontend can't connect?**
- Check BACKEND_URL is correct
- Ensure backend is running
- Check CORS settings

**Need help?** Check [DEPLOYMENT.md](DEPLOYMENT.md) for detailed guide.
