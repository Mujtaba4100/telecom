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
- 🤖 Gemini LLM for intelligent insights
- 🔍 Semantic search with HuggingFace embeddings
- 💡 Personalized package recommendations

### ✅ Advanced Analytics
- 📊 KMeans & DBSCAN clustering
- 🎨 Interactive visualizations (Plotly)
- 🔬 On-demand clustering analysis

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
$env:GEMINI_API_KEY="your_key_here"  # PowerShell
# OR
export GEMINI_API_KEY="your_key_here"  # Bash
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
3. Set `GEMINI_API_KEY` secret on backend
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

The dashboard has **5 main pages** (select from sidebar):

---

### 1️⃣ 🏠 Overview Dashboard

**Purpose:** See statistics for ALL customers

**What you see:**
- **Top metrics:** Total customers, international users, avg calls/data/SMS
- **Three tabs:**
  - **📞 Communication:** Voice stats, time-of-day breakdown (Morning/Evening/Night), pie chart
  - **🌐 Internet:** Download/Upload breakdown, usage stats, pie chart
  - **💬 SMS:** Total messages, frequency distribution, bar chart

**How to use:** Just scroll and explore - everything updates automatically!

---

### 2️⃣ 👤 Customer Lookup

**Purpose:** Search for ONE specific customer by ID

**How to use:**
1. Enter **Subscriber ID** (e.g., 864226)
2. Click **"🔎 Search Customer"**

**What you get:**
- **📞 Communication Analysis:** Calls, duration, time breakdown
- **🌍 International Details:** Countries called, durations, call history
- **🌐 Internet Usage:** Download/Upload with pie chart
- **💬 SMS Activity:** Message count and frequency level
- **🤖 AI Suggestions:** Personalized package recommendations from Gemini

**Example:** Enter ID 864226 → See their complete profile + AI recommendation

---

### 3️⃣ 📈 Visual Insights

**Purpose:** Generate interactive charts on-demand

**How to use:**
1. Select from dropdown:
   - **Time Distribution** → Calls by time of day
   - **Data Breakdown** → Download vs Upload
   - **Customer Segments** → Cluster distribution
2. Chart appears instantly (interactive - hover, zoom, pan)

---

### 4️⃣ 🔬 Clustering Analysis

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

### 5️⃣ 💬 AI Assistant

**Purpose:** Ask questions in natural language

**How to use:**
1. Type your question
2. Click **"🔍 Ask AI"**
3. Get intelligent answer from Gemini

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
| Visualize patterns | 📈 Visual Insights |
| Group similar customers | 🔬 Clustering Analysis |
| Ask data questions | 💬 AI Assistant |

---

## 🛠️ Tech Stack

**Backend:**
- FastAPI
- scikit-learn (KMeans, DBSCAN)
- Google Gemini AI
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
| `/api/query` | POST | AI query (Gemini) |
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
- Verify GEMINI_API_KEY is set

**Frontend can't connect?**
- Check BACKEND_URL is correct
- Ensure backend is running
- Check CORS settings

**Need help?** Check [DEPLOYMENT.md](DEPLOYMENT.md) for detailed guide.
