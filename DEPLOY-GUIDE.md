# 🚀 HuggingFace Deployment - Step by Step

## 📦 What You'll Deploy

Two separate HuggingFace Spaces:
1. **Backend (Docker)** - API server
2. **Frontend (Streamlit)** - Dashboard

---

## 1️⃣ BACKEND SPACE

### Create Space
1. Go to https://huggingface.co/new-space
2. **Owner:** Your username
3. **Space name:** `telecom-backend` (or any name)
4. **License:** MIT
5. **Select SDK:** **Docker** ⚠️ IMPORTANT!
6. **Space hardware:** CPU basic - Free
7. Click **Create Space**

### Files to Upload

Upload **EXACTLY** these files to the backend space:

```
telecom-backend/         (Your HF Space root)
├── app.py              ← Copy from backend/app.py
├── requirements.txt    ← Copy from backend/requirements.txt
├── Dockerfile          ← Copy from backend/Dockerfile
├── merged_subscriber_data.csv         ← From root
├── international_calls.csv            ← From root
└── golden_table_clustered.csv        ← From root (or backend/)
```

### Set Secret (IMPORTANT!)
1. In your HF Space, click **Settings** (⚙️)
2. Go to **Repository secrets**
3. Click **New secret**
4. Name: `GEMINI_API_KEY`
5. Value: `AIzaSyAMnF1TlsgVwuk_4ozncUf6tIMBqN-Ll_s` (your actual key)
6. Click **Save**

### How to Upload
**Option A: Web Interface**
- Click "Files" → "Add file" → Upload each file

**Option B: Git (Recommended)**
```bash
# Clone your space
git clone https://huggingface.co/spaces/YOUR-USERNAME/telecom-backend
cd telecom-backend

# Copy files
copy ..\backend\app.py .
copy ..\backend\requirements.txt .
copy ..\backend\Dockerfile .
copy ..\merged_subscriber_data.csv .
copy ..\international_calls.csv .
copy ..\golden_table_clustered.csv .

# Commit and push
git add .
git commit -m "Deploy backend"
git push
```

✅ **Backend will be live at:** `https://YOUR-USERNAME-telecom-backend.hf.space`

---

## 2️⃣ FRONTEND SPACE

### Create Space
1. Go to https://huggingface.co/new-space
2. **Space name:** `telecom-dashboard` (or any name)
3. **Select SDK:** **Streamlit** ⚠️ IMPORTANT!
4. **Space hardware:** CPU basic - Free
5. Click **Create Space**

### Files to Upload

Upload **EXACTLY** these files:

```
telecom-dashboard/      (Your HF Space root)
├── app.py              ← Copy from frontend/app.py
└── requirements.txt    ← Copy from frontend/requirements.txt
```

**That's it! Only 2 files for frontend!**

### Set Environment Variable (Not Secret!)
1. In your HF Space, click **Settings** (⚙️)
2. Go to **Variables and secrets** section
3. Under "Variables" (NOT secrets), click **New variable**
4. Name: `BACKEND_URL`
5. Value: `https://YOUR-USERNAME-telecom-backend.hf.space`
   (Replace YOUR-USERNAME with your actual username)
6. Click **Save**

⚠️ **Note:** Use Variable (not Secret) because it's just a URL, not sensitive data!

### How to Upload
**Option A: Web Interface**
- Click "Files" → "Add file" → Upload app.py and requirements.txt

**Option B: Git**
```bash
# Clone your space
git clone https://huggingface.co/spaces/YOUR-USERNAME/telecom-dashboard
cd telecom-dashboard

# Copy files
copy ..\frontend\app.py .
copy ..\frontend\requirements.txt .

# Commit and push
git add .
git commit -m "Deploy frontend"
git push
```

✅ **Dashboard will be live at:** `https://YOUR-USERNAME-telecom-dashboard.hf.space`

---

## 🎯 Quick Checklist

### Backend ✓
- [ ] SDK = **Docker** (not Streamlit!)
- [ ] Uploaded: app.py, requirements.txt, Dockerfile
- [ ] Uploaded: 3 CSV files
- [ ] Secret: `GEMINI_API_KEY` = your API key
- [ ] Wait for build (5-10 mins)

### Frontend ✓
- [ ] SDK = **Streamlit** (not Docker!)
- [ ] Uploaded: app.py, requirements.txt (just 2 files!)
- [ ] Variable: `BACKEND_URL` = https://YOUR-USERNAME-telecom-backend.hf.space
- [ ] Wait for deployment (2-3 mins)

---

## 🔍 Troubleshooting

### Backend Issues

**"Build failed"**
- Check Dockerfile is uploaded
- Check all CSV files are uploaded
- Look at build logs in HF Space

**"API not responding"**
- Wait 5-10 minutes for Docker build
- Check if `GEMINI_API_KEY` secret is set
- Click on your space URL to see if it loads

### Frontend Issues

**"Module not found"**
- Check requirements.txt is uploaded
- Wait for dependencies to install

**"Backend connection error"**
- Make sure backend is running first
- Check `BACKEND_URL` variable is correct
- Must include `https://` (not `http://`)
- No trailing slash

---

## 📊 Final URLs

After deployment, you'll have:

**Backend API:**
```
https://YOUR-USERNAME-telecom-backend.hf.space
```
Test: Open in browser, should show:
```json
{"status": "healthy", "version": "2.0", "customers": 266322}
```

**Frontend Dashboard:**
```
https://YOUR-USERNAME-telecom-dashboard.hf.space
```
Opens directly to the dashboard!

---

## 💰 Cost

**Everything is FREE!** ✅
- HuggingFace Spaces: FREE
- CPU basic hardware: FREE
- Gemini API: FREE (60 req/min)

---

## 🎉 That's It!

Your complete system will be live on the internet, accessible from anywhere!

**Need help?** Check build logs in HF Space interface.
