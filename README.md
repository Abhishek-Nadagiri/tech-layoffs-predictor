# 🧠 Tech Layoffs Intelligence Dashboard

> An end-to-end ML-powered data analysis project on 12,000 tech layoff records across 23 features, served through an interactive Streamlit Q&A dashboard.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange?logo=scikitlearn)
![Plotly](https://img.shields.io/badge/Plotly-6.7-3F4F75?logo=plotly)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📸 Features

- 💬 **Natural Language Q&A** — ask questions in plain English, get answers + charts
- 📊 **15+ Chart Types** — bar, line, pie, choropleth, scatter, heatmap, cluster plots
- 🤖 **5 ML Models** — classification, regression, and clustering
- 🔮 **Live Predictions** — adjust sliders in the sidebar to get instant ML predictions
- 🚀 **Auto-training** — models are built on first launch automatically (no pre-built `.pkl` needed)
- 🌙 **Dark Theme** — GitHub-style dark UI out of the box

---

## 📁 Project Structure

```
tech_layoffs_project/
│
├── app.py                          ← Streamlit dashboard (main entry point)
├── train_models.py                 ← ML training script
├── requirements.txt                ← Python dependencies (pinned)
├── packages.txt                    ← OS-level deps for Streamlit Cloud
├── runtime.txt                     ← Python version for Heroku/Cloud
├── Procfile                        ← Heroku deployment config
├── setup.sh                        ← Shell entrypoint for platforms
├── .gitignore                      ← Git ignore rules
├── .gitattributes                  ← Git LFS rules for large .pkl files
│
├── .streamlit/
│   ├── config.toml                 ← Theme + server settings
│   └── secrets.toml.example        ← API key template (never commit real one)
│
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py            ← Shared encode/decode helpers
│   └── visualizations.py          ← Reusable Plotly chart factories
│
├── models/                         ← Auto-generated on first run (Git LFS tracked)
│   ├── hiring_trend_classifier.pkl
│   ├── layoff_count_regressor.pkl
│   ├── ai_risk_classifier.pkl
│   ├── job_security_regressor.pkl
│   ├── kmeans_cluster.pkl
│   ├── cluster_scaler.pkl
│   ├── label_encoders.pkl
│   ├── feature_importances.csv
│   └── model_metrics.pkl
│
└── tech_layoffs_hiring_trends_elite_v2.csv   ← Dataset (12,000 rows × 23 cols)
```

---

## 🚀 Quick Start (Local)

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/tech-layoffs-dashboard.git
cd tech-layoffs-dashboard

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train ML models (first time only — ~60 seconds)
python train_models.py

# 5. Launch the dashboard
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

> **Note:** Step 4 is optional. The app detects missing models on startup and trains them automatically.

---

## ☁️ Deploy to Streamlit Cloud (Free)

1. Push this repo to GitHub (public or private)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, branch `main`, and set **Main file path** to `app.py`
4. Click **Deploy** — models train automatically on first boot (~60–90 sec)

> **Git LFS note:** The `models/` folder is tracked by Git LFS (`.gitattributes`).  
> If you want to skip LFS entirely, delete the `models/` folder from git — the app will auto-train on first run.

---

## 🤖 ML Models

| # | Model | Algorithm | Task | Metric |
|---|-------|-----------|------|--------|
| 1 | Hiring Trend Classifier | Random Forest | 4-class: Freeze / Moderate / Aggressive / Downsizing | Acc: **1.00** |
| 2 | Layoff Count Regressor | Random Forest | Predict # of layoffs | R²: **0.55** |
| 3 | AI Replacement Risk Classifier | Gradient Boosting | 3-class: Low / Medium / High | Acc: **1.00** |
| 4 | Job Security Regressor | Random Forest | Predict job security score 0–10 | R²: **0.87** |
| 5 | Company Risk Clustering | KMeans (k=4) | Segment companies by risk profile | Sil: **0.16** |

---

## 💬 Sample Questions

| Category | Example Questions |
|----------|------------------|
| Layoffs | *"Show layoffs by industry"*, *"Which countries have most layoffs?"*, *"Top 10 companies by layoffs"* |
| Trends | *"Show layoff trend over the years"*, *"What is the hiring trend distribution?"* |
| AI Impact | *"What is the AI impact on tech jobs?"*, *"Show AI replacement risk by industry"* |
| Financials | *"How does stock growth relate to layoffs?"*, *"Show salary budget changes"* |
| Workforce | *"Show employee sentiment by industry"*, *"What is the job security score?"* |
| Analytics | *"Show feature importances"*, *"Cluster analysis of companies"*, *"Dataset overview"* |
| ML Predict | *"Predict hiring trend"*, *"Predict layoff count"* → use sidebar sliders |

---

## 📊 Dataset Columns (23)

| Column | Type | Description |
|--------|------|-------------|
| `record_id` | str | Unique record ID |
| `company_name` | str | Company name |
| `industry` | cat | 7 sectors: AI, Cloud, FinTech, Gaming, Cybersecurity, Social Media, E-Commerce |
| `country` | cat | 6 countries: USA, UK, Canada, India, Germany, Singapore |
| `company_size` | cat | Startup / Mid-size / Big Tech / Enterprise |
| `month` | cat | Month of record |
| `year` | int | 2024 / 2025 / 2026 |
| `layoffs_count` | int | Number of employees laid off |
| `layoff_percentage` | float | % of workforce laid off |
| `reason_for_layoffs` | cat | AI Automation / Cost Cutting / Overhiring Correction / Market Slowdown / Restructuring |
| `ai_automation_impact` | float | 0–10 AI automation score |
| `ai_replacement_risk` | float | 0–10 AI replacement risk |
| `open_roles` | int | Number of currently open positions |
| `hiring_trend` | cat | Hiring Freeze / Moderate / Aggressive / Downsizing |
| `remote_jobs_percentage` | float | % of remote-eligible roles |
| `top_hiring_role` | cat | Most in-demand role |
| `stock_growth_percent` | float | Stock growth % |
| `revenue_growth_percent` | float | Revenue growth % |
| `salary_budget_change` | float | Salary budget delta % |
| `ai_adoption_level` | float | 0–10 AI adoption maturity |
| `employee_sentiment` | float | 0–10 sentiment score |
| `job_security_score` | float | 0–10 job security score |
| `market_condition` | cat | Bull Market / Recession / Stable |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| UI / Frontend | Streamlit 1.58, custom CSS (GitHub dark theme) |
| Charts | Plotly 6.7 (bar, line, pie, choropleth, scatter, heatmap) |
| ML Models | scikit-learn 1.8 (RandomForest, GradientBoosting, KMeans) |
| Data | Pandas 3.0, NumPy |
| Serialization | Joblib |
| NL Intent Engine | Regex-based intent classifier (15 intents) |

---

## 📂 GitHub Setup with Git LFS

If you want to commit the pre-trained `.pkl` files (saves ~60s on first load):

```bash
# Install Git LFS
git lfs install

# Clone / init repo
git init
git lfs track "models/*.pkl"
git add .gitattributes
git add .
git commit -m "Initial commit: Tech Layoffs Dashboard"
git remote add origin https://github.com/<you>/tech-layoffs-dashboard.git
git push -u origin main
```

If you skip LFS, just don't commit the `models/` folder — the app trains on startup.

---

## 📄 License

MIT — free to use, modify, and distribute.

---

## 🙋 Author

Built with ❤️ using Python, Streamlit, and scikit-learn.
