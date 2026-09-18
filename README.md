# 💼 DS Jobs Salary Predictor

> A full-stack Machine Learning web application that predicts data science job salaries using the **Cleaned_DS_Jobs** dataset from Glassdoor.
> Built with **Python**, **FastAPI**, **scikit-learn**, and **Streamlit**.

---

## 📌 Project Overview

This project was built as part of an internship to demonstrate end-to-end ML engineering skills:

- **Data preprocessing & feature engineering** on a real-world HR dataset
- **Model training** using Gradient Boosting (scikit-learn)
- **REST API** development using FastAPI
- **Interactive web UI** using Streamlit + Plotly

---

## 🗂️ Dataset

**Glassdoor Data Science Jobs Dataset (Cleaned_DS_Jobs)**

- 📥 Download from Kaggle: [https://www.kaggle.com/datasets/rashikrahmanpritom/data-science-job-posting-on-glassdoor](https://www.kaggle.com/datasets/rashikrahmanpritom/data-science-job-posting-on-glassdoor)
- File name: `Cleaned_DS_Jobs.csv`
- Place it in the **project root directory** (same level as `requirements.txt`)

| Column | Description |
|--------|-------------|
| `Job Title` | Title of the job posting |
| `Salary Estimate` | Salary range (e.g. $80K–$110K) |
| `Rating` | Glassdoor company rating |
| `Company Name` | Name of the hiring company |
| `Location` | Job location (City, State) |
| `Founded` | Year the company was founded |
| `Size` | Employee count band |
| `Type of ownership` | Public / Private / Government etc. |
| `Industry` | Industry category |
| `Sector` | Sector category |

---

## 🏗️ Project Structure

```
salary_predictor/
├── train.py                  # Data cleaning, feature engineering, model training
├── backend/
│   └── app.py                # FastAPI REST API
├── frontend/
│   └── app.py                # Streamlit web UI
└── model/                    # Auto-created after training
    ├── pipeline.pkl           # Serialised sklearn pipeline
    └── meta.json             # Feature metadata + eval metrics

Cleaned_DS_Jobs.csv    # ← Dataset goes here (project root)
requirements.txt
README.md
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.11+ |
| Data Processing | pandas, numpy |
| Machine Learning | scikit-learn (GradientBoostingRegressor) |
| Backend API | FastAPI, uvicorn |
| Frontend UI | Streamlit, Plotly |
| Serialisation | pickle, JSON |

---

## ⚡ Setup & Installation

### Prerequisites
- Python 3.11 or higher
- pip

### 1 – Clone / download the project

```bash
git clone <your-repo-url>
cd salary_predictor
```

### 2 – Install dependencies

```bash
pip install -r requirements.txt
```

### 3 – Download and place the dataset

Download `Cleaned_DS_Jobs.csv` from the Kaggle link above and place it in the project root:

```
salary_predictor/
Cleaned_DS_Jobs.csv   ← here
requirements.txt
README.md
```

### 4 – Train the model

```bash
python -m salary_predictor.train
```

Expected output:
```
Loaded 742 rows, 15 columns
✓ Training complete  |  MAE: 14.32K  |  R²: 0.7241
Pipeline saved  → salary_predictor/model/pipeline.pkl
Metadata saved  → salary_predictor/model/meta.json
```

### 5 – Start the backend API

```bash
uvicorn salary_predictor.backend.app:app --reload --port 8000
```

- API runs at: **http://localhost:8000**
- Swagger docs: **http://localhost:8000/docs**

### 6 – Launch the frontend UI

Open a **new terminal** and run:

```bash
streamlit run salary_predictor/frontend/app.py
```

- App opens at: **http://localhost:8501**

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/meta` | Model metadata + dropdown options |
| `POST` | `/predict` | Predict salary from job features |
| `GET` | `/feature-importance` | Top 20 feature importances |

### Sample `/predict` Request

```json
POST http://localhost:8000/predict
Content-Type: application/json

{
  "job_title": "Senior Data Scientist",
  "rating": 4.0,
  "company_age": 15,
  "size": "1001 to 5000 employees",
  "type_of_ownership": "Company - Public",
  "industry": "Internet",
  "sector": "Information Technology",
  "state": "CA"
}
```

### Sample Response

```json
{
  "predicted_salary_k": 118.45,
  "predicted_salary_annual": 118450.0,
  "confidence_note": "Estimate based on 593 training samples. Typical error ≈ $14.32K.",
  "model_r2": 0.7241,
  "model_mae_k": 14.32
}
```

---

## 🧠 Model Details

### Feature Engineering

| Feature | Type | How Derived |
|---------|------|-------------|
| `Rating` | Numeric | Glassdoor company rating (1–5) |
| `company_age` | Numeric | `current_year − Founded` |
| `is_senior` | Binary | Regex match on job title (senior/lead/principal…) |
| `is_manager` | Binary | Regex match on job title (manager/director/vp…) |
| `is_remote` | Binary | "remote" keyword in location |
| `Size` | Categorical | Employee count band |
| `Type of ownership` | Categorical | Public / Private / Govt etc. |
| `Industry` | Categorical | Glassdoor industry tag |
| `Sector` | Categorical | Glassdoor sector tag |
| `state` | Categorical | US state extracted from location |

### ML Pipeline

```
ColumnTransformer
├── Numeric  → SimpleImputer(median)        → StandardScaler
└── Categorical → SimpleImputer(frequent)   → OneHotEncoder
                              ↓
            GradientBoostingRegressor
            n_estimators=300 | lr=0.05 | max_depth=5
```

### Target Variable

Mid-point of the Glassdoor salary range in **$K**  
Example: `$80K–$110K (Glassdoor est.)` → target = **95**

---

## 🖥️ Frontend Features

- 🎛️ **Sidebar input form** — job title, rating, company size, industry, state, etc.
- 🔮 **One-click prediction** — instant salary estimate
- 📊 **Gauge chart** — visual salary meter with colour zones
- 📈 **Salary range bar** — low / predicted / high estimate
- 🏆 **Feature importance chart** — top 15 drivers of salary
- ℹ️ **Model info panel** — training stats and methodology

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | Gradient Boosting Regressor |
| Train / Test Split | 80% / 20% |
| R² Score | ~0.72 |
| Mean Absolute Error | ~$14K |

---

## 👨‍💻 Author

Built as part of an internship project demonstrating end-to-end ML application development.

---

## 📄 License

This project is for educational and internship demonstration purposes.
