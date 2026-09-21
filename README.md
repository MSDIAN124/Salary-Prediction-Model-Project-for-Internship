# 💼 DS Jobs Salary Predictor

An end-to-end machine learning project that estimates data science job salaries from the **Glassdoor Data Science Jobs** dataset (`Cleaned_DS_Jobs.csv`).

**Stack:** Python · pandas · scikit-learn · FastAPI · Streamlit · Plotly

> Built as the final project of my **IBM internship**, developed with **IBM Bob** (AI coding assistant). The main goal was to practise the full ML workflow: data cleaning → feature engineering → model pipeline → REST API → web UI → honest evaluation.

---

## 📌 What this project covers

- Data cleaning and feature preparation on a real-world job-postings dataset
- A scikit-learn `Pipeline` (imputation + scaling + one-hot encoding + Gradient Boosting)
- A FastAPI service that serves predictions and feature importances
- A Streamlit + Plotly interface for interactive use
- A cross-validated comparison against simple baselines (see [Results](#-results--honest-evaluation))

---

## 🗂️ Dataset

**Glassdoor Data Science Jobs** – [Kaggle link](https://www.kaggle.com/datasets/rashikrahmanpritom/data-science-job-posting-on-glassdoor)

`Cleaned_DS_Jobs.csv` is included in this repo: **660 job postings, 27 columns**. Salary values are Glassdoor *estimates* (in $K), not confirmed pay.

Columns used by the model:

| Feature | Type | Description |
|---|---|---|
| `Rating` | Numeric | Glassdoor company rating (-1 treated as missing) |
| `company_age` | Numeric | Age of the company in years |
| `python`, `excel`, `hadoop`, `spark`, `aws`, `tableau`, `big_data` | Binary | 1 if the skill appears in the job description |
| `same_state` | Binary | 1 if job location is in the same state as headquarters |
| `Size` | Categorical | Employee-count band |
| `Type of ownership` | Categorical | Public / Private / Nonprofit / etc. |
| `Industry`, `Sector` | Categorical | Company industry and sector |
| `job_simp` | Categorical | Simplified job title (data scientist, analyst, mle, ...) |
| `seniority` | Categorical | senior / jr / na |
| `job_state` | Categorical | US state of the job |

**Target:** `avg_salary` – average of the salary range, in $K.

---

## 🏗️ Project structure

```
.
├── Cleaned_DS_Jobs.csv               # Dataset (660 rows)
├── MukeshSamarit_SalaryPredictor.py   # Data cleaning + model training
├── model_comparison.py               # Baseline vs. model comparison (5-fold CV)
├── backend/
│   └── app.py                        # FastAPI REST API
├── frontend/
│   └── app.py                        # Streamlit web UI
├── MukeshSamarit_ProjectReport.docx  # Detailed project report
├── requirements.txt
├── README.md
└── model/                            # Auto-created after training
    ├── pipeline.pkl                  # Serialised sklearn pipeline
    └── meta.json                     # Feature lists, dropdown options, metrics
```

---

## ⚡ Setup & how to run

**Prerequisites:** Python 3.11+ and pip.

```bash
# 1. Clone and install
git clone https://github.com/MSDIAN124/Salary-Prediction-Model-Project-for-Internship.git
cd Salary-Prediction-Model-Project-for-Internship
pip install -r requirements.txt

# 2. Train the model (creates the model/ folder)
python MukeshSamarit_SalaryPredictor.py

# 3. (Optional) Compare against baselines
python model_comparison.py

# 4. Start the API   → http://localhost:8000  (Swagger docs at /docs)
uvicorn backend.app:app --reload --port 8000

# 5. In a new terminal, start the UI   → http://localhost:8501
streamlit run frontend/app.py
```

Run all commands from the project root folder.

---

## 🔌 API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/meta` | Model metadata and dropdown options |
| `POST` | `/predict` | Predict salary from job features |
| `GET` | `/feature-importance` | Top 20 feature importances |

**Sample `/predict` request**

```json
{
  "job_title": "Senior Data Scientist",
  "rating": 4.0,
  "company_age": 15,
  "size": "1001 to 5000 employees",
  "type_of_ownership": "Company - Public",
  "industry": "Internet",
  "sector": "Information Technology",
  "job_state": "CA",
  "seniority": "senior",
  "same_state": 1,
  "python": 1,
  "spark": 1,
  "aws": 1
}
```

**Sample response** (values depend on your training run)

```json
{
  "predicted_salary_k": 116.29,
  "predicted_salary_annual": 116291.5,
  "confidence_note": "Estimate based on 528 training samples. Typical error ≈ $33.18K.",
  "model_r2": -0.5694,
  "model_mae_k": 33.18
}
```

---

## 🧠 Model

```
ColumnTransformer
├── Numeric      → SimpleImputer(median)         → StandardScaler
└── Categorical  → SimpleImputer(most_frequent)  → OneHotEncoder(handle_unknown="ignore")
                          ↓
      GradientBoostingRegressor
      n_estimators=300 | learning_rate=0.05 | max_depth=5 | subsample=0.8
```

Split: 80% train (528 rows) / 20% test (132 rows), `random_state=42`.

---

## 📊 Results & honest evaluation

**Hold-out test set (training script, 132 rows):**

| Metric | Value |
|---|---|
| R² | -0.57 |
| MAE | ≈ $33.2K |

**5-fold cross-validation (`model_comparison.py`):**

| Model | R² | MAE ($K) |
|---|---|---|
| Mean baseline (always predict the average) | -0.00 | 28.5 |
| Ridge regression | -0.06 | 29.2 |
| Random Forest | -0.11 | 29.6 |
| Gradient Boosting (depth 5, 300 trees) – current model | -0.41 | 32.9 |
| Gradient Boosting (depth 3, 150 trees) | -0.10 | 29.5 |

**What this tells us**

- None of the tested models beats the simple "predict the average salary" baseline on this dataset.
- The deep Gradient Boosting model overfits: ~528 training rows against many one-hot columns (industry, sector, state).
- With only 660 rows and salaries that are Glassdoor estimates, the available features carry very little salary signal. Even seniority shows almost no difference in average salary in this data.
- So the app is best seen as a **working end-to-end ML pipeline and demo**, not a reliable salary estimator.

---

## 🔭 Possible improvements

- Use a larger, more recent dataset with real (not estimated) salaries
- Add text features from job descriptions and more precise location / company data
- Regularise or simplify the model (shallower trees, fewer categories) and tune with cross-validation
- Predict salary bands (low / medium / high) instead of exact values
- Deploy the app (e.g. Streamlit Community Cloud or Hugging Face Spaces)

---

## 👨‍💻 Author

**Mukesh Samarit** – civil engineering graduate moving into data analytics / data science.
[LinkedIn](#) · [GitHub](https://github.com/MSDIAN124)

*For educational and internship-demonstration purposes only.*
