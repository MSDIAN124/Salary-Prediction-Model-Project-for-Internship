"""
frontend/app.py – Streamlit UI for the DS Jobs Salary Predictor.

Run with:  streamlit run salary_predictor/frontend/app.py
Requires the FastAPI backend to be running on http://localhost:8000
"""

import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

API_BASE = "http://localhost:8000"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="💼 DS Jobs Salary Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #0f3460;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #6c757d;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .salary-display {
        font-size: 3rem;
        font-weight: 800;
        color: #16a34a;
        text-align: center;
    }
    .salary-label {
        font-size: 1rem;
        color: #6c757d;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_meta() -> dict:
    try:
        r = requests.get(f"{API_BASE}/meta", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"⚠️ Cannot reach API at {API_BASE}. Start the backend first.\n\n`{e}`")
        return {}


@st.cache_data(ttl=3600)
def fetch_feature_importance() -> list[dict]:
    try:
        r = requests.get(f"{API_BASE}/feature-importance", timeout=5)
        r.raise_for_status()
        return r.json().get("features", [])
    except Exception:
        return []


def predict_salary(payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}/predict", json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        st.error(f"Prediction failed: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Request error: {e}")
        return None


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar(meta: dict) -> dict:
    st.sidebar.markdown("## 🔧 Job Details")
    st.sidebar.markdown("Fill in as many fields as possible for the best estimate.")
    st.sidebar.markdown("---")

    cat_opts = meta.get("cat_options", {})

    # ── Basic Info
    st.sidebar.markdown("### 📝 Basic Info")
    job_title = st.sidebar.text_input(
        "Job Title *",
        value="Data Scientist",
        help="e.g. Senior Data Scientist, ML Engineer, Data Analyst",
    )

    rating = st.sidebar.slider(
        "Company Rating (Glassdoor)", min_value=1.0, max_value=5.0,
        value=3.8, step=0.1,
    )

    company_age = st.sidebar.number_input(
        "Company Age (years)", min_value=0, max_value=200, value=15, step=1,
    )

    seniority_opts = [""] + cat_opts.get("seniority", ["na", "junior", "senior"])
    seniority = st.sidebar.selectbox(
        "Seniority Level",
        options=seniority_opts,
        format_func=lambda x: "— Auto-detect from title —" if x == "" else x.capitalize(),
    )

    # ── Location
    st.sidebar.markdown("### 📍 Location")
    state_opts = [""] + cat_opts.get("job_state", [
        "CA", "NY", "TX", "WA", "IL", "MA", "GA", "FL", "VA", "NC",
    ])
    job_state = st.sidebar.selectbox(
        "State",
        options=state_opts,
        format_func=lambda x: "— Select —" if x == "" else x,
    )
    same_state = st.sidebar.checkbox(
        "Job location is same as headquarters?", value=False
    )

    # ── Company Info
    st.sidebar.markdown("### 🏢 Company Info")
    size_opts = [""] + cat_opts.get("Size", [
        "1 to 50 employees", "51 to 200 employees", "201 to 500 employees",
        "501 to 1000 employees", "1001 to 5000 employees",
        "5001 to 10000 employees", "10000+ employees",
    ])
    size = st.sidebar.selectbox(
        "Company Size", options=size_opts,
        format_func=lambda x: "— Select —" if x == "" else x,
    )

    ownership_opts = [""] + cat_opts.get("Type of ownership", [
        "Company - Public", "Company - Private", "Nonprofit Organization",
        "Government", "Hospital", "College / University",
    ])
    ownership = st.sidebar.selectbox(
        "Type of Ownership", options=ownership_opts,
        format_func=lambda x: "— Select —" if x == "" else x,
    )

    industry_opts = [""] + cat_opts.get("Industry", [])
    industry = st.sidebar.selectbox(
        "Industry", options=industry_opts,
        format_func=lambda x: "— Select —" if x == "" else x,
    )

    sector_opts = [""] + cat_opts.get("Sector", [])
    sector = st.sidebar.selectbox(
        "Sector", options=sector_opts,
        format_func=lambda x: "— Select —" if x == "" else x,
    )

    # ── Skills
    st.sidebar.markdown("### 🛠️ Required Skills")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        python   = int(st.checkbox("Python",  value=True))
        hadoop   = int(st.checkbox("Hadoop",  value=False))
        aws      = int(st.checkbox("AWS",     value=False))
        big_data = int(st.checkbox("Big Data",value=False))
    with col2:
        excel    = int(st.checkbox("Excel",   value=False))
        spark    = int(st.checkbox("Spark",   value=False))
        tableau  = int(st.checkbox("Tableau", value=False))

    return {
        "job_title":         job_title,
        "rating":            rating,
        "company_age":       float(company_age),
        "seniority":         seniority or None,
        "job_state":         job_state or None,
        "same_state":        int(same_state),
        "size":              size or None,
        "type_of_ownership": ownership or None,
        "industry":          industry or None,
        "sector":            sector or None,
        "python":            python,
        "excel":             excel,
        "hadoop":            hadoop,
        "spark":             spark,
        "aws":               aws,
        "tableau":           tableau,
        "big_data":          big_data,
    }


# ── Charts ────────────────────────────────────────────────────────────────────
def render_gauge(predicted_k: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=predicted_k,
        delta={"reference": 80, "valueformat": ".1f", "suffix": "K"},
        title={"text": "Predicted Salary ($K)", "font": {"size": 18}},
        number={"suffix": "K", "valueformat": ".1f"},
        gauge={
            "axis": {"range": [20, 250], "tickwidth": 1},
            "bar":  {"color": "#3b82d4"},
            "steps": [
                {"range": [20,  60],  "color": "#fee2e2"},
                {"range": [60,  100], "color": "#fef9c3"},
                {"range": [100, 150], "color": "#dcfce7"},
                {"range": [150, 250], "color": "#d1fae5"},
            ],
            "threshold": {
                "line":      {"color": "#16a34a", "width": 4},
                "thickness": 0.75,
                "value":     predicted_k,
            },
        },
    ))
    fig.update_layout(height=300, margin=dict(t=40, b=10, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)


def render_feature_importance(features: list[dict]):
    if not features:
        st.info("Feature importance not available.")
        return
    df = pd.DataFrame(features).head(15)
    fig = px.bar(
        df.sort_values("importance"),
        x="importance", y="name",
        orientation="h",
        color="importance",
        color_continuous_scale="Blues",
        labels={"importance": "Importance Score", "name": "Feature"},
        title="Top 15 Feature Importances",
    )
    fig.update_layout(
        height=450, coloraxis_showscale=False,
        yaxis_title=None, margin=dict(l=10, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_salary_range(predicted_k: float, mae_k: float):
    low  = max(0, predicted_k - mae_k)
    high = predicted_k + mae_k
    fig  = go.Figure()
    fig.add_trace(go.Bar(
        x=["Low Estimate", "Predicted", "High Estimate"],
        y=[low, predicted_k, high],
        marker_color=["#f87171", "#3b82d4", "#34d399"],
        text=[f"${v:.1f}K" for v in [low, predicted_k, high]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Salary Range Estimate", yaxis_title="Salary ($K)",
        height=320, showlegend=False, margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    st.markdown('<div class="main-header">💼 DS Jobs Salary Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">ML-powered salary estimates based on Cleaned_DS_Jobs dataset (Glassdoor)</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    meta = fetch_meta()
    if not meta:
        st.stop()

    payload = render_sidebar(meta)

    # Model stats banner
    col1, col2, col3 = st.columns(3)
    col1.metric("Training Samples", f"{meta.get('n_train', '?'):,}")
    col2.metric("Model R²",         f"{meta.get('r2', 0):.4f}")
    col3.metric("Mean Abs Error",   f"${meta.get('mae', 0):.1f}K")
    st.markdown("---")

    predict_clicked = st.button("🔮 Predict Salary", type="primary", use_container_width=True)

    if predict_clicked:
        if not payload.get("job_title", "").strip():
            st.warning("Please enter a Job Title.")
        else:
            with st.spinner("Running prediction…"):
                result = predict_salary(payload)

            if result:
                pred_k   = result["predicted_salary_k"]
                pred_ann = result["predicted_salary_annual"]
                mae_k    = result["model_mae_k"]

                st.success("✅ Prediction complete!")
                st.markdown("---")

                st.markdown(
                    f'<div class="salary-display">${pred_k:.1f}K / yr</div>'
                    f'<div class="salary-label">≈ ${pred_ann:,.0f} annual</div>',
                    unsafe_allow_html=True,
                )
                st.markdown("<br>", unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                with c1:
                    render_gauge(pred_k)
                with c2:
                    render_salary_range(pred_k, mae_k)

                st.info(result.get("confidence_note", ""))
                st.markdown("---")

    with st.expander("📊 Model: Feature Importance", expanded=False):
        feats = fetch_feature_importance()
        render_feature_importance(feats)

    with st.expander("ℹ️ About this model", expanded=False):
        st.markdown(f"""
**Model:** Gradient Boosting Regressor (scikit-learn)

**Target:** `avg_salary` — average salary in $K (pre-computed in dataset)

**Features used:**
- Company Glassdoor rating, company age
- Skills: Python, Excel, Hadoop, Spark, AWS, Tableau, Big Data
- Seniority level, simplified job title
- Company size, industry, sector, ownership type, state

**Training data:** Glassdoor Data Science Jobs (Cleaned_DS_Jobs.csv)
- Train / Test split: 80% / 20%
- R² = **{meta.get('r2', '?')}** | MAE = **${meta.get('mae', '?')}K**

> Predictions are statistical estimates and may not reflect current market conditions.
        """)

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center;color:#aaa;font-size:0.85rem;'>"
        "Built with FastAPI · scikit-learn · Streamlit · Plotly"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
