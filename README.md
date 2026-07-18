# 🚀 Software Release Risk Prediction

An end-to-end machine learning project that predicts the **risk level** (Low / Medium / High) of a software release based on code, testing, and team metrics — with an interactive Streamlit app, a SQL analytics layer, and a Power BI dashboard.

## 🔗 Live Links

| Resource | Link |
|---|---|
| 🔮 Live Demo (Streamlit) | `<add your Streamlit Cloud URL here>` |
| 📊 Power BI Dashboard | `<add your published Power BI link here>` |
| 📁 GitHub Repo | `<this repo>` |

## 📌 Overview

Software teams often ship releases without a clear, data-driven signal of how risky that release actually is. This project trains a machine learning model on historical release data — commits, bugs, test coverage, code churn, security vulnerabilities, and more — to classify a release as **Low**, **Medium**, or **High** risk, and gives teams a recommendation before deployment.

## 🧰 Tech Stack

- **Python** — data processing and modeling
- **scikit-learn** — Random Forest classifier
- **Streamlit** — interactive web app (prediction tool + SQL insights dashboard)
- **SQLite** — SQL analytics layer on top of the dataset
- **Plotly** — interactive charts inside the Streamlit dashboard
- **Power BI** — standalone business-intelligence dashboard
- **joblib** — model persistence

## 📊 Model Performance

Trained on 5,000 historical release records using a `RandomForestClassifier` (200 trees):

- **Accuracy:** 87.9%
- Strong precision/recall on **Low** and **High** risk classes
- **Medium** risk is the hardest class to separate (lower recall) — a natural byproduct of class imbalance in the dataset, worth revisiting with techniques like class weighting or resampling in future iterations

See `train_model.py` for the full training pipeline and classification report.

## 🗂️ Project Structure

```
new project/
│
├── app.py                          # Streamlit app (Prediction + SQL Insights tabs)
├── train_model.py                  # Trains the model from the CSV
├── requirements.txt                # Python dependencies
├── software_release_risk_dataset.csv   # Dataset (5,000 releases, 14 columns)
├── release_risk_model.pkl          # Trained model
├── label_encoder.pkl               # Label encoder for risk levels
├── analysis_queries.sql            # SQL analysis queries
├── release_risk.db                 # SQLite database (auto-created if missing)
├── Power_BI_Dashboard_Guide.docx   # Step-by-step guide to build the Power BI dashboard
└── README.md                       # This file
```

## ✨ Features

### 1. Risk Prediction Tab
Enter release metrics (commits, bugs, test coverage, code churn, developer experience, security vulnerabilities, etc.) in the sidebar and get:
- Predicted risk level with confidence score
- Probability breakdown across all three risk classes
- A concrete deployment recommendation based on the result

### 2. SQL Insights Dashboard Tab
Every chart is powered by a live SQL query against a SQLite database built from the dataset:
- KPI cards (total releases, high-risk count, avg test coverage, total security vulnerabilities)
- Risk level distribution (donut chart)
- Average critical bugs & security vulnerabilities by risk level
- Test coverage buckets vs. risk level
- Code churn vs. test coverage (scatter, sized by critical bugs)
- Developer experience vs. bug/vulnerability trends
- A free-form SQL query box to explore the data yourself

### 3. Power BI Dashboard
A separate, polished analytics dashboard built in Power BI Desktop, following `Power_BI_Dashboard_Guide.docx`: KPI cards, risk distribution, risk drivers, coverage impact, churn-vs-coverage scatter, experience trend, and a top-risky-releases table — all cross-filterable via slicers.

## 🖥️ How to Run Locally

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd "new project"

# 2. (Recommended) create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) retrain the model if you update the dataset
python train_model.py

# 5. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501` with two tabs: **Risk Prediction** and **SQL Insights Dashboard**.

## ☁️ Deployment

Deployed on **Streamlit Community Cloud**. If redeploying:
1. Push changes to the `main` branch
2. Streamlit Cloud auto-detects the push and redeploys
3. Ensure `requirements.txt` and all `.pkl` files are committed (not `.gitignore`d)
4. If you hit Python version issues during build, set the Python version explicitly in the app's **Settings → Advanced settings** on Streamlit Cloud (3.12 is a safe choice)

## 📈 Building the Power BI Dashboard

See `Power_BI_Dashboard_Guide.docx` for full step-by-step instructions, including:
- Connecting Power BI to the CSV or the SQLite database
- Power Query cleanup steps
- DAX measures to create
- Recommended visual layout
- Color scheme matched to the Streamlit app (green = Low, amber = Medium, red = High)

## 🔮 Future Improvements

- Address class imbalance to improve Medium-risk recall
- Add model explainability (e.g., SHAP values) to show which features drove a given prediction
- Add historical release tracking (store predictions over time)
- CI/CD pipeline to auto-retrain the model when the dataset updates

## 📄 License

Add your preferred license here (e.g., MIT).
