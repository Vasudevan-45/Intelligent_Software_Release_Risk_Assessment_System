import sys
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_fscore_support
)

# -------------------------------
# Compatibility shim (numpy pickle)
# -------------------------------
if "numpy._core" not in sys.modules:
    try:
        import numpy._core  # noqa: F401
    except ImportError:
        import numpy.core as core
        sys.modules["numpy._core"] = core

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Software Release Risk Assessment System",
    page_icon="🚀",
    layout="wide"
)

# -------------------------------
# CUSTOM CSS
# -------------------------------
st.markdown("""
<style>
.main{
    background-color:#0f172a;
}
.block-container{
    padding-top:2rem;
}
.metric-card{
    background:#1e293b;
    padding:15px;
    border-radius:10px;
}
h1,h2,h3{
color:white;
}
p{
color:#d1d5db;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# CONSTANTS
# -------------------------------
REQUIRED_FEATURES = [
    "Commits", "Bugs_Reported", "Critical_Bugs", "Test_Coverage",
    "Failed_Test_Cases", "Code_Churn", "Cyclomatic_Complexity", "Files_Changed",
    "Developer_Experience", "Security_Vulnerabilities", "Build_Time_Minutes",
    "Deployment_Frequency",
]
RISK_COLORS = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}


def style_fig(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e5e7eb",
        margin=dict(t=30, b=20, l=20, r=20),
    )
    return fig


@st.cache_resource
def load_default_model():
    model = joblib.load("release_risk_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, label_encoder


@st.cache_resource
def evaluate_default_model():
    """Re-create the exact train/test split used in train_model.py and evaluate
    the bundled model on the genuine held-out test set (not data it was trained on)."""
    model, label_encoder = load_default_model()
    df = pd.read_csv("software_release_risk_dataset.csv")
    X = df.drop(["Risk_Level", "Release_ID"], axis=1)
    y = label_encoder.transform(df["Risk_Level"])
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    return {
        "X_test": X_test, "y_test": y_test, "y_pred": y_pred, "y_proba": y_proba,
        "classes": label_encoder.classes_,
        "accuracy": accuracy_score(y_test, y_pred),
    }


def render_model_performance(y_test, y_pred, y_proba, classes, key_prefix=""):
    """Shared model performance panel: confusion matrix, ROC curves, classification report."""
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{acc*100:.1f}%")
    m2.metric("Precision (weighted)", f"{precision*100:.1f}%")
    m3.metric("Recall (weighted)", f"{recall*100:.1f}%")
    m4.metric("F1 Score (weighted)", f"{f1*100:.1f}%")

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm, text_auto=True, x=list(classes), y=list(classes),
            color_continuous_scale="Blues", labels=dict(x="Predicted", y="Actual", color="Count")
        )
        st.plotly_chart(style_fig(fig_cm), use_container_width=True, key=f"{key_prefix}_cm")

    with c2:
        st.subheader("ROC Curve (One-vs-Rest)")
        n_classes = len(classes)
        if n_classes >= 2 and y_proba is not None:
            y_test_bin = label_binarize(y_test, classes=range(n_classes))
            if n_classes == 2:
                y_test_bin = np.hstack([1 - y_test_bin, y_test_bin])
            fig_roc = go.Figure()
            for i, cls_name in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{cls_name} (AUC={roc_auc:.2f})"))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Random"))
            fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            st.plotly_chart(style_fig(fig_roc), use_container_width=True, key=f"{key_prefix}_roc")
        else:
            st.info("ROC curve needs at least 2 classes and predicted probabilities.")

    st.markdown("---")
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, target_names=[str(c) for c in classes], output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose().round(3)
    st.dataframe(report_df, use_container_width=True)


# -------------------------------
# TITLE
# -------------------------------
st.markdown("""
<div style="background:linear-gradient(90deg,#2563eb,#7c3aed);
padding:25px;
border-radius:15px;
text-align:center;
box-shadow:0px 5px 20px rgba(0,0,0,0.3);">

<h1 style="color:white;">
🛡️ Intelligent Software Release Risk Assessment System
</h1>

<h4 style="color:white;">
AI-Powered Software Quality Analytics & Release Risk Prediction
</h4>

<p style="color:white;">
✔ Machine Learning
|
✔ SQL Analytics
|
✔ Interactive Dashboard
|
✔ Business Intelligence
</p>

</div>
""", unsafe_allow_html=True)

st.write("")

st.write(
"""
Analyze software release datasets using
✅ Machine Learning
✅ SQL Analytics
✅ Interactive Charts
✅ KPI Dashboard
✅ Risk Prediction

Upload **any software defect dataset** to begin.
"""
)

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.image(
    "https://img.icons8.com/color/96/artificial-intelligence.png",
    width=90
)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go To",
    [
        "🏠 Home",
        "📂 Upload Dataset",
        "📊 Dashboard",
        "🗄 SQL Explorer",
        "🤖 Risk Prediction",
        "📈 Analytics",
        "🔮 Future Scope"
    ]
)

# =========================================================================
# HOME
# =========================================================================
if page == "🏠 Home":
    st.header("Welcome")
    col1, col2, col3 = st.columns(3)
    col1.metric("Machine Learning", "✔")
    col2.metric("SQL Dashboard", "✔")
    col3.metric("Interactive Charts", "✔")
    st.info(
"""
This application can analyze software engineering datasets from:
• Kaggle
• NASA
• PROMISE
• Custom CSV Files

It automatically generates dashboards and performs risk prediction whenever compatible features are available.
"""
    )
    if "df" in st.session_state:
        st.success(f"A dataset is currently loaded: {st.session_state['df'].shape[0]} rows, "
                   f"{st.session_state['df'].shape[1]} columns.")
    else:
        st.warning("No dataset loaded yet. Go to **📂 Upload Dataset** to get started.")

# =========================================================================
# UPLOAD DATASET
# =========================================================================
elif page == "📂 Upload Dataset":
    st.header("📂 Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Dataset Uploaded Successfully")

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Dataset Shape")
        c1, c2 = st.columns(2)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])

        st.subheader("Column Names")
        st.write(df.columns.tolist())

        st.subheader("Data Types")
        st.dataframe(df.dtypes.astype(str))

        st.subheader("Missing Values")
        st.dataframe(df.isnull().sum())

        st.session_state["df"] = df
        st.session_state["dataset_name"] = uploaded_file.name

        # reset any previously trained custom model since the dataset changed
        for key in ["custom_model", "custom_target_col", "custom_target_encoder",
                    "custom_feature_cols", "custom_feature_encoders", "custom_feature_info",
                    "custom_accuracy"]:
            st.session_state.pop(key, None)
    else:
        st.info("Upload a CSV dataset to continue.")
        if "df" in st.session_state:
            st.caption(f"Currently loaded: **{st.session_state.get('dataset_name', 'dataset')}** "
                       f"({st.session_state['df'].shape[0]} rows)")

# =========================================================================
# DASHBOARD
# =========================================================================
elif page == "📊 Dashboard":
    st.header("📊 Dashboard")

    if "df" not in st.session_state:
        st.warning("Please upload a dataset first from **📂 Upload Dataset**.")
    else:
        df = st.session_state["df"]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        has_schema = all(c in df.columns for c in REQUIRED_FEATURES)

        st.subheader("🎯 Key Performance Indicators")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Rows", f"{df.shape[0]:,}")
        k2.metric("Total Columns", df.shape[1])
        k3.metric("Numeric Columns", len(numeric_cols))
        k4.metric("Categorical Columns", len(categorical_cols))

        k5, k6, k7, k8 = st.columns(4)
        k5.metric("Missing Values", int(df.isnull().sum().sum()))
        k6.metric("Duplicate Rows", int(df.duplicated().sum()))

        if has_schema and "Risk_Level" in df.columns:
            high_count = int((df["Risk_Level"] == "High").sum())
            k7.metric("High Risk Releases", high_count)
            k8.metric("Avg Test Coverage", f"{df['Test_Coverage'].mean():.1f}%")
        elif "custom_accuracy" in st.session_state:
            k7.metric("Model Trained On", st.session_state.get("custom_target_col", "—"))
            k8.metric("Model Accuracy", f"{st.session_state['custom_accuracy']*100:.1f}%")
        else:
            k7.metric("Model Trained On", "—")
            k8.metric("Model Accuracy", "Not trained")

        st.markdown("---")

        # ---- Executive Summary ----
        st.subheader("📋 Executive Summary")
        missing_pct = 100 * df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        summary_lines = [
            f"This dataset contains **{df.shape[0]:,} rows** and **{df.shape[1]} columns** "
            f"({len(numeric_cols)} numeric, {len(categorical_cols)} categorical).",
            f"Data completeness is **{100 - missing_pct:.1f}%** "
            f"({int(df.isnull().sum().sum())} missing values across the dataset).",
        ]
        if has_schema and "Risk_Level" in df.columns:
            high_pct = 100 * (df["Risk_Level"] == "High").sum() / len(df)
            summary_lines.append(
                f"**{high_pct:.1f}%** of releases in this dataset are classified as **High risk**, "
                f"using the pretrained release-risk model."
            )
        elif "custom_accuracy" in st.session_state:
            summary_lines.append(
                f"A model was trained on this data to predict **{st.session_state['custom_target_col']}**, "
                f"achieving **{st.session_state['custom_accuracy']*100:.1f}%** test accuracy."
            )
        st.info("\n\n".join(summary_lines))

        # ---- Model Performance Summary ----
        st.subheader("🧠 Model Performance Summary")
        if has_schema and "Risk_Level" in df.columns:
            perf = evaluate_default_model()
            p1, p2, p3 = st.columns(3)
            p1.metric("Held-out Test Accuracy", f"{perf['accuracy']*100:.1f}%")
            precision, recall, f1, _ = precision_recall_fscore_support(
                perf["y_test"], perf["y_pred"], average="weighted", zero_division=0
            )
            p2.metric("Weighted F1 Score", f"{f1*100:.1f}%")
            p3.metric("Weighted Recall", f"{recall*100:.1f}%")
            st.caption("Full confusion matrix, ROC curves, and classification report are available on the **🤖 Risk Prediction** page.")
        elif "custom_accuracy" in st.session_state:
            st.metric("Test Accuracy", f"{st.session_state['custom_accuracy']*100:.1f}%")
            st.caption("Full model performance details are available on the **🤖 Risk Prediction** page.")
        else:
            st.caption("No model trained yet for this dataset. Visit **🤖 Risk Prediction** to train one.")

        st.markdown("---")

        if has_schema and "Risk_Level" in df.columns:
            st.subheader("Risk Level Distribution")
            dist = df["Risk_Level"].value_counts().reset_index()
            dist.columns = ["Risk_Level", "Count"]
            fig = px.pie(dist, names="Risk_Level", values="Count",
                        color="Risk_Level", color_discrete_map=RISK_COLORS, hole=0.45)
            st.plotly_chart(style_fig(fig), use_container_width=True)
        elif categorical_cols:
            st.subheader(f"Distribution of '{categorical_cols[0]}'")
            col_choice = st.selectbox("Choose a categorical column", categorical_cols)
            dist = df[col_choice].value_counts().reset_index()
            dist.columns = [col_choice, "Count"]
            fig = px.pie(dist, names=col_choice, values="Count", hole=0.45)
            st.plotly_chart(style_fig(fig), use_container_width=True)

        if len(numeric_cols) >= 2:
            st.subheader("Correlation Heatmap")
            corr = df[numeric_cols].corr()
            fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
            st.plotly_chart(style_fig(fig_corr), use_container_width=True)

        st.subheader("Summary Statistics")
        st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

# =========================================================================
# SQL EXPLORER
# =========================================================================
elif page == "🗄 SQL Explorer":
    st.header("🗄 SQL Explorer")

    if "df" not in st.session_state:
        st.warning("Please upload a dataset first from **📂 Upload Dataset**.")
    else:
        df = st.session_state["df"]
        conn = sqlite3.connect(":memory:")
        df.to_sql("data", conn, if_exists="replace", index=False)

        st.caption("Your uploaded data is available as a table named `data`. Only SELECT statements are allowed.")

        st.subheader("Quick Queries")
        qc1, qc2, qc3 = st.columns(3)
        quick_query = None
        if qc1.button("Preview first 10 rows"):
            quick_query = "SELECT * FROM data LIMIT 10;"
        if qc2.button("Row count"):
            quick_query = "SELECT COUNT(*) AS Total_Rows FROM data;"
        if qc3.button("Column summary (numeric)"):
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                agg = ", ".join([f'AVG("{c}") AS avg_{c}' for c in numeric_cols[:5]])
                quick_query = f"SELECT {agg} FROM data;"
            else:
                quick_query = "SELECT * FROM data LIMIT 5;"

        default_query = quick_query if quick_query else "SELECT * FROM data LIMIT 10;"
        user_query = st.text_area("SQL Query", value=default_query, height=120)

        if st.button("Run Query"):
            try:
                if not user_query.strip().lower().startswith("select"):
                    st.error("Only SELECT queries are allowed.")
                else:
                    result_df = pd.read_sql_query(user_query, conn)
                    st.dataframe(result_df, use_container_width=True)
            except Exception as e:
                st.error(f"Query failed: {e}")

        conn.close()

# =========================================================================
# RISK PREDICTION
# =========================================================================
elif page == "🤖 Risk Prediction":
    st.header("🤖 Risk Prediction")

    if "df" not in st.session_state:
        st.warning("Please upload a dataset first from **📂 Upload Dataset**.")
    else:
        df = st.session_state["df"]
        has_schema = all(c in df.columns for c in REQUIRED_FEATURES)

        if has_schema:
            st.success("This dataset matches the original release-risk schema — using the pretrained model.")
            default_model, default_label_encoder = load_default_model()

            st.sidebar.header("Enter Software Release Details")
            commits = st.sidebar.number_input("Commits", 0, 1000, 150)
            bugs = st.sidebar.number_input("Bugs Reported", 0, 100, 10)
            critical = st.sidebar.number_input("Critical Bugs", 0, 20, 2)
            coverage = st.sidebar.slider("Test Coverage (%)", 0.0, 100.0, 85.0)
            failed = st.sidebar.number_input("Failed Test Cases", 0, 100, 5)
            churn = st.sidebar.number_input("Code Churn", 0, 20000, 2500)
            complexity = st.sidebar.number_input("Cyclomatic Complexity", 1, 100, 15)
            files = st.sidebar.number_input("Files Changed", 1, 300, 30)
            experience = st.sidebar.slider("Developer Experience (Years)", 0.0, 20.0, 5.0)
            security = st.sidebar.number_input("Security Vulnerabilities", 0, 20, 1)
            build = st.sidebar.number_input("Build Time (Minutes)", 1, 120, 20)
            deployment = st.sidebar.number_input("Deployment Frequency", 1, 30, 6)

            input_data = pd.DataFrame({
                "Commits": [commits], "Bugs_Reported": [bugs], "Critical_Bugs": [critical],
                "Test_Coverage": [coverage], "Failed_Test_Cases": [failed], "Code_Churn": [churn],
                "Cyclomatic_Complexity": [complexity], "Files_Changed": [files],
                "Developer_Experience": [experience], "Security_Vulnerabilities": [security],
                "Build_Time_Minutes": [build], "Deployment_Frequency": [deployment]
            })

            st.subheader("Input Data")
            st.dataframe(input_data)

            if st.button("Predict Risk"):
                prediction = default_model.predict(input_data)
                prediction_label = default_label_encoder.inverse_transform(prediction)[0]
                probability = default_model.predict_proba(input_data)
                confidence = probability.max() * 100

                st.markdown("---")
                st.header("Prediction Result")
                if prediction_label == "Low":
                    st.success(f"✅ Release Risk : {prediction_label}")
                elif prediction_label == "Medium":
                    st.warning(f"⚠️ Release Risk : {prediction_label}")
                else:
                    st.error(f"🚨 Release Risk : {prediction_label}")

                st.write(f"### Confidence : {confidence:.2f}%")
                prob_df = pd.DataFrame({
                    "Risk Level": default_label_encoder.classes_,
                    "Probability (%)": probability[0] * 100
                })
                st.dataframe(prob_df)
                st.bar_chart(prob_df.set_index("Risk Level"))

            st.markdown("---")
            with st.expander("🧠 Model Performance Details (held-out test set)", expanded=False):
                perf = evaluate_default_model()
                render_model_performance(
                    perf["y_test"], perf["y_pred"], perf["y_proba"], perf["classes"],
                    key_prefix="default_model"
                )

        else:
            st.info("This dataset doesn't match the release-risk schema. "
                     "Pick a target column below and train a model on the spot.")

            all_cols = list(df.columns)
            default_target_idx = len(all_cols) - 1
            for i, c in enumerate(all_cols):
                if df[c].nunique() <= 10 and df[c].dtype == object:
                    default_target_idx = i
                    break

            target_col = st.selectbox("Target column to predict", options=all_cols, index=default_target_idx)
            exclude_default = [c for c in all_cols if c != target_col and df[c].nunique() == len(df)]
            exclude_cols = st.multiselect(
                "Columns to exclude (e.g. ID columns)",
                options=[c for c in all_cols if c != target_col],
                default=exclude_default,
            )

            if st.button("🚀 Train Model on This Dataset"):
                work_df = df.dropna(subset=[target_col]).copy()
                feature_cols = [c for c in all_cols if c != target_col and c not in exclude_cols]

                if len(feature_cols) == 0:
                    st.error("No feature columns left after exclusions.")
                else:
                    target_encoder = LabelEncoder()
                    y = target_encoder.fit_transform(work_df[target_col].astype(str))

                    if len(np.unique(y)) < 2:
                        st.error("Target column only has one class — choose a different column.")
                    else:
                        X = pd.DataFrame(index=work_df.index)
                        feature_encoders = {}
                        feature_info = {}

                        for col in feature_cols:
                            if work_df[col].dtype == object or work_df[col].nunique() <= 15:
                                work_df[col] = work_df[col].astype(str)
                                enc = LabelEncoder()
                                X[col] = enc.fit_transform(work_df[col])
                                feature_encoders[col] = enc
                                feature_info[col] = {"type": "categorical", "options": list(enc.classes_)}
                            else:
                                fill_val = work_df[col].median()
                                X[col] = work_df[col].fillna(fill_val)
                                feature_info[col] = {
                                    "type": "numeric",
                                    "min": float(X[col].min()),
                                    "max": float(X[col].max()),
                                    "mean": float(X[col].mean()),
                                }

                        stratify_arg = y if min(np.bincount(y)) >= 2 else None
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42, stratify=stratify_arg
                        )
                        clf = RandomForestClassifier(n_estimators=200, random_state=42)
                        clf.fit(X_train, y_train)
                        y_pred_test = clf.predict(X_test)
                        y_proba_test = clf.predict_proba(X_test)
                        acc = accuracy_score(y_test, y_pred_test)

                        st.session_state["custom_model"] = clf
                        st.session_state["custom_target_col"] = target_col
                        st.session_state["custom_target_encoder"] = target_encoder
                        st.session_state["custom_feature_cols"] = feature_cols
                        st.session_state["custom_feature_encoders"] = feature_encoders
                        st.session_state["custom_feature_info"] = feature_info
                        st.session_state["custom_accuracy"] = acc
                        st.session_state["custom_y_test"] = y_test
                        st.session_state["custom_y_pred"] = y_pred_test
                        st.session_state["custom_y_proba"] = y_proba_test
                        st.session_state["custom_classes"] = target_encoder.classes_

                        st.success(f"✅ Model trained — test accuracy: **{acc*100:.1f}%**")

            if "custom_model" in st.session_state and st.session_state.get("custom_target_col") == target_col:
                st.markdown("---")
                st.subheader(f"Predict '{target_col}'")

                feature_info = st.session_state["custom_feature_info"]
                feature_encoders = st.session_state["custom_feature_encoders"]
                target_encoder = st.session_state["custom_target_encoder"]
                clf = st.session_state["custom_model"]

                st.sidebar.header(f"Enter values to predict '{target_col}'")
                input_values = {}
                for col, info in feature_info.items():
                    if info["type"] == "categorical":
                        input_values[col] = st.sidebar.selectbox(col, options=info["options"])
                    else:
                        max_v = info["max"] if info["max"] > info["min"] else info["min"] + 1
                        input_values[col] = st.sidebar.number_input(
                            col, min_value=float(info["min"]), max_value=float(max_v),
                            value=float(round(info["mean"], 2))
                        )

                input_row = {}
                for col, info in feature_info.items():
                    if info["type"] == "categorical":
                        enc = feature_encoders[col]
                        val = input_values[col]
                        input_row[col] = enc.transform([val])[0] if val in enc.classes_ else 0
                    else:
                        input_row[col] = input_values[col]

                input_df = pd.DataFrame([input_row])[st.session_state["custom_feature_cols"]]
                st.dataframe(input_df)

                if st.button("Predict"):
                    pred = clf.predict(input_df)
                    pred_label = target_encoder.inverse_transform(pred)[0]
                    proba = clf.predict_proba(input_df)
                    confidence = proba.max() * 100

                    st.success(f"Predicted **{target_col}**: {pred_label}")
                    st.write(f"### Confidence : {confidence:.2f}%")
                    prob_df = pd.DataFrame({
                        target_col: target_encoder.classes_,
                        "Probability (%)": proba[0] * 100
                    })
                    st.dataframe(prob_df)
                    st.bar_chart(prob_df.set_index(target_col))

                st.markdown("---")
                with st.expander("🧠 Model Performance Details (held-out test set)", expanded=False):
                    render_model_performance(
                        st.session_state["custom_y_test"],
                        st.session_state["custom_y_pred"],
                        st.session_state["custom_y_proba"],
                        st.session_state["custom_classes"],
                        key_prefix="custom_model"
                    )

# =========================================================================
# ANALYTICS
# =========================================================================
elif page == "📈 Analytics":
    st.header("📈 Analytics")

    if "df" not in st.session_state:
        st.warning("Please upload a dataset first from **📂 Upload Dataset**.")
    else:
        df = st.session_state["df"]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        if numeric_cols:
            st.subheader("Distribution of a Numeric Column")
            col_choice = st.selectbox("Choose a numeric column", numeric_cols)
            fig_hist = px.histogram(df, x=col_choice, nbins=30, marginal="box")
            st.plotly_chart(style_fig(fig_hist), use_container_width=True)

        if numeric_cols and categorical_cols:
            st.subheader("Numeric Column by Category")
            c1, c2 = st.columns(2)
            num_choice = c1.selectbox("Numeric column", numeric_cols, key="box_num")
            cat_choice = c2.selectbox("Category column", categorical_cols, key="box_cat")
            fig_box = px.box(df, x=cat_choice, y=num_choice, color=cat_choice)
            st.plotly_chart(style_fig(fig_box), use_container_width=True)

        if len(numeric_cols) >= 2:
            st.subheader("Scatter Plot Explorer")
            c1, c2, c3 = st.columns(3)
            x_col = c1.selectbox("X axis", numeric_cols, index=0, key="scatter_x")
            y_col = c2.selectbox("Y axis", numeric_cols, index=min(1, len(numeric_cols) - 1), key="scatter_y")
            color_col = c3.selectbox("Color by (optional)", ["None"] + categorical_cols, key="scatter_color")
            color_arg = None if color_col == "None" else color_col
            fig_scatter = px.scatter(df, x=x_col, y=y_col, color=color_arg, opacity=0.6)
            st.plotly_chart(style_fig(fig_scatter), use_container_width=True)
        else:
            st.info("Not enough numeric columns for a scatter plot.")

# =========================================================================
# FUTURE SCOPE
# =========================================================================
elif page == "🔮 Future Scope":
    st.header("🔮 Future Scope")
    st.markdown("""
    Planned improvements for this project:

    - **Model explainability** — SHAP values to show which features drive each prediction
    - **Class imbalance handling** — better recall on minority risk classes via resampling or class weighting
    - **Historical tracking** — store predictions over time to monitor release quality trends
    - **CI/CD integration** — auto-retrain the model whenever new release data is added
    - **Multi-model comparison** — let users compare Random Forest against Logistic Regression, XGBoost, etc.
    - **Export reports** — generate a downloadable PDF/Word summary of dashboard insights
    - **Power BI integration** — publish the same analytics as a standalone BI dashboard
    """)
