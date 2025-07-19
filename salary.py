import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc

st.title("💼 Employee Salary Prediction App")
st.markdown("Predict whether an employee earns >50K or <=50K using ML and visualization.")

# Upload CSV
uploaded_file = st.file_uploader("Upload the dataset (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Raw Dataset")
    st.dataframe(df.head())

    # Clean & preprocess
    df.replace("?", pd.NA, inplace=True)
    df.dropna(inplace=True)

    X = df.drop("income", axis=1)
    y = df["income"]

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Preprocessing pipelines
    numerical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])

    # Model pipeline
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    # Predict & evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    st.subheader("✅ Model Accuracy")
    st.metric("Accuracy", f"{accuracy * 100:.2f}%")

    # -----------------------
    # Confusion Matrix
    # -----------------------
    st.subheader("🔍 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred, labels=["<=50K", ">50K"])
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["<=50K", ">50K"], yticklabels=["<=50K", ">50K"], ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

    # -----------------------
    # Feature Importance
    # -----------------------
    st.subheader("📌 Top 15 Important Features")
    encoded_features = model.named_steps["preprocessor"].transformers_[1][1].named_steps["onehot"].get_feature_names_out(categorical_cols)
    feature_names = numerical_cols + list(encoded_features)
    importances = model.named_steps["classifier"].feature_importances_
    feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    top_feats = feat_df.sort_values(by="Importance", ascending=False).head(15)

    fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=top_feats, ax=ax_fi)
    st.pyplot(fig_fi)

    # -----------------------
    # ROC Curve
    # -----------------------
    st.subheader("📈 ROC Curve")
    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_bin, y_proba)
    roc_auc = auc(fpr, tpr)

    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    ax_roc.plot([0, 1], [0, 1], linestyle="--")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)

    # -----------------------
    # Classification Report
    # -----------------------
    st.subheader("🧾 Classification Report")
    st.dataframe(pd.DataFrame(report).transpose())
