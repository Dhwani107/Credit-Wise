import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = "loan_approval_data.csv"


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create the same features used by the notebook model."""
    data = data.copy()
    data["Total_Income"] = data["Applicant_Income"] + data["Coapplicant_Income"]
    data["Loan_to_Income"] = data["Loan_Amount"] / (data["Total_Income"] + 1)
    data["Savings_to_Loan"] = data["Savings"] / (data["Loan_Amount"] + 1)
    data["Collateral_to_Loan"] = data["Collateral_Value"] / (data["Loan_Amount"] + 1)
    data["Income_per_Dependent"] = data["Total_Income"] / (data["Dependents"] + 1)
    data["Credit_DTI_Interaction"] = data["Credit_Score"] / (data["DTI_Ratio"] + 0.05)
    data["Credit_Score_sq"] = data["Credit_Score"] ** 2
    data["DTI_Ratio_sq"] = data["DTI_Ratio"] ** 2
    return data


@st.cache_resource
def train_model() -> tuple[Pipeline, list[str], list[str]]:
    data = pd.read_csv(DATA_PATH).dropna(subset=["Loan_Approved"])
    data = data.drop(columns=["Applicant_ID"])
    target = (data.pop("Loan_Approved") == "Yes").astype(int)
    features = build_features(data)

    numeric_features = features.select_dtypes(include="number").columns.tolist()
    categorical_features = features.select_dtypes(exclude="number").columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=23, weights="uniform", p=1, leaf_size=15
                ),
            ),
        ]
    )
    model.fit(features, target)
    return model, numeric_features, categorical_features


st.set_page_config(
    page_title="Credit Wise | Loan Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #f5f7fb; color: #172554; }
    section[data-testid="stMain"] { color: #172554; }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp label, .stApp [data-testid="stWidgetLabel"] p,
    .stApp [data-testid="stMarkdownContainer"] p {
        color: #172554 !important;
    }
    section[data-testid="stMain"] label,
    section[data-testid="stMain"] [data-testid="stWidgetLabel"],
    section[data-testid="stMain"] [data-testid="stWidgetLabel"] *,
    section[data-testid="stMain"] [data-testid="stWidgetLabel"] p {
        color: #172554 !important;
        opacity: 1 !important;
    }
    .stApp [data-testid="stCaptionContainer"] p { color: #475569 !important; }
    [data-testid="stSidebar"] { background: #111827; }
    [data-testid="stSidebar"] * { color: #e5e7eb; }
    .stApp [data-testid="stSidebar"] h1,
    .stApp [data-testid="stSidebar"] h2,
    .stApp [data-testid="stSidebar"] h3,
    .stApp [data-testid="stSidebar"] p,
    .stApp [data-testid="stSidebar"] label,
    .stApp [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
        color: #f8fafc !important;
    }
    .hero {
        padding: 2.2rem 2.5rem; border-radius: 22px; color: white;
        background: linear-gradient(135deg, #172554 0%, #2563eb 55%, #06b6d4 100%);
        box-shadow: 0 12px 30px rgba(37, 99, 235, .18); margin-bottom: 1.5rem;
    }
    .stApp .hero h1 { margin: 0; font-size: 2.5rem; color: white !important; }
    .stApp .hero p { margin: .55rem 0 0; color: #dbeafe !important; font-size: 1.05rem; }
    .section-title { color: #172554; font-size: 1.35rem; font-weight: 700; margin: 1.2rem 0 .4rem; }
    .result {
        padding: 1.4rem; border-radius: 18px; text-align: center;
        background: white; border: 1px solid #dbeafe;
        box-shadow: 0 8px 25px rgba(15, 23, 42, .07);
    }
    .result h2 { margin: 0; color: #172554; }
    .result p { color: #64748b; margin-bottom: 0; }
    .stApp [data-baseweb="select"] *,
    .stApp input,
    div.stButton > button,
    div.stButton > button * { color: #f8fafc !important; }
    div.stButton > button {
        width: 100%; border: 0; border-radius: 10px; padding: .7rem;
        color: white; background: linear-gradient(90deg, #2563eb, #0891b2);
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

model, _, _ = train_model()

st.markdown(
    """
    <div class="hero">
        <h1 style="color: #ffffff !important;">💳 Credit Wise</h1>
        <p>Smart, data-driven loan approval screening powered by an optimized KNN model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Enter the applicant's personal, employment, income, credit, and loan details below. "
    "The app will provide a preliminary KNN-based approval prediction."
)

with st.sidebar:
    st.markdown("## About Credit Wise")
    st.caption(
        "Complete the application form to receive an instant preliminary prediction."
    )
    st.markdown("---")
    st.markdown("**Model:** Tuned KNN")
    st.markdown("**Precision:** 89.4%")
    st.markdown("**F1 score:** 78.5%")
    st.info("This prediction is for screening support only, not a final lending decision.")

st.header("Applicant Profile")
profile_left, profile_mid, profile_right = st.columns(3)
with profile_left:
    st.markdown("#### Personal Details")
    age = st.number_input(
        "Age (years)", min_value=18, max_value=100, value=35, step=1,
        help="Applicant's age in years.",
    )
    dependents = st.number_input(
        "Number of dependents", min_value=0, max_value=15, value=1, step=1,
        help="Number of people financially dependent on the applicant.",
    )
    gender = st.selectbox("Gender", ["Female", "Male"], help="Applicant's gender.")
with profile_mid:
    st.markdown("#### Employment & Family")
    employment = st.selectbox(
        "Employment status",
        ["Salaried", "Self-employed", "Contract", "Unemployed"],
        help="Current employment arrangement.",
    )
    marital = st.selectbox("Marital status", ["Single", "Married"], help="Marital status.")
    education = st.selectbox(
        "Education level", ["Graduate", "Not Graduate"], help="Highest education level."
    )
with profile_right:
    st.markdown("#### Location & Purpose")
    employer = st.selectbox(
        "Employer category",
        ["Private", "Government", "MNC", "Unemployed"],
        help="Type of employer or employment category.",
    )
    property_area = st.selectbox(
        "Property area", ["Urban", "Semiurban", "Rural"], help="Area where the property is located."
    )
    purpose = st.selectbox(
        "Loan purpose",
        ["Home", "Car", "Education", "Personal", "Business"],
        help="Reason for requesting the loan.",
    )

st.header("Financial Information")
finance_left, finance_mid, finance_right = st.columns(3)
with finance_left:
    st.markdown("#### Income & Savings")
    applicant_income = st.number_input(
        "Applicant income",
        min_value=0.0,
        value=10000.0,
        step=500.0,
        help="Applicant's regular income in the dataset's currency units.",
    )
    coapplicant_income = st.number_input(
        "Coapplicant income",
        min_value=0.0,
        value=3000.0,
        step=500.0,
        help="Regular income contributed by a coapplicant.",
    )
    savings = st.number_input(
        "Savings", min_value=0.0, value=10000.0, step=500.0,
        help="Applicant's available savings.",
    )
with finance_mid:
    st.markdown("#### Credit Profile")
    credit_score = st.number_input(
        "Credit score",
        min_value=300.0,
        max_value=900.0,
        value=680.0,
        step=1.0,
        help="Credit score, typically between 300 and 900.",
    )
    existing_loans = st.number_input(
        "Existing loans",
        min_value=0.0,
        max_value=20.0,
        value=1.0,
        step=1.0,
        help="Number of currently active loans.",
    )
    dti_ratio = st.number_input(
        "DTI ratio",
        min_value=0.0,
        max_value=1.0,
        value=0.30,
        step=0.01,
        help="Debt-to-income ratio as a decimal. For example, 0.30 means 30%.",
    )
with finance_right:
    st.markdown("#### Loan & Collateral")
    collateral = st.number_input(
        "Collateral value",
        min_value=0.0,
        value=25000.0,
        step=1000.0,
        help="Estimated value of the collateral offered.",
    )
    loan_amount = st.number_input(
        "Requested loan amount",
        min_value=0.0,
        value=20000.0,
        step=1000.0,
        help="Amount the applicant wants to borrow.",
    )
    loan_term = st.number_input(
        "Loan term (months)",
        min_value=6.0,
        max_value=120.0,
        value=60.0,
        step=12.0,
        help="Requested repayment period in months.",
    )

if st.button("Predict loan approval"):
    application = pd.DataFrame(
        [
            {
                "Applicant_Income": applicant_income,
                "Coapplicant_Income": coapplicant_income,
                "Employment_Status": employment,
                "Age": age,
                "Marital_Status": marital,
                "Dependents": dependents,
                "Credit_Score": credit_score,
                "Existing_Loans": existing_loans,
                "DTI_Ratio": dti_ratio,
                "Savings": savings,
                "Collateral_Value": collateral,
                "Loan_Amount": loan_amount,
                "Loan_Term": loan_term,
                "Loan_Purpose": purpose,
                "Property_Area": property_area,
                "Education_Level": education,
                "Gender": gender,
                "Employer_Category": employer,
            }
        ]
    )
    prediction = int(model.predict(build_features(application))[0])
    probability = float(model.predict_proba(build_features(application))[0, 1])
    if prediction:
        title, message, color = "Likely to be approved", "Your profile matches approval patterns in the training data.", "#059669"
    else:
        title, message, color = "Additional review recommended", "Your profile does not strongly match approval patterns in the training data.", "#dc2626"
    st.markdown(
        f'<div class="result"><h2 style="color:{color}">{title}</h2><p>{message}</p></div>',
        unsafe_allow_html=True,
    )
    metric_left, metric_right = st.columns(2)
    metric_left.metric("Approval likelihood", f"{probability:.1%}")
    metric_right.metric("Model", "Tuned KNN")
