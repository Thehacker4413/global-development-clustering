import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Global Development Clustering",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}

h1 {
    color: #12345b;
}

h2, h3 {
    color: #163d68;
}

.metric-card {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    background-color: white;
    border: 1px solid #e1e8f0;
}

.cluster-card {
    padding: 20px;
    border-radius: 15px;
    background-color: white;
    border: 1px solid #dce5ef;
    min-height: 180px;
}

.predict-button {
    font-size: 20px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca.pkl")
    kmeans = joblib.load("kmeans_model.pkl")
    feature_names = joblib.load("feature_names.pkl")

    return scaler, pca, kmeans, feature_names


try:

    scaler, pca, kmeans, feature_names = load_models()

except Exception as e:

    st.error("Model files could not be loaded.")

    st.write(
        "Make sure these files are in the same folder as app.py:"
    )

    st.code("""
scaler.pkl
pca.pkl
kmeans_model.pkl
feature_names.pkl
    """)

    st.stop()


# ============================================================
# CLUSTER INFORMATION
# ============================================================

cluster_names = {

    0: "High Population & Resource Usage",

    1: "Lower Scale & Higher Life Expectancy",

    2: "Young Population & Higher Mortality",

    3: "High Health & Digital Connectivity"
}


cluster_descriptions = {

    0:
    "Large population with relatively high energy usage and "
    "CO2 emissions, with comparatively lower health expenditure "
    "and internet usage.",

    1:
    "Lower population and economic/resource scale, with "
    "relatively higher life expectancy and working-age population.",

    2:
    "Younger population characteristics, with higher birth rate "
    "and infant mortality rate and comparatively lower life "
    "expectancy and urbanization.",

    3:
    "Higher health expenditure, internet usage and mobile phone "
    "usage, with a relatively larger elderly population and "
    "lower birth rate."
}


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🌍 Global Development")

    st.markdown("---")

    st.subheader("Navigation")

    st.write("🏠 Home")
    st.write("📊 Model Information")
    st.write("🎯 Prediction")
    st.write("📈 Cluster Analysis")

    st.markdown("---")

    st.subheader("Model Details")

    st.write("**Algorithm**")
    st.write("K-Means")

    st.write("**Number of Clusters**")
    st.write("4")

    st.write("**Original Features**")
    st.write("22")

    st.write("**PCA Components**")
    st.write("13")

    st.write("**Scaling**")
    st.write("StandardScaler")

    st.markdown("---")

    st.info(
        "The model uses PCA and K-Means clustering "
        "to group observations based on development indicators."
    )


# ============================================================
# HEADER
# ============================================================

st.title("🌍 Global Development Clustering")

st.markdown(
    """
    ### PCA + K-Means Machine Learning Application

    Enter values for the 22 development indicators and predict
    the corresponding cluster.
    """
)

st.divider()


# ============================================================
# MODEL OVERVIEW
# ============================================================

st.subheader("📊 Model Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        label="Input Features",
        value="22"
    )

with col2:

    st.metric(
        label="PCA Components",
        value="13"
    )

with col3:

    st.metric(
        label="Clusters",
        value="4"
    )

with col4:

    st.metric(
        label="Algorithm",
        value="K-Means"
    )


st.divider()


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader("📝 Enter Development Indicators")

st.caption(
    "Enter values for all 22 features used during model training."
)


inputs = {}


# Split features into two columns

midpoint = len(feature_names) // 2

left_features = feature_names[:midpoint]
right_features = feature_names[midpoint:]


col1, col2 = st.columns(2)


# ------------------------------------------------------------
# LEFT COLUMN
# ------------------------------------------------------------

with col1:

    for feature in left_features:

        inputs[feature] = st.number_input(
            feature,
            value=0.0,
            step=0.01,
            format="%.4f"
        )


# ------------------------------------------------------------
# RIGHT COLUMN
# ------------------------------------------------------------

with col2:

    for feature in right_features:

        inputs[feature] = st.number_input(
            feature,
            value=0.0,
            step=0.01,
            format="%.4f"
        )


st.markdown("")


# ============================================================
# PREDICT BUTTON
# ============================================================

predict = st.button(
    "🔍 Predict Cluster",
    use_container_width=True,
    type="primary"
)


# ============================================================
# PREDICTION
# ============================================================

if predict:

    # Create DataFrame
    input_df = pd.DataFrame(
        [inputs],
        columns=feature_names
    )

    # --------------------------------------------------------
    # STANDARD SCALING
    # --------------------------------------------------------

    scaled_data = scaler.transform(input_df)

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    pca_data = pca.transform(scaled_data)

    # --------------------------------------------------------
    # K-MEANS
    # --------------------------------------------------------

    predicted_cluster = int(
        kmeans.predict(pca_data)[0]
    )


    st.divider()

    # ========================================================
    # PREDICTION RESULT
    # ========================================================

    st.subheader("🎯 Prediction Result")

    result_col1, result_col2 = st.columns([1, 2])


    with result_col1:

        st.success(
            f"Cluster {predicted_cluster}"
        )

        st.metric(
            "Predicted Cluster",
            f"Cluster {predicted_cluster}"
        )


    with result_col2:

        st.markdown(
            f"""
            ### {cluster_names[predicted_cluster]}

            {cluster_descriptions[predicted_cluster]}
            """
        )


    # ========================================================
    # CLUSTER CARDS
    # ========================================================

    st.divider()

    st.subheader("📌 Cluster Descriptions")

    c0, c1, c2, c3 = st.columns(4)


    with c0:

        st.markdown(
            f"""
            ### 🔵 Cluster 0

            **{cluster_names[0]}**

            {cluster_descriptions[0]}
            """
        )


    with c1:

        st.markdown(
            f"""
            ### 🟢 Cluster 1

            **{cluster_names[1]}**

            {cluster_descriptions[1]}
            """
        )


    with c2:

        st.markdown(
            f"""
            ### 🟠 Cluster 2

            **{cluster_names[2]}**

            {cluster_descriptions[2]}
            """
        )


    with c3:

        st.markdown(
            f"""
            ### 🟣 Cluster 3

            **{cluster_names[3]}**

            {cluster_descriptions[3]}
            """
        )


    # ========================================================
    # PCA INFORMATION
    # ========================================================

    st.divider()

    st.subheader("🔬 PCA Information")

    pca_col1, pca_col2, pca_col3 = st.columns(3)


    with pca_col1:

        st.metric(
            "Original Features",
            len(feature_names)
        )


    with pca_col2:

        st.metric(
            "PCA Components",
            pca.n_components_
        )


    with pca_col3:

        variance = pca.explained_variance_ratio_.sum()

        st.metric(
            "Explained Variance",
            f"{variance * 100:.2f}%"
        )


    # ========================================================
    # INPUT VALUES
    # ========================================================

    st.divider()

    st.subheader("📋 Input Values")

    st.dataframe(
        input_df.T.rename(
            columns={0: "Value"}
        ),
        use_container_width=True
    )


    # ========================================================
    # PCA VALUES
    # ========================================================

    st.subheader("📈 PCA Transformed Values")

    pca_columns = [
        f"PC{i+1}"
        for i in range(
            pca_data.shape[1]
        )
    ]

    pca_df = pd.DataFrame(
        pca_data,
        columns=pca_columns
    )

    st.dataframe(
        pca_df,
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Global Development Clustering | "
    "StandardScaler → PCA → K-Means"
)