# 🌍 Global Development Clustering Using PCA and K-Means

## 📌 Project Overview

Global Development Clustering is a machine learning project that groups observations based on development indicators using **Principal Component Analysis (PCA)** and **K-Means Clustering**.

The project uses an interactive **Streamlit dashboard** that allows users to enter development indicator values, predict cluster assignments, explore cluster characteristics, and analyze the dataset.

## 🎯 Objectives

- Analyze global development indicators using machine learning.
- Reduce feature dimensionality using PCA.
- Group similar observations using K-Means clustering.
- Predict cluster assignments for user-provided input.
- Visualize cluster distributions and explore cluster characteristics through an interactive dashboard.

## 🛠️ Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Joblib
- Matplotlib
- Excel
- Jupyter Notebook

## 🤖 Machine Learning Workflow

The project follows this pipeline:

**Input Features → StandardScaler → PCA → K-Means Clustering**

1. **Data Preparation:** Load and prepare the development indicator dataset.
2. **Feature Scaling:** Standardize the input features using `StandardScaler`.
3. **Dimensionality Reduction:** Apply PCA to reduce the original 22 features to 13 principal components.
4. **Clustering:** Use K-Means to group observations into four clusters.
5. **Prediction:** Transform user inputs through the saved preprocessing pipeline and predict a cluster.
6. **Visualization:** Display cluster information and analysis results through Streamlit.

The clusters represent patterns in the dataset and should not automatically be interpreted as official developed or developing country classifications.

## 📊 Model Information

| Component | Configuration |
|---|---|
| Original input features | 22 |
| Feature scaling | StandardScaler |
| Dimensionality reduction | PCA |
| PCA components | 13 |
| Clustering algorithm | K-Means |
| Number of clusters | 4 |

## 📁 Project Structure

```text
Global_Development_Project/
│
├── app.py
├── Global_Cluster_Development.ipynb
├── feature_names.pkl
├── kmeans_model.pkl
├── pca.pkl
├── scaler.pkl
├── P706_ World_development_mesurement.xlsx
├── requirements.txt
├── README.md
└── .gitignore
```

## 💻 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Thehacker4413/global-development-clustering.git
```

### 2. Navigate to the project directory

```bash
cd global-development-clustering
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt instead:

```cmd
venv\Scripts\activate.bat
```

### 5. Install the required packages

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser. If it does not open automatically, use the local URL displayed in your terminal, usually `http://localhost:8501`.

## 🚀 Dashboard Features

- **Home:** View the project overview and model summary.
- **Model Information:** Explore the machine learning pipeline and cluster descriptions.
- **Prediction:** Enter the 22 development indicators to predict a cluster.
- **Cluster Analysis:** Analyze the Excel dataset, view cluster distributions, and download cluster assignments.

## 📦 Model Artifacts

The application uses saved model artifacts:

- `scaler.pkl` — fitted feature scaler.
- `pca.pkl` — fitted PCA transformation.
- `kmeans_model.pkl` — trained K-Means clustering model.
- `feature_names.pkl` — original feature names and ordering.

These files must correspond to the same training pipeline used by the application.

## 📈 Results and Interpretation

The model assigns observations to one of four clusters based on similarities in the selected development indicators.

Cluster assignments can help explore patterns in population, economic activity, healthcare, resource usage, and digital connectivity. The clusters are exploratory groups, not verified country rankings or official development classifications.

## 🌐 Deployment

The Streamlit application can be deployed using Streamlit Community Cloud.

1. Push the project to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Sign in with GitHub.
4. Select the `global-development-clustering` repository.
5. Set the main file path to `app.py`.
6. Deploy the application.

Ensure the required model files and dataset are available to the deployed application.

## 🔮 Future Enhancements

- Add interactive visualizations for cluster characteristics.
- Improve dataset validation and preprocessing.
- Add downloadable cluster summaries and reports.
- Compare alternative clustering algorithms.
- Deploy the application for public access.

## 👨‍💻 Author

**Thehacker4413**

GitHub: [https://github.com/Thehacker4413](https://github.com/Thehacker4413)

## 📄 License

Add a license file if you intend to distribute this project publicly. Until a license is added, do not assume others have permission to reuse or redistribute the code or dataset.
