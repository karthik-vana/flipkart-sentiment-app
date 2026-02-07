# 🛍️ Flipkart Sentiment Analysis MLOps Project

A production-grade End-to-End MLOps project for Sentiment Analysis on Flipkart Product Reviews. This project demonstrates industry-standard practices including Experiment Tracking, Model Registry, Automated Pipelines, and Deployment.

## Click here for live Demo[Live Demo](https://flipkart-sentiment-app-100.streamlit.app/) 

## 🚀 Project Overview

This project builds a machine learning system to classify product reviews as **Positive** or **Negative**.

**Key Features:**
- **Data Pipeline:** Automated cleaning and preprocessing of raw text data.
- **Experiment Tracking:** Uses **MLflow** to log parameters, metrics, and artifacts (confusion matrices, models).
- **Model Registry:** Best model versioning and stage management (Staging -> Production).
- **Orchestration:** **Prefect** workflow for automated training and evaluation.
- **Comparison:** Evaluates models (Logistic Regression, Naive Bayes, SVM) + Hyperparameter Tuning.
- **Deployment:** Interactive **Streamlit** web application for real-time inference.
- **Code Quality:** Modular, clean, and beginner-friendly codebase.

---

## 📂 Project Structure

```
Flipkart-Sentiment-MLflow/
│── data/                   # Raw and Cleaned Data
│── notebooks/              # Jupyter Notebooks for experiments
│── src/                    # Source Code
│   │── preprocess.py       # Data cleaning pipeline
│   │── train.py            # Training, Evaluation, and MLflow logging
│   │── evaluate.py         # Evaluation metrics and plotting
│   │── predict.py          # Inference class loading model from Registry
│── workflows/              # Workflow Orchestration
│   │── prefect_flow.py     # Prefect Flow definition
│── artifacts/              # Local path for saved artifacts (plots, etc.)
│── mlruns/                 # MLflow tracking store
│── app.py                  # Streamlit Web App
│── requirements.txt        # Dependencies
│── README.md               # Project Documentation
```

---

## 🛠️ Installation & Setup

1. **Clone the Repository** (if applicable)
   ```bash
   git clone <repo_url>
   cd Flipkart-Sentiment-MLflow
   ```

2. **Create Virtual Environment** (Optional but Recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize Project & Data**
   ```bash
   python setup_project.py
   ```

---

## 📊 How to Run the Experiments

### 1. Run the Training Pipeline (Manual)
To preprocess data, train models, tune hyperparameters, and register the best model:
```bash
python src/train.py
```
*Check the terminal output for F1-scores and run IDs.*

### 2. View MLflow UI
To visualize experiments, compare runs, and see the Model Registry:
```bash
mlflow ui
```
Open [http://localhost:5000](http://localhost:5000) in your browser.

**What to look for in MLflow:**
- **Experiments**: Check "Flipkart Sentiment Analysis".
- **Compare**: Select multiple runs -> "Compare" to see charts.
- **Artifacts**: View Confusion Matrix and vectorizer files.
- **Models**: Click "Models" tab to see "FlipkartSentimentClassifier".

---

## 🌊 Automated Workflow with Prefect

This project uses Prefect to orchestrate the pipeline.

### Run the Flow Locally
```bash
python workflows/prefect_flow.py
```

### Setup Prefect Server & Deployment (Bonus)
1. **Start Prefect Server** (in a simplified setup, just running the flow script works, but for full UI):
   ```bash
   prefect server start
   ```
2. **Build Deployment** (in another terminal):
   ```bash
   prefect deployment build workflows/prefect_flow.py:sentiment_analysis_pipeline -n "Daily Sentiment Run" -q test
   prefect deployment apply sentiment_analysis_pipeline-deployment.yaml
   ```
3. **Start Agent**:
   ```bash
   prefect agent start -q test
   ```
View the Dashboard at [http://127.0.0.1:4200](http://127.0.0.1:4200).

---

## 🌐 Run the Web App
 
 To start the Streamlit interface for inference:
 ```bash
 streamlit run app.py
 ```
 Open the provided URL (usually [http://localhost:8501](http://localhost:8501)).
 *Note: The app loads the model marked as "Production" or the latest version from MLflow.*
 
 ---
 
 ## ☁️ Deploy on Streamlit Cloud
 
 To deploy this app for free on Streamlit Cloud:
 
 1. **Push to GitHub**:
    - Initialize Git: `git init`
    - Add files: `git add .`
    - Commit: `git commit -m "Initial commit"`
    - Push to your GitHub repository.
 
 2. **Deploy**:
    - Go to [Streamlit Cloud](https://streamlit.io/cloud).
    - Connect your GitHub account.
    - Select your repository (`Flipkart-Sentiment-MLflow`).
    - Set the **Main file path** to `app.py`.
    - Click **Deploy!** 🚀
 
 *Note: This project is configured to read artifacts locally (`artifacts/`) for cloud deployment to avoid complex MLflow setup in the cloud demo.*
 
 ---

## ☁️ Deployment Guide (AWS EC2)

1. **Launch EC2 Instance**: Use Ubuntu/Amazon Linux (t2.medium recommended).
2. **Setup Environment**:
   ```bash
   sudo apt update
   sudo apt install python3-pip git
   git clone <your-repo>
   pip install -r requirements.txt
   ```
3. **Run Training**: Execute `python src/train.py`.
4. **Run Streamlit in Background**:
   ```bash
   nohup streamlit run app.py --server.port 8501 &
   ```
5. **Security Groups**: Open port 8501 in AWS Security Groups.
6. **Access**: Go to `http://<EC2_PUBLIC_IP>:8501`.

---

## 📝 Authors

Built by **karthik Vana**.
```
