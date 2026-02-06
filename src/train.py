import pandas as pd
import numpy as np
import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from evaluate import get_metrics, save_confusion_matrix, save_classification_report

# Setup Check directories
os.makedirs("artifacts", exist_ok=True)
os.makedirs("mlruns", exist_ok=True)

# 1. Setup MLflow
# Using absolute path for tracking URI to avoid Windows issues
TRACKING_URI = "file:///" + os.path.abspath("mlruns").replace("\\", "/")
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("Flipkart Sentiment Analysis")

def load_data(path):
    df = pd.read_csv(path)
    # Ensure no NaN in text
    df['cleaned_text'] = df['cleaned_text'].fillna("")
    return df['cleaned_text'], df['sentiment']

def train_and_log(run_name, model, vectorizer, X_train, X_test, y_train, y_test, params):
    """
    Train a model, evaluate it, and log everything to MLflow.
    """
    with mlflow.start_run(run_name=run_name) as run:
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        metrics = get_metrics(y_test, y_pred)
        
        # Log Parameters
        mlflow.log_params(params)
        mlflow.log_param("vectorizer_max_features", vectorizer.max_features)
        mlflow.log_param("vectorizer_ngram_range", str(vectorizer.ngram_range))
        
        # Log Metrics
        mlflow.log_metrics(metrics)
        print(f"Run: {run_name} | F1-Score: {metrics['f1_score']:.4f}")
        
        # Save and Log Artifacts
        # 1. Confusion Matrix
        cm_path = "artifacts/confusion_matrix.png"
        save_confusion_matrix(y_test, y_pred, cm_path)
        mlflow.log_artifact(cm_path)
        
        # 2. Classification Report
        cr_path = "artifacts/classification_report.txt"
        save_classification_report(y_test, y_pred, cr_path)
        mlflow.log_artifact(cr_path)
        
        # 3. Vectorizer
        vec_path = "artifacts/vectorizer.joblib"
        joblib.dump(vectorizer, vec_path)
        mlflow.log_artifact(vec_path)
        
        # 4. Model
        model_path = "artifacts/model.joblib"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
        
        # Log Model to MLflow (Standard)
        mlflow.sklearn.log_model(model, "model")
        
        return run.info.run_id, metrics['f1_score']

def main():
    print("Loading data...")
    X, y = load_data("data/cleaned_data.csv")
    
    # Split Data (Stratified)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # --- PHASE 1: Baseline Models (Fixed TF-IDF) ---
    print("\n--- PHASE 1: Baseline Training ---")
    
    # Baseline Vectorizer
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 1))
    X_train_vec = vectorizer.fit_transform(X_train_raw)
    X_test_vec = vectorizer.transform(X_test_raw)
    
    models = [
        ("LogReg", LogisticRegression(max_iter=1000), {"C": 1.0}),
        ("NaiveBayes", MultinomialNB(), {"alpha": 1.0}),
        ("SVM", SGDClassifier(loss='hinge'), {"alpha": 0.0001})
    ]
    
    best_f1 = 0
    best_run_id = None
    best_art_path = ""
    
    for name, model, params in models:
        run_name = f"{name}_TFIDF_5k_1gram"
        run_id, f1 = train_and_log(run_name, model, vectorizer, X_train_vec, X_test_vec, y_train, y_test, params)
        
        if f1 > best_f1:
            best_f1 = f1
            best_run_id = run_id
            
    # --- PHASE 2: Hyperparameter Tuning ---
    print("\n--- PHASE 2: Hyperparameter Tuning ---")
    # We will manually loop to log each run as a separate MLflow run as requested
    
    tfidf_params = [
        # (max_features, ngram_range) - reduced subset for speed
        (5000, (1,1)), 
        (10000, (1,2))
    ]
    
    logreg_cs = [0.1, 1, 10]
    
    for max_feat, ngram in tfidf_params:
        print(f"Vectorizing with max_features={max_feat}, ngram={ngram}...")
        vec = TfidfVectorizer(max_features=max_feat, ngram_range=ngram)
        X_tr = vec.fit_transform(X_train_raw)
        X_te = vec.transform(X_test_raw)
        
        for c in logreg_cs:
            model = LogisticRegression(C=c, max_iter=1000)
            run_name = f"Tuning_LogReg_TFIDF_{max_feat}_{ngram}_C{c}"
            params = {"C": c, "max_features": max_feat, "ngram_range": str(ngram)}
            
            run_id, f1 = train_and_log(run_name, model, vec, X_tr, X_te, y_train, y_test, params)
            
            if f1 > best_f1:
                best_f1 = f1
                best_run_id = run_id
                
    print(f"\nBest Run ID: {best_run_id} with F1-Score: {best_f1}")
    
    # --- PHASE 3: Register Best Model ---
    print("\n--- PHASE 3: Registering Best Model ---")
    if best_run_id:
        model_uri = f"runs:/{best_run_id}/model"
        mv = mlflow.register_model(model_uri, "FlipkartSentimentClassifier")
        print(f"Registered model: {mv.name}, version: {mv.version}")
        
        # Transition to Production (Optional demo)
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name="FlipkartSentimentClassifier",
            version=mv.version,
            stage="Production"
        )
        
        # Update description/tags
        client.update_model_version(
            name="FlipkartSentimentClassifier",
            version=mv.version,
            description="Best model selected based on F1 Score."
        )
        client.set_model_version_tag(
            name="FlipkartSentimentClassifier",
            version=mv.version,
            key="approved_by",
            value="Karthik"
        )
        print("Model transitioned to Production and tagged.")

if __name__ == "__main__":
    main()
