import mlflow
import mlflow.sklearn
import joblib
import os
import pandas as pd
from mlflow.tracking import MlflowClient
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

class SentimentPredictor:
    def __init__(self, model_name="FlipkartSentimentClassifier", stage="Production", use_local_artifacts=False):
        self.model_name = model_name
        self.stage = stage
        self.use_local_artifacts = use_local_artifacts
        self.tracking_uri = "file:///" + os.path.abspath("mlruns").replace("\\", "/")
        
        # Only setup MLflow if not using local artifacts
        if not self.use_local_artifacts:
            mlflow.set_tracking_uri(self.tracking_uri)
            self.client = MlflowClient()
            
        self.model = None
        self.vectorizer = None
        self._load_resources()
        
    def _load_resources(self):
        print(f"Loading resources...")
        
        # METHOD 1: Local Artifacts (For Deployment / Streamlit Cloud)
        if self.use_local_artifacts:
            print("Loading from local 'artifacts/' directory...")
            try:
                vec_path = "artifacts/vectorizer.joblib"
                model_path = "artifacts/model.joblib"
                
                if not os.path.exists(vec_path) or not os.path.exists(model_path):
                    raise FileNotFoundError("Artifacts not found in 'artifacts/' folder.")
                
                self.vectorizer = joblib.load(vec_path)
                self.model = joblib.load(model_path)
                print("Local resources loaded successfully.")
                return
            except Exception as e:
                print(f"Failed to load local artifacts: {e}")
                raise e

        # METHOD 2: MLflow Registry (For Development / MLOps)
        print(f"Loading model '{self.model_name}' (Stage: {self.stage}) from MLflow Registry...")
        try:
            # 1. Get run_id from Registry
            versions = self.client.get_latest_versions(self.model_name, stages=[self.stage])
            if not versions:
                # Fallback to None stage (latest) if Production not set yet
                versions = self.client.get_latest_versions(self.model_name, stages=["None"])
                if not versions:
                    raise Exception("No model found in registry.")
                print(f"Warning: Model not found in '{self.stage}'. Using latest version from 'None' stage.")
                
            latest_version = versions[0]
            run_id = latest_version.run_id
            print(f"Found Run ID: {run_id}")
            
            # 2. Load Vectorizer (download artifact)
            local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="vectorizer.joblib")
            self.vectorizer = joblib.load(local_path)
            
            # 3. Load Model
            model_uri = f"models:/{self.model_name}/{latest_version.current_stage}"
            if latest_version.current_stage == "None":
                 model_uri = f"models:/{self.model_name}/{latest_version.version}"

            self.model = mlflow.sklearn.load_model(model_uri)
            print("Resources loaded successfully from MLflow.")
            
        except Exception as e:
            print(f"Error loading resources: {e}")
            raise e

    def preprocess(self, text):
        # Copy of preprocess logic or import it
        # For simplicity, duplicating basic cleaning or better: import from predict.py? 
        # But predict.py is this file.
        # import from src.preprocess is risky if run from root.
        # I'll implement a simple clean function here to be self-contained for inference.
        
        text = str(text).lower()
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text

    def predict(self, text):
        clean_text = self.preprocess(text)
        vec_text = self.vectorizer.transform([clean_text])
        pred = self.model.predict(vec_text)[0]
        prob = self.model.predict_proba(vec_text)[0] if hasattr(self.model, "predict_proba") else [0, 0]
        
        # Prob of Positive class (1)
        # Note: classes_ might need inspection, but usually 0 at index 0, 1 at index 1 for binary
        pos_prob = prob[1] if len(prob) > 1 else 0
        
        return {
            "prediction": "Positive" if pred == 1 else "Negative",
            "sentiment_code": int(pred),
            "probability": pos_prob,
            "raw_text": text
        }

if __name__ == "__main__":
    predictor = SentimentPredictor()
    while True:
        user_input = input("\nEnter review text (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        result = predictor.predict(user_input)
        print(f"Prediction: {result['prediction']} (Prob: {result['probability']:.4f})")
