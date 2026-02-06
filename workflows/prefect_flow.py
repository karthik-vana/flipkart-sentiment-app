from prefect import flow, task
import subprocess
import os
import sys

# Ensure src is in path to import modules if needed, 
# although we wil primarily use subprocess to run the standalone scripts 
# to ensure isolation and environment consistency.

@task(name="Load and Explore Data")
def load_data_task():
    """
    Simulates loading data and initial checks.
    In a real scenario, this might return a dataframe, 
    but our preprocess script handles loading from raw zip/csv.
    """
    print("Loading data configuration...")
    if not os.path.exists("data"):
        raise Exception("Data directory missing!")
    print("Data directory confirmed.")

@task(name="Data Cleaning & Preprocessing")
def clean_data_task():
    """
    Runs the preprocessing script to clean text and generate target columns.
    """
    print("Running src/preprocess.py...")
    # using sys.executable to ensure we use the same python interpreter
    subprocess.run([sys.executable, "src/preprocess.py"], check=True)
    print("Preprocessing complete. Cleaned data saved.")

@task(name="Feature Engineering & Model Training")
def train_model_task():
    """
    Runs the main training pipeline which includes:
    - Feature Engineering (TF-IDF)
    - Model Training (Baseline + Tuning)
    - Evaluation
    - MLflow Logging
    """
    print("Running src/train.py...")
    subprocess.run([sys.executable, "src/train.py"], check=True)
    print("Training and Experimentation complete.")

@task(name="Model Registration")
def register_model_task():
    """
    This step is handled inside train.py in our current architecture 
    assuming the best model is identified and registered there.
    We just verify the registration took place or perform promotion here.
    """
    print("Verifying model artifacts...")
    if not os.path.exists("mlruns"):
        print("Warning: mlruns directory not found.")
    else:
        print("MLflow runs generated successfully.")

@flow(name="Flipkart Sentiment Analysis Workflow", log_prints=True)
def sentiment_analysis_pipeline():
    """
    End-to-End MLOps Pipeline for Flipkart Sentiment Analysis.
    """
    print("Starting Pipeline...")
    
    # 1. Load Data
    load_data_task()
    
    # 2. Clean Data
    clean_data_task()
    
    # 3. Train, Evaluate, Feature Engineer, Register
    # (Bundled in train.py for cohesive MLflow experiment tracking)
    train_model_task()
    
    # 4. Confirmation
    register_model_task()
    
    print("Pipeline Finished Successfully! Data flow complete.")

if __name__ == "__main__":
    sentiment_analysis_pipeline()
