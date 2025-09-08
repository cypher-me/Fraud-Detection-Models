from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
import pickle
import time
import json
import os

class FraudDetectionStreaming:
    def __init__(self):
        self.spark = SparkSession.builder.appName("FraudDetectionStreaming").getOrCreate()
        self.load_models()
        
    def load_models(self):
        """Load preprocessor and train Isolation Forest"""
        with open('../models/feature_names.pkl', 'rb') as f:
            self.feature_names = pickle.load(f)
        with open('../models/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
            
        # Train Isolation Forest
        X_train = np.load('../models/X_train.npy')
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.isolation_forest.fit(X_train)
    
    def preprocess_batch(self, df):
        """Preprocess streaming batch"""
        pdf = df.toPandas()
        pdf['Amount_scaled'] = self.scaler.transform(pdf[['Amount']])
        pdf['Time_scaled'] = self.scaler.transform(pdf[['Time']])
        pdf['Hour'] = (pdf['Time'] / 3600) % 24
        pdf['Amount_log'] = np.log1p(pdf['Amount'])
        return pdf[self.feature_names].values, pdf
    
    def detect_anomalies(self, X):
        """Detect anomalies using Isolation Forest"""
        anomaly_scores = self.isolation_forest.decision_function(X)
        predictions = self.isolation_forest.predict(X)
        fraud_predictions = np.where(predictions == -1, 1, 0)
        return fraud_predictions, anomaly_scores
    
    def process_batch(self, batch_df, batch_id):
        """Process streaming batch"""
        if batch_df.count() == 0:
            return
            
        X, pdf = self.preprocess_batch(batch_df)
        fraud_predictions, anomaly_scores = self.detect_anomalies(X)
        
        pdf['fraud_prediction'] = fraud_predictions
        pdf['anomaly_score'] = anomaly_scores
        
        fraud_cases = pdf[pdf['fraud_prediction'] == 1]
        if len(fraud_cases) > 0:
            self.log_alerts(fraud_cases, batch_id)
            print(f"ALERT: {len(fraud_cases)} fraud cases detected in batch {batch_id}")
    
    def log_alerts(self, fraud_cases, batch_id):
        """Log fraud alerts"""
        os.makedirs('../logs', exist_ok=True)
        alert_data = {
            'batch_id': batch_id,
            'timestamp': time.time(),
            'fraud_count': len(fraud_cases),
            'cases': fraud_cases[['Time', 'Amount', 'anomaly_score']].to_dict('records')
        }
        with open(f'../logs/fraud_alerts_batch_{batch_id}.json', 'w') as f:
            json.dump(alert_data, f)
    
    def simulate_streaming(self, batch_size=1000, max_batches=10):
        """Simulate streaming processing"""
        df = pd.read_csv('../data/creditcard.csv')
        spark_df = self.spark.createDataFrame(df)
        
        print("Starting fraud detection streaming...")
        for batch_id in range(max_batches):
            start_idx = batch_id * batch_size
            batch_df = spark_df.limit(batch_size).offset(start_idx)
            self.process_batch(batch_df, batch_id)
            time.sleep(1)
        print("Streaming simulation completed!")
    
    def stop(self):
        self.spark.stop()

if __name__ == "__main__":
    detector = FraudDetectionStreaming()
    try:
        detector.simulate_streaming()
    finally:
        detector.stop()