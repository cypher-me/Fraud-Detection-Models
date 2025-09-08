import numpy as np
import pandas as pd
import pickle
import json
import time
import os
from datetime import datetime
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

class RealTimeFraudDetector:
    def __init__(self):
        self.load_models()
        self.setup_logging()
        
    def load_models(self):
        """Load all trained models and preprocessors"""
        # Load preprocessor
        with open('models/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        with open('models/feature_names.pkl', 'rb') as f:
            self.feature_names = pickle.load(f)
            
        # Load ensemble model
        try:
            with open('models/ensemble_model.pkl', 'rb') as f:
                self.ensemble_model = pickle.load(f)
        except:
            print("Ensemble model not found, using Isolation Forest only")
            self.ensemble_model = None
            
        # Initialize Isolation Forest
        X_train = np.load('models/X_train.npy')
        self.iso_forest = IsolationForest(contamination=0.1, random_state=42)
        self.iso_forest.fit(X_train)
        
        print("Models loaded successfully!")
    
    def setup_logging(self):
        """Setup logging directories"""
        os.makedirs('logs', exist_ok=True)
        self.alert_log = 'logs/real_time_alerts.jsonl'
        self.stats_log = 'logs/real_time_stats.jsonl'
        
    def preprocess_transaction(self, transaction):
        """Preprocess single transaction"""
        # Convert to DataFrame
        df = pd.DataFrame([transaction])
        
        # Feature engineering
        df['Amount_scaled'] = self.scaler.transform(df[['Amount']])
        df['Time_scaled'] = self.scaler.transform(df[['Time']])
        df['Hour'] = (df['Time'] / 3600) % 24
        df['Amount_log'] = np.log1p(df['Amount'])
        
        return df[self.feature_names].values[0]
    
    def predict_fraud(self, transaction):
        """Predict fraud for single transaction"""
        # Preprocess
        features = self.preprocess_transaction(transaction)
        
        # Isolation Forest prediction
        iso_score = self.iso_forest.decision_function([features])[0]
        iso_pred = (self.iso_forest.predict([features])[0] == -1).astype(int)
        
        # If ensemble model available, use it
        if self.ensemble_model is not None:
            # Create ensemble features (simplified)
            ensemble_features = np.array([
                0.1,  # GNN placeholder
                abs(iso_score),  # Autoencoder-like score
                abs(iso_score),  # Isolation Forest score
                0,    # GNN pred placeholder
                iso_pred,  # Autoencoder pred placeholder
                iso_pred   # Isolation Forest pred
            ]).reshape(1, -1)
            
            ensemble_prob = self.ensemble_model.predict_proba(ensemble_features)[0, 1]
            final_prediction = int(ensemble_prob > 0.5)
            confidence = ensemble_prob
        else:
            final_prediction = iso_pred
            confidence = abs(iso_score)
            
        return {
            'fraud_prediction': final_prediction,
            'confidence': float(confidence),
            'iso_score': float(iso_score),
            'timestamp': datetime.now().isoformat()
        }
    
    def log_alert(self, transaction, prediction):
        """Log fraud alert"""
        alert = {
            'timestamp': prediction['timestamp'],
            'transaction_id': transaction.get('id', 'unknown'),
            'amount': transaction['Amount'],
            'prediction': prediction['fraud_prediction'],
            'confidence': prediction['confidence'],
            'iso_score': prediction['iso_score']
        }
        
        with open(self.alert_log, 'a') as f:
            f.write(json.dumps(alert) + '\n')
            
        print(f"🚨 FRAUD ALERT: Transaction {alert['transaction_id']} - Amount: ${alert['amount']:.2f} - Confidence: {alert['confidence']:.3f}")
    
    def log_stats(self, batch_stats):
        """Log batch statistics"""
        with open(self.stats_log, 'a') as f:
            f.write(json.dumps(batch_stats) + '\n')
    
    def process_stream(self, data_source='data/creditcard.csv', batch_size=100, max_batches=50):
        """Process streaming transactions"""
        print("🚀 Starting real-time fraud detection...")
        
        # Load data
        df = pd.read_csv(data_source)
        total_processed = 0
        total_fraud_detected = 0
        
        for batch_id in range(max_batches):
            start_idx = batch_id * batch_size
            end_idx = min(start_idx + batch_size, len(df))
            
            if start_idx >= len(df):
                break
                
            batch_df = df.iloc[start_idx:end_idx]
            batch_fraud_count = 0
            
            # Process each transaction
            for _, transaction in batch_df.iterrows():
                prediction = self.predict_fraud(transaction.to_dict())
                
                if prediction['fraud_prediction'] == 1:
                    self.log_alert(transaction.to_dict(), prediction)
                    batch_fraud_count += 1
                    total_fraud_detected += 1
            
            # Log batch statistics
            batch_stats = {
                'batch_id': batch_id,
                'timestamp': datetime.now().isoformat(),
                'transactions_processed': len(batch_df),
                'fraud_detected': batch_fraud_count,
                'fraud_rate': batch_fraud_count / len(batch_df),
                'total_processed': total_processed + len(batch_df),
                'total_fraud_detected': total_fraud_detected
            }
            
            self.log_stats(batch_stats)
            total_processed += len(batch_df)
            
            print(f"Batch {batch_id}: {len(batch_df)} transactions, {batch_fraud_count} fraud detected")
            
            # Simulate real-time delay
            time.sleep(2)
        
        print(f"✅ Processing complete! Total: {total_processed} transactions, {total_fraud_detected} fraud cases detected")

def main():
    """Main application entry point"""
    detector = RealTimeFraudDetector()
    
    try:
        detector.process_stream()
    except KeyboardInterrupt:
        print("\n⏹️  Stopping fraud detection system...")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()