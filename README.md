# Real-Time Fraud Detection in Financial Transactions

A comprehensive fraud detection system using multiple ML approaches including Spark Streaming, PyTorch GNN, and TensorFlow autoencoders.

## Project Structure
```
fraud-detection/
├── data/                    # Dataset storage
├── src/                     # Source code modules
├── notebooks/               # Jupyter notebooks
├── models/                  # Trained models
├── reports/                 # Generated plots and reports
├── logs/                    # Application logs
└── requirements.txt         # Dependencies
```

## Setup
```bash
pip install -r requirements.txt
```

## Usage

### Step-by-step execution:
1. **Data Preprocessing**: `01_data_preprocessing.ipynb`
2. **Exploratory Analysis**: `02_eda.ipynb`
3. **GNN Training**: `03_gnn_training.ipynb`
4. **Autoencoder Training**: `04_tensorflow_anomaly.ipynb`
5. **Ensemble Modeling**: `05_model_ensemble.ipynb`
6. **Monitoring Dashboard**: `06_monitoring.ipynb`

### Real-time deployment:
```bash
python fraud_detection_app.py
```

### Spark streaming simulation:
```bash
python src/spark_streaming.py
```

## Features
- 🔥 Real-time Spark Streaming with Isolation Forest
- 🧠 PyTorch GNN for transaction relationship modeling
- 🤖 TensorFlow Autoencoder for anomaly detection
- 🎯 Ensemble model with Optuna hyperparameter tuning
- 📊 Interactive monitoring dashboard
- 📝 Comprehensive logging and alerting