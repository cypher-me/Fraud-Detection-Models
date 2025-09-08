import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

class FraudGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.2):
        super(FraudGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x, edge_index, batch=None):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        h = F.relu(self.conv3(h, edge_index))
        
        if batch is None:
            h = torch.mean(h, dim=0, keepdim=True)
        else:
            h = global_mean_pool(h, batch)
            
        return torch.sigmoid(self.classifier(h))

class GraphDataProcessor:
    def create_transaction_graph(self, df, feature_cols):
        """Create graph from transaction data"""
        node_features = torch.FloatTensor(df[feature_cols].values)
        edges = self._create_edges(df)
        edge_index = torch.LongTensor(edges).t().contiguous()
        labels = torch.FloatTensor(df['Class'].values)
        return Data(x=node_features, edge_index=edge_index, y=labels)
    
    def _create_edges(self, df):
        """Create edges based on transaction similarity"""
        edges = []
        n_nodes = min(len(df), 2000)  # Limit for efficiency
        
        for i in range(n_nodes):
            for j in range(i+1, min(i+20, n_nodes)):
                time_diff = abs(df.iloc[i]['Time'] - df.iloc[j]['Time'])
                amount_diff = abs(df.iloc[i]['Amount'] - df.iloc[j]['Amount'])
                if time_diff < 3600 and amount_diff < 100:  # Similar transactions
                    edges.extend([[i, j], [j, i]])
            edges.append([i, i])  # Self-loop
        return edges

class GNNTrainer:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()
        
    def train_epoch(self, data_loader):
        self.model.train()
        total_loss = 0
        for batch in data_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(batch.x, batch.edge_index, batch.batch)
            loss = self.criterion(out.squeeze(), batch.y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(data_loader)
    
    def evaluate(self, data_loader):
        self.model.eval()
        predictions, labels = [], []
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.edge_index, batch.batch)
                predictions.extend(out.squeeze().cpu().numpy())
                labels.extend(batch.y.cpu().numpy())
        return roc_auc_score(labels, predictions), np.array(predictions), np.array(labels)
    
    def train(self, train_loader, val_loader, epochs=30):
        best_auc = 0
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_auc, _, _ = self.evaluate(val_loader)
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(self.model.state_dict(), '../models/gnn_best_model.pth')
            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Loss: {train_loss:.4f}, Val AUC: {val_auc:.4f}')
        return best_auc