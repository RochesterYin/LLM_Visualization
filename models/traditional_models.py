#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import joblib

class TraditionalModels:
    def __init__(self, data_path='../processed/features.csv'):
        """Initialize with path to processed data"""
        self.data_path = data_path
        self.models = {
            'decision_tree': DecisionTreeClassifier(random_state=42),
            'random_forest': RandomForestClassifier(random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42)
        }
        self.results = {}
        
    def load_data(self):
        """Load and prepare data"""
        print(f"Loading data from {self.data_path}")
        
        # Check if processed data exists
        if not os.path.exists(self.data_path):
            print(f"Data file {self.data_path} not found. Run preprocessing first.")
            return False
            
        # Load features
        df = pd.read_csv(self.data_path)
        
        # Separate features and labels
        y = df['label']
        X = df.drop(['label', 'file'], axis=1)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Save feature names for interpretability
        self.feature_names = X.columns
        
        # Save scaler for later use
        os.makedirs('../models/saved', exist_ok=True)
        joblib.dump(scaler, '../models/saved/scaler.pkl')
        
        print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        return True
    
    def train_models(self):
        """Train all traditional models"""
        if not hasattr(self, 'X_train'):
            if not self.load_data():
                return
        
        print("Training traditional ML models...")
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            self.results[name] = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1': f1_score(self.y_test, y_pred),
                'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist()
            }
            
            # Calculate ROC curve
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(self.X_test)[:, 1]
                fpr, tpr, _ = roc_curve(self.y_test, y_prob)
                self.results[name]['roc'] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'auc': auc(fpr, tpr)
                }
            
            # Save trained model
            os.makedirs('../models/saved', exist_ok=True)
            model_path = f'../models/saved/{name}.pkl'
            joblib.dump(model, model_path)
            print(f"Model saved to {model_path}")
        
        # Save results
        results_path = '../models/saved/traditional_results.pkl'
        with open(results_path, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"Results saved to {results_path}")
        
        return self.results
    
    def print_results(self):
        """Print evaluation results"""
        if not self.results:
            print("No results available. Train models first.")
            return
            
        print("\n===== Traditional ML Models Results =====")
        for name, result in self.results.items():
            print(f"\n--- {name.upper()} ---")
            print(f"Accuracy: {result['accuracy']:.4f}")
            print(f"Precision: {result['precision']:.4f}")
            print(f"Recall: {result['recall']:.4f}")
            print(f"F1 Score: {result['f1']:.4f}")
            if 'roc' in result:
                print(f"AUC: {result['roc']['auc']:.4f}")

if __name__ == '__main__':
    models = TraditionalModels()
    models.train_models()
    models.print_results() 