#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import pickle
import joblib
import json
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class ModelInterpretability:
    def __init__(self, data_path='processed/features.csv'):
        """Initialize with path to processed data"""
        self.data_path = data_path
        self.models = {}
        self.feature_names = None
        self.X_test = None
        self.X_train = None
        self.y_test = None
        self.interpretability_results = {
            'shap': {},
            'lime': {}
        }
        
    def load_data_and_models(self):
        """Load data and trained models"""
        print("Loading data and models...")
        
        # Check if all necessary files exist
        models_dir = 'models/saved'
        if not os.path.exists(models_dir):
            print(f"Directory {models_dir} not found. Train models first.")
            return False
            
        # Load data
        if not os.path.exists(self.data_path):
            print(f"Data file {self.data_path} not found. Run preprocessing first.")
            return False
            
        # Load features
        df = pd.read_csv(self.data_path)
        
        # Separate features and labels
        y = df['label']
        X = df.drop(['label', 'file'], axis=1, errors='ignore')
        
        # Save feature names
        self.feature_names = X.columns.tolist()
        
        # Load scaler
        scaler_path = os.path.join(models_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            X_scaled = scaler.transform(X)
        else:
            print(f"Scaler not found at {scaler_path}. Using unscaled features.")
            X_scaled = X.values
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Load traditional models
        model_files = {
            'decision_tree': 'decision_tree.pkl',
            'random_forest': 'random_forest.pkl',
            'svm': 'svm.pkl',
            'logistic_regression': 'logistic_regression.pkl'
        }
        
        for name, filename in model_files.items():
            model_path = os.path.join(models_dir, filename)
            if os.path.exists(model_path):
                self.models[name] = joblib.load(model_path)
                print(f"Loaded {name} model")
            else:
                print(f"Model {name} not found at {model_path}")
        
        # Create output directory
        os.makedirs('visualization/interpretability', exist_ok=True)
        
        return len(self.models) > 0
    
    def generate_shap_explanations(self):
        """Generate SHAP explanations for models"""
        if not self.models or self.X_test is None:
            if not self.load_data_and_models():
                return
        
        print("Generating SHAP explanations...")
        
        for name, model in self.models.items():
            print(f"Generating SHAP explanation for {name}...")
            
            try:
                # For tree-based models (Decision Tree, Random Forest)
                if name in ['decision_tree', 'random_forest']:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(self.X_test)
                    
                    # For Random Forest (returns a list of arrays for each class)
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]  # Take values for class 1 (vulnerability)
                    
                # For other models
                else:
                    # Create a kernel explainer
                    explainer = shap.KernelExplainer(
                        model.predict_proba, 
                        shap.sample(self.X_train, 100)  # Use a sample of training data as background
                    )
                    shap_values = explainer.shap_values(self.X_test[:100])  # Use a subset for efficiency
                    
                    # For models returning a list of arrays for each class
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]  # Take values for class 1 (vulnerability)
                
                # Save SHAP values
                self.interpretability_results['shap'][name] = {
                    'shap_values': shap_values.tolist() if hasattr(shap_values, 'tolist') else shap_values,
                    'feature_names': self.feature_names
                }
                
                # Create and save summary plot
                plt.figure(figsize=(12, 8))
                shap.summary_plot(
                    shap_values, 
                    self.X_test if name in ['decision_tree', 'random_forest'] else self.X_test[:100],
                    feature_names=self.feature_names,
                    show=False
                )
                plt.tight_layout()
                plt.savefig(f'visualization/interpretability/shap_summary_{name}.png')
                plt.close()
                
                # Create and save bar plot
                plt.figure(figsize=(12, 8))
                shap.summary_plot(
                    shap_values, 
                    self.X_test if name in ['decision_tree', 'random_forest'] else self.X_test[:100],
                    feature_names=self.feature_names,
                    plot_type='bar',
                    show=False
                )
                plt.tight_layout()
                plt.savefig(f'visualization/interpretability/shap_bar_{name}.png')
                plt.close()
                
                print(f"SHAP explanations for {name} saved")
                
            except Exception as e:
                print(f"Error generating SHAP explanation for {name}: {e}")
    
    def generate_lime_explanations(self):
        """Generate LIME explanations for models"""
        if not self.models or self.X_test is None:
            if not self.load_data_and_models():
                return
        
        print("Generating LIME explanations...")
        
        # Create a LIME explainer
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            self.X_train,
            feature_names=self.feature_names,
            class_names=['Non-Vulnerable', 'Vulnerable'],
            mode='classification'
        )
        
        # For each model, generate explanations for a sample
        for name, model in self.models.items():
            print(f"Generating LIME explanation for {name}...")
            
            try:
                # Choose samples to explain
                samples_to_explain = [0, 1, 2]  # Explain first 3 samples
                lime_explanations = []
                
                for i in samples_to_explain:
                    # Get prediction function
                    if hasattr(model, "predict_proba"):
                        predict_fn = model.predict_proba
                    else:
                        # Create a wrapper function if predict_proba is not available
                        def predict_fn(x):
                            return np.column_stack((1 - model.predict(x), model.predict(x)))
                    
                    # Generate explanation
                    explanation = lime_explainer.explain_instance(
                        self.X_test[i], 
                        predict_fn, 
                        num_features=10
                    )
                    
                    # Save explanation as image
                    fig = explanation.as_pyplot_figure(label=1)  # Label 1 for vulnerable
                    plt.tight_layout()
                    plt.savefig(f'visualization/interpretability/lime_{name}_sample_{i}.png')
                    plt.close()
                    
                    # Store explanation data
                    lime_explanations.append({
                        'sample_idx': i,
                        'features': explanation.as_list(label=1)
                    })
                
                # Save LIME explanations
                self.interpretability_results['lime'][name] = lime_explanations
                
                print(f"LIME explanations for {name} saved")
                
            except Exception as e:
                print(f"Error generating LIME explanation for {name}: {e}")
    
    def save_results(self):
        """Save interpretability results"""
        output_path = 'models/saved/interpretability_results.json'
        
        # Convert results to a serializable format
        serializable_results = {
            'shap': {},
            'lime': self.interpretability_results['lime']
        }
        
        # For SHAP, we only save feature importances, not all values (to save space)
        for model_name, shap_data in self.interpretability_results['shap'].items():
            if isinstance(shap_data['shap_values'], list):
                # Calculate feature importance as mean absolute SHAP value
                feature_importance = np.abs(np.array(shap_data['shap_values'])).mean(axis=0)
                
                serializable_results['shap'][model_name] = {
                    'feature_importance': feature_importance.tolist(),
                    'feature_names': shap_data['feature_names']
                }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f)
        
        print(f"Interpretability results saved to {output_path}")

if __name__ == '__main__':
    interpreter = ModelInterpretability()
    interpreter.load_data_and_models()
    interpreter.generate_shap_explanations()
    interpreter.generate_lime_explanations()
    interpreter.save_results() 