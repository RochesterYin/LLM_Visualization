#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import json

def create_dummy_traditional_results():
    """Create dummy results for traditional ML models"""
    results = {}
    
    # Models
    models = ['decision_tree', 'random_forest', 'svm', 'logistic_regression']
    
    # Create results for each model with slightly different values
    for i, model in enumerate(models):
        base_acc = 0.75 + i * 0.03  # Different base accuracy for each model
        
        # Add random variation
        results[model] = {
            'accuracy': min(0.95, base_acc + np.random.uniform(-0.02, 0.02)),
            'precision': min(0.95, base_acc + np.random.uniform(-0.03, 0.03)),
            'recall': min(0.95, base_acc - 0.05 + np.random.uniform(-0.02, 0.02)),
            'f1': min(0.95, base_acc - 0.02 + np.random.uniform(-0.02, 0.02)),
            'confusion_matrix': [[80 + i, 20 - i], [15 - i, 85 + i]]
        }
        
        # Add ROC curve
        # Create a smooth ROC curve with some random points
        fpr = np.sort(np.unique(np.concatenate([np.linspace(0, 1, 20), np.random.uniform(0, 1, 10)])))
        tpr = np.minimum(1, np.maximum(0, fpr + base_acc * (1 - fpr) + np.random.uniform(-0.05, 0.05, size=len(fpr))))
        tpr[0] = 0  # Force start at 0
        tpr[-1] = 1  # Force end at 1
        
        # Calculate AUC (approximate)
        auc_val = base_acc + np.random.uniform(-0.02, 0.02)
        
        results[model]['roc'] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'auc': auc_val
        }
    
    # Save to file
    os.makedirs('models/saved', exist_ok=True)
    with open('models/saved/traditional_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Created dummy results for {len(models)} traditional models")
    return results

def create_dummy_deep_results():
    """Create dummy results for deep learning models"""
    results = {}
    
    # Models
    models = ['lstm', 'cnn']
    
    # Create results for each model with slightly different values
    for i, model in enumerate(models):
        base_acc = 0.83 + i * 0.02  # Different base accuracy for each model
        
        # Add random variation
        results[model] = {
            'accuracy': min(0.95, base_acc + np.random.uniform(-0.02, 0.02)),
            'precision': min(0.95, base_acc + np.random.uniform(-0.03, 0.03)),
            'recall': min(0.95, base_acc - 0.03 + np.random.uniform(-0.02, 0.02)),
            'f1': min(0.95, base_acc - 0.01 + np.random.uniform(-0.02, 0.02)),
            'confusion_matrix': [[85 + i, 15 - i], [12 - i, 88 + i]]
        }
        
        # Add ROC curve
        # Create a smooth ROC curve with some random points
        fpr = np.sort(np.unique(np.concatenate([np.linspace(0, 1, 20), np.random.uniform(0, 1, 10)])))
        tpr = np.minimum(1, np.maximum(0, fpr + base_acc * (1 - fpr) + np.random.uniform(-0.05, 0.05, size=len(fpr))))
        tpr[0] = 0  # Force start at 0
        tpr[-1] = 1  # Force end at 1
        
        # Calculate AUC (approximate)
        auc_val = base_acc + np.random.uniform(-0.02, 0.02)
        
        results[model]['roc'] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'auc': float(auc_val)
        }
    
    # Save to file
    os.makedirs('models/saved', exist_ok=True)
    with open('models/saved/deep_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Created dummy results for {len(models)} deep learning models")
    return results

def create_dummy_interpretability_results(trad_results, deep_results):
    """Create dummy interpretability results"""
    # Combine all models
    all_models = list(trad_results.keys()) + list(deep_results.keys())
    
    # Features for traditional models
    feature_names = [
        'code_length', 'num_lines', 'num_function_calls', 'num_variables',
        'has_strcpy', 'has_strcat', 'has_sprintf', 'has_gets', 'has_memcpy', 'has_scanf',
        'num_if', 'num_for', 'num_while', 'num_switch', 'num_pointers'
    ]
    
    # Create dummy interpretability results
    interp_results = {
        'shap': {},
        'lime': {}
    }
    
    # Create SHAP results for each model
    for model in all_models:
        # Create feature importance with some high values for dangerous functions
        importance = np.abs(np.random.normal(size=len(feature_names)))
        
        # Make dangerous functions more important
        for i, feat in enumerate(feature_names):
            if feat.startswith('has_'):
                importance[i] *= 2.0
        
        interp_results['shap'][model] = {
            'feature_importance': importance.tolist(),
            'feature_names': feature_names
        }
        
        # Create LIME explanations
        lime_explanations = []
        
        for sample_idx in range(3):
            # Create sample explanations
            features = []
            
            # Add some features with both positive and negative contributions
            for i in range(10):
                feat_idx = np.random.randint(0, len(feature_names))
                value = np.random.uniform(-0.5, 0.5)
                features.append([feature_names[feat_idx], float(value)])
            
            lime_explanations.append({
                'sample_idx': sample_idx,
                'features': features
            })
        
        interp_results['lime'][model] = lime_explanations
    
    # Save to file
    with open('models/saved/interpretability_results.json', 'w') as f:
        json.dump(interp_results, f)
    
    print(f"Created dummy interpretability results for {len(all_models)} models")

def create_dummy_visualization_images():
    """Create some dummy visualization placeholder images"""
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create directory
    os.makedirs('visualization/interpretability', exist_ok=True)
    
    # Models
    trad_models = ['decision_tree', 'random_forest', 'svm', 'logistic_regression']
    deep_models = ['lstm', 'cnn']
    all_models = trad_models + deep_models
    
    # Create SHAP summary images for each model
    for model in all_models:
        # Create a figure with random data
        plt.figure(figsize=(10, 8))
        
        # Create a dummy SHAP summary plot
        # Generate random feature importances
        importance = np.random.normal(size=10)
        features = [f'Feature {i+1}' for i in range(10)]
        
        # Create horizontal bar plot
        colors = ['blue' if i > 0 else 'red' for i in importance]
        plt.barh(features, importance, color=colors)
        plt.title(f'SHAP Summary for {model}')
        plt.tight_layout()
        plt.savefig(f'visualization/interpretability/shap_summary_{model}.png')
        plt.close()
        
        # Create a SHAP bar plot
        plt.figure(figsize=(10, 8))
        plt.barh(features, np.abs(importance), color='skyblue')
        plt.title(f'Feature Importance for {model}')
        plt.tight_layout()
        plt.savefig(f'visualization/interpretability/shap_bar_{model}.png')
        plt.close()
    
    # Create LIME explanations for each model and sample
    for model in all_models:
        for sample_idx in range(3):
            # Create a figure with random data
            plt.figure(figsize=(10, 8))
            
            # Create a dummy LIME explanation
            # Generate random feature contributions
            contributions = np.random.uniform(-0.5, 0.5, size=10)
            features = [f'Feature {i+1}' for i in range(10)]
            
            # Sort by absolute value
            idx = np.argsort(np.abs(contributions))[::-1]
            contributions = contributions[idx]
            features = [features[i] for i in idx]
            
            # Create horizontal bar plot
            colors = ['green' if c > 0 else 'red' for c in contributions]
            plt.barh(features, contributions, color=colors)
            plt.title(f'LIME Explanation for {model} - Sample {sample_idx+1}')
            plt.axvline(x=0, color='black', linestyle='-')
            plt.tight_layout()
            plt.savefig(f'visualization/interpretability/lime_{model}_sample_{sample_idx}.png')
            plt.close()
    
    print(f"Created dummy visualization images for {len(all_models)} models")

def main():
    """Main function to create all dummy data"""
    print("Creating dummy model results for testing...")
    
    # Create traditional model results
    trad_results = create_dummy_traditional_results()
    
    # Create deep learning model results
    deep_results = create_dummy_deep_results()
    
    # Create interpretability results
    create_dummy_interpretability_results(trad_results, deep_results)
    
    # Create visualization images
    create_dummy_visualization_images()
    
    print("Dummy data creation complete!")

if __name__ == '__main__':
    main() 