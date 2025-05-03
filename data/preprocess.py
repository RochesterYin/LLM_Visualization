#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import pickle
import re
from collections import Counter
import numpy as np

def load_data(data_dir):
    """Load CodeXGLUE dataset from directory"""
    data = []
    
    for label in [0, 1]:  # 0: non-vulnerable, 1: vulnerable
        dir_path = os.path.join(data_dir, str(label))
        if not os.path.exists(dir_path):
            print(f"Directory {dir_path} not found. Please download the dataset first.")
            continue
            
        for filename in os.listdir(dir_path):
            if filename.endswith('.c') or filename.endswith('.cpp'):
                file_path = os.path.join(dir_path, filename)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    code = f.read()
                    data.append({
                        'code': code,
                        'label': label,
                        'file': filename
                    })
    
    return pd.DataFrame(data)

def extract_basic_features(code):
    """Extract basic features from code"""
    features = {}
    
    # Code length
    features['code_length'] = len(code)
    
    # Number of lines
    features['num_lines'] = len(code.split('\n'))
    
    # Function calls
    function_calls = re.findall(r'\b\w+\s*\(', code)
    features['num_function_calls'] = len(function_calls)
    
    # Variables
    variables = re.findall(r'\b(int|char|float|double|void|long|unsigned|bool)\s+(\w+)', code)
    features['num_variables'] = len(variables)
    
    # Potentially dangerous functions
    dangerous_funcs = ['strcpy', 'strcat', 'sprintf', 'gets', 'memcpy', 'scanf']
    for func in dangerous_funcs:
        features[f'has_{func}'] = 1 if re.search(r'\b' + func + r'\s*\(', code) else 0
    
    # Control structures
    features['num_if'] = len(re.findall(r'\bif\s*\(', code))
    features['num_for'] = len(re.findall(r'\bfor\s*\(', code))
    features['num_while'] = len(re.findall(r'\bwhile\s*\(', code))
    features['num_switch'] = len(re.findall(r'\bswitch\s*\(', code))
    
    # Pointer usage
    features['num_pointers'] = len(re.findall(r'\*\s*\w+', code))
    
    return features

def tokenize_code(code):
    """Simple tokenization for code"""
    # Remove comments
    code = re.sub(r'//.*?\n|/\*.*?\*/', '', code, flags=re.DOTALL)
    
    # Split by symbols while keeping the symbols
    tokens = re.findall(r'[a-zA-Z_]\w*|[^\w\s]|\d+', code)
    return tokens

def main():
    print("Loading and preprocessing CodeXGLUE dataset...")
    
    # Check if dataset exists
    if not os.path.exists('./dataset'):
        print("Dataset not found. Please download it first.")
        return
    
    # Load data
    df = load_data('./dataset')
    print(f"Loaded {len(df)} code samples.")
    
    # Extract features
    print("Extracting features...")
    features_list = []
    tokens_list = []
    
    for _, row in df.iterrows():
        code = row['code']
        features = extract_basic_features(code)
        features['label'] = row['label']
        features['file'] = row['file']
        features_list.append(features)
        
        tokens = tokenize_code(code)
        tokens_list.append(tokens)
    
    # Create features dataframe
    features_df = pd.DataFrame(features_list)
    
    # Save processed data
    os.makedirs('../processed', exist_ok=True)
    features_df.to_csv('../processed/features.csv', index=False)
    
    with open('../processed/tokens.pkl', 'wb') as f:
        pickle.dump(tokens_list, f)
    
    print("Preprocessing complete!")
    print(f"Features saved to '../processed/features.csv'")
    print(f"Tokens saved to '../processed/tokens.pkl'")

if __name__ == '__main__':
    main() 