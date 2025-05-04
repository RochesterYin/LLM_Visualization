#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import pickle
import re
from collections import Counter
import numpy as np

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
    print("Converting CodeXGLUE dataset to features.csv format...")
    
    # Define file paths for CodeXGLUE data
    data_files = [
        ('data/dataset/codexglue_train.csv', 'train'),
        ('data/dataset/codexglue_validation.csv', 'valid'),
        ('data/dataset/codexglue_test.csv', 'test')
    ]
    
    # Process all data
    all_features = []
    all_tokens = []
    
    # Check if files exist
    missing_files = [path for path, _ in data_files if not os.path.exists(path)]
    if missing_files:
        print(f"Warning: Missing data files: {missing_files}")
        return
    
    # Process each file
    for file_path, split in data_files:
        print(f"Processing {file_path}...")
        
        # Read CSV file
        df = pd.read_csv(file_path)
        print(f"CSV file contains {len(df)} rows and columns: {df.columns.tolist()}")
        
        # Determine which columns contain code and label
        code_column = None
        label_column = None
        
        # Try to identify columns by name
        possible_code_columns = ['func', 'code', 'source', 'source_code', 'text']
        possible_label_columns = ['target', 'label', 'vulnerable', 'is_vulnerable', 'class']
        
        for col in df.columns:
            col_lower = col.lower()
            if code_column is None:
                for possible in possible_code_columns:
                    if possible in col_lower:
                        code_column = col
                        break
            if label_column is None:
                for possible in possible_label_columns:
                    if possible in col_lower:
                        label_column = col
                        break
        
        # If we couldn't identify columns, try to make educated guesses
        if code_column is None:
            # Look for a column with string values that are likely code
            for col in df.columns:
                if df[col].dtype == 'object' and len(df) > 0:
                    sample = str(df[col].iloc[0])
                    if len(sample) > 50 and ('{' in sample or '(' in sample):
                        code_column = col
                        break
        
        if label_column is None:
            # Look for binary columns that could be labels
            for col in df.columns:
                if set(df[col].unique()).issubset({0, 1}):
                    label_column = col
                    break
        
        print(f"Using columns - Code: {code_column}, Label: {label_column}")
        
        if code_column is None or label_column is None:
            print(f"Error: Could not identify code and label columns in {file_path}")
            print(f"Please check file format. First few rows:")
            print(df.head())
            continue
        
        # Process each row
        for i, row in df.iterrows():
            try:
                code = str(row[code_column])
                target = int(row[label_column])
                
                # Extract features
                features = extract_basic_features(code)
                features['label'] = target
                features['file'] = f"{split}_{i}"
                all_features.append(features)
                
                # Tokenize code
                tokens = tokenize_code(code)
                all_tokens.append(tokens)
                
                if i % 1000 == 0:
                    print(f"Processed {i} examples from {split}")
            except Exception as e:
                print(f"Error processing row {i} in {file_path}: {e}")
    
    # Create features dataframe
    print("Creating features dataframe...")
    features_df = pd.DataFrame(all_features)
    
    # Save processed data
    print("Saving processed data...")
    os.makedirs('processed', exist_ok=True)
    features_df.to_csv('processed/features.csv', index=False)
    
    with open('processed/tokens.pkl', 'wb') as f:
        pickle.dump(all_tokens, f)
    
    print("Conversion complete!")
    print(f"Features saved to 'processed/features.csv'")
    print(f"Tokens saved to 'processed/tokens.pkl'")
    print(f"Total examples processed: {len(all_features)}")

if __name__ == '__main__':
    main() 