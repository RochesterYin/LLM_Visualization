#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Conv1D, MaxPooling1D, Flatten, Input, GlobalMaxPooling1D, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

class DeepModels:
    def __init__(self, tokens_path='processed/tokens.pkl', features_path='processed/features.csv'):
        """Initialize with paths to processed data"""
        self.tokens_path = tokens_path
        self.features_path = features_path
        self.results = {}
        self.models = {}
        self.tokenizer = None
        self.max_length = 500  # Maximum sequence length
        self.vocab_size = 10000  # Maximum vocabulary size
        self.embedding_dim = 128  # Embedding dimension
        
    def load_data(self):
        """Load and prepare data"""
        print(f"Loading data from {self.tokens_path} and {self.features_path}")
        
        # Check if processed data exists
        if not os.path.exists(self.tokens_path) or not os.path.exists(self.features_path):
            print("Data files not found. Run preprocessing first.")
            return False
            
        try:
            # Load tokens
            with open(self.tokens_path, 'rb') as f:
                tokens_list = pickle.load(f)
            print(f"Loaded {len(tokens_list)} token sequences")
            
            # Load features for labels
            df = pd.read_csv(self.features_path)
            print(f"Loaded features with shape: {df.shape}")
            labels = df['label'].values
            
            # Create vocabulary and encode tokens
            from tensorflow.keras.preprocessing.text import Tokenizer
            self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
            
            # Convert tokens to strings for tokenizer
            token_strings = [' '.join([str(t) for t in tokens]) for tokens in tokens_list]
            print("Fitting tokenizer on token sequences...")
            self.tokenizer.fit_on_texts(token_strings)
            print(f"Vocabulary size: {len(self.tokenizer.word_index)}")
            
            # Convert tokens to sequences
            print("Converting to sequences and padding...")
            sequences = self.tokenizer.texts_to_sequences(token_strings)
            padded_sequences = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
            print(f"Padded sequences shape: {padded_sequences.shape}")
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                padded_sequences, labels, test_size=0.2, random_state=42
            )
            
            # Save tokenizer for later use
            os.makedirs('models/saved', exist_ok=True)
            with open('models/saved/tokenizer.pkl', 'wb') as f:
                pickle.dump(self.tokenizer, f)
            
            # Save vocabulary size and max length
            with open('models/saved/dl_params.json', 'w') as f:
                json.dump({
                    'vocab_size': self.vocab_size,
                    'max_length': self.max_length,
                    'embedding_dim': self.embedding_dim
                }, f)
            
            print(f"Data loaded: {len(padded_sequences)} samples")
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            # Use a simplified approach if there's an error with the tokens
            print("Trying alternative approach with features only...")
            
            # Load features
            df = pd.read_csv(self.features_path)
            # Use only the numeric features, drop the label and file columns
            X = df.drop(['label', 'file'], axis=1, errors='ignore')
            y = df['label'].values
            
            # Create a simpler model input (sequence of numeric features)
            # Convert each row to a sequence for LSTM/CNN
            feature_sequences = X.values
            feature_sequences = feature_sequences.reshape(feature_sequences.shape[0], feature_sequences.shape[1], 1)
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                feature_sequences, y, test_size=0.2, random_state=42
            )
            
            # Set flag to use simplified model
            self.use_simplified = True
            print(f"Using simplified model with feature sequences: {feature_sequences.shape}")
            
            return True
    
    def build_lstm_model(self):
        """Build LSTM model for code vulnerability detection"""
        # Check if we're using the simplified approach
        if hasattr(self, 'use_simplified') and self.use_simplified:
            # Build a simpler LSTM model directly on features
            input_shape = self.X_train.shape[1:]
            model = Sequential([
                LSTM(128, input_shape=input_shape, return_sequences=True),
                LSTM(64),
                Dense(32, activation='relu'),
                Dropout(0.5),
                Dense(1, activation='sigmoid')
            ])
        else:
            # Build the standard LSTM model with embeddings
            model = Sequential([
                Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_length),
                LSTM(128, return_sequences=True),
                LSTM(64),
                Dense(32, activation='relu'),
                Dropout(0.5),
                Dense(1, activation='sigmoid')
            ])
        
        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        model.summary()
        return model
    
    def build_cnn_model(self):
        """Build CNN model for code vulnerability detection"""
        # Check if we're using the simplified approach
        if hasattr(self, 'use_simplified') and self.use_simplified:
            # Build a simpler CNN model directly on features
            input_shape = self.X_train.shape[1:]
            model = Sequential([
                Conv1D(128, 3, activation='relu', input_shape=input_shape),
                MaxPooling1D(3),
                Conv1D(128, 3, activation='relu'),
                GlobalMaxPooling1D(),
                Dense(64, activation='relu'),
                Dropout(0.5),
                Dense(1, activation='sigmoid')
            ])
        else:
            # Build the standard CNN model with embeddings
            model = Sequential([
                Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_length),
                Conv1D(128, 5, activation='relu'),
                MaxPooling1D(5),
                Conv1D(128, 5, activation='relu'),
                GlobalMaxPooling1D(),
                Dense(64, activation='relu'),
                Dropout(0.5),
                Dense(1, activation='sigmoid')
            ])
        
        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        model.summary()
        return model
    
    def train_models(self):
        """Train deep learning models"""
        if not hasattr(self, 'X_train'):
            if not self.load_data():
                return
        
        print("Training deep learning models...")
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        
        # Train LSTM model
        print("Training LSTM model...")
        try:
            lstm_model = self.build_lstm_model()
            lstm_history = lstm_model.fit(
                self.X_train, self.y_train,
                epochs=5,  # Reduced epochs for faster training
                batch_size=64,  # Larger batch size
                validation_split=0.1,
                callbacks=[early_stopping],
                verbose=1
            )
            self.models['lstm'] = lstm_model
        except Exception as e:
            print(f"Error training LSTM model: {e}")
        
        # Train CNN model
        print("Training CNN model...")
        try:
            cnn_model = self.build_cnn_model()
            cnn_history = cnn_model.fit(
                self.X_train, self.y_train,
                epochs=5,  # Reduced epochs for faster training
                batch_size=64,  # Larger batch size
                validation_split=0.1,
                callbacks=[early_stopping],
                verbose=1
            )
            self.models['cnn'] = cnn_model
        except Exception as e:
            print(f"Error training CNN model: {e}")
        
        # Evaluate models
        for name, model in self.models.items():
            print(f"Evaluating {name}...")
            
            # Make predictions
            y_pred_prob = model.predict(self.X_test)
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()
            
            # Calculate metrics
            self.results[name] = {
                'accuracy': float(accuracy_score(self.y_test, y_pred)),
                'precision': float(precision_score(self.y_test, y_pred, zero_division=0)),
                'recall': float(recall_score(self.y_test, y_pred, zero_division=0)),
                'f1': float(f1_score(self.y_test, y_pred, zero_division=0)),
                'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist()
            }
            
            # Calculate ROC curve
            try:
                fpr, tpr, _ = roc_curve(self.y_test, y_pred_prob)
                roc_auc = auc(fpr, tpr)
                self.results[name]['roc'] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'auc': float(roc_auc)
                }
            except Exception as e:
                print(f"Error calculating ROC: {e}")
                self.results[name]['roc'] = {
                    'fpr': [0, 1],
                    'tpr': [0, 1],
                    'auc': 0.5
                }
            
            # Save trained model
            try:
                model_path = f'models/saved/{name}_model'
                model.save(model_path)
                print(f"Model saved to {model_path}")
            except Exception as e:
                print(f"Error saving model: {e}")
        
        # Save results
        results_path = 'models/saved/deep_results.pkl'
        with open(results_path, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"Results saved to {results_path}")
        
        return self.results
    
    def print_results(self):
        """Print evaluation results"""
        if not self.results:
            print("No results available. Train models first.")
            return
            
        print("\n===== Deep Learning Models Results =====")
        for name, result in self.results.items():
            print(f"\n--- {name.upper()} ---")
            print(f"Accuracy: {result['accuracy']:.4f}")
            print(f"Precision: {result['precision']:.4f}")
            print(f"Recall: {result['recall']:.4f}")
            print(f"F1 Score: {result['f1']:.4f}")
            if 'roc' in result:
                print(f"AUC: {result['roc']['auc']:.4f}")

if __name__ == '__main__':
    models = DeepModels()
    models.train_models()
    models.print_results() 