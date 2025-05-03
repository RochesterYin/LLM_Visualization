#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import subprocess
import sys
import time

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        'data/dataset',
        'processed',
        'models/saved',
        'visualization/interpretability'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def download_dataset():
    """Download the CodeXGLUE dataset"""
    print("\n=== Downloading CodeXGLUE Dataset ===")
    
    # Check if dataset already exists
    if os.path.exists('data/dataset') and os.listdir('data/dataset'):
        print("Dataset already exists. Skipping download.")
        return True
    
    try:
        # Create a wrapper script to handle dataset download (can be customized)
        print("Please download the dataset manually from:")
        print("https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection")
        print("and place it in the 'data/dataset' directory.")
        
        # For demonstration, we'll create a sample dataset
        print("Creating a small sample dataset for demonstration...")
        
        # Create sample vulnerable code
        os.makedirs('data/dataset/1', exist_ok=True)
        with open('data/dataset/1/sample1.c', 'w') as f:
            f.write("""
void vulnerable_function(char *input) {
    char buffer[10];
    strcpy(buffer, input);  // No bounds checking - buffer overflow vulnerability
}

int main(int argc, char *argv[]) {
    if (argc > 1) {
        vulnerable_function(argv[1]);
    }
    return 0;
}
            """)
        
        with open('data/dataset/1/sample2.c', 'w') as f:
            f.write("""
void process_data(char *src) {
    char dest[50];
    sprintf(dest, "%s", src);  // Potential buffer overflow
    printf("Processed: %s\\n", dest);
}

int main() {
    char *user_input = get_input();
    process_data(user_input);
    return 0;
}
            """)
        
        # Create sample non-vulnerable code
        os.makedirs('data/dataset/0', exist_ok=True)
        with open('data/dataset/0/sample1.c', 'w') as f:
            f.write("""
void safe_function(const char *input) {
    char buffer[100];
    strncpy(buffer, input, sizeof(buffer) - 1);
    buffer[sizeof(buffer) - 1] = '\\0';  // Ensure null termination
}

int main(int argc, char *argv[]) {
    if (argc > 1) {
        safe_function(argv[1]);
    }
    return 0;
}
            """)
        
        with open('data/dataset/0/sample2.c', 'w') as f:
            f.write("""
#define MAX_INPUT 1024

void process_data_safely(const char *src) {
    char dest[50];
    snprintf(dest, sizeof(dest), "%s", src);  // Safe alternative to sprintf
    printf("Processed: %s\\n", dest);
}

int main() {
    char user_input[MAX_INPUT];
    fgets(user_input, MAX_INPUT, stdin);
    process_data_safely(user_input);
    return 0;
}
            """)
        
        print("Sample dataset created successfully.")
        return True
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

def preprocess_data():
    """Preprocess the dataset"""
    print("\n=== Preprocessing Data ===")
    
    try:
        result = subprocess.run(['python', 'data/preprocess.py'], 
                               check=True, 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               universal_newlines=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error preprocessing data: {e}")
        print(e.stderr)
        return False

def train_models():
    """Train traditional and deep learning models"""
    print("\n=== Training Models ===")
    
    # Train traditional models
    print("Training traditional ML models...")
    try:
        result = subprocess.run(['python', 'models/traditional_models.py'], 
                               check=True, 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               universal_newlines=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error training traditional models: {e}")
        print(e.stderr)
        return False
    
    # Train deep learning models
    print("\nTraining deep learning models...")
    try:
        result = subprocess.run(['python', 'models/deep_models.py'], 
                               check=True, 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               universal_newlines=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error training deep learning models: {e}")
        print(e.stderr)
        return False
    
    return True

def generate_interpretability():
    """Generate model interpretability visualizations"""
    print("\n=== Generating Model Interpretability ===")
    
    try:
        result = subprocess.run(['python', 'models/model_interpretability.py'], 
                               check=True, 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               universal_newlines=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error generating interpretability: {e}")
        print(e.stderr)
        return False

def launch_dashboard():
    """Launch the visualization dashboard"""
    print("\n=== Launching Visualization Dashboard ===")
    print("Starting dashboard at http://localhost:8050")
    
    try:
        # This will block until the user terminates
        subprocess.run(['python', 'visualization/dashboard.py'], 
                      check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error launching dashboard: {e}")
        return False
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
        return True

def main():
    parser = argparse.ArgumentParser(description='Run ML/DL Model Comparison Pipeline')
    parser.add_argument('--skip-preprocessing', action='store_true', 
                        help='Skip data preprocessing')
    parser.add_argument('--skip-training', action='store_true', 
                        help='Skip model training')
    parser.add_argument('--skip-interpretability', action='store_true', 
                        help='Skip interpretability generation')
    parser.add_argument('--dashboard-only', action='store_true', 
                        help='Only launch the dashboard')
    
    args = parser.parse_args()
    
    # Create necessary directories
    create_directories()
    
    if args.dashboard_only:
        launch_dashboard()
        return
    
    # Download dataset
    if not download_dataset():
        print("Failed to download dataset. Exiting.")
        return
    
    # Preprocess data
    if not args.skip_preprocessing:
        if not preprocess_data():
            print("Failed to preprocess data. Exiting.")
            return
    else:
        print("\n=== Skipping Data Preprocessing ===")
    
    # Train models
    if not args.skip_training:
        if not train_models():
            print("Failed to train models. Exiting.")
            return
    else:
        print("\n=== Skipping Model Training ===")
    
    # Generate interpretability
    if not args.skip_interpretability:
        if not generate_interpretability():
            print("Failed to generate interpretability. Exiting.")
            return
    else:
        print("\n=== Skipping Interpretability Generation ===")
    
    # Launch dashboard
    launch_dashboard()

if __name__ == '__main__':
    main() 