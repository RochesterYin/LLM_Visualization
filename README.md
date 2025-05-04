# Machine Learning for Code Vulnerability Detection

This project develops a visualization and analysis framework for comparing traditional machine learning and deep learning models in static code vulnerability detection using the CodeXGLUE dataset.

## Project Overview

This system analyzes C/C++ code to detect potential security vulnerabilities without executing the code. It:

- **Compares multiple model approaches**:
  - Traditional ML: Decision Trees, Random Forests, SVM, Logistic Regression
  - Deep Learning: LSTM, CNN
  
- **Visualizes model performance and interpretability**:
  - Performance metrics (accuracy, precision, recall, F1)
  - ROC curves and AUC values
  - Feature importance and LIME explanations

## Dataset

<<<<<<< HEAD
This project uses the **CodeXGLUE C/C++ vulnerability detection dataset**, which contains:
- C/C++ functions labeled as either vulnerable (1) or non-vulnerable (0)
- A diverse set of vulnerability types including buffer overflows, memory leaks, and more
=======
1. **Clone this repository**:
   ```bash
   git clone https://github.com/RochesterYin/LLM_Visualization.git
   cd code-vulnerability-detection
   ```
>>>>>>> 2647e0eb9a2c45f3515e63f0d2863c69445f79b9

## Setup and Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Pipeline

1. **With Existing Data**
   
   The project already includes preprocessed files from the CodeXGLUE dataset:
   ```bash
   python run_pipeline.py --dashboard-only
   ```

2. **Full Training Pipeline**
   
   To train models from scratch:
   ```bash
   # Start with existing preprocessed dataset
   python models/traditional_models.py  # Train traditional ML models
   python models/deep_models.py         # Train deep learning models
   ```

3. **Visualizing Results**
   ```bash
   python run_pipeline.py --dashboard-only
   ```
   Open http://localhost:8050 in your web browser to access the dashboard.

## Project Structure

```
├── data/
│   ├── dataset/              # CodeXGLUE dataset files
│   │   ├── codexglue_train.csv
│   │   ├── codexglue_test.csv
│   │   └── codexglue_validation.csv
│   └── convert_data.py       # Data preprocessing script
├── models/
│   ├── traditional_models.py # Decision Tree, Random Forest, SVM, Logistic Regression
│   ├── deep_models.py        # LSTM and CNN models
│   └── model_interpretability.py # SHAP and LIME explanations
├── processed/
│   ├── features.csv          # Extracted features from code samples
│   └── tokens.pkl            # Tokenized source code
├── visualization/
│   ├── dashboard.py          # Interactive web dashboard
│   └── interpretability/     # Model explanation visualizations
└── requirements.txt          # Project dependencies
```

## Model Performance

Results from the CodeXGLUE dataset show varying performance across models:

<<<<<<< HEAD
| Model | Accuracy | F1 Score | Key Strengths |
|-------|----------|----------|---------------|
| CNN | ~61% | ~55% | Best at capturing code patterns |
| SVM | ~55% | ~25% | Good with high-dimensional feature spaces |
| Decision Tree | ~54% | ~49% | Most interpretable results |
| Random Forest | ~53% | ~44% | Robust to noise in the data |
| Logistic Regression | ~56% | ~20% | Simple and fast |
| LSTM | ~55% | <1% | Good with sequential data |

## Contributing

Feel free to submit issues or pull requests to improve the system. For major changes, please open an issue first to discuss what you'd like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/) 
=======
```bash
python run_pipeline.py
```

This will:
1. Download/prepare sample dataset
2. Preprocess the data for model training
3. Train traditional ML and DL models
4. Generate interpretability visualizations
5. Launch the dashboard

## Command-line Options

The pipeline supports several options:
- `--skip-preprocessing`: Skip data preprocessing
- `--skip-training`: Skip model training
- `--skip-interpretability`: Skip interpretability generation
- `--dashboard-only`: Only launch the dashboard

## Dataset

The project uses the CodeXGLUE dataset for code vulnerability detection. For demo purposes, a small sample dataset is generated automatically.

## Models Evaluated

### Traditional ML Models
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Logistic Regression

### Deep Learning Models
- Long Short-Term Memory (LSTM)
- Convolutional Neural Network (CNN)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 
>>>>>>> 2647e0eb9a2c45f3515e63f0d2863c69445f79b9
