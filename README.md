# ML/DL Model Comparison for Code Vulnerability Detection

A visualization and analysis framework for comparing machine learning and deep learning models in static code vulnerability detection.

## Project Overview

This project provides tools to evaluate, compare, and visualize the performance of various ML and DL models for detecting security vulnerabilities in source code. It features:

- **Model Comparison**: Compare traditional ML models (Decision Trees, Random Forests, SVMs, Logistic Regression) with DL models (LSTM, CNN)
- **Interpretability Analysis**: Understand model decisions using SHAP and LIME techniques
- **Interactive Visualization**: Web-based dashboard for exploring model performance and behavior

## Quick Start

1. **Clone this repository**:
   ```bash
   git clone https://github.com/yourusername/code-vulnerability-detection.git
   cd code-vulnerability-detection
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the demo with pre-generated data**:
   ```bash
   # Generate sample data and visualizations
   python create_dummy_results.py
   
   # Launch the dashboard
   python run_pipeline.py --dashboard-only
   ```

4. **Open the dashboard** in your browser at: http://localhost:8050

## Project Structure

```
├── data/               # Dataset and preprocessing scripts
├── models/             # ML and DL model implementations
│   ├── saved/          # Saved model files and results
├── visualization/      # Visualization and dashboard components
│   ├── interpretability/ # Interpretability visualizations
├── requirements.txt    # Project dependencies
├── run_pipeline.py     # Main execution script
└── README.md
```

## Full Pipeline Execution

To run the complete pipeline:

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

## Pushing to GitHub

If you want to share this project on GitHub:

1. **Create a new repository** on GitHub (without initializing with README or .gitignore)

2. **Initialize your local Git repository**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Code vulnerability detection project"
   ```

3. **Add the remote repository**:
   ```bash
   git remote add origin https://github.com/yourusername/repository-name.git
   ```

4. **Push to GitHub**:
   ```bash
   git push -u origin main
   ```
   (Use `master` instead of `main` if your default branch is named "master")

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 