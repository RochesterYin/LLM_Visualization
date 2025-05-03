# Dataset Directory

This directory contains the CodeXGLUE dataset for code vulnerability detection and scripts for preprocessing.

## CodeXGLUE Dataset

The CodeXGLUE dataset can be downloaded from the official repository:
https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection

### Download Instructions

1. Download the dataset:
   ```
   wget https://github.com/microsoft/CodeXGLUE/raw/main/Code-Code/Defect-detection/dataset.zip
   ```

2. Unzip the dataset:
   ```
   unzip dataset.zip -d ./
   ```

3. The dataset contains C/C++ functions labeled as either vulnerable (1) or non-vulnerable (0).

## Preprocessing

Run the preprocessing script to extract features:
```
python preprocess.py
```

This will generate the processed data files ready for model training:
- `features.csv`: Extracted code features
- `tokens.pkl`: Tokenized source code
- `ast_features.pkl`: Abstract Syntax Tree features 