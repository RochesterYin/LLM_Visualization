#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import pickle
import json
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
from io import BytesIO
import sys
import traceback

# Initialize Dash app
app = dash.Dash(__name__, title='ML/DL Model Comparison for Code Vulnerability Detection')
server = app.server

# Debug settings
DEBUG = True

# Directory paths - use absolute paths to avoid issues
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
MODELS_DIR = os.path.join(project_dir, 'models', 'saved')
INTERP_DIR = os.path.join(project_dir, 'visualization', 'interpretability')

if DEBUG:
    print(f"Current directory: {current_dir}")
    print(f"Project directory: {project_dir}")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Using models from: {os.path.abspath(MODELS_DIR)}")
    print(f"Files in models directory: {os.listdir(MODELS_DIR) if os.path.exists(MODELS_DIR) else 'Directory not found'}")

# Load results
def load_model_results():
    results = {}
    
    try:
        # Traditional models
        trad_path = os.path.join(MODELS_DIR, 'traditional_results.pkl')
        if os.path.exists(trad_path):
            if DEBUG:
                print(f"Loading traditional models from: {trad_path}")
            with open(trad_path, 'rb') as f:
                trad_results = pickle.load(f)
                if DEBUG:
                    print(f"Traditional models loaded: {list(trad_results.keys())}")
                results.update(trad_results)
        else:
            print(f"Warning: Traditional model results not found at {trad_path}")
        
        # Deep learning models
        deep_path = os.path.join(MODELS_DIR, 'deep_results.pkl')
        if os.path.exists(deep_path):
            if DEBUG:
                print(f"Loading deep learning models from: {deep_path}")
            with open(deep_path, 'rb') as f:
                deep_results = pickle.load(f)
                if DEBUG:
                    print(f"Deep learning models loaded: {list(deep_results.keys())}")
                results.update(deep_results)
        else:
            print(f"Warning: Deep learning model results not found at {deep_path}")
        
        if DEBUG:
            print(f"Total models loaded: {len(results)}")
            print(f"All models: {list(results.keys())}")
        
        return results
    except Exception as e:
        print(f"Error loading model results: {e}")
        traceback.print_exc()
        # Return empty dict on error
        return {}

# Load interpretability results
def load_interpretability_results():
    interp_path = os.path.join(MODELS_DIR, 'interpretability_results.json')
    if os.path.exists(interp_path):
        with open(interp_path, 'r') as f:
            return json.load(f)
    print(f"Warning: Interpretability results not found at {interp_path}")
    return None

# Load image as base64
def load_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    print(f"Warning: Image not found at {image_path}")
    return None

# Get available models
def get_available_models():
    results = load_model_results()
    if results:
        return list(results.keys())
    return []

# Create layout
app.layout = html.Div([
    html.H1('ML/DL Model Comparison for Code Vulnerability Detection', 
            style={'textAlign': 'center', 'marginBottom': '30px', 'marginTop': '20px'}),
    
    html.Div([
        html.Div([
            html.H2('Model Performance Comparison', style={'textAlign': 'center'}),
            
            # Model selection
            html.Div([
                html.Label('Select Models to Compare:'),
                dcc.Checklist(
                    id='model-checklist',
                    options=[],
                    value=[],
                    inline=True
                ),
            ], style={'marginBottom': '20px'}),
            
            # Performance metrics graph
            dcc.Graph(id='performance-graph'),
            
            # ROC curves
            dcc.Graph(id='roc-graph'),
            
        ], className='six columns', style={'padding': '10px'}),
        
        html.Div([
            html.H2('Model Interpretability', style={'textAlign': 'center'}),
            
            # Model selection for interpretability
            html.Div([
                html.Label('Select Model:'),
                dcc.Dropdown(
                    id='interp-model-dropdown',
                    options=[],
                    value=None
                ),
            ], style={'marginBottom': '20px'}),
            
            # Interpretability method selection
            html.Div([
                html.Label('Select Method:'),
                dcc.RadioItems(
                    id='interp-method-radio',
                    options=[
                        {'label': 'SHAP Summary', 'value': 'shap_summary'},
                        {'label': 'SHAP Bar Plot', 'value': 'shap_bar'},
                        {'label': 'LIME Explanation', 'value': 'lime'}
                    ],
                    value='shap_summary'
                ),
            ], style={'marginBottom': '20px'}),
            
            # Sample selection for LIME
            html.Div([
                html.Label('Select Sample (for LIME):'),
                dcc.Dropdown(
                    id='lime-sample-dropdown',
                    options=[
                        {'label': f'Sample {i+1}', 'value': i} for i in range(3)
                    ],
                    value=0,
                    disabled=True
                ),
            ], style={'marginBottom': '20px', 'display': 'none'}, id='lime-sample-div'),
            
            # Interpretability visualization
            html.Div([
                html.Img(id='interp-image', style={'width': '100%'})
            ], id='interp-img-div', style={'textAlign': 'center', 'marginTop': '20px'}),
            
            # Feature importance
            html.Div([
                dcc.Graph(id='feature-importance-graph')
            ], id='feature-importance-div', style={'display': 'none'}),
            
        ], className='six columns', style={'padding': '10px'}),
    ], className='row'),
    
    # Footer
    html.Div([
        html.Hr(),
        html.P('ML/DL Model Comparison for Code Vulnerability Detection', 
               style={'textAlign': 'center'})
    ], style={'marginTop': '50px'})
], className='container')

# Callbacks
@app.callback(
    [Output('model-checklist', 'options'),
     Output('model-checklist', 'value'),
     Output('interp-model-dropdown', 'options'),
     Output('interp-model-dropdown', 'value')],
    [dash.dependencies.Input('model-checklist', 'options')]  # Dummy input for initialization
)
def initialize_options(dummy):
    models = get_available_models()
    
    if DEBUG:
        print(f"Available models for dropdown: {models}")
    
    if not models:
        # If no models are available, create dummy models
        print("Warning: No models available. Creating dummy models for display.")
        models = ['decision_tree', 'random_forest', 'svm', 'logistic_regression', 'lstm', 'cnn']
    
    model_options = [{'label': model.replace('_', ' ').title(), 'value': model} for model in models]
    interp_model_options = model_options.copy()
    
    # Default selections
    model_values = models[:4] if len(models) > 4 else models
    interp_model_value = models[0] if models else None
    
    return model_options, model_values, interp_model_options, interp_model_value

@app.callback(
    Output('performance-graph', 'figure'),
    [Input('model-checklist', 'value')]
)
def update_performance_graph(selected_models):
    if not selected_models:
        return go.Figure()
    
    results = load_model_results()
    
    # Get metric values for each model
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    performance_data = []
    
    for model in selected_models:
        if model in results:
            for metric in metrics:
                # Check if metric exists and is valid
                if metric in results[model] and isinstance(results[model][metric], (int, float)) and not np.isnan(results[model][metric]):
                    performance_data.append({
                        'Model': model.replace('_', ' ').title(),
                        'Metric': metric.title(),
                        'Value': results[model][metric]
                    })
        else:
            # If model doesn't exist in results, create dummy data
            for metric in metrics:
                performance_data.append({
                    'Model': model.replace('_', ' ').title(),
                    'Metric': metric.title(),
                    'Value': 0.5  # Default value
                })
            if DEBUG:
                print(f"Warning: Model {model} not found in results. Using dummy data.")
    
    if not performance_data:
        # If no valid performance data, create a simple figure
        fig = go.Figure()
        fig.add_annotation(
            text="No performance data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig
    
    # Create performance comparison bar chart
    try:
        fig = px.bar(
            pd.DataFrame(performance_data),
            x='Model',
            y='Value',
            color='Metric',
            barmode='group',
            title='Performance Metrics Comparison',
            labels={'Value': 'Score', 'Model': 'Model'},
            category_orders={'Metric': [m.title() for m in metrics]}
        )
        
        fig.update_layout(
            xaxis_title='Model',
            yaxis_title='Score',
            legend_title='Metric',
            yaxis=dict(range=[0, 1]),
            plot_bgcolor='white'
        )
        
        return fig
    except Exception as e:
        print(f"Error creating performance graph: {e}")
        traceback.print_exc()
        # Return empty figure on error
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating graph: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig

@app.callback(
    Output('roc-graph', 'figure'),
    [Input('model-checklist', 'value')]
)
def update_roc_graph(selected_models):
    if not selected_models:
        return go.Figure()
    
    try:
        results = load_model_results()
        
        # Create ROC curves figure
        fig = go.Figure()
        
        has_valid_data = False
        
        for model in selected_models:
            if model in results and 'roc' in results[model]:
                roc = results[model]['roc']
                if 'fpr' in roc and 'tpr' in roc and 'auc' in roc:
                    try:
                        fig.add_trace(go.Scatter(
                            x=roc['fpr'],
                            y=roc['tpr'],
                            mode='lines',
                            name=f"{model.replace('_', ' ').title()} (AUC={roc['auc']:.3f})"
                        ))
                        has_valid_data = True
                    except Exception as e:
                        print(f"Error adding ROC trace for {model}: {e}")
            else:
                # Create dummy ROC curve if model not found
                if DEBUG:
                    print(f"Warning: ROC data for {model} not found. Using dummy data.")
                x = np.linspace(0, 1, 10)
                y = np.linspace(0, 1, 10)
                fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode='lines',
                    name=f"{model.replace('_', ' ').title()} (Dummy)",
                    line=dict(dash='dash')
                ))
                has_valid_data = True
        
        # Add diagonal line (random classifier)
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash', color='gray')
        ))
        
        fig.update_layout(
            title='ROC Curves',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain'),
            plot_bgcolor='white'
        )
        
        if not has_valid_data:
            fig.add_annotation(
                text="No ROC data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
        
        return fig
    
    except Exception as e:
        print(f"Error creating ROC graph: {e}")
        traceback.print_exc()
        # Return empty figure on error
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating ROC graph: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig

@app.callback(
    [Output('lime-sample-div', 'style'),
     Output('lime-sample-dropdown', 'disabled')],
    [Input('interp-method-radio', 'value')]
)
def toggle_lime_sample(method):
    if method == 'lime':
        return {'marginBottom': '20px', 'display': 'block'}, False
    return {'marginBottom': '20px', 'display': 'none'}, True

@app.callback(
    [Output('interp-img-div', 'style'),
     Output('feature-importance-div', 'style')],
    [Input('interp-method-radio', 'value')]
)
def toggle_display_divs(method):
    if method == 'shap_bar':
        return {'textAlign': 'center', 'marginTop': '20px', 'display': 'none'}, {'display': 'block'}
    return {'textAlign': 'center', 'marginTop': '20px'}, {'display': 'none'}

@app.callback(
    Output('interp-image', 'src'),
    [Input('interp-model-dropdown', 'value'),
     Input('interp-method-radio', 'value'),
     Input('lime-sample-dropdown', 'value')]
)
def update_interpretability_image(model, method, lime_sample):
    if not model or not method:
        return None
    
    # Get image path based on method
    if method == 'shap_summary':
        image_path = os.path.join(INTERP_DIR, f'shap_summary_{model}.png')
    elif method == 'shap_bar':
        image_path = os.path.join(INTERP_DIR, f'shap_bar_{model}.png')
    elif method == 'lime':
        image_path = os.path.join(INTERP_DIR, f'lime_{model}_sample_{lime_sample}.png')
    else:
        return None
    
    # Load image
    image_data = load_image(image_path)
    if image_data:
        return f'data:image/png;base64,{image_data}'
    
    return None

@app.callback(
    Output('feature-importance-graph', 'figure'),
    [Input('interp-model-dropdown', 'value'),
     Input('interp-method-radio', 'value')]
)
def update_feature_importance(model, method):
    if not model or method != 'shap_bar':
        return go.Figure()
    
    # Load interpretability results
    interp_results = load_interpretability_results()
    
    if not interp_results or 'shap' not in interp_results or model not in interp_results['shap']:
        return go.Figure()
    
    # Get SHAP feature importance
    model_data = interp_results['shap'][model]
    feature_names = model_data['feature_names']
    feature_importance = model_data['feature_importance']
    
    # Create dataframe
    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })
    
    # Sort by importance
    df = df.sort_values('Importance', ascending=False).head(15)
    
    # Create bar chart
    fig = px.bar(
        df,
        x='Importance',
        y='Feature',
        orientation='h',
        title=f'Feature Importance for {model.replace("_", " ").title()}'
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        plot_bgcolor='white'
    )
    
    return fig

# Add a basic route to serve the main page
@app.server.route('/')
def serve_index():
    return """
    <html>
    <head>
        <meta http-equiv="refresh" content="0;url=/"/>
        <title>Redirecting...</title>
    </head>
    <body>
        <p>If you are not redirected automatically, click <a href="/">here</a>.</p>
    </body>
    </html>
    """

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050) 