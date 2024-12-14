#!/usr/bin/env python
# uv: dependencies="pandas==2.0.3 numpy==1.26.0 seaborn==0.12.2 matplotlib==3.7.1 requests==2.31.0"



import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests

# Ensure AIPROXY_TOKEN is set
API_TOKEN = os.getenv("AIPROXY_TOKEN")
if not API_TOKEN:
    print("Error: AIPROXY_TOKEN environment variable not set.")
    sys.exit(1)

# Set API headers for AI Proxy
API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

def load_data(file_path):
    """Load CSV file with fallback for encoding issues."""
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
        print(f"Successfully loaded {file_path}")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="latin1")
        print(f"Fallback to Latin-1 encoding for {file_path}")
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)
    return df

def analyze_data(df):
    """Perform basic dataset analysis."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # Safely attempt to coerce non-numeric columns to numeric for correlation analysis
    numeric_df = df.apply(pd.to_numeric, errors='coerce')
    correlation_matrix = numeric_df.corr().to_dict() if len(numeric_cols) > 1 else {}

    stats = df.describe(include="all").to_dict()
    missing_values = df.isnull().sum().to_dict()

    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "missing_values": missing_values,
        "stats": stats,
        "correlation_matrix": correlation_matrix
    }


def generate_visualizations(df, output_dir):
    """Generate up to three visualizations and save as PNG files."""
    os.makedirs(output_dir, exist_ok=True)
    numeric_cols = df.select_dtypes(include=np.number).columns

    # Visualization 1: Histogram
    if len(numeric_cols) >= 1:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[numeric_cols[0]], kde=True, color="blue")
        plt.title(f"Distribution of {numeric_cols[0]}")
        plt.savefig(os.path.join(output_dir, "histogram.png"))
        plt.close()

    # Visualization 2: Scatterplot
    if len(numeric_cols) >= 2:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=numeric_cols[0], y=numeric_cols[1], data=df)
        plt.title(f"Scatterplot: {numeric_cols[0]} vs {numeric_cols[1]}")
        plt.savefig(os.path.join(output_dir, "scatterplot.png"))
        plt.close()

    # Visualization 3: Correlation Heatmap
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.savefig(os.path.join(output_dir, "heatmap.png"))
        plt.close()

def generate_readme(summary, output_dir):
    """Use AI Proxy to generate README content."""
    prompt = f"""
    Summarize the following dataset analysis in Markdown format for a README.md file:

    - Shape: {summary['shape']}
    - Columns: {summary['columns']}
    - Missing Values: {summary['missing_values']}
    - Correlation Matrix: {summary['correlation_matrix']}

    Include:
    - A brief introduction
    - Key findings and insights
    - Mention of visualizations
    """

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a data analyst generating Markdown summaries."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(API_URL, headers=HEADERS, json=data)
        response.raise_for_status()
        readme_content = response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error generating README: {e}")
        readme_content = "README generation failed. Please check your analysis."

    # Save README.md
    with open(os.path.join(output_dir, "README.md"), "w") as file:
        file.write(readme_content)

def main():
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    output_dir = os.path.splitext(file_path)[0]

    # Step 1: Load the dataset
    df = load_data(file_path)

    # Step 2: Analyze the dataset
    summary = analyze_data(df)

    # Step 3: Generate visualizations
    generate_visualizations(df, output_dir)

    # Step 4: Generate README
    generate_readme(summary, output_dir)

if __name__ == "__main__":
    main()
