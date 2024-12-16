#!/usr/bin/env python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "scikit-learn",
#   "python-dotenv",
#   "requests",
# ]
# ///


import os
import sys
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    print("Error: AIPROXY_TOKEN not set in environment.")
    sys.exit(1)

def load_data(file_path: str) -> pd.DataFrame:
    """Load CSV data with fallback for encoding issues."""
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
        print(f"Loaded file successfully: {file_path}")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="latin1")
        print(f"Used fallback encoding to load file: {file_path}")
    except Exception as e:
        print(f"Failed to read the file due to: {e}")
        sys.exit(1)
    return df

def analyze_dataset(df: pd.DataFrame) -> dict:
    """Perform basic dataset analysis."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime"]).columns.tolist()

    summary = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_columns": numeric_cols,
        "datetime_columns": datetime_cols,
    }

    # Generate statistics for numeric columns
    numeric_stats = df.select_dtypes(include=["number"]).describe().to_dict()
    summary["statistics"] = numeric_stats

    return summary

def visualize_data(df: pd.DataFrame, output_dir: str):
    """Generate up to 3 visualizations and save them."""
    os.makedirs(output_dir, exist_ok=True)
    plots = 0

    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        plt.figure(figsize=(6, 6), dpi=150)
        sns.histplot(df[numeric_cols[0]], kde=True, color="blue")
        plt.title(f"Distribution of {numeric_cols[0]}")
        plt.savefig(os.path.join(output_dir, "histogram.png"))
        plt.close()
        plots += 1

    if len(numeric_cols) >= 2:
        plt.figure(figsize=(6, 6), dpi=150)
        sns.scatterplot(x=numeric_cols[0], y=numeric_cols[1], data=df)
        plt.title(f"Scatterplot: {numeric_cols[0]} vs {numeric_cols[1]}")
        plt.savefig(os.path.join(output_dir, "scatterplot.png"))
        plt.close()
        plots += 1

        plt.figure(figsize=(6, 6), dpi=150)
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.savefig(os.path.join(output_dir, "heatmap.png"))
        plt.close()
        plots += 1

    print(f"Generated {plots} visualizations.")

def generate_readme(summary: dict, output_dir: str):
    """Generate a README.md file using the AI Proxy."""
    prompt = f"""
    Summarize the following dataset analysis in Markdown format for a README.md file:

    - Shape: {summary['shape']}
    - Columns: {summary['columns']}
    - Missing Values: {summary['missing_values']}
    - Statistics: {summary.get('statistics', 'N/A')}

    Include:
    - A brief introduction
    - Key findings and insights
    - Mention of visualizations
    """

    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are an expert data analyst."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            readme_content = response.json()['choices'][0]['message']['content']
        else:
            print(f"Failed to generate README.md: {response.text}")
            readme_content = "README generation failed."
    except Exception as e:
        print(f"OpenAI failed: {e}")
        readme_content = f"""# Dataset Analysis

## Summary
- **Shape**: {summary['shape']}
- **Columns**: {summary['columns']}
- **Missing Values**: {summary['missing_values']}

## Insights
OpenAI assistance was unavailable, but here are key insights based on the analysis:

- The dataset contains {summary['shape'][0]} rows and {summary['shape'][1]} columns.
- Missing values were found in the following columns: {', '.join([k for k, v in summary['missing_values'].items() if v > 0])}.
- Summary statistics for numeric columns are available in the dataset.

## Visualizations
Visualizations have been saved in the output directory:
1. Histogram of the first numeric column.
2. Scatterplot of the first two numeric columns.
3. Correlation heatmap of numeric columns.

"""

    # Write README.md
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as file:
        file.write(readme_content)
    print(f"README.md generated at {readme_path}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    output_dir = os.path.splitext(file_path)[0]

    # Load, analyze, visualize, and summarize data
    df = load_data(file_path)
    summary = analyze_dataset(df)
    visualize_data(df, output_dir)
    generate_readme(summary, output_dir)

if __name__ == "__main__":
    main()
