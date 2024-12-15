#!/usr/bin/env python
# uv: dependencies="pandas, numpy, seaborn, matplotlib, requests"

import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests

# Ensure API Token is set
API_TOKEN = os.getenv("AIPROXY_TOKEN")
if not API_TOKEN:
    raise EnvironmentError("Error: AIPROXY_TOKEN environment variable not set.")

# Set API headers for AI Proxy
API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

def check_and_install_dependencies():
    """Ensure required dependencies are installed."""
    try:
        import seaborn
    except ImportError:
        print("Seaborn not found. Installing now...")
        os.system("pip install seaborn")
        import seaborn

check_and_install_dependencies()

def load_data(file_path):
    """Load CSV file with fallback for encoding issues."""
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
        print(f"Successfully loaded {file_path}")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="latin1")
        print(f"Fallback to Latin-1 encoding for {file_path}")
    except Exception as e:
        raise FileNotFoundError(f"Error loading file: {e}")
    return df

def analyze_data(df):
    """Perform enhanced dataset analysis."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_df = df.apply(pd.to_numeric, errors='coerce')

    correlation_matrix = numeric_df.corr()
    significant_correlations = (
        correlation_matrix[correlation_matrix.abs() > 0.5]
        .stack()
        .reset_index()
        .query("level_0 != level_1")  # Avoid self-correlations
        .to_dict(orient="records")
    )
    stats = df.describe(include="all").to_dict()
    missing_values = df.isnull().sum().to_dict()

    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "missing_values": missing_values,
        "stats": stats,
        "significant_correlations": significant_correlations,
    }

def generate_visualizations(df, output_dir):
    """Generate annotated visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    numeric_cols = df.select_dtypes(include=np.number).columns

    # Histogram with annotation
    if len(numeric_cols) >= 1:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[numeric_cols[0]], kde=True, color="blue")
        plt.title(f"Distribution of {numeric_cols[0]}")
        plt.xlabel(numeric_cols[0])
        plt.ylabel("Frequency")
        plt.annotate("Data Distribution", xy=(0.5, 0.9), xycoords='axes fraction')
        plt.legend(["Histogram"])
        plt.savefig(os.path.join(output_dir, "histogram.png"))
        plt.close()

    # Scatterplot with legend
    if len(numeric_cols) >= 2:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=numeric_cols[0], y=numeric_cols[1], data=df)
        plt.title(f"Scatterplot: {numeric_cols[0]} vs {numeric_cols[1]}")
        plt.xlabel(numeric_cols[0])
        plt.ylabel(numeric_cols[1])
        plt.legend(["Scatterplot"])
        plt.savefig(os.path.join(output_dir, "scatterplot.png"))
        plt.close()

    # Correlation Heatmap
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.savefig(os.path.join(output_dir, "heatmap.png"))
        plt.close()

def generate_readme(summary, output_dir):
    """Generate README content using dynamic AI Proxy prompts."""
    correlation_summary = "\n".join(
        [f"- {item['level_0']} and {item['level_1']}: {item[0]:.2f}" for item in summary['significant_correlations']]
    ) if summary["significant_correlations"] else "No significant correlations found."

    prompt = f"""
    Create a README summarizing the dataset analysis:
    - Dataset shape: {summary['shape']}
    - Columns: {', '.join(summary['columns'])}
    - Missing Values: {summary['missing_values']}
    - Significant Correlations: {correlation_summary}

    Include:
    - Insights derived from the data
    - Visualizations and their interpretations
    - Any trends or patterns detected.
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

    with open(os.path.join(output_dir, "README.md"), "w") as file:
        file.write(readme_content)

def main():
    if len(sys.argv) != 2:
        raise ValueError("Usage: python autolysis.py <dataset.csv>")

    file_path = sys.argv[1]
    output_dir = os.path.splitext(file_path)[0]

    try:
        df = load_data(file_path)
        summary = analyze_data(df)
        generate_visualizations(df, output_dir)
        generate_readme(summary, output_dir)
        print("Analysis completed successfully. Check the output folder for results.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
