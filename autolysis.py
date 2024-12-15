#!/usr/bin/env python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "pandas",
#   "numpy",
#   "matplotlib",
#   "seaborn",
#   "requests",
# ]
# ///

import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests

API_TOKEN = os.getenv("AIPROXY_TOKEN")
if not API_TOKEN:
    print("Error: AIPROXY_TOKEN environment variable not set.")
    sys.exit(1)

API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

def load_data(file_path):
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
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    correlation_matrix = numeric_df.corr().to_dict() if len(numeric_cols) > 1 else {}

    stats = df.describe(include="all").to_dict()
    missing_values = df.isnull().sum().to_dict()

    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "missing_values": missing_values,
        "stats": stats,
        "correlation_matrix": correlation_matrix,
    }

def generate_visualizations(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    numeric_cols = df.select_dtypes(include=np.number).columns

    file_links = []
    if len(numeric_cols) >= 1:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[numeric_cols[0]], kde=True, color="blue")
        plt.title(f"Distribution of {numeric_cols[0]}")
        hist_path = os.path.join(output_dir, "histogram.png")
        plt.savefig(hist_path)
        file_links.append(hist_path)
        plt.close()

    if len(numeric_cols) >= 2:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=numeric_cols[0], y=numeric_cols[1], data=df)
        plt.title(f"Scatterplot: {numeric_cols[0]} vs {numeric_cols[1]}")
        scatter_path = os.path.join(output_dir, "scatterplot.png")
        plt.savefig(scatter_path)
        file_links.append(scatter_path)
        plt.close()

    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        heatmap_path = os.path.join(output_dir, "heatmap.png")
        plt.savefig(heatmap_path)
        file_links.append(heatmap_path)
        plt.close()

    return file_links

def generate_readme(summary, visual_links, output_dir):
    """
    Generate a README using multiple LLM calls and improved formatting.
    """
    # Step 1: Generate Key Insights
    insights_prompt = f"""
    Analyze the following dataset summary and derive key insights:

    - Shape: {summary['shape']}
    - Columns: {summary['columns']}
    - Missing Values: {summary['missing_values']}
    - Correlation Matrix: {summary['correlation_matrix']}
    """

    insights_response = call_ai(insights_prompt)

    # Step 2: Generate Visualization Descriptions
    visualization_prompt = f"""
    Describe the following visualizations in Markdown format:

    - Histogram: {visual_links[0] if len(visual_links) > 0 else "No visualization"}
    - Scatterplot: {visual_links[1] if len(visual_links) > 1 else "No visualization"}
    - Heatmap: {visual_links[2] if len(visual_links) > 2 else "No visualization"}
    """

    visualization_response = call_ai(visualization_prompt)

    # Step 3: Combine Results into a README
    readme_content = f"""
    # Dataset Analysis

    ## Overview
    This document provides an analysis of the dataset, summarizing key findings and visualizations.

    ## Key Insights
    {insights_response}

    ## Visualizations
    {visualization_response}

    ## Files
    - [Histogram]({visual_links[0]}) if exists.
    - [Scatterplot]({visual_links[1]}) if exists.
    - [Heatmap]({visual_links[2]}) if exists.
    """

    # Save README
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme_content)
    print(f"README generated successfully at: {readme_path}")

def call_ai(prompt):
    """
    Helper function to call AI Proxy and return the response.
    """
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are an AI assistant specializing in dataset summaries."},
            {"role": "user", "content": prompt},
        ],
    }
    try:
        response = requests.post(API_URL, headers=HEADERS, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error during AI call: {e}")
        return "Error generating content. Please review the analysis."

def main():
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    output_dir = os.path.splitext(file_path)[0]

    # Step 1: Load Data
    df = load_data(file_path)

    # Step 2: Analyze Data
    summary = analyze_data(df)

    # Step 3: Generate Visualizations
    visual_links = generate_visualizations(df, output_dir)

    # Step 4: Generate README
    generate_readme(summary, visual_links, output_dir)

if __name__ == "__main__":
    main()
