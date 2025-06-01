import pandas as pd
import numpy as np
import os
import glob
import json
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
BASE_RESULTS_DIR = os.path.join(os.getcwd(), "api_generations", "lr_experiments", "spatial_reasoning_experiment")
INPUTS_DIR = os.path.join(os.getcwd(), "inputs", "lr_experiments_inputs")
ANALYSIS_OUTPUT_DIR = os.path.join(os.getcwd(), "lr_experiment_results", "spatial_reasoning_analysis")
os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)

# --- Helper Functions ---
def load_spatial_experiment_results():
    """Loads all CSV results for the spatial_reasoning_experiment."""
    if not os.path.isdir(BASE_RESULTS_DIR):
        print(f"Results directory not found: {BASE_RESULTS_DIR}")
        return pd.DataFrame()
    csv_files = glob.glob(os.path.join(BASE_RESULTS_DIR, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {BASE_RESULTS_DIR}")
        return pd.DataFrame()
    df_list = [pd.read_csv(f) for f in csv_files]
    return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

def load_spatial_experiment_definition():
    """Loads the spatial experiment definition JSON which contains task details and optimal answers."""
    json_path = os.path.join(INPUTS_DIR, "spatial_reasoning_experiment.json")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Spatial experiment definition file not found: {json_path}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {json_path}")
    return None

def parse_and_score_spatial_responses(df, experiment_definition):
    """
    Parses responses and scores them. This requires knowing which task the response refers to.
    Assumes `api_calls.py` saves one task response per row with a 'TaskID' column.
    """
    if df.empty or experiment_definition is None:
        return df

    print("Note: `parse_and_score_spatial_responses` assumes each CSV row contains a 'TaskID' and 'Response'.")
    print("This parsing and scoring logic needs to be adapted based on actual CSV output structure from `api_calls.py`.")

    df['ParsedAnswer'] = df['Response'].astype(str).str.strip().str.lower()
    df['IsOptimal'] = False # Default, update based on TaskID and scoring criteria
    df['PathLengthScore'] = np.nan # For tasks with path lengths

    # Detailed scoring logic would go here, mapping TaskID to experiment_definition["task_scoring"]
    # and comparing df['ParsedAnswer'] to optimal choices or evaluating path descriptions.
    # For tasks with options (e.g., SRT-1-REL, SRT-2, MN-1), check if ParsedAnswer is in optimal list.
    # For MN-2 (path description), this would require qualitative analysis or more complex NLP scoring.

    return df

def analyze_spatial_accuracy_optimality(df, experiment_name, group_by_cols=None):
    """Analyzes accuracy/optimality for spatial tasks with defined correct/optimal answers."""
    if df.empty or 'IsOptimal' not in df.columns:
        print(f"No data or 'IsOptimal' column for analysis of {experiment_name}.")
        return

    if group_by_cols is None:
        group_by_cols = ['Language', 'Model', 'SystemPromptType'] # Potentially 'TaskID'

    print(f"\n--- Optimality/Accuracy Analysis for {experiment_name} ---")
    optimality_results = df.groupby(group_by_cols)['IsOptimal'].mean().reset_index()
    optimality_results.rename(columns={'IsOptimal': 'OptimalityRate'}, inplace=True)
    print(optimality_results)

    # Plotting similar to color_experiment accuracy
    for key, group_data in optimality_results.groupby([col for col in group_by_cols if col not in ['Language']]):
        plt.figure(figsize=(10, 6))
        sns.barplot(data=group_data, x='Language', y='OptimalityRate', hue='Model')
        plt.title(f"Optimality ({experiment_name}) - {key}")
        plt.ylabel("Optimality Rate")
        plt.xlabel("Language")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plot_filename = f"optimality_{experiment_name}_{str(key).replace(' ','_').replace('#','')}.png"
        plt.savefig(os.path.join(ANALYSIS_OUTPUT_DIR, plot_filename))
        plt.close()
        print(f"Saved optimality plot: {plot_filename}")

def run_spatial_reasoning_analysis():
    print("\nAnalyzing Spatial Reasoning Experiment...")
    df_results = load_spatial_experiment_results()
    experiment_definition = load_spatial_experiment_definition()

    if df_results.empty or experiment_definition is None:
        print("Cannot proceed with spatial reasoning analysis due to missing data or definition.")
        return

    print("WARNING: Spatial reasoning analysis depends on `api_calls.py` saving one task response per CSV row, including a 'TaskID' column. The current implementation may not do this, limiting analysis accuracy.")
    df_processed = parse_and_score_spatial_responses(df_results, experiment_definition)
    
    analyze_spatial_accuracy_optimality(df_processed[df_processed['IsOptimal'].notna()], "spatial_reasoning_tasks")
    
    # Further analysis could include path length comparisons, qualitative analysis of descriptions for MN-2, etc.

    print(f"Spatial Reasoning analysis outputs saved in: {ANALYSIS_OUTPUT_DIR}")

def run_spatial_reasoning_experiment():
    """Runs the spatial reasoning experiment analysis."""
    run_spatial_reasoning_analysis()

if __name__ == '__main__':
    run_spatial_reasoning_analysis()
    print("\nSpatial Reasoning Experiment Analysis Complete.") 