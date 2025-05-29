import pandas as pd
import numpy as np
import os
import glob
import json
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
BASE_RESULTS_DIR = os.path.join(os.getcwd(), "api_generations", "lr_experiments", "color_reasoning_experiment")
INPUTS_DIR = os.path.join(os.getcwd(), "inputs", "lr_experiments_inputs")
ANALYSIS_OUTPUT_DIR = os.path.join(os.getcwd(), "lr_experiment_results", "color_reasoning_analysis")
os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)

# --- Helper Functions ---
def load_color_experiment_results():
    """Loads all CSV results for the color_reasoning_experiment."""
    if not os.path.isdir(BASE_RESULTS_DIR):
        print(f"Results directory not found: {BASE_RESULTS_DIR}")
        return pd.DataFrame()
    csv_files = glob.glob(os.path.join(BASE_RESULTS_DIR, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {BASE_RESULTS_DIR}")
        return pd.DataFrame()
    df_list = [pd.read_csv(f) for f in csv_files]
    return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

def load_color_experiment_definition():
    """Loads the color experiment definition JSON which contains correct answers/stimuli info."""
    json_path = os.path.join(INPUTS_DIR, "color_reasoning_experiment.json")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Color experiment definition file not found: {json_path}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {json_path}")
    return None

def parse_and_score_color_responses(df, experiment_definition):
    """
    Parses responses and scores them against ground truth from the experiment definition.
    This is highly specific to how `api_calls.py` formats prompts and saves individual question responses.
    It currently assumes the 'Response' column contains the direct answer from the LLM for one task/stimulus.
    And that there's a 'StimulusID' or similar column in the CSV to map back to the definition.
    """
    if df.empty or experiment_definition is None:
        return df

    print("Note: `parse_and_score_color_responses` assumes each CSV row contains a 'StimulusID' and 'Response' for a single stimulus.")
    print("This parsing and scoring logic needs to be adapted based on actual CSV output structure from `api_calls.py`.")

    df['ParsedAnswer'] = df['Response'].astype(str).str.strip().str.lower()
    df['IsCorrect'] = False # Default to False, update when StimulusID is present

    # Detailed parsing and scoring logic would go here, requiring 'StimulusID' in CSV
    # and a robust way to map it to the correct answer in experiment_definition.

    return df

def analyze_color_accuracy(df, experiment_name, group_by_cols=None):
    """Analyzes accuracy for color tasks that have correct answers."""
    if df.empty or 'IsCorrect' not in df.columns:
        print(f"No data or 'IsCorrect' column for accuracy analysis of {experiment_name}.")
        return

    if group_by_cols is None:
        group_by_cols = ['Language', 'Model', 'SystemPromptType']

    print(f"\n--- Accuracy Analysis for {experiment_name} ---")
    accuracy_results = df.groupby(group_by_cols)['IsCorrect'].mean().reset_index()
    accuracy_results.rename(columns={'IsCorrect': 'Accuracy'}, inplace=True)
    print(accuracy_results)

    for key, group_data in accuracy_results.groupby([col for col in group_by_cols if col not in ['Language']]):
        plt.figure(figsize=(10, 6))
        sns.barplot(data=group_data, x='Language', y='Accuracy', hue='Model')
        title_str = f"Accuracy ({experiment_name}) - {key}"
        plt.title(title_str)
        plt.ylabel("Accuracy")
        plt.xlabel("Language")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plot_filename = f"accuracy_{experiment_name}_{str(key).replace(' ', '_').replace('#','')}.png"
        plt.savefig(os.path.join(ANALYSIS_OUTPUT_DIR, plot_filename))
        plt.close()
        print(f"Saved accuracy plot: {plot_filename}")

def analyze_color_choice_frequencies(df, experiment_name, task_filter_column=None, task_filter_value=None, group_by_cols=None):
    """Analyzes choice frequencies for association/scenario tasks."""
    if df.empty:
        print(f"No data for frequency analysis of {experiment_name}.")
        return
    
    if task_filter_column and task_filter_value and task_filter_column in df.columns:
        df_filtered = df[df[task_filter_column].str.contains(task_filter_value, case=False, na=False)]
    else:
        df_filtered = df

    if df_filtered.empty:
        print(f"No data after filtering for {task_filter_value} in {experiment_name}.")
        return
        
    if group_by_cols is None:
        group_by_cols = ['Language', 'Model', 'SystemPromptType']

    print(f"\n--- Choice Frequency Analysis for {experiment_name} (Filter: {task_filter_value or 'All'}) ---")
    
    if 'ParsedAnswer' not in df_filtered.columns:
        print("'ParsedAnswer' column missing.")
        return

    for group_key, group_df in df_filtered.groupby(group_by_cols):
        plt.figure(figsize=(12, 7))
        sns.countplot(data=group_df, y='ParsedAnswer', order=group_df['ParsedAnswer'].value_counts().index)
        title_str = f"Choices ({experiment_name} - {task_filter_value or 'All'}) - {group_key}"
        plt.title(title_str)
        plt.tight_layout()
        plot_filename = f"freq_color_{experiment_name}_{task_filter_value or 'all'}_{str(group_key).replace(' ','_').replace('#','')}.png"
        plt.savefig(os.path.join(ANALYSIS_OUTPUT_DIR, plot_filename))
        plt.close()
        print(f"Saved frequency plot: {plot_filename}")

def run_color_experiment_analysis():
    print("\nAnalyzing Color Reasoning Experiment...")
    df_results = load_color_experiment_results()
    experiment_definition = load_color_experiment_definition()

    if df_results.empty or experiment_definition is None:
        print("Cannot proceed with color experiment analysis due to missing data or definition.")
        return

    print("WARNING: Color experiment analysis relies heavily on `api_calls.py` saving one stimulus response per CSV row, including a 'StimulusID' column. The current implementation may not do this, which will limit the accuracy of this analysis script.")
    df_processed = parse_and_score_color_responses(df_results, experiment_definition)

    analyze_color_accuracy(df_processed[df_processed['IsCorrect'].notna()], "color_reasoning_accuracy_tasks")
    analyze_color_choice_frequencies(df_processed, "color_reasoning_all_choices")

    print(f"Color Reasoning analysis outputs saved in: {ANALYSIS_OUTPUT_DIR}")

if __name__ == '__main__':
    run_color_experiment_analysis()
    print("\nColor Experiment Analysis Complete.")

def run_color_experiment():
    print("Color Perception and Categorization experiment is not yet implemented.")
    print("If this experiment requires the language gender map, it can be passed as an argument.")
    # Example: if lang_gender_map:
    # print("Language gender map received.") 