# multiple choice
# open ended

import pandas as pd
import numpy as np
import os
import glob
import json
from scipy.stats import chi2_contingency, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
BASE_RESULTS_DIR = os.path.join(os.getcwd(), "api_generations", "lr_experiments")
ANALYSIS_OUTPUT_DIR = os.path.join(os.getcwd(), "lr_experiment_results", "preference_analysis")
os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)

def load_experiment_results(experiment_name):
    """Loads all CSV results for a given experiment."""
    experiment_dir = os.path.join(BASE_RESULTS_DIR, experiment_name)
    if not os.path.isdir(experiment_dir):
        print(f"Results directory not found for experiment: {experiment_name}")
        return pd.DataFrame()
    
    csv_files = glob.glob(os.path.join(experiment_dir, "*.csv"))
    if not csv_files:
        print(f"No CSV files found for experiment: {experiment_name}")
        return pd.DataFrame()
    
    df_list = []
    for f in csv_files:
        try:
            df_list.append(pd.read_csv(f))
        except Exception as e:
            print(f"Error reading {f}: {e}")
    
    if not df_list:
        return pd.DataFrame()
        
    return pd.concat(df_list, ignore_index=True)

def parse_responses_from_experiment(df, experiment_name):
    """
    Parses the 'Response' column based on the experiment type.
    For preference tasks, it assumes the response column contains lines of "Question: ...\nChoices: ...\nAnswer: <Chosen Answer>\n\n"
    or just the chosen answer directly if the API was instructed so.
    This function will need to be tailored if response formats are more complex.
    """
    df['ParsedChoice'] = df['Response'].astype(str).str.strip().str.lower()
    return df

def analyze_choice_frequencies(df, experiment_name, group_by_cols=None):
    """
    Analyzes and plots frequency of choices for preference tasks.
    Groups by language, model, and system prompt by default.
    Also requires original questions data to know the available choices and map them.
    """
    if df.empty:
        print(f"No data to analyze for {experiment_name}.")
        return

    if group_by_cols is None:
        group_by_cols = ['Language', 'Model', 'SystemPromptType']

    experiment_json_path = os.path.join(os.getcwd(), "inputs", "lr_experiments_inputs", f"{experiment_name}.json")
    try:
        with open(experiment_json_path, 'r', encoding='utf-8') as f:
            experiment_questions_data = json.load(f)
    except FileNotFoundError:
        print(f"Could not find experiment definition file: {experiment_json_path}")
        return
    except json.JSONDecodeError:
        print(f"Could not decode experiment definition file: {experiment_json_path}")
        return

    print(f"\n--- Frequency Analysis for {experiment_name} ---")
    
    if 'ParsedChoice' not in df.columns:
        print("Column 'ParsedChoice' not found. Ensure responses are parsed.")
        return

    for group_key, group_df in df.groupby(group_by_cols):
        title_str = f"Choice Frequencies ({experiment_name}) - {', '.join(map(str,group_key))}"
        plt.figure(figsize=(12, 7))
        
        try:
            sns.countplot(data=group_df, y='ParsedChoice', order = group_df['ParsedChoice'].value_counts().index)
        except Exception as e:
            print(f"Error during plotting for group {group_key}: {e}. Skipping this plot.")
            plt.close()
            continue
            
        plt.title(title_str)
        plt.xlabel("Frequency")
        plt.ylabel("Chosen Answer")
        plt.tight_layout()
        
        plot_filename = f"freq_{experiment_name}_{'_'.join(map(str,group_key)).replace(' ', '_').replace('#', '')}.png"
        plt.savefig(os.path.join(ANALYSIS_OUTPUT_DIR, plot_filename))
        plt.close()
        print(f"Saved plot: {plot_filename}")

def analyze_multiple_choice_experiment():
    experiment_name = "multiple_choice_experiment"
    df_results = load_experiment_results(experiment_name)
    if df_results.empty:
        return
    
    df_parsed = parse_responses_from_experiment(df_results, experiment_name)
    
    print("Note: Multiple choice analysis currently assumes 'Response' column is the direct answer to a single question per row.")
    print("This requires modification in `api_calls.py` to save results per question.")
    
    analyze_choice_frequencies(df_parsed, experiment_name)

def analyze_neutral_animals_experiment():
    experiment_name = "neutral_animals_experiment"
    df_results = load_experiment_results(experiment_name)
    if df_results.empty:
        return

    df_parsed = parse_responses_from_experiment(df_results, experiment_name)
    print("Note: Neutral animals analysis currently assumes 'Response' column is the direct answer to a single question per row.")
    print("This requires modification in `api_calls.py` to save results per question.")
    analyze_choice_frequencies(df_parsed, experiment_name)

if __name__ == '__main__':
    print("Running Preference Task Analysis...")
    
    print("\nAnalyzing Multiple Choice Experiment...")
    analyze_multiple_choice_experiment()
    
    print("\nAnalyzing Neutral Animals Experiment...")
    analyze_neutral_animals_experiment()

    # Grammatical gender questions are also preference, but might need specific parsing/linking to gender categories.
    # This script could be extended or a separate one used.
    # print("\nAnalyzing Grammatical Gender Questions Experiment...")
    # analyze_grammatical_gender_questions_experiment() # Needs to be implemented

    print("\nPreference Task Analysis Complete.")
    print(f"Analysis outputs saved in: {ANALYSIS_OUTPUT_DIR}")