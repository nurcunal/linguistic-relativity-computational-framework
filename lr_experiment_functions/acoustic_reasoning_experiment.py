import pandas as pd
import numpy as np
import os
import glob
import json # Though not strictly needed for this one if labels are in code
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
BASE_RESULTS_DIR = os.path.join(os.getcwd(), "api_generations", "lr_experiments", "sound_waves")
ANALYSIS_OUTPUT_DIR = os.path.join(os.getcwd(), "lr_experiment_results", "acoustic_reasoning_analysis")
os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)

# Ground truth mapping (as defined in api_calls.py for the sound_waves experiment)
IMAGE_TO_EMOTION_GROUND_TRUTH = {
    'a.png': "neutral", 'b.png': "calm", 'c.png': "happy", 'd.png': "sad",
    'e.png': "angry", 'f.png': "fearful", 'g.png': "disgust", 'h.png': "surprised"
}
EMOTION_LABELS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

def load_acoustic_results():
    """Loads all CSV results for the sound_waves experiment."""
    if not os.path.isdir(BASE_RESULTS_DIR):
        print(f"Results directory not found: {BASE_RESULTS_DIR}")
        return pd.DataFrame()
    csv_files = glob.glob(os.path.join(BASE_RESULTS_DIR, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {BASE_RESULTS_DIR}")
        return pd.DataFrame()
    df_list = [pd.read_csv(f) for f in csv_files]
    return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

def score_acoustic_responses(df):
    """Scores responses against the ground truth for sound wave emotions."""
    if df.empty:
        return df
    
    # Ensure PredictedEmotion is lowercase and stripped
    df['PredictedEmotion_clean'] = df['PredictedEmotion'].astype(str).str.strip().str.lower()
    
    # Ensure TrueEmotionLabel is also clean (it should be from the CSV already)
    df['TrueEmotionLabel_clean'] = df['TrueEmotionLabel'].astype(str).str.strip().str.lower()

    df['IsCorrect'] = df['PredictedEmotion_clean'] == df['TrueEmotionLabel_clean']
    return df

def analyze_acoustic_accuracy(df, group_by_cols=None):
    """Analyzes accuracy for the sound wave emotion prediction task."""
    if df.empty or 'IsCorrect' not in df.columns:
        print(f"No data or 'IsCorrect' column for acoustic accuracy analysis.")
        return

    if group_by_cols is None:
        group_by_cols = ['Language', 'Model', 'SystemPromptType']

    print(f"\n--- Accuracy Analysis for Sound Waves Experiment ---")
    accuracy_results = df.groupby(group_by_cols)['IsCorrect'].mean().reset_index()
    accuracy_results.rename(columns={'IsCorrect': 'Accuracy'}, inplace=True)
    print("Overall Accuracy per group:")
    print(accuracy_results)

    # Plotting overall accuracy
    for key, group_data in accuracy_results.groupby([col for col in group_by_cols if col not in ['Language']]):
        plt.figure(figsize=(10, 6))
        sns.barplot(data=group_data, x='Language', y='Accuracy', hue='Model') # Example
        plt.title(f"Sound Wave Emotion Accuracy - {key}")
        plt.ylabel("Accuracy")
        plt.xlabel("Language")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plot_filename = f"accuracy_soundwaves_{str(key).replace(' ','_').replace('#','')}.png"
        plt.savefig(os.path.join(ANALYSIS_OUTPUT_DIR, plot_filename))
        plt.close()
        print(f"Saved accuracy plot: {plot_filename}")

    # Confusion Matrix per group
    for group_key_tuple, group_df in df.groupby(group_by_cols):
        group_key_str = '_'.join(map(str, group_key_tuple)).replace(' ', '_').replace('#', '')
        if group_df.empty or group_df['PredictedEmotion_clean'].isnull().all() or group_df['TrueEmotionLabel_clean'].isnull().all():
            print(f"Skipping confusion matrix for group {group_key_str} due to missing data.")
            continue
        
        cm = confusion_matrix(group_df['TrueEmotionLabel_clean'], group_df['PredictedEmotion_clean'], labels=EMOTION_LABELS)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS)
        plt.title(f'Confusion Matrix - Sound Waves - {group_key_str}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        cm_filename = f"confusion_matrix_soundwaves_{group_key_str}.png"
        plt.savefig(os.path.join(ANALYSIS_OUTPUT_DIR, cm_filename))
        plt.close()
        print(f"Saved confusion matrix: {cm_filename}")
        
        # Classification Report
        try:
            report = classification_report(group_df['TrueEmotionLabel_clean'], group_df['PredictedEmotion_clean'], labels=EMOTION_LABELS, zero_division=0)
            print(f"\nClassification Report for {group_key_str}:\n{report}")
            with open(os.path.join(ANALYSIS_OUTPUT_DIR, f"report_soundwaves_{group_key_str}.txt"), 'w') as f_report:
                f_report.write(report)
        except Exception as e:
            print(f"Could not generate classification report for {group_key_str}: {e}")

def run_acoustic_reasoning_analysis():
    print("\nAnalyzing Acoustic Reasoning (Sound Waves) Experiment...")
    df_results = load_acoustic_results()
    if df_results.empty:
        return

    df_scored = score_acoustic_responses(df_results)
    analyze_acoustic_accuracy(df_scored)
    print(f"Acoustic Reasoning analysis outputs saved in: {ANALYSIS_OUTPUT_DIR}")

def run_acoustic_reasoning_experiment():
    """Runs the acoustic reasoning experiment analysis."""
    run_acoustic_reasoning_analysis()

if __name__ == '__main__':
    run_acoustic_reasoning_analysis()
    print("\nAcoustic Reasoning (Sound Waves) Experiment Analysis Complete.") 