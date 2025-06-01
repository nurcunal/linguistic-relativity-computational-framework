import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial import procrustes
import json
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
import umap
import concurrent.futures

# Helper function to parse temperature and proficiency from config_id for METADATA
def parse_config_details(config_id):
    details = {"temperature": "unknown", "proficiency_from_id": "unknown", "system_prompt": "unknown"}
    # Temperature (e.g., 0T, 1.5T)
    temp_match = re.search(r"(\d+\.?\d*T)", config_id)
    if temp_match:
        numeric_temp_str = temp_match.group(1)[:-1]
        try: details["temperature"] = float(numeric_temp_str)
        except ValueError: details["temperature"] = "error_parsing_temp"
    else:
        temp_match_fallback = re.search(r"[Tt]emp(?:erature)?_?(\d+\.?\d*)", config_id)
        if temp_match_fallback:
            try: details["temperature"] = float(temp_match_fallback.group(1))
            except ValueError: details["temperature"] = "error_parsing_temp_fallback"

    # Proficiency from ID (e.g., profB1, P_C2) - for metadata only
    prof_id_match = re.search(r"[Pp]rof(?:iciency)?_?([A-Ca-c][1-2])", config_id)
    if prof_id_match:
        details["proficiency_from_id"] = prof_id_match.group(1).upper()

    # System Prompt (e.g., -BB-, -TN-, -ME- preceding temperature)
    # Looks for XX (two uppercase letters) or XXX (three) between hyphens, before a temperature string if present
    # Example: GG2-BB-0T, XG3M-ME-1T, SOME-LONGER-PROMPT-TN-0T (less likely with current IDs but more robust)
    # This regex tries to find a capitalized code like BB, TN, ME.
    # It looks for a hyphen, then capital letters, then a hyphen followed by a digit and T (temperature).
    prompt_match = re.search(r"-([A-Z]{2,3})-(?=\d+\.?\d*T)", config_id) 
    if not prompt_match: # Fallback if temp is not directly after, or different format
        # Try to find known prompts if regex is too strict or format varies.
        # This is a simpler heuristic looking for known patterns if the above fails.
        if "-BB-" in config_id: details["system_prompt"] = "BB"
        elif "-TN-" in config_id: details["system_prompt"] = "TN"
        elif "-ME-" in config_id: details["system_prompt"] = "ME"
        # Add other known prompt codes if necessary
    else:
        details["system_prompt"] = prompt_match.group(1)
        
    return details

# Helper function to calculate Cohen's d
def cohen_d(group1, group2):
    group1 = np.array(group1)[~np.isnan(group1)]
    group2 = np.array(group2)[~np.isnan(group2)]
    if len(group1) < 2 or len(group2) < 2: return np.nan
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    if n1 + n2 -2 == 0: return np.nan
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    if pooled_std == 0: return np.nan
    # Ensure Cohen's D is always positive (absolute magnitude)
    return abs((mean1 - mean2) / pooled_std)

# Utility functions to find and load LLM configurations
def find_llm_configurations(base_dir="api_generations"):
    configs = []
    for provider in os.listdir(base_dir):
        provider_path = os.path.join(base_dir, provider)
        if not os.path.isdir(provider_path) or provider.startswith('.'): continue
        for model in os.listdir(provider_path):
            model_path = os.path.join(provider_path, model)
            if not os.path.isdir(model_path) or model.startswith('.'): continue
            for config_dir_name in os.listdir(model_path):
                config_path = os.path.join(model_path, config_dir_name)
                if os.path.isdir(config_path) and not config_dir_name.startswith('.'):
                    configs.append({
                        "provider": provider, "model": model,
                        "config_id": config_dir_name, "path": config_path
                    })
    return configs

def load_embedding_results(config, result_type="language_distances", 
                           embedding_model_name="paraphrase-multilingual-mpnet-base-v2", 
                           config_id_prefix="PMMB-"):
    provider = config["provider"]
    model_api_name = config["model"]
    original_config_id = config["config_id"]
    # Minimal prints to confirm file loading status
    print(f"LOAD_ATTEMPT: Config {original_config_id}")

    temp_dir_match = re.search(r"(\d+\.?\d*T)", original_config_id)
    if not temp_dir_match:
        temp_dir_match_fallback = re.search(r"temp_?(\d+\.?\d*)", original_config_id, re.IGNORECASE)
        if temp_dir_match_fallback:
            num_part = temp_dir_match_fallback.group(1)
            temperature_dir_str = f"{num_part}T"
        else:
            print(f"  LOAD_ERROR (L_E_R): Could not extract temp dir from '{original_config_id}'.")
            return None 
    else:
        temperature_dir_str = temp_dir_match.group(1)

    prefixed_config_id_folder_name = config_id_prefix + original_config_id
    base_path_to_config_specific_folder = os.path.join(
        "embedding_analysis", embedding_model_name, provider, 
        model_api_name, temperature_dir_str, prefixed_config_id_folder_name
    )
    file_path = None
    if result_type == "language_distances":
        file_name = f"all_proficiencies_consolidated_{original_config_id}.csv"
        file_path = os.path.join(base_path_to_config_specific_folder, "02_distances", "comprehensive_analysis", file_name)
    else:
        print(f"  LOAD_ERROR (L_E_R): Unknown result_type '{result_type}' for {original_config_id}.")
        return None
    
    if file_path and os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            print(f"  LOAD_SUCCESS (L_E_R): Loaded {original_config_id}, Shape: {df.shape}")
            return df
        except Exception as e_read:
            print(f"  LOAD_ERROR (L_E_R): Failed to read {file_path} for {original_config_id}: {e_read}")
            return None
    else:
        if file_path: print(f"  LOAD_ERROR (L_E_R): File not found at: {file_path} for {original_config_id}")
        return None

# Comparison metrics - Procrustes is effectively disabled
def procrustes_alignment_score(matrix1, matrix2): return np.nan 

def interpret_cohens_d(d_value):
    """Interpret Cohen's D value into qualitative categories."""
    if pd.isna(d_value):
        return "N/A (insufficient data)"
    abs_d = abs(d_value)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"

def compute_distance_correlation(array1_1d, array2_1d):
    arr1 = np.array(array1_1d).flatten()
    arr2 = np.array(array2_1d).flatten()
    min_len = min(len(arr1), len(arr2))
    if min_len < 2:
        # print(f"DEBUG compute_corr: Arrays too short before common length. len1: {len(arr1)}, len2: {len(arr2)}")
        return {"correlation": np.nan, "p_value": np.nan}
    arr1_common, arr2_common = arr1[:min_len], arr2[:min_len]
    valid_mask = ~np.isnan(arr1_common) & ~np.isnan(arr2_common)
    arr1_final, arr2_final = arr1_common[valid_mask], arr2_common[valid_mask]
    if len(arr1_final) < 2:
        print(f"DEBUG compute_corr: Not enough valid (non-NaN) pairs after alignment. Original lengths: ({len(arr1)}, {len(arr2)}), Min_len: {min_len}, Valid_pairs: {len(arr1_final)}")
        return {"correlation": np.nan, "p_value": np.nan}
    try:
        r, p = stats.pearsonr(arr1_final, arr2_final)
        return {"correlation": r, "p_value": p}
    except ValueError: 
        # print(f"DEBUG compute_corr: ValueError in pearsonr. Valid_pairs: {len(arr1_final)}")
        return {"correlation": np.nan, "p_value": np.nan}
    except Exception as e: 
        # print(f"DEBUG compute_corr: Other error in pearsonr: {e}")
        return {"correlation": np.nan, "p_value": np.nan}

def create_config_comparison_data(configs_with_processed_data):
    n_configs = len(configs_with_processed_data)
    correlation_matrix = np.full((n_configs, n_configs), np.nan)
    for i in range(n_configs):
        correlation_matrix[i,i] = 1.0
        for j in range(i + 1, n_configs):
            if "all_distance_values" in configs_with_processed_data[i] and \
               "all_distance_values" in configs_with_processed_data[j]:
                array1 = configs_with_processed_data[i]["all_distance_values"]
                array2 = configs_with_processed_data[j]["all_distance_values"]
                corr_result = compute_distance_correlation(array1, array2)
                correlation_matrix[i, j] = correlation_matrix[j, i] = corr_result["correlation"]
    return {"correlation": correlation_matrix, "configs": configs_with_processed_data}

def calculate_pairwise_stats_from_matrix(configs_with_processed_data, correlation_matrix):
    pairs = []
    n_configs = len(configs_with_processed_data)
    for i in range(n_configs):
        for j in range(i + 1, n_configs):
            config1, config2 = configs_with_processed_data[i], configs_with_processed_data[j]
            correlation = correlation_matrix[i, j]
            p_value = np.nan
            if not np.isnan(correlation) and "all_distance_values" in config1 and "all_distance_values" in config2:
                corr_details = compute_distance_correlation(config1["all_distance_values"], config2["all_distance_values"])
                p_value = corr_details["p_value"]
            pairs.append({
                "config1": config1['config_id'],
                "config2": config2['config_id'],
                "correlation": correlation if not np.isnan(correlation) else None,
                "p_value": p_value if not np.isnan(p_value) else None,
                "is_significant": (p_value < 0.05) if p_value is not None and not np.isnan(p_value) else False
            })
    return pd.DataFrame(pairs) if pairs else pd.DataFrame(columns=["config1", "config2", "correlation", "p_value", "is_significant"])

# Visualization functions - kept as is, assuming they work with the structure of plot_data_for_viz
def plot_config_distance_heatmap(result_data, metric="distance", title=None):
    matrix = result_data[metric]
    configs = result_data["configs"]
    labels = [c['config_id'] for c in configs]
    fig = plt.figure(figsize=(max(12, len(labels) * 0.5), max(10, len(labels) * 0.4)))
    
    current_cmap = "viridis_r" # Default for distances
    vmin, vmax = 0.0, None    # Default vmin for distance

    if metric == "distance":
        # For a distance matrix where 0 is identical (e.g., 1-correlation)
        actual_matrix_for_scaling = matrix[~np.eye(matrix.shape[0], dtype=bool)].flatten() # Off-diagonal elements
        actual_matrix_for_scaling = actual_matrix_for_scaling[~np.isnan(actual_matrix_for_scaling)]
        if actual_matrix_for_scaling.size > 0:
            vmax = np.max(actual_matrix_for_scaling)
            if vmax == 0: vmax = 0.1 # Avoid all same color if all off-diagonals are 0
        else:
            vmax = 0.1 # Fallback if matrix is all NaNs or single element
        if vmin == vmax and vmin == 0 : vmax = 0.1 # ensure some range if all values are 0
        if vmin > vmax : vmin=vmax - 0.1 if vmax >0 else 0 # ensure vmin <= vmax

    elif metric == "correlation": 
        current_cmap = "coolwarm" 
        vmin, vmax = -1.0, 1.0
    
    sns.heatmap(matrix, annot=True, cmap=current_cmap, 
                xticklabels=labels, yticklabels=labels, 
                fmt=".2f", vmin=vmin, vmax=vmax, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    plt.title(title if title else f"Configuration Comparison - {metric.capitalize()}")
    plt.tight_layout()
    return fig

def plot_config_umap(result_data, metric="distance", n_neighbors=5, min_dist=0.1):
    matrix = result_data[metric]
    configs = result_data["configs"]
    if matrix.shape[0] <= 1 or np.all(np.isnan(matrix)):
        print("UMAP cannot be generated: matrix is too small or all NaNs.")
        return go.Figure()
    
    df = pd.DataFrame([{
        "provider": c["provider"], "model": c["model"], "config_id": c["config_id"],
        "label": c['config_id']
    } for c in configs])
    
    effective_n_neighbors = min(n_neighbors, matrix.shape[0] - 1) if matrix.shape[0] > 1 else 1 
    if effective_n_neighbors <=0: effective_n_neighbors = 1

    reducer = umap.UMAP(n_neighbors=effective_n_neighbors, min_dist=min_dist, metric="precomputed", random_state=42)
    embedding = reducer.fit_transform(matrix)
    df["x"], df["y"] = embedding[:, 0], embedding[:, 1]
    fig = px.scatter(df, x="x", y="y", color="provider", symbol="model", text="label", size_max=10,
                     title=f"UMAP Projection of Configuration {metric.capitalize()}",
                     hover_data=["provider", "model", "config_id"])
    fig.update_traces(textposition="top center")
    fig.update_layout(height=800, width=1000)
    return fig

def calculate_intra_config_proficiency_cohens_d(config_item):
    """Calculate Cohen's D between proficiency levels within a single configuration."""
    if "distance_values_by_proficiency" not in config_item or not config_item["distance_values_by_proficiency"]:
        return {}

    prof_levels = sorted(list(config_item["distance_values_by_proficiency"].keys()))
    cohens_d_results = {}

    for i in range(len(prof_levels)):
        for j in range(i + 1, len(prof_levels)):
            prof1_key = prof_levels[i]
            prof2_key = prof_levels[j]

            data1 = config_item["distance_values_by_proficiency"].get(prof1_key, np.array([]))
            data2 = config_item["distance_values_by_proficiency"].get(prof2_key, np.array([]))

            # cohen_d() now returns absolute value or NaN directly
            effect_size_to_store = cohen_d(data1, data2)
            cohens_d_results[f"{prof1_key}_vs_{prof2_key}"] = effect_size_to_store
    return cohens_d_results

def canonical_lang_pair(lang1, lang2):
    """Create a canonical string representation for a language pair."""
    # Ensure inputs are strings before sorting, to handle potential non-string data if columns are mixed.
    return "--".join(sorted([str(lang1).strip(), str(lang2).strip()]))

# Helper worker function for parallelizing Cohen's D per language pair
def _worker_calc_lang_pair_prof_cohens_d(args_tuple):
    lang_pair, prof_data, process_id_for_debug = args_tuple # Added process_id for potential debug
    # print(f"WORKER LP {process_id_for_debug if process_id_for_debug else '-'}: Processing lang pair {lang_pair}")
    
    result_for_lang_pair = {}
    valid_prof_levels = {p: d for p, d in prof_data.items() if len(d) >= 2}
    prof_levels = sorted(list(valid_prof_levels.keys()))

    if len(prof_levels) < 2:
        return lang_pair, {} # Return empty dict if no comparisons can be made

    for i in range(len(prof_levels)):
        for j in range(i + 1, len(prof_levels)):
            prof1_key, prof2_key = prof_levels[i], prof_levels[j]
            data1, data2 = np.array(valid_prof_levels[prof1_key]), np.array(valid_prof_levels[prof2_key])
            
            # cohen_d() now returns absolute value or NaN directly
            effect_size_to_store = cohen_d(data1, data2)
            interpretation = interpret_cohens_d(effect_size_to_store) 
            
            result_for_lang_pair[f"{prof1_key}_vs_{prof2_key}"] = effect_size_to_store
            result_for_lang_pair[f"{prof1_key}_vs_{prof2_key}_interpretation"] = interpretation
    
    if not result_for_lang_pair:
        return lang_pair, {} # Return empty if no valid comparisons were made
    return lang_pair, result_for_lang_pair

def calculate_language_pair_proficiency_cohens_d(master_aggregated_data_by_lang_pair_prof, num_workers=12):
    # This function now RECEIVES the aggregated data
    print(f"-- Calculating Language Pair Proficiency Cohen's D (using up to {num_workers} workers on pre-aggregated data) --")
    
    if not master_aggregated_data_by_lang_pair_prof:
        print("No data aggregated for language pair proficiency Cohen's D.")
        return {}

    final_cohens_d_results = {}
    lang_pair_items_to_process = [
        (lang_pair, prof_data, None) 
        for lang_pair, prof_data in master_aggregated_data_by_lang_pair_prof.items()
    ]

    if not lang_pair_items_to_process:
        print("No language pairs with sufficient data to process for Cohen's D.")
        return {}

    print(f"Pre-aggregated data for {len(lang_pair_items_to_process)} language pairs. Starting parallel Cohen's D calculation...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        parallel_results = list(executor.map(_worker_calc_lang_pair_prof_cohens_d, lang_pair_items_to_process))
    
    for lang_pair_key, result_for_pair in parallel_results:
        if result_for_pair: 
            final_cohens_d_results[lang_pair_key] = result_for_pair
            
    return final_cohens_d_results

# Worker function for parallel processing
def process_single_config(config_item_tuple):
    config_item, distance_column_name, config_id_prefix, embedding_model_name = config_item_tuple
    config_id_for_debug = config_item['config_id'] # For easier reference in prints
    # Limit very verbose prints to a known-good config if possible, or first few.
    # Let's pick one that worked before: 'GG2-BB-2T-24L-5P-144N-1AP'
    should_print_detailed_debug = (config_id_for_debug == 'GG2-BB-2T-24L-5P-144N-1AP')

    if should_print_detailed_debug: print(f"PSC_DEBUG: Worker starting for {config_id_for_debug}")
    
    parsed_details = parse_config_details(config_item["config_id"])
    config_item.update(parsed_details)
    lang_dist_df_original = load_embedding_results(config_item, "language_distances", 
                                          embedding_model_name=embedding_model_name, 
                                          config_id_prefix=config_id_prefix)
    
    if lang_dist_df_original is None: 
        if should_print_detailed_debug: print(f"PSC_DEBUG: {config_id_for_debug} - lang_dist_df_original is None. Returning (None, None).")
        return None, None 
    config_item["raw_df"] = lang_dist_df_original
    if should_print_detailed_debug: print(f"PSC_DEBUG: {config_id_for_debug} - raw_df assigned. Shape: {lang_dist_df_original.shape}")

    local_aggregated_lp_data = {}
    df_for_lp_aggregation = lang_dist_df_original.copy() 
    required_cols_lp = [distance_column_name, "Proficiency", "Language1", "Language2"]
    if not all(col in df_for_lp_aggregation.columns for col in required_cols_lp):
        if should_print_detailed_debug: print(f"PSC_DEBUG: {config_id_for_debug} - LP Agg - Missing required columns. Has: {df_for_lp_aggregation.columns.tolist()}")
        # Don't return yet, main processing might still work. local_aggregated_lp_data remains empty.
    else:
        df_for_lp_aggregation[distance_column_name] = pd.to_numeric(df_for_lp_aggregation[distance_column_name], errors='coerce')
        df_for_lp_aggregation.dropna(subset=[distance_column_name, "Proficiency", "Language1", "Language2"], inplace=True)
        if not df_for_lp_aggregation.empty:
            try:
                df_for_lp_aggregation["LanguagePair"] = df_for_lp_aggregation.apply(
                    lambda row: canonical_lang_pair(row["Language1"], row["Language2"]), axis=1)
                for _, row in df_for_lp_aggregation.iterrows():
                    lang_pair, prof, dist = row["LanguagePair"], str(row["Proficiency"]).strip(), row[distance_column_name]
                    local_aggregated_lp_data.setdefault(lang_pair, {}).setdefault(prof, []).append(dist)
            except Exception as e_lp_agg:
                 if should_print_detailed_debug: print(f"PSC_DEBUG: {config_id_for_debug} - LP Agg - Error: {e_lp_agg}")

    processed_config_successfully = False
    try:
        df_for_main_processing = lang_dist_df_original.copy()
        if df_for_main_processing.empty: 
            if should_print_detailed_debug: print(f"PSC_DEBUG: {config_id_for_debug} - MainProc - df_for_main_processing is empty.")
            raise ValueError("DataFrame for main processing is empty")

        if distance_column_name not in df_for_main_processing.columns:
            if should_print_detailed_debug: print(f"PSC_DEBUG: {config_id_for_debug} - MainProc - Distance column '{distance_column_name}' not found. Columns: {df_for_main_processing.columns.tolist()}")
            raise ValueError(f"Distance column '{distance_column_name}' not found")
            
        if "Proficiency" not in df_for_main_processing.columns:
            if should_print_detailed_debug: print(f"PSC_DEBUG: {config_id_for_debug} - MainProc - 'Proficiency' column not found. Columns: {df_for_main_processing.columns.tolist()}")
            raise ValueError("'Proficiency' column not found")
        if should_print_detailed_debug: print(f"PSC_DEBUG: {config_id_for_debug} - MainProc - All required columns present.")

        config_item["distance_values_by_proficiency"] = {}
        all_distances_for_this_config = []
        df_for_main_processing[distance_column_name] = pd.to_numeric(df_for_main_processing[distance_column_name], errors='coerce')
        if should_print_detailed_debug: print(f"PSC_DEBUG: {config_id_for_debug} - MainProc - Distance column converted to numeric.")

        unique_prof_levels = df_for_main_processing["Proficiency"].unique()
        if should_print_detailed_debug: print(f"PSC_DEBUG: {config_id_for_debug} - MainProc - Unique proficiencies in df_for_main_processing: {unique_prof_levels}")

        for prof_level_orig in unique_prof_levels:
            if pd.isna(prof_level_orig): continue
            prof_level = str(prof_level_orig).strip()
            prof_df = df_for_main_processing[df_for_main_processing["Proficiency"] == prof_level_orig]
            distances = prof_df[distance_column_name].dropna().values.astype(float)
            if should_print_detailed_debug: print(f"PSC_DEBUG: {config_id_for_debug} - MainProc - Prof: {prof_level}, Num distances extracted: {distances.size}")
            if distances.size > 0:
                config_item["distance_values_by_proficiency"][prof_level] = distances
                all_distances_for_this_config.extend(distances)
        
        if all_distances_for_this_config:
            config_item["all_distance_values"] = np.array(all_distances_for_this_config)
            config_item["intra_config_prof_cohens_d"] = calculate_intra_config_proficiency_cohens_d(config_item)
            processed_config_successfully = True
            if should_print_detailed_debug: print(f"PSC_DEBUG: {config_id_for_debug} - MainProc - SUCCESS, all_distances size: {len(all_distances_for_this_config)}.")
        else:
            if should_print_detailed_debug: print(f"PSC_DEBUG: {config_id_for_debug} - MainProc - FAILED, all_distances_for_this_config is empty.")

    except Exception as e:
        print(f"WORKER ERROR processing main distances for {config_id_for_debug}: {e}. Main processing failed.")

    if processed_config_successfully:
        if should_print_detailed_debug: print(f"PSC_DEBUG: {config_id_for_debug} - Returning successfully processed config_item and lp_data.")
        return config_item, local_aggregated_lp_data
    else:
        if should_print_detailed_debug: print(f"PSC_DEBUG: {config_id_for_debug} - Main processing failed. Returning (None, lp_data). config_item had keys: {list(config_item.keys())}")
        return None, local_aggregated_lp_data

def calculate_language_deviation_analysis(configs_with_processed_data, distance_column_name):
    print("-- Calculating Language Deviation from Mean Context Cohen's D --")
    overall_results = {}

    for config_idx, config in enumerate(configs_with_processed_data):
        config_id = config['config_id']
        # Limit verbose debugging to the first config to avoid excessive output initially
        is_first_config_for_debug = (config_idx == 0) 

        if is_first_config_for_debug:
            print(f"DEBUG_DEV: Processing deviations for config: {config_id}")
        overall_results[config_id] = {}

        if "raw_df" not in config or config["raw_df"] is None or config["raw_df"].empty:
            if is_first_config_for_debug: print(f"  DEBUG_DEV: Skipping {config_id}: missing raw_df or empty.")
            continue
        
        df_original = config["raw_df"].copy()
        required_cols = [distance_column_name, "Proficiency", "Language1", "Language2"]
        if not all(col in df_original.columns for col in required_cols):
            if is_first_config_for_debug: print(f"  DEBUG_DEV: Skipping {config_id}: missing one of {required_cols}.")
            continue
        
        df_original[distance_column_name] = pd.to_numeric(df_original[distance_column_name], errors='coerce')
        df_original.dropna(subset=[distance_column_name, "Proficiency", "Language1", "Language2"], inplace=True)

        if df_original.empty:
            if is_first_config_for_debug: print(f"  DEBUG_DEV: Skipping {config_id}: DataFrame became empty after initial dropna.")
            continue

        unique_langs_l1 = df_original["Language1"].astype(str).str.strip().unique()
        unique_langs_l2 = df_original["Language2"].astype(str).str.strip().unique()
        all_unique_languages_in_config = sorted(list(set(unique_langs_l1) | set(unique_langs_l2)))
        if is_first_config_for_debug: print(f"  DEBUG_DEV: Config {config_id}, Unique languages: {len(all_unique_languages_in_config)}")

        for prof_level_orig in df_original["Proficiency"].unique():
            prof_level = str(prof_level_orig).strip()
            if is_first_config_for_debug:
                print(f"    DEBUG_DEV: Processing proficiency: {prof_level} for {config_id}")
            overall_results[config_id][prof_level] = {}

            df_prof = df_original[df_original["Proficiency"] == prof_level_orig].copy()
            if is_first_config_for_debug: print(f"      DEBUG_DEV: df_prof shape for {prof_level}: {df_prof.shape}")
            if df_prof.empty: continue

            for lang_x in all_unique_languages_in_config:
                # Define conditions for S1 and S2
                cond_s1 = (df_prof["Language1"].astype(str).str.strip() == lang_x) | \
                          (df_prof["Language2"].astype(str).str.strip() == lang_x)
                cond_s2 = (df_prof["Language1"].astype(str).str.strip() != lang_x) & \
                          (df_prof["Language2"].astype(str).str.strip() != lang_x)

                s1_df = df_prof[cond_s1]
                s2_df = df_prof[cond_s2]
                s1_distances = s1_df[distance_column_name].values # .values implicitly handles earlier dropna for this col
                s2_distances = s2_df[distance_column_name].values

                if is_first_config_for_debug:
                    print(f"        DEBUG_DEV: Lang={lang_x}, Prof={prof_level}")
                    print(f"          S1 (involves {lang_x}): selected {len(s1_df)} rows, resulting distances size: {s1_distances.size}")
                    print(f"          S2 (not involve {lang_x}): selected {len(s2_df)} rows, resulting distances size: {s2_distances.size}")

                effect_size = cohen_d(s1_distances, s2_distances)
                interpretation = interpret_cohens_d(effect_size)
                
                if not pd.isna(effect_size): # Only add if Cohen's D was calculable
                    overall_results[config_id][prof_level][lang_x] = effect_size
                    overall_results[config_id][prof_level][f"{lang_x}_interpretation"] = interpretation
                elif is_first_config_for_debug: # If effect_size is NaN, print why for the first config
                    print(f"          DEBUG_DEV: Cohen's D is NaN for Lang={lang_x}, Prof={prof_level}. S1 size: {s1_distances.size}, S2 size: {s2_distances.size}")
            
            if not overall_results[config_id][prof_level]:
                if is_first_config_for_debug: print(f"    DEBUG_DEV: No deviation data for {prof_level} in {config_id}, removing prof level entry.")
                del overall_results[config_id][prof_level]
        
        if not overall_results[config_id]:
            if is_first_config_for_debug: print(f"  DEBUG_DEV: No deviation data for any proficiency in {config_id}, removing config entry.")
            del overall_results[config_id]
            
    return overall_results

# Predefined list of languages for Table 14 output order
TABLE14_LANGUAGES_ORDER = [
    "Arabic", "Bengali", "German", "Spanish", "Persian", "French", "Irish", 
    "Hebrew", "Hindi", "Italian", "Japanese", "Korean", "Latvian", "Polish", 
    "Pashto", "Portuguese", "Russian", "Albanian", "Swahili", "Thai", 
    "Turkish", "Urdu", "Vietnamese", "Mandarin"
]
# Helper to normalize language names from CSV (e.g., "AR # Arabic" -> "Arabic")
def normalize_language_name(lang_csv_name):
    if pd.isna(lang_csv_name): return None
    parts = str(lang_csv_name).split("#")
    return parts[-1].strip() if len(parts) > 0 else str(lang_csv_name).strip()

def generate_table14_data(configs_with_processed_data, distance_column_name):
    print("-- Generating data for Table 14 (Cross-Model Language Mean Distances, Perfect, Temp=0) --")
    table_data_intermediate = {lang: {} for lang in TABLE14_LANGUAGES_ORDER}
    all_models_in_table = set()
    temp0_configs = [c for c in configs_with_processed_data if c.get("temperature") == 0.0]
    if not temp0_configs:
        print("  INFO_T14: No Temp=0.0 configurations found to generate Table 14 data.")
        return pd.DataFrame(index=TABLE14_LANGUAGES_ORDER, columns=sorted(list(all_models_in_table)) if all_models_in_table else []) 
    for config_idx, config in enumerate(temp0_configs):
        config_id = config['config_id']
        model_name = config['model']
        is_gemini_for_detailed_debug = (model_name == 'gemini-2.0-flash') 
        should_print_gemini_stats = is_gemini_for_detailed_debug and (config_idx < 2) 
        if "raw_df" not in config or config["raw_df"] is None or config["raw_df"].empty: continue
        df_original = config["raw_df"].copy()
        required_cols = [distance_column_name, "Proficiency", "Language1", "Language2"]
        if not all(col in df_original.columns for col in required_cols): continue
        df_original[distance_column_name] = pd.to_numeric(df_original[distance_column_name], errors='coerce')
        df_original.dropna(subset=[distance_column_name, "Proficiency", "Language1", "Language2"], inplace=True)
        if df_original.empty: continue
        df_perfect_temp0 = df_original[df_original["Proficiency"].astype(str).str.strip().str.lower() == "perfect"].copy()
        if df_perfect_temp0.empty: continue
        all_models_in_table.add(model_name)
        df_perfect_temp0["NormLang1"] = df_perfect_temp0["Language1"].apply(normalize_language_name)
        df_perfect_temp0["NormLang2"] = df_perfect_temp0["Language2"].apply(normalize_language_name)
        current_unique_norm_langs = sorted(list(set(df_perfect_temp0["NormLang1"].unique()) | set(df_perfect_temp0["NormLang2"].unique())))
        current_unique_norm_langs = [l for l in current_unique_norm_langs if l is not None]
        if should_print_gemini_stats:
            print(f"  DEBUG_GEMINI (Config {config_id}, Model {model_name}, Temp 0, Perfect Prof):")
            print(f"    DataFrame df_perfect_temp0 shape: {df_perfect_temp0.shape}")
        for lang_x_norm in current_unique_norm_langs:
            if lang_x_norm not in TABLE14_LANGUAGES_ORDER: continue
            s1_distances = df_perfect_temp0[(df_perfect_temp0["NormLang1"] == lang_x_norm) | (df_perfect_temp0["NormLang2"] == lang_x_norm)][distance_column_name].values
            s2_distances = df_perfect_temp0[(df_perfect_temp0["NormLang1"] != lang_x_norm) & (df_perfect_temp0["NormLang2"] != lang_x_norm)][distance_column_name].values
            effect_size = cohen_d(s1_distances, s2_distances) 
            interpretation = interpret_cohens_d(effect_size)
            value_to_store = f"{effect_size:.3f} ({interpretation})" if not pd.isna(effect_size) else "N/A"
            table_data_intermediate[lang_x_norm][model_name] = value_to_store
            if should_print_gemini_stats and lang_x_norm in ["Arabic", "German", "Spanish"]:
                m1, std1, n1 = (np.mean(s1_distances), np.std(s1_distances, ddof=1), s1_distances.size) if s1_distances.size >=1 else (np.nan, np.nan, 0)
                m2, std2, n2 = (np.mean(s2_distances), np.std(s2_distances, ddof=1), s2_distances.size) if s2_distances.size >=1 else (np.nan, np.nan, 0)
                print(f"      Lang: {lang_x_norm}")
                print(f"        S1 (involves {lang_x_norm}): N={n1}, Mean={m1:.4f}, STD={std1:.4f}")
                print(f"        S2 (not involve {lang_x_norm}): N={n2}, Mean={m2:.4f}, STD={std2:.4f}")
                if n1>=2 and n2>=2 and (n1+n2-2 > 0):
                    pooled_std_debug = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)) if (n1 > 1 or n2 > 1) else np.nan
                    print(f"        Pooled STD: {pooled_std_debug:.4f}, Raw Cohen's D: {effect_size:.4f} (abs already applied in cohen_d)")
                else:
                    print(f"        Pooled STD: N/A (due to N1 or N2 < 2), Raw Cohen's D: {effect_size} (abs already applied)")
    df_table14 = pd.DataFrame.from_dict(table_data_intermediate, orient='index', columns=sorted(list(all_models_in_table)) if all_models_in_table else [])
    df_table14 = df_table14.reindex(TABLE14_LANGUAGES_ORDER)
    return df_table14

def generate_table15_data(configs_with_processed_data, distance_column_name, 
                          target_temperature=0.0, target_system_prompt=None):
    title_prompt_part = f"Prompt '{target_system_prompt}'" if target_system_prompt else "All Prompts"
    print(f"-- Generating data for Table 15 (Cross-Proficiency Lang Mean Dists, Temp={target_temperature}, {title_prompt_part}, All LLMs Aggregated) --")
    
    aggregated_s1_s2_data = {lang: {} for lang in TABLE14_LANGUAGES_ORDER}
    all_proficiencies_in_table = set()

    # 1. Filter for Target Temperature and Target System Prompt configurations
    filtered_configs = []
    for c in configs_with_processed_data:
        temp_match = (c.get("temperature") == target_temperature)
        prompt_meta = c.get("system_prompt", "unknown") # Get prompt from parsed_details
        prompt_match = (target_system_prompt is None or prompt_meta == target_system_prompt)
        if temp_match and prompt_match:
            filtered_configs.append(c)

    if not filtered_configs:
        print(f"  INFO_T15: No configs found for Temp={target_temperature} & Prompt='{target_system_prompt}'.")
        return pd.DataFrame(index=TABLE14_LANGUAGES_ORDER, columns=[])
    print(f"  INFO_T15: Found {len(filtered_configs)} configs for Temp={target_temperature} & Prompt='{target_system_prompt}'.")

    # 2. Aggregate S1 and S2 distances for each language and proficiency from these configs
    for config in filtered_configs:
        if "raw_df" not in config or config["raw_df"] is None or config["raw_df"].empty: continue
        df_original = config["raw_df"].copy()
        required_cols = [distance_column_name, "Proficiency", "Language1", "Language2"]
        if not all(col in df_original.columns for col in required_cols): continue
        df_original[distance_column_name] = pd.to_numeric(df_original[distance_column_name], errors='coerce')
        df_original.dropna(subset=[distance_column_name, "Proficiency", "Language1", "Language2"], inplace=True)
        if df_original.empty: continue
        df_original["NormLang1"] = df_original["Language1"].apply(normalize_language_name)
        df_original["NormLang2"] = df_original["Language2"].apply(normalize_language_name)
        for prof_level_orig in df_original["Proficiency"].unique():
            prof_level_str = str(prof_level_orig).strip()
            if pd.isna(prof_level_str) or prof_level_str == '': continue
            all_proficiencies_in_table.add(prof_level_str)
            df_prof_specific = df_original[df_original["Proficiency"] == prof_level_orig]
            if df_prof_specific.empty: continue
            for lang_x_norm in TABLE14_LANGUAGES_ORDER:
                aggregated_s1_s2_data[lang_x_norm].setdefault(prof_level_str, {"s1": [], "s2": []})
                s1_rows = df_prof_specific[(df_prof_specific["NormLang1"] == lang_x_norm) | (df_prof_specific["NormLang2"] == lang_x_norm)]
                s2_rows = df_prof_specific[(df_prof_specific["NormLang1"] != lang_x_norm) & (df_prof_specific["NormLang2"] != lang_x_norm)]
                aggregated_s1_s2_data[lang_x_norm][prof_level_str]["s1"].extend(s1_rows[distance_column_name].tolist())
                aggregated_s1_s2_data[lang_x_norm][prof_level_str]["s2"].extend(s2_rows[distance_column_name].tolist())
    if not all_proficiencies_in_table: return pd.DataFrame(index=TABLE14_LANGUAGES_ORDER, columns=[])
    table_output_data = {lang: {} for lang in TABLE14_LANGUAGES_ORDER}
    sorted_prof_columns = sorted(list(all_proficiencies_in_table))
    for lang_x_norm in TABLE14_LANGUAGES_ORDER:
        for prof_level_str in sorted_prof_columns:
            s1s2_data = aggregated_s1_s2_data.get(lang_x_norm, {}).get(prof_level_str, None)
            if s1s2_data and s1s2_data["s1"] and s1s2_data["s2"]:
                s1_distances, s2_distances = np.array(s1s2_data["s1"]), np.array(s1s2_data["s2"])
                effect_size = cohen_d(s1_distances, s2_distances)
                interpretation = interpret_cohens_d(effect_size)
                table_output_data[lang_x_norm][prof_level_str] = f"{effect_size:.3f} ({interpretation})" if not pd.isna(effect_size) else "N/A"
            else: table_output_data[lang_x_norm][prof_level_str] = "N/A"
    df_table15 = pd.DataFrame.from_dict(table_output_data, orient='index', columns=sorted_prof_columns)
    df_table15 = df_table15.reindex(TABLE14_LANGUAGES_ORDER)
    return df_table15

# Main analysis function
def analyze_llm_configurations(output_dir="analysis_results/llm_comparison", 
                               distance_column_name="CosineDistance",
                               config_id_prefix="PMMB-",
                               embedding_model_name="paraphrase-multilingual-mpnet-base-v2",
                               num_workers=12):
    print(f"INFO: Using column '{distance_column_name}' from CSVs for distance values.")
    print(f"INFO: Using up to {num_workers} worker processes for initial config processing AND lang pair Cohen's D.")
    os.makedirs(output_dir, exist_ok=True)
    
    initial_configs = find_llm_configurations()
    if not initial_configs: print("No LLM configurations found!"); return
    print(f"Found {len(initial_configs)} LLM configurations to scan.")

    process_args = [
        (config_item.copy(), distance_column_name, config_id_prefix, embedding_model_name) 
        for config_item in initial_configs
    ]

    configs_with_processed_data = []
    master_aggregated_lp_data = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results_from_workers = list(executor.map(process_single_config, process_args))
    for config_result, local_lp_data in results_from_workers:
        if config_result is not None:
            configs_with_processed_data.append(config_result)
        if local_lp_data:
            for lang_pair, prof_map in local_lp_data.items():
                target_lp_map = master_aggregated_lp_data.setdefault(lang_pair, {})
                for prof, distances in prof_map.items():
                    target_prof_list = target_lp_map.setdefault(prof, [])
                    target_prof_list.extend(distances)

    if not configs_with_processed_data:
        print("FINAL_STATUS: No config data processed for ANY configuration. Aborting."); return
    print(f"FINAL_STATUS: Successfully processed main data for {len(configs_with_processed_data)} configs.")
    configs_with_processed_data.sort(key=lambda c: (c['provider'], c['model'], c['config_id']))

    comparison_data = create_config_comparison_data(configs_with_processed_data)
    correlation_matrix = comparison_data["correlation"]

    distance_matrix_for_plotting = 1 - correlation_matrix 
    np.fill_diagonal(distance_matrix_for_plotting, 0)
    plot_data_for_viz = {
        "distance": distance_matrix_for_plotting,
        "correlation": correlation_matrix,
        "procrustes": np.full((len(configs_with_processed_data), len(configs_with_processed_data)), np.nan),
        "configs": configs_with_processed_data 
    }
    try:
        heatmap_fig = plot_config_distance_heatmap(plot_data_for_viz, metric="distance", title="Config Distance Matrix (1 - Correlation)")
        heatmap_fig.savefig(os.path.join(output_dir, "config_distance_heatmap.png"))
        plt.close(heatmap_fig)
    except Exception as e: print(f"Error plotting distance heatmap: {e}")
    if not np.all(np.isnan(plot_data_for_viz["distance"])):
        try:
            umap_distance_matrix = plot_data_for_viz["distance"].copy()
            # Robust nanmax check
            nan_max_val = np.nanmax(umap_distance_matrix) if not np.all(np.isnan(umap_distance_matrix)) else 2.0
            if pd.isna(nan_max_val): nan_max_val = 2.0 # If nanmax is still nan (e.g. all NaNs), default to 2.0
            umap_distance_matrix[np.isnan(umap_distance_matrix)] = nan_max_val
            np.fill_diagonal(umap_distance_matrix, 0) 
            temp_plot_data_for_umap = plot_data_for_viz.copy()
            temp_plot_data_for_umap["distance"] = umap_distance_matrix
            umap_fig = plot_config_umap(temp_plot_data_for_umap, metric="distance") 
            umap_fig.write_html(os.path.join(output_dir, "config_all_distances_umap_projection.html"))
        except Exception as e: print(f"Error plotting UMAP: {e}")
    
    pairwise_stats_df = calculate_pairwise_stats_from_matrix(configs_with_processed_data, correlation_matrix)
    pairwise_stats_df.to_csv(os.path.join(output_dir, "pairwise_stats_all_distances.csv"), index=False)
    
    cohens_d_cross_attribute_results = {}
    attributes_to_compare = ["provider", "model", "temperature", "proficiency"]
    for attribute in attributes_to_compare:
        raw_cohens_d_abs = perform_cohens_d_group_comparisons(configs_with_processed_data, attribute, distance_column_name)
        interpreted_cohens_d = {f"{k}_interpretation": interpret_cohens_d(v_abs) for k, v_abs in raw_cohens_d_abs.items()}
        cohens_d_cross_attribute_results[f"cross_{attribute}"] = {**raw_cohens_d_abs, **interpreted_cohens_d}
        numeric_cohens_d_for_heatmap = {k: v for k,v in raw_cohens_d_abs.items() if not "_interpretation" in k}
        if numeric_cohens_d_for_heatmap:
            heatmap_title = f"Cohen's D: Cross-{attribute.capitalize()} Comparison"
            heatmap_path = os.path.join(output_dir, f"cohens_d_cross_{attribute}_heatmap.png")
            plot_cohens_d_heatmap(numeric_cohens_d_for_heatmap, heatmap_title, heatmap_path)
    
    intra_config_summary_cohens_d = {}
    for config in configs_with_processed_data:
        if "intra_config_prof_cohens_d" in config and config["intra_config_prof_cohens_d"]:
            config_key = config['config_id']
            raw_cohens_d_abs = config["intra_config_prof_cohens_d"]
            interpreted_cohens_d = {f"{k}_interpretation": interpret_cohens_d(v_abs) for k, v_abs in raw_cohens_d_abs.items()}
            intra_config_summary_cohens_d[config_key] = {**raw_cohens_d_abs, **interpreted_cohens_d}
    
    lang_pair_prof_cohens_d_results = calculate_language_pair_proficiency_cohens_d(master_aggregated_lp_data, num_workers)

    # New: Language Deviation from Mean Analysis
    language_deviation_results = calculate_language_deviation_analysis(configs_with_processed_data, distance_column_name)

    # Table 14 call (existing)
    if configs_with_processed_data: 
        df_table14 = generate_table14_data(configs_with_processed_data, distance_column_name)
        table14_csv_path = os.path.join(output_dir, "table14_lang_dev_perfect_temp0_cohens_d.csv")
        try: df_table14.to_csv(table14_csv_path); print(f"Saved Table 14 data to: {table14_csv_path}")
        except Exception as e: print(f"Error saving Table 14: {e}")
    else:
        print("Skipping Table 14/15 generation: no processed configs.")
        df_table14 = pd.DataFrame() # Ensure it exists for summary

    # Table 15 generation - for BB and TN prompts
    df_table15_bb = pd.DataFrame() # Initialize empty
    df_table15_tn = pd.DataFrame() # Initialize empty
    if configs_with_processed_data:
        df_table15_bb = generate_table15_data(configs_with_processed_data, distance_column_name, target_system_prompt="BB")
        table15_bb_csv_path = os.path.join(output_dir, "table15_lang_dev_cross_prof_temp0_BB_cohens_d.csv")
        try: df_table15_bb.to_csv(table15_bb_csv_path); print(f"Saved Table 15 (BB) data to: {table15_bb_csv_path}")
        except Exception as e: print(f"Error saving Table 15 (BB): {e}")

        df_table15_tn = generate_table15_data(configs_with_processed_data, distance_column_name, target_system_prompt="TN")
        table15_tn_csv_path = os.path.join(output_dir, "table15_lang_dev_cross_prof_temp0_TN_cohens_d.csv")
        try: df_table15_tn.to_csv(table15_tn_csv_path); print(f"Saved Table 15 (TN) data to: {table15_tn_csv_path}")
        except Exception as e: print(f"Error saving Table 15 (TN): {e}")
    else:
        # This else was part of Table 14, now covered above
        pass 

    valid_correlations = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
    valid_correlations = valid_correlations[~np.isnan(valid_correlations)]
    summary = {
        "info": f"Analysis based on '{distance_column_name}' column. Temp for metadata from config ID, Proficiency from CSV.",
        "num_configs_with_data": len(configs_with_processed_data),
        "total_configs_scanned": len(initial_configs),
        "mean_correlation_all_dists": np.mean(valid_correlations) if len(valid_correlations) > 0 else np.nan,
        "std_correlation_all_dists": np.std(valid_correlations) if len(valid_correlations) > 0 else np.nan,
        "min_correlation_all_dists": np.min(valid_correlations) if len(valid_correlations) > 0 else np.nan,
        "max_correlation_all_dists": np.max(valid_correlations) if len(valid_correlations) > 0 else np.nan,
        "significant_pairs_correlation": sum(pairwise_stats_df["is_significant"]) if not pairwise_stats_df.empty else 0,
        "total_comparable_pairs_correlation": len(pairwise_stats_df[pairwise_stats_df['correlation'].notna()]),
        "cohens_d_analysis": cohens_d_cross_attribute_results,
        "intra_config_proficiency_cohens_d": intra_config_summary_cohens_d,
        "language_pair_proficiency_cohens_d": lang_pair_prof_cohens_d_results,
        "language_deviation_from_mean_cohens_d": language_deviation_results,
        "table14_data_as_dict": df_table14.to_dict(orient='index') if not df_table14.empty else "No data for Table 14",
        "table15_data_BB_as_dict": df_table15_bb.to_dict(orient='index') if not df_table15_bb.empty else "No data for Table 15 (BB)",
        "table15_data_TN_as_dict": df_table15_tn.to_dict(orient='index') if not df_table15_tn.empty else "No data for Table 15 (TN)"
    }
    with open(os.path.join(output_dir, "summary_statistics.json"), "w") as f:
        json.dump(summary, f, indent=2, default=lambda x: None if pd.isna(x) else x)
    print("Analysis complete! Results saved to:", output_dir)

def perform_cohens_d_group_comparisons(configs_with_data, attribute_to_group_by, distance_column_name):
    print(f"-- Performing Cohen's D for attribute: {attribute_to_group_by} --")
    grouped_values = {}

    if attribute_to_group_by == "proficiency":
        for config in configs_with_data:
            if "distance_values_by_proficiency" in config:
                for prof_level, distances in config["distance_values_by_proficiency"].items():
                    if prof_level not in grouped_values: grouped_values[prof_level] = []
                    grouped_values[prof_level].extend(distances)
    else: 
        for config in configs_with_data:
            group_key = config.get(attribute_to_group_by)
            if isinstance(group_key, float) and np.isnan(group_key): group_key = "unknown"
            if group_key is None : group_key = "unknown"
            
            if group_key not in grouped_values: grouped_values[group_key] = []
            if "all_distance_values" in config and config["all_distance_values"].size > 0:
                grouped_values[group_key].extend(config["all_distance_values"])

    results = {}
    group_keys = list(grouped_values.keys())

    for i in range(len(group_keys)):
        for j in range(i + 1, len(group_keys)):
            key1, key2 = group_keys[i], group_keys[j]
            data1, data2 = grouped_values[key1], grouped_values[key2]
            
            # cohen_d() now returns absolute value or NaN directly
            effect_size_to_store = cohen_d(data1, data2)
            results[f"{key1}_vs_{key2}"] = effect_size_to_store
    return results

def plot_cohens_d_heatmap(data_dict, title, output_path, vmax_val=1.0):
    """Plots a heatmap for Cohen's D values stored in a dictionary format {key1_vs_key2: value}."""
    if not data_dict:
        print(f"No data for Cohen's D heatmap: {title}")
        return

    labels_set = set()
    for key in data_dict.keys():
        if "_interpretation" in key: continue
        parts = key.split("_vs_")
        if len(parts) == 2:
            labels_set.add(parts[0])
            labels_set.add(parts[1])
    
    sorted_labels = sorted(list(labels_set))
    if not sorted_labels or len(sorted_labels) < 2:
        print(f"Not enough labels to create a Cohen's D heatmap for {title}. Labels: {sorted_labels}")
        return

    matrix = pd.DataFrame(np.nan, index=sorted_labels, columns=sorted_labels)

    for key, value in data_dict.items():
        if "_interpretation" in key: continue
        parts = key.split("_vs_")
        if len(parts) == 2 and not pd.isna(value):
            if parts[0] in matrix.index and parts[1] in matrix.columns:
                 matrix.loc[parts[0], parts[1]] = value
            if parts[1] in matrix.index and parts[0] in matrix.columns: 
                 matrix.loc[parts[1], parts[0]] = value 
    
    np.fill_diagonal(matrix.values, 0) 

    fig = plt.figure(figsize=(max(8, len(sorted_labels) * 0.6), max(6, len(sorted_labels) * 0.5)))
    # Use a reversed sequential colormap (e.g., viridis_r) so 0 is light and higher values are darker.
    sns.heatmap(matrix.astype(float), annot=True, cmap="viridis_r", fmt=".2f", vmin=0, vmax=vmax_val, 
                square=True, linewidths=.5, cbar_kws={"shrink": .75}, annot_kws={"size": 8})
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved Cohen's D heatmap: {output_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare results across LLM configurations")
    parser.add_argument("--output", default="analysis_results/llm_comparison", 
                       help="Output directory for analysis results")
    parser.add_argument("--distance_col_name", type=str, default="CosineDistance", 
                        help="Name of the column in the CSV that contains the distance values.")
    parser.add_argument("--config_id_prefix", type=str, default="PMMB-", help="Prefix for config ID folder names in embedding_analysis.")
    parser.add_argument("--embedding_model_name", type=str, default="paraphrase-multilingual-mpnet-base-v2", help="Name of the embedding model subfolder.")
    parser.add_argument("--num_workers", type=int, default=12, help="Number of worker processes for parallel execution.")

    args = parser.parse_args()
    analyze_llm_configurations(args.output, 
                               distance_column_name=args.distance_col_name,
                               config_id_prefix=args.config_id_prefix,
                               embedding_model_name=args.embedding_model_name,
                               num_workers=args.num_workers)

if __name__ == "__main__":
    # This check is crucial for ProcessPoolExecutor on some platforms (like Windows)
    # to prevent issues with re-importing and re-executing the main script in child processes.
    main() 