"""
Language processing functions for Language Representation Analysis.

This module contains functions for processing language embeddings,
calculating distances between languages, and generating various
language-level comparisons.
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
import multiprocessing as mp
import time
import traceback
from scipy.spatial.distance import jensenshannon

# Import the statistical analysis functions we need
from .statistical_analysis import (
    analyze_language_level_statistics,
    analyze_language_family_statistics
)

# Import the utils module
from .utils import sanitize_name

# Import visualization functions
from .visualization import (
    plot_language_dendrogram,
    plot_interactive_heatmap,
    plot_pca,
    plot_tsne,
    plot_umap,
    plot_combined_family_language_pca
)


def calculate_language_embeddings(df_with_embeddings):
    """
    Aggregates embeddings by language to produce language-level embeddings.
    
    Args:
        df_with_embeddings (pd.DataFrame): DataFrame with embeddings and language information
        
    Returns:
        pd.DataFrame: DataFrame with aggregated embeddings per language (and optionally per proficiency)
    """
    if df_with_embeddings.empty or 'Embedding' not in df_with_embeddings.columns:
        print("Error: Input DataFrame is empty or missing 'Embedding' column")
        return pd.DataFrame()
    
    print("Calculating language-level embeddings...")
    
    language_embedding_records = []
    
    if 'Language' not in df_with_embeddings.columns:
        print("Error: DataFrame missing 'Language' column")
        return pd.DataFrame()
    
    # Check if we have proficiency information
    has_proficiency = 'Proficiency' in df_with_embeddings.columns
    
    # Group by appropriate columns
    if has_proficiency:
        groupby_cols = ['Language', 'Proficiency']
    else:
        groupby_cols = ['Language']
    
    # Aggregate embeddings by language (and optionally proficiency)
    for group_key, group_data in df_with_embeddings.groupby(groupby_cols):
        if len(group_data) > 0:
            # Extract embeddings and calculate mean
            embeddings = np.vstack(group_data['Embedding'].values)
            mean_embedding = np.mean(embeddings, axis=0)
            
            # Produce record
            record = {'Embedding': mean_embedding}
            
            # Handle different group keys based on groupby columns
            if has_proficiency:
                lang, prof = group_key
                record['Language'] = lang
                record['Proficiency'] = prof
            else:
                record['Language'] = group_key
            
            language_embedding_records.append(record)
    
    if not language_embedding_records:
        print("Warning: No language embeddings could be calculated")
        return pd.DataFrame()
    
    language_embeddings_df = pd.DataFrame(language_embedding_records)
    print(f"Generated {len(language_embeddings_df)} language-level embeddings")
    
    return language_embeddings_df


def compute_language_distances(language_embeddings_df, output_csv):
    """
    Computes pairwise cosine distances between language embeddings.
    
    Args:
        language_embeddings_df (pd.DataFrame): DataFrame with language embeddings
        output_csv (str): Path to save distance matrix CSV
        
    Returns:
        pd.DataFrame: Distance matrix between languages
    """
    if language_embeddings_df.empty or 'Embedding' not in language_embeddings_df.columns:
        print("Error: language_embeddings_df is empty or missing 'Embedding' column")
        return pd.DataFrame()
    
    if 'Language' not in language_embeddings_df.columns:
        print("Error: language_embeddings_df is missing 'Language' column")
        return pd.DataFrame()
    
    print("Computing language distances...")
    
    # Check if we have proficiency information
    has_proficiency = 'Proficiency' in language_embeddings_df.columns
    proficiency_values = language_embeddings_df['Proficiency'].unique() if has_proficiency else []
    
    # Generate distance records and matrices
    distance_records = []
    matrices = {}  # For visualization
    
    # Process each proficiency level separately if applicable
    if has_proficiency:
        for prof in proficiency_values:
            prof_df = language_embeddings_df[language_embeddings_df['Proficiency'] == prof]
            if len(prof_df) < 2:
                continue
            
            languages = prof_df['Language'].tolist()
            embeddings = np.vstack(prof_df['Embedding'])
            
            # Compute distance matrix
            dist_matrix = 1 - cosine_similarity(embeddings)
            matrices[prof] = pd.DataFrame(dist_matrix, index=languages, columns=languages)
            
            # Extract pairwise distances
            for i in range(len(languages)):
                for j in range(i + 1, len(languages)):
                    distance_records.append({
                        'Language1': languages[i],
                        'Language2': languages[j],
                        'Distance': dist_matrix[i, j],
                        'Proficiency': prof
                    })
    else:
        # No proficiency, process all languages together
        languages = language_embeddings_df['Language'].tolist()
        if len(languages) < 2:
            print("Error: At least two languages are required to compute distances")
            return pd.DataFrame()
        
        embeddings = np.vstack(language_embeddings_df['Embedding'])
        
        # Compute distance matrix
        dist_matrix = 1 - cosine_similarity(embeddings)
        matrices['all'] = pd.DataFrame(dist_matrix, index=languages, columns=languages)
        
        # Extract pairwise distances
        for i in range(len(languages)):
            for j in range(i + 1, len(languages)):
                distance_records.append({
                    'Language1': languages[i],
                    'Language2': languages[j],
                    'Distance': dist_matrix[i, j]
                })
    
    if not distance_records:
        print("Warning: No language distances could be computed")
        return pd.DataFrame()
    
    # Generate DataFrame with all pairwise distances
    distances_df = pd.DataFrame(distance_records)
    
    # Sort for consistency
    sort_cols = ['Language1', 'Language2']
    if has_proficiency:
        sort_cols.append('Proficiency')
    distances_df.sort_values(by=sort_cols, inplace=True)
    
    # Save to CSV if path provided
    if output_csv:
        try:
            distances_df.to_csv(output_csv, index=False)
            print(f"Saved language distances to: {output_csv}")
        except Exception as e:
            print(f"Error saving distances to {output_csv}: {e}")
    
    return matrices.get('all') if len(matrices) == 1 else matrices


def _process_language_pair_comparison(args):
    """Helper function for parallel processing in generate_language_level_comparisons."""
    lang1, lang2, proficiency, df_with_embeddings_subset, lang_noun_counts_global, embeddings_for_prof_map = args

    # Get all nouns that are in both languages for this specific proficiency
    lang1_nouns_set = set(df_with_embeddings_subset[
        df_with_embeddings_subset['Language'] == lang1
    ]['Noun'])
    
    lang2_nouns_set = set(df_with_embeddings_subset[
        df_with_embeddings_subset['Language'] == lang2
    ]['Noun'])
    
    shared_nouns_set = lang1_nouns_set.intersection(lang2_nouns_set)
    
    avg_jaccard_val = 0.0
    avg_noun_distances_val = 0.0
    all_adjectives_lang1_list = []
    all_adjectives_lang2_list = []
    shared_nouns_list_sorted = sorted(list(shared_nouns_set))
    
    noun_distances_list = []
    noun_jaccards_list = []

    if shared_nouns_set:
        for noun_item in shared_nouns_set:
            # Filter df_with_embeddings_subset for the current noun and languages
            adj1_rows_df = df_with_embeddings_subset[
                (df_with_embeddings_subset['Noun'] == noun_item) & 
                (df_with_embeddings_subset['Language'] == lang1)
            ]
            adj2_rows_df = df_with_embeddings_subset[
                (df_with_embeddings_subset['Noun'] == noun_item) & 
                (df_with_embeddings_subset['Language'] == lang2)
            ]
            
            if not adj1_rows_df.empty and not adj2_rows_df.empty:
                adj1_list = adj1_rows_df.iloc[0]['Adjectives']
                adj2_list = adj2_rows_df.iloc[0]['Adjectives']
                
                if isinstance(adj1_list, list):
                    all_adjectives_lang1_list.extend(adj1_list)
                if isinstance(adj2_list, list):
                    all_adjectives_lang2_list.extend(adj2_list)
                
                if isinstance(adj1_list, list) and isinstance(adj2_list, list):
                    set1 = set(adj1_list)
                    set2 = set(adj2_list)
                    union_set = set1.union(set2)
                    if union_set:
                        intersection_set = set1.intersection(set2)
                        jaccard_dist_val = 1.0 - (len(intersection_set) / len(union_set))
                        noun_jaccards_list.append(jaccard_dist_val)
                
                # Noun-level embedding distance (if embeddings are present at this level)
                # This uses the *original* embeddings from df_with_embeddings, not the aggregated language embeddings
                if 'Embedding' in adj1_rows_df.columns and 'Embedding' in adj2_rows_df.columns: 
                    emb1_arr_noun_level = adj1_rows_df.iloc[0]['Embedding']
                    emb2_arr_noun_level = adj2_rows_df.iloc[0]['Embedding']
                    if isinstance(emb1_arr_noun_level, np.ndarray) and isinstance(emb2_arr_noun_level, np.ndarray):
                        norm1_nl = np.linalg.norm(emb1_arr_noun_level)
                        norm2_nl = np.linalg.norm(emb2_arr_noun_level)
                        if norm1_nl > 0 and norm2_nl > 0:
                            cosine_sim_val_nl = np.dot(emb1_arr_noun_level, emb2_arr_noun_level) / (norm1_nl * norm2_nl)
                            noun_dist_val = 1.0 - cosine_sim_val_nl
                            noun_distances_list.append(noun_dist_val)
        
        if noun_jaccards_list:
            avg_jaccard_val = sum(noun_jaccards_list) / len(noun_jaccards_list)
        if noun_distances_list:
            avg_noun_distances_val = sum(noun_distances_list) / len(noun_distances_list)
    
    # Calculate LanguageEmbeddingDistance using the pre-aggregated embeddings_for_prof_map
    lang_emb_dist_val = np.nan # Default to NaN
    emb1_lang_level = embeddings_for_prof_map.get(lang1)
    emb2_lang_level = embeddings_for_prof_map.get(lang2)
    if emb1_lang_level is not None and emb2_lang_level is not None:
        norm1_ll = np.linalg.norm(emb1_lang_level)
        norm2_ll = np.linalg.norm(emb2_lang_level)
        if norm1_ll > 0 and norm2_ll > 0:
            cosine_sim_ll = np.dot(emb1_lang_level, emb2_lang_level) / (norm1_ll * norm2_ll)
            lang_emb_dist_val = 1.0 - cosine_sim_ll

    # Return only the required columns
    return {
        "Language1": lang1,
        "Language2": lang2,
        "Proficiency": proficiency,
        "LanguageEmbeddingDistance": lang_emb_dist_val, 
        "AvgNounDistance": avg_noun_distances_val,
        "AvgJaccardDistance": avg_jaccard_val
    }


def generate_language_level_comparisons(df_with_embeddings, provider_folder_ref, base_filename, api_model, distances_target_dir, visualizations_target_dir):
    """
    Generate comprehensive language-level comparisons by aggregating embeddings across nouns.
    Produces a separate file with detailed language comparisons for each proficiency level.
    
    Args:
        df_with_embeddings: DataFrame with embeddings
        provider_folder_ref: Base output directory (provider_folder from analysis_pipeline) - USED FOR REFERENCE OR IF A NON-STRUCTURED PATH IS NEEDED.
        base_filename: Base name for output files
        api_model: Name of the API model used
        distances_target_dir: Specific subdirectory for distance CSV files (e.g., ".../02_distances/language_level")
        visualizations_target_dir: Specific subdirectory for plot files (e.g., ".../03_visualizations/scatter_plots")
        
    Returns:
        Dictionary mapping proficiency levels to language comparison DataFrames
    """
    print("\nGenerating comprehensive language-level comparisons...")
    
    # Calculate language embeddings (aggregates all nouns for each language at each proficiency)
    lang_embeddings_df = calculate_language_embeddings(df_with_embeddings) # This returns df with Lang, Proficiency, Embedding
    
    if lang_embeddings_df.empty:
        print("Warning: No language embeddings could be calculated")
        return {}
    
    language_comparisons_by_prof_level = {}
    # Determine number of cores to use, consistent with other parallel operations
    n_cores = min(12, max(1, (os.cpu_count() - 4) if os.cpu_count() is not None else 1))
    print(f"Using up to {n_cores} CPU cores for generating language-level comparisons.")

    # Group the original df_with_embeddings by proficiency for efficient subsetting later
    grouped_df_by_prof = dict(tuple(df_with_embeddings.groupby('Proficiency')))

    for proficiency_level, prof_specific_lang_embeddings_df in lang_embeddings_df.groupby('Proficiency'):
        if len(prof_specific_lang_embeddings_df) < 2:
            print(f"Skipping proficiency '{proficiency_level}': Only {len(prof_specific_lang_embeddings_df)} languages with aggregated embeddings")
            continue
            
        print(f"  Producing language-level comparisons for: {proficiency_level} ({len(prof_specific_lang_embeddings_df)} languages)")
        
        # These are the languages that have an aggregated embedding for this proficiency
        languages_in_prof = prof_specific_lang_embeddings_df['Language'].tolist()
        
        # Produce a map of language to its aggregated embedding for this proficiency for quick lookup in helper
        embeddings_map_for_this_prof = {row['Language']: row['Embedding'] for _, row in prof_specific_lang_embeddings_df.iterrows()}

        # Get the subset of the original df_with_embeddings for the current proficiency_level
        # This subset contains all original noun-level data (including adjectives and noun-level embeddings)
        df_subset_for_prof = grouped_df_by_prof.get(proficiency_level)
        if df_subset_for_prof is None or df_subset_for_prof.empty:
            print(f"  Warning: No original data found in df_with_embeddings for proficiency '{proficiency_level}'. Skipping.")
            continue

        # Count unique nouns per language within this proficiency from the original data
        lang_noun_counts_for_prof = {}
        for lang_item in languages_in_prof: # Only iterate over languages that have aggregated embeddings
            lang_nouns_unique = df_subset_for_prof[df_subset_for_prof['Language'] == lang_item]['Noun'].unique()
            lang_noun_counts_for_prof[lang_item] = len(lang_nouns_unique)
        
        tasks = []
        # Generate pairs only from languages present in prof_specific_lang_embeddings_df
        for i in range(len(languages_in_prof)):
            for j in range(i + 1, len(languages_in_prof)):
                lang1, lang2 = languages_in_prof[i], languages_in_prof[j]
                # df_subset_for_prof is the relevant part of original df_with_embeddings for this proficiency
                tasks.append((lang1, lang2, proficiency_level, df_subset_for_prof, lang_noun_counts_for_prof, embeddings_map_for_this_prof))

        collected_results_for_prof = []
        if n_cores > 1 and len(tasks) > 0:
            print(f"  Processing {len(tasks)} language pairs for proficiency '{proficiency_level}' using {n_cores} cores...")
            with ProcessPoolExecutor(max_workers=n_cores) as executor:
                future_to_task_map = {executor.submit(_process_language_pair_comparison, task_args): task_args for task_args in tasks}
                for future_obj in as_completed(future_to_task_map):
                    # task_debug_info = future_to_task_map[future_obj][:3] # lang1, lang2, proficiency
                    try:
                        pair_result = future_obj.result()
                        collected_results_for_prof.append(pair_result)
                    except Exception as exc:
                        original_task_params = future_to_task_map[future_obj]
                        print(f'Task for ({original_task_params[0]}, {original_task_params[1]}, {original_task_params[2]}) generated an exception: {exc}')
        elif len(tasks) > 0: # Single core or n_cores <=1 but tasks exist
            print(f"  Processing {len(tasks)} language pairs for proficiency '{proficiency_level}' using single core...")
            for task_args_single_core in tasks:
                try:
                    result_single = _process_language_pair_comparison(task_args_single_core)
                    collected_results_for_prof.append(result_single)
                except Exception as exc_single:
                    print(f'Task {task_args_single_core[:3]} generated an exception during single core execution: {exc_single}')
        else:
            print(f"  No language pairs to process for proficiency '{proficiency_level}'.")

        if collected_results_for_prof:
            lang_comp_df_for_prof = pd.DataFrame(collected_results_for_prof)
            
            lang_comp_df_for_prof.sort_values(by=["Language1", "Language2"], ascending=[True, True], inplace=True)
            
            prof_output_filename = f"{base_filename}_{proficiency_level}_language_level_comparisons.csv"
            prof_specific_distances_dir = os.path.join(distances_target_dir, "proficiency_specific")
            os.makedirs(prof_specific_distances_dir, exist_ok=True)
            prof_output_file = os.path.join(prof_specific_distances_dir, prof_output_filename)
            lang_comp_df_for_prof.to_csv(prof_output_file, index=False)
            print(f"  Saved language-level comparisons for '{proficiency_level}' to {prof_output_file}")
            
            # Add statistical analysis for this proficiency level
            print(f"  Generating statistical analysis for language-level comparisons ('{proficiency_level}')...")
            stats_dir = provider_folder_ref.replace("02_distances", "04_statistical_reports").replace("language_level", "")
            os.makedirs(stats_dir, exist_ok=True)
            
            # Create distinct directories for each distance metric
            lang_stats_base_dir = os.path.join(stats_dir, "language_level_statistics")
            lang_emb_stats_dir = os.path.join(lang_stats_base_dir, "LanguageEmbeddingDistance", "proficiency_specific")
            avg_noun_stats_dir = os.path.join(lang_stats_base_dir, "AvgNounDistance", "proficiency_specific")
            avg_jaccard_stats_dir = os.path.join(lang_stats_base_dir, "AvgJaccardDistance", "proficiency_specific")
            
            # Create all necessary directories
            os.makedirs(lang_emb_stats_dir, exist_ok=True)
            os.makedirs(avg_noun_stats_dir, exist_ok=True)
            os.makedirs(avg_jaccard_stats_dir, exist_ok=True)
            
            # Generate statistical analysis for LanguageEmbeddingDistance
            if "LanguageEmbeddingDistance" in lang_comp_df_for_prof.columns:
                lang_emb_stats_prefix = os.path.join(lang_emb_stats_dir, f"{base_filename}_{proficiency_level}")
                analyze_language_level_statistics(
                    lang_comp_df_for_prof, 
                    output_file_path_prefix=lang_emb_stats_prefix, 
                    distance_metric="LanguageEmbeddingDistance"
                )
            
            # Generate statistical analysis for AvgNounDistance
            if "AvgNounDistance" in lang_comp_df_for_prof.columns:
                noun_dist_stats_prefix = os.path.join(avg_noun_stats_dir, f"{base_filename}_{proficiency_level}")
                analyze_language_level_statistics(
                    lang_comp_df_for_prof, 
                    output_file_path_prefix=noun_dist_stats_prefix, 
                    distance_metric="AvgNounDistance"
                )
            
            # Generate statistical analysis for AvgJaccardDistance
            if "AvgJaccardDistance" in lang_comp_df_for_prof.columns:
                jaccard_stats_prefix = os.path.join(avg_jaccard_stats_dir, f"{base_filename}_{proficiency_level}")
                analyze_language_level_statistics(
                    lang_comp_df_for_prof, 
                    output_file_path_prefix=jaccard_stats_prefix, 
                    distance_metric="AvgJaccardDistance"
                )
            
            sorted_by_dist_df = lang_comp_df_for_prof.sort_values(by="LanguageEmbeddingDistance", ascending=True) # Ascending for closest pairs first
            sorted_output_filename = f"{base_filename}_{proficiency_level}_language_level_by_distance.csv"
            sorted_output_file = os.path.join(prof_specific_distances_dir, sorted_output_filename)
            sorted_by_dist_df.to_csv(sorted_output_file, index=False)
            print(f"  Saved sorted language-level comparisons for '{proficiency_level}' to {sorted_output_file}")
            
            if "AvgJaccardDistance" in lang_comp_df_for_prof.columns and "LanguageEmbeddingDistance" in lang_comp_df_for_prof.columns:
                fig = plt.figure(figsize=(10, 8))
                plt.scatter(
                    lang_comp_df_for_prof["LanguageEmbeddingDistance"],
                    lang_comp_df_for_prof["AvgJaccardDistance"],
                    alpha=0.7
                )
                # Add labels carefully to avoid clutter
                # for _, row_plot in lang_comp_df_for_prof.iterrows():
                #     if pd.notna(row_plot["LanguageEmbeddingDistance"]) and pd.notna(row_plot["AvgJaccardDistance"]):
                #        plt.annotate(f"{row_plot['Language1']}-{row_plot['Language2']}", 
                #                     (row_plot["LanguageEmbeddingDistance"], row_plot["AvgJaccardDistance"]),
                #                     fontsize=8, alpha=0.6)
                
                plt.title(f"Language Distances: Embedding vs Jaccard ({api_model}, {proficiency_level})")
                plt.xlabel("Language Embedding Distance (Cosine-based, from aggregated embeddings)")
                plt.ylabel("Average Jaccard Distance (from shared nouns' adjectives)")
                plt.grid(True, linestyle='--', alpha=0.7)
                
                if len(lang_comp_df_for_prof) > 1:
                    # Drop NaN before correlation
                    corr_df = lang_comp_df_for_prof[["LanguageEmbeddingDistance", "AvgJaccardDistance"]].dropna()
                    if len(corr_df) > 1:
                        corr_val, p_val = stats.pearsonr(corr_df["LanguageEmbeddingDistance"], corr_df["AvgJaccardDistance"])
                        plt.figtext(0.5, 0.01, f"Pearson r: {corr_val:.3f} (p={p_val:.3f}, n={len(corr_df)})", ha="center")
                    else:
                        plt.figtext(0.5, 0.01, "Correlation: Not enough data points after NaN removal", ha="center")
                
                plot_filename = f"{base_filename}_{proficiency_level}_language_distances_comparison.png"
                plot_file = os.path.join(visualizations_target_dir, plot_filename)
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close(fig) # Close the figure
                print(f"  Generated distance comparison plot: {plot_file}")
            
            language_comparisons_by_prof_level[proficiency_level] = lang_comp_df_for_prof
        else:
            print(f"  No comparison results generated for proficiency '{proficiency_level}'.")
    
    if language_comparisons_by_prof_level:
        all_comparisons_list = pd.concat(language_comparisons_by_prof_level.values(), ignore_index=True)
        all_output_filename = f"all_proficiencies_lang_level_comparisons_{base_filename}.csv"
        all_output_file = os.path.join(distances_target_dir, all_output_filename)
        all_comparisons_list.to_csv(all_output_file, index=False)
        print(f"Saved consolidated language-level comparisons for all proficiency levels to {all_output_file}")
        
        # Add consolidated statistical analysis
        print("Generating statistical analysis for all-proficiencies language-level comparisons...")
        
        # Create distinct directories for consolidated statistics
        lang_stats_base_dir = os.path.join(stats_dir, "language_level_statistics")
        lang_emb_stats_dir = os.path.join(lang_stats_base_dir, "LanguageEmbeddingDistance")
        avg_noun_stats_dir = os.path.join(lang_stats_base_dir, "AvgNounDistance")
        avg_jaccard_stats_dir = os.path.join(lang_stats_base_dir, "AvgJaccardDistance")
        
        # Create directories
        os.makedirs(lang_emb_stats_dir, exist_ok=True)
        os.makedirs(avg_noun_stats_dir, exist_ok=True)
        os.makedirs(avg_jaccard_stats_dir, exist_ok=True)
        
        # Generate statistical analysis for LanguageEmbeddingDistance
        if "LanguageEmbeddingDistance" in all_comparisons_list.columns:
            all_lang_emb_stats_prefix = os.path.join(lang_emb_stats_dir, f"all_proficiencies_{base_filename}")
            analyze_language_level_statistics(
                all_comparisons_list, 
                output_file_path_prefix=all_lang_emb_stats_prefix, 
                distance_metric="LanguageEmbeddingDistance"
            )
        
        # Generate statistical analysis for AvgNounDistance
        if "AvgNounDistance" in all_comparisons_list.columns:
            all_noun_dist_stats_prefix = os.path.join(avg_noun_stats_dir, f"all_proficiencies_{base_filename}")
            analyze_language_level_statistics(
                all_comparisons_list, 
                output_file_path_prefix=all_noun_dist_stats_prefix, 
                distance_metric="AvgNounDistance"
            )
        
        # Generate statistical analysis for AvgJaccardDistance
        if "AvgJaccardDistance" in all_comparisons_list.columns:
            all_jaccard_stats_prefix = os.path.join(avg_jaccard_stats_dir, f"all_proficiencies_{base_filename}")
            analyze_language_level_statistics(
                all_comparisons_list, 
                output_file_path_prefix=all_jaccard_stats_prefix, 
                distance_metric="AvgJaccardDistance"
            )
    
    return language_comparisons_by_prof_level


def generate_language_comparisons_df(category_df, output_csv=None):
    """
    Generates language comparison DataFrame for the given category data.
    Computes pairwise distances between languages for the same nouns.
    
    Args:
        category_df: DataFrame with embeddings and language information
        output_csv: Optional path to save CSV file
        
    Returns:
        DataFrame with language comparisons
    """
    print(f"Generating language comparisons...")
    languages = sorted(category_df['Language'].unique())
    
    records = []
    for noun in category_df['Noun'].unique():
        noun_df = category_df[category_df['Noun'] == noun]
        
        for i, lang1 in enumerate(languages):
            lang1_rows = noun_df[noun_df['Language'] == lang1]
            if lang1_rows.empty:
                continue
                
            for lang2 in languages[i+1:]:
                lang2_rows = noun_df[noun_df['Language'] == lang2]
                if lang2_rows.empty:
                    continue
                
                # Get embeddings for comparison
                emb1 = lang1_rows.iloc[0]['Embedding']
                emb2 = lang2_rows.iloc[0]['Embedding']
                
                # Calculate cosine distance
                sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                cosine_dist = 1.0 - sim
                
                # Get adjectives
                adj1 = lang1_rows.iloc[0]['Adjectives']
                adj2 = lang2_rows.iloc[0]['Adjectives']
                
                # Format adjectives for CSV
                if isinstance(adj1, list):
                    adj1_str = ";".join(adj1)
                else:
                    adj1_str = adj1
                    
                if isinstance(adj2, list):
                    adj2_str = ";".join(adj2)
                else:
                    adj2_str = adj2
                
                # Calculate Jaccard distance between adjective sets
                if isinstance(adj1, str):
                    adj1_set = set(adj1.split(";"))
                elif isinstance(adj1, list):
                    adj1_set = set(adj1)
                else:
                    adj1_set = set()
                
                if isinstance(adj2, str):
                    adj2_set = set(adj2.split(";"))
                elif isinstance(adj2, list):
                    adj2_set = set(adj2)
                else:
                    adj2_set = set()
                
                # Calculate Jaccard distance
                union = adj1_set.union(adj2_set)
                if union:  # Avoid division by zero
                    intersection = adj1_set.intersection(adj2_set)
                    jaccard_dist = 1.0 - (len(intersection) / len(union))
                else:
                    jaccard_dist = 1.0  # Maximum distance for empty sets
                
                records.append({
                    'Noun': noun,
                    'NounCategory': lang1_rows.iloc[0]['NounCategory'],
                    'Language1': lang1,
                    'Language2': lang2,
                    'CosineDistance': cosine_dist,
                    'JaccardDistance': jaccard_dist,
                    'Adjectives1': adj1_str,
                    'Adjectives2': adj2_str
                })
    
    # Generate DataFrame
    if records:
        result_df = pd.DataFrame(records)
        result_df.sort_values(by=['NounCategory', 'Noun', 'Language1', 'Language2'], inplace=True)
        
        # Save to CSV if path provided
        if output_csv:
            result_df.to_csv(output_csv, index=False)
            print(f"Saved language comparisons to {output_csv}")
            
        return result_df
    else:
        print("No language comparisons generated")
        return pd.DataFrame(columns=['Noun', 'NounCategory', 'Language1', 'Language2', 'CosineDistance', 'JaccardDistance', 'Adjectives1', 'Adjectives2'])


def calculate_language_family_embeddings_and_distances(language_embeddings_df, lang_to_family_map, output_embeddings_dir, output_distances_dir, base_filename_prefix="all_proficiencies"):
    """
    Calculates embeddings and pairwise distances for language families.

    Args:
        language_embeddings_df: DataFrame with columns ['Language', 'Embedding', Optional['Proficiency']里斯].
                                  Embeddings should be aggregated per language (and optionally per proficiency).
        lang_to_family_map: Dictionary mapping language codes (e.g., 'FR') to family names (e.g., 'Romance').
        output_embeddings_dir: Directory path to save family embeddings .pkl file.
        output_distances_dir: Directory path to save family distances .csv file.
        base_filename_prefix: Prefix for output filenames (e.g., specific proficiency or 'all_proficiencies').

    Returns:
        Tuple: (family_embeddings_df, family_distances_df)
               Returns (pd.DataFrame(), pd.DataFrame()) if data is insufficient or error occurs.
    """
    print(f"\nCalculating language family embeddings and distances for: {base_filename_prefix}...")

    if language_embeddings_df.empty or 'Language' not in language_embeddings_df.columns or 'Embedding' not in language_embeddings_df.columns:
        print("Error: language_embeddings_df is empty or missing required columns ('Language', 'Embedding').")
        return pd.DataFrame(), pd.DataFrame()
    if not lang_to_family_map:
        print("Error: lang_to_family_map is empty.")
        return pd.DataFrame(), pd.DataFrame()

    df = language_embeddings_df.copy()
    df['LanguageCode'] = df['Language'].apply(lambda x: x.split(' # ')[0] if isinstance(x, str) and ' # ' in x else x)
    df['LanguageFamily'] = df['LanguageCode'].map(lang_to_family_map)
    df.dropna(subset=['LanguageFamily'], inplace=True)

    if df.empty:
        print("No languages could be mapped to families, or no data left after mapping.")
        return pd.DataFrame(), pd.DataFrame()

    group_by_cols = ['LanguageFamily']
    proficiency_present_and_variable = 'Proficiency' in df.columns and df['Proficiency'].nunique() > 0 

    if proficiency_present_and_variable:
        group_by_cols.append('Proficiency')
        print("Aggregating family embeddings per LanguageFamily and Proficiency.")
    else:
        print("Aggregating family embeddings per LanguageFamily (either no Proficiency column or only one Proficiency value).")

    family_embedding_records = []
    for name_parts, group_df in df.groupby(group_by_cols):
        if not group_df.empty:
            mean_embedding = np.mean(np.vstack(group_df['Embedding']), axis=0)
            record = {'Embedding': mean_embedding}
            if isinstance(name_parts, tuple):
                record['LanguageFamily'] = name_parts[0]
                if len(name_parts) > 1: 
                     record['Proficiency'] = name_parts[1]
            else:
                record['LanguageFamily'] = name_parts
            family_embedding_records.append(record)
    
    if not family_embedding_records:
        print("No family embeddings calculated.")
        return pd.DataFrame(), pd.DataFrame()

    family_embeddings_df = pd.DataFrame(family_embedding_records)

    # Create a subdirectory based on the prefix to avoid file collisions
    prefix_dir = base_filename_prefix.split('_')[0]  # Use the first part of the prefix as subdirectory
    embeddings_subdir = os.path.join(output_embeddings_dir, prefix_dir)
    distances_subdir = os.path.join(output_distances_dir, prefix_dir)
    
    # Create subdirectories if they don't exist
    os.makedirs(embeddings_subdir, exist_ok=True)
    os.makedirs(distances_subdir, exist_ok=True)
    
    family_embeddings_filename = f"family_embeddings_{base_filename_prefix}.pkl"
    family_embeddings_filepath = os.path.join(embeddings_subdir, family_embeddings_filename)
    try:
        family_embeddings_df.to_pickle(family_embeddings_filepath)
        print(f"Saved language family embeddings to: {family_embeddings_filepath}")
    except Exception as e:
        print(f"Error saving family embeddings: {e}")
        return pd.DataFrame(), pd.DataFrame()

    all_family_distances_records = []
    
    if 'Proficiency' in family_embeddings_df.columns:
        for prof_level, group_df in family_embeddings_df.groupby('Proficiency'):
            if len(group_df) < 2:
                continue
            families_in_group = group_df['LanguageFamily'].tolist()
            embeddings_in_group = np.vstack(group_df['Embedding'])
            dist_matrix = 1 - cosine_similarity(embeddings_in_group)
            for i in range(len(families_in_group)):
                for j in range(i + 1, len(families_in_group)):
                    all_family_distances_records.append({
                        'Family1': families_in_group[i],
                        'Family2': families_in_group[j],
                        'Distance': dist_matrix[i, j],
                        'Proficiency': prof_level
                    })
    else: 
        if len(family_embeddings_df) < 2:
            print("Not enough families to compute distances.")
        else:
            families_in_group = family_embeddings_df['LanguageFamily'].tolist()
            embeddings_in_group = np.vstack(family_embeddings_df['Embedding'])
            dist_matrix = 1 - cosine_similarity(embeddings_in_group)
            for i in range(len(families_in_group)):
                for j in range(i + 1, len(families_in_group)):
                    all_family_distances_records.append({
                        'Family1': families_in_group[i],
                        'Family2': families_in_group[j],
                        'Distance': dist_matrix[i, j]
                    })

    # Create the family distances DataFrame and save it
    if all_family_distances_records:
        family_distances_df = pd.DataFrame(all_family_distances_records)
        
        # Save the distances to a CSV file
        family_distances_filename = f"family_distances_{base_filename_prefix}.csv"
        family_distances_filepath = os.path.join(distances_subdir, family_distances_filename)
        
        try:
            family_distances_df.to_csv(family_distances_filepath, index=False)
            print(f"Saved language family distances to: {family_distances_filepath}")
        except Exception as e:
            print(f"Error saving family distances: {e}")
        
        # Generate statistical analysis for family-level comparisons
        print(f"Generating statistical analysis for language family comparisons ({base_filename_prefix})...")
        
        # Determine the base directory structure
        if output_embeddings_dir and "embeddings" in output_embeddings_dir:
            base_dir = os.path.dirname(os.path.dirname(output_embeddings_dir))
            stats_base_dir = os.path.join(base_dir, "04_statistical_reports", "language_family_statistics")
            
            # Create separate directories for each distance metric
            stats_dir_lang_emb = os.path.join(stats_base_dir, "LanguageEmbeddingDistance")
            stats_dir_avg_noun = os.path.join(stats_base_dir, "AvgNounDistance")
            
            # Ensure directories exist
            os.makedirs(stats_dir_lang_emb, exist_ok=True)
            os.makedirs(stats_dir_avg_noun, exist_ok=True)
            
            # Determine which stats directory to use based on the distance metric in the DataFrame
            if 'Distance' in family_distances_df.columns:
                # Check if this is LanguageEmbeddingDistance or AvgNounDistance
                is_lang_emb_distance = True  # Default assumption
                
                # Look for clues in the provider folder path
                if "AvgNounDistance" in output_embeddings_dir or "avg_noun" in output_embeddings_dir:
                    is_lang_emb_distance = False
                
                # Use the appropriate directory
                stats_dir = stats_dir_lang_emb if is_lang_emb_distance else stats_dir_avg_noun
            else:
                # Default to LanguageEmbeddingDistance if we can't determine
                stats_dir = stats_dir_lang_emb
        else:
            # Fallback to creating a stats directory alongside the embedding directory
            stats_dir = os.path.join(os.path.dirname(output_embeddings_dir), "statistics")
        
        # Ensure directory exists
        os.makedirs(stats_dir, exist_ok=True)
        
        # Create statistics for the whole dataset
        stats_prefix = os.path.join(stats_dir, f"family_{base_filename_prefix}")
        analyze_language_family_statistics(
            family_distances_df,
            output_file_path_prefix=stats_prefix,
            distance_metric="Distance"
        )
        
        # If we have proficiency information, also generate per-proficiency statistics
        if 'Proficiency' in family_distances_df.columns and family_distances_df['Proficiency'].nunique() > 1:
            # Create a directory for proficiency-specific stats
            prof_stats_dir = os.path.join(stats_dir, "proficiency_specific")
            os.makedirs(prof_stats_dir, exist_ok=True)
            
            # Process each proficiency level separately
            for prof, prof_df in family_distances_df.groupby('Proficiency'):
                if len(prof_df) >= 2:  # Only analyze if we have enough data
                    prof_stats_prefix = os.path.join(prof_stats_dir, f"family_{base_filename_prefix}_{prof}")
                    analyze_language_family_statistics(
                        prof_df, 
                        output_file_path_prefix=prof_stats_prefix,
                        distance_metric="Distance"
                    )
        
        return family_embeddings_df, family_distances_df
    else:
        print("No family distance records generated.")
        return family_embeddings_df, pd.DataFrame()

def run_language_level_analysis_and_visualization(
    df_with_embeddings,
    provider_folder, # Used for stat report output path construction
    base_filename,
    api_model,
    embeddings_dir, 
    distances_dir, 
    visualizations_dir,
    lang_to_family_map=None # For combined PCA plots
):
    """
    Orchestrates the full suite of language-level analysis, including:
    - Calculating language embeddings.
    - Computing language distances.
    - Generating language comparison CSVs and statistical reports.
    - Producing all language-level visualizations (dendrograms, heatmaps, PCA, t-SNE, UMAP)
      for LanguageEmbeddingDistance and AvgNounDistance metrics.

    Args:
        df_with_embeddings (pd.DataFrame): DataFrame with original embeddings.
        provider_folder (str): Base output directory for the current analysis run.
        base_filename (str): Base name for output files.
        api_model (str): Name of the API model used.
        embeddings_dir (str): Directory to save language-level embeddings.
        distances_dir (str): Directory to save language-level distances and comparisons.
        visualizations_dir (str): Directory to save language-level visualizations.
        lang_to_family_map (dict, optional): Mapping of language codes to families for combined PCA.

    Returns:
        pd.DataFrame: language_embeddings_df, or an empty DataFrame if an error occurs.
    """
    print("\n--- Starting Comprehensive Language-Level Analysis and Visualization ---")
    language_embeddings_df = pd.DataFrame()
    stats_dir = os.path.join(provider_folder, "04_statistical_reports") # Define stats_dir based on provider_folder

    try:
        print("\nCalculating language-level embeddings...")
        language_embeddings_df = calculate_language_embeddings(df_with_embeddings)
        
        if language_embeddings_df.empty:
            print("Error: No language embeddings calculated. Skipping language-level analysis.")
            return pd.DataFrame()

        lang_embeddings_filename = f"language_level_all_proficiencies_{base_filename}.pkl"
        lang_embeddings_file_path = os.path.join(embeddings_dir, "language_level", lang_embeddings_filename)
        os.makedirs(os.path.dirname(lang_embeddings_file_path), exist_ok=True)
        language_embeddings_df.to_pickle(lang_embeddings_file_path)
        print(f"Language embeddings saved to {lang_embeddings_file_path}")
        
        language_dist_filename = f"all_proficiencies_lang_distances_{base_filename}.csv"
        lang_dist_dir = os.path.join(distances_dir, "language_level")
        os.makedirs(lang_dist_dir, exist_ok=True)
        language_dist_file_path = os.path.join(lang_dist_dir, language_dist_filename)
        # compute_language_distances returns a dict of matrices if by proficiency, or a single df
        language_distance_matrices = compute_language_distances(language_embeddings_df, language_dist_file_path)

        # Ensure directories for comparisons and visualizations exist
        language_comparisons_distances_dir = os.path.join(distances_dir, "language_level") # Main dir for consolidated
        prof_specific_lang_comp_dist_dir = os.path.join(language_comparisons_distances_dir, "proficiency_specific")
        language_comparisons_visualizations_dir = os.path.join(visualizations_dir, "scatter_plots") # No proficiency split here for now
        os.makedirs(prof_specific_lang_comp_dist_dir, exist_ok=True)
        os.makedirs(language_comparisons_visualizations_dir, exist_ok=True)

        # This function generates both proficiency-specific and a consolidated comparison CSV.
        # It also calls analyze_language_level_statistics internally.
        generate_language_level_comparisons(
            df_with_embeddings, 
            provider_folder, # Pass this for stats output path construction inside generate_language_level_comparisons
            base_filename,
            api_model,
            language_comparisons_distances_dir, # Base for consolidated and prof-specific output
            language_comparisons_visualizations_dir
        )
        
        # --- Visualizations --- 
        print(f"\nGenerating language-level dendrograms, heatmaps, and dimensionality reduction plots...")
        
        # Path for consolidated language comparisons (all proficiencies)
        consolidated_lang_comp_path = os.path.join(language_comparisons_distances_dir, f"all_proficiencies_lang_level_comparisons_{base_filename}.csv")

        # Determine if analysis is per-proficiency or consolidated based on language_distance_matrices type
        is_per_proficiency_viz = isinstance(language_distance_matrices, dict) and not all(k == 'all' for k in language_distance_matrices.keys())

        if is_per_proficiency_viz:
            print("  Generating visualizations per proficiency level.")
            for prof, lang_dist_matrix_prof in language_distance_matrices.items():
                if not isinstance(lang_dist_matrix_prof, pd.DataFrame) or lang_dist_matrix_prof.empty:
                    print(f"    Skipping visualizations for proficiency '{prof}': Invalid or empty distance matrix.")
                    continue
                
                prof_safe = sanitize_name(str(prof))
                languages_list_prof = lang_dist_matrix_prof.index.tolist()
                prof_base_filename = f"{base_filename}_{prof_safe}"

                # Create a proficiency-specific language_pca_df for visualizations
                # Filter language_embeddings_df for the current proficiency
                current_prof_lang_embeddings_df = language_embeddings_df[language_embeddings_df['Proficiency'] == prof]
                if current_prof_lang_embeddings_df.empty:
                    print(f"    Skipping PCA for proficiency '{prof}': No embeddings found.")
                    continue

                lang_pca_df_prof = pd.DataFrame({
                    'Language': current_prof_lang_embeddings_df['Language'].unique(),
                    'Label': current_prof_lang_embeddings_df['Language'].unique(),
                    'Embedding': [current_prof_lang_embeddings_df[current_prof_lang_embeddings_df['Language'] == lang]['Embedding'].values[0] 
                                 for lang in current_prof_lang_embeddings_df['Language'].unique()]
                })
                if len(lang_pca_df_prof) < 2:
                    print(f"    Skipping visualizations for proficiency '{prof}': Less than 2 languages with embeddings.")
                    continue

                # 1. LanguageEmbeddingDistance Visualizations (per proficiency)
                # Dendrogram
                dendrogram_path_prof = os.path.join(visualizations_dir, "dendrograms", "language_level", "LanguageEmbeddingDistance", f"{prof_base_filename}_language_dendrogram.png")
                try:
                    plot_language_dendrogram(lang_dist_matrix_prof, languages_list_prof, dendrogram_path_prof, f"Language Clustering ({prof}) - {api_model}")
                except Exception as e: print(f"Error generating dendrogram for {prof}: {e}")
                # Heatmap
                heatmap_path_prof = os.path.join(visualizations_dir, "heatmaps", "language_level", "LanguageEmbeddingDistance", f"{prof_base_filename}_language_heatmap.html")
                try:
                    plot_interactive_heatmap(lang_dist_matrix_prof, heatmap_path_prof, f"Language Distance Heatmap ({prof}) - {api_model}")
                except Exception as e: print(f"Error generating heatmap for {prof}: {e}")
                
                # PCA (2D and 3D)
                pca_dir_prof = os.path.join(visualizations_dir, "pca", "language_level", "LanguageEmbeddingDistance")
                try:
                    plot_pca(lang_pca_df_prof, os.path.join(pca_dir_prof, f"{prof_base_filename}_language_pca_2d"), f"Language PCA ({prof}) - LanguageEmbeddingDistance - {api_model}", 'Language', True, dimensions=2)
                    plot_pca(lang_pca_df_prof, os.path.join(pca_dir_prof, f"{prof_base_filename}_language_pca_3d"), f"3D Language PCA ({prof}) - LanguageEmbeddingDistance - {api_model}", 'Language', True, dimensions=3)
                except Exception as e: print(f"Error generating PCA (LanguageEmbeddingDistance) for {prof}: {e}")

                # t-SNE (2D and 3D)
                tsne_dir_prof = os.path.join(visualizations_dir, "tsne", "language_level", "LanguageEmbeddingDistance")
                try:
                    plot_tsne(lang_pca_df_prof, os.path.join(tsne_dir_prof, f"{prof_base_filename}_language_tsne_2d.html"), f"Language t-SNE ({prof}) - LanguageEmbeddingDistance - {api_model}", 'Language', True, original_distance_matrix=lang_dist_matrix_prof, dimensions=2)
                    plot_tsne(lang_pca_df_prof, os.path.join(tsne_dir_prof, f"{prof_base_filename}_language_tsne_3d.html"), f"3D Language t-SNE ({prof}) - LanguageEmbeddingDistance - {api_model}", 'Language', True, original_distance_matrix=lang_dist_matrix_prof, dimensions=3)
                except Exception as e: print(f"Error generating t-SNE (LanguageEmbeddingDistance) for {prof}: {e}")

                # UMAP (2D and 3D)
                umap_dir_prof = os.path.join(visualizations_dir, "umap", "language_level", "LanguageEmbeddingDistance")
                try:
                    plot_umap(lang_pca_df_prof, os.path.join(umap_dir_prof, f"{prof_base_filename}_language_umap_2d.html"), f"Language UMAP ({prof}) - LanguageEmbeddingDistance - {api_model}", 'Language', True, original_distance_matrix=lang_dist_matrix_prof, dimensions=2)
                    plot_umap(lang_pca_df_prof, os.path.join(umap_dir_prof, f"{prof_base_filename}_language_umap_3d.html"), f"3D Language UMAP ({prof}) - LanguageEmbeddingDistance - {api_model}", 'Language', True, original_distance_matrix=lang_dist_matrix_prof, dimensions=3)
                except Exception as e: print(f"Error generating UMAP (LanguageEmbeddingDistance) for {prof}: {e}")

                # 2. AvgNounDistance Visualizations (per proficiency)
                prof_lang_comp_path = os.path.join(prof_specific_lang_comp_dist_dir, f"{base_filename}_{prof_safe}_language_level_comparisons.csv")
                if os.path.exists(prof_lang_comp_path):
                    try:
                        prof_comp_df = pd.read_csv(prof_lang_comp_path)
                        if 'AvgNounDistance' in prof_comp_df.columns:
                            avg_noun_dist_matrix_prof = pd.pivot_table(prof_comp_df, values='AvgNounDistance', index='Language1', columns='Language2', fill_value=0)
                            # Ensure symmetry and correct labels
                            all_langs_in_prof_comp = sorted(list(set(prof_comp_df['Language1']) | set(prof_comp_df['Language2'])))
                            avg_noun_dist_matrix_prof = avg_noun_dist_matrix_prof.reindex(index=all_langs_in_prof_comp, columns=all_langs_in_prof_comp, fill_value=0)
                            for lang1_idx, lang1_val in enumerate(all_langs_in_prof_comp):
                                for lang2_idx, lang2_val in enumerate(all_langs_in_prof_comp):
                                    if lang1_idx < lang2_idx:
                                        val = avg_noun_dist_matrix_prof.loc[lang1_val, lang2_val]
                                        if pd.isna(val) or val == 0:
                                            val_rev = avg_noun_dist_matrix_prof.loc[lang2_val, lang1_val]
                                            if not pd.isna(val_rev):
                                                avg_noun_dist_matrix_prof.loc[lang1_val, lang2_val] = val_rev
                                        avg_noun_dist_matrix_prof.loc[lang2_val, lang1_val] = avg_noun_dist_matrix_prof.loc[lang1_val, lang2_val] # Symmetrize
                            np.fill_diagonal(avg_noun_dist_matrix_prof.values, 0)

                            if not avg_noun_dist_matrix_prof.empty and len(all_langs_in_prof_comp) >=2:
                                # Heatmap for AvgNounDistance
                                heatmap_avg_noun_path_prof = os.path.join(visualizations_dir, "heatmaps", "language_level", "AvgNounDistance", f"{prof_base_filename}_avg_noun_dist_heatmap.html")
                                plot_interactive_heatmap(avg_noun_dist_matrix_prof, heatmap_avg_noun_path_prof, f"Average Noun Distance Heatmap ({prof}) - {api_model}")
                                
                                # PCA, t-SNE, UMAP for AvgNounDistance require coordinates from MDS or similar
                                # For simplicity, we will focus on direct embeddings for PCA/tSNE/UMAP here
                                # If MDS-based plots for AvgNounDistance are desired, that logic would go here.
                                # As an example, PCA using MDS for AvgNounDistance:
                                from sklearn.manifold import MDS
                                mds_2d_avg_noun = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
                                coords_2d_avg_noun = mds_2d_avg_noun.fit_transform(avg_noun_dist_matrix_prof.values)
                                mds_df_2d_avg_noun = pd.DataFrame(coords_2d_avg_noun, columns=['x', 'y'], index=all_langs_in_prof_comp)
                                mds_df_2d_avg_noun['Language'] = mds_df_2d_avg_noun.index
                                mds_df_2d_avg_noun['Label'] = mds_df_2d_avg_noun.index

                                mds_3d_avg_noun = MDS(n_components=3, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
                                coords_3d_avg_noun = mds_3d_avg_noun.fit_transform(avg_noun_dist_matrix_prof.values)
                                mds_df_3d_avg_noun = pd.DataFrame(coords_3d_avg_noun, columns=['x', 'y', 'z'], index=all_langs_in_prof_comp)
                                mds_df_3d_avg_noun['Language'] = mds_df_3d_avg_noun.index
                                mds_df_3d_avg_noun['Label'] = mds_df_3d_avg_noun.index

                                pca_avg_noun_dir_prof = os.path.join(visualizations_dir, "pca", "language_level", "AvgNounDistance")
                                if len(mds_df_2d_avg_noun) >=2:
                                     plot_pca(mds_df_2d_avg_noun, os.path.join(pca_avg_noun_dir_prof, f"{prof_base_filename}_avg_noun_pca_2d"), f"Language PCA ({prof}) - AvgNounDistance (MDS) - {api_model}", 'Language', True, dimensions=2, use_existing_coords=True)
                                if len(mds_df_3d_avg_noun) >=2:
                                     plot_pca(mds_df_3d_avg_noun, os.path.join(pca_avg_noun_dir_prof, f"{prof_base_filename}_avg_noun_pca_3d"), f"3D Language PCA ({prof}) - AvgNounDistance (MDS) - {api_model}", 'Language', True, dimensions=3, use_existing_coords=True)

                    except Exception as e: print(f"Error generating AvgNounDistance visualizations for {prof}: {e}")

                # 3. Combined Language-Family PCA (per proficiency)
                if lang_to_family_map:
                    try:
                        prof_family_embeddings = []
                        prof_lang_to_family_dict = {}
                        family_counts_prof = {}

                        for lang_pca_entry in lang_pca_df_prof['Language']:
                            lang_code_prof = lang_pca_entry.split(' # ')[0] if ' # ' in lang_pca_entry else lang_pca_entry
                            family_prof = lang_to_family_map.get(lang_code_prof, "Unknown")
                            prof_lang_to_family_dict[lang_pca_entry] = family_prof
                            family_counts_prof[family_prof] = family_counts_prof.get(family_prof, 0) + 1
                        
                        for family_name_prof, count_prof in family_counts_prof.items():
                            if count_prof > 0:
                                family_langs_prof = [l for l in lang_pca_df_prof['Language'] if prof_lang_to_family_dict.get(l) == family_name_prof]
                                if not family_langs_prof: continue # Should not happen if count > 0

                                embeddings_for_family_prof = np.vstack([
                                    lang_pca_df_prof[lang_pca_df_prof['Language'] == lang_val]['Embedding'].values[0]
                                    for lang_val in family_langs_prof
                                ])
                                avg_embedding_prof = np.mean(embeddings_for_family_prof, axis=0)
                                prof_family_embeddings.append({
                                    'LanguageFamily': family_name_prof,
                                    'Label': family_name_prof,
                                    'Embedding': avg_embedding_prof
                                })
                        prof_family_df_for_plot = pd.DataFrame(prof_family_embeddings)
                        if not prof_family_df_for_plot.empty and len(prof_family_df_for_plot) >=1:
                            combined_pca_path_prof = os.path.join(visualizations_dir, "pca", "language_level", f"{prof_base_filename}_language_family_combined_pca.html") # Store with other lang-level PCAs
                            plot_combined_family_language_pca(lang_pca_df_prof, prof_family_df_for_plot, combined_pca_path_prof, f"Languages and Families PCA ({prof}) - {api_model}", prof_lang_to_family_dict)
                    except Exception as e: print(f"Error generating combined family-language PCA for {prof}: {e}")

        else: # Consolidated visualizations (not per-proficiency)
            print("  Generating consolidated visualizations for all proficiencies combined.")
            if not isinstance(language_distance_matrices, pd.DataFrame) or language_distance_matrices.empty:
                print("    Skipping consolidated visualizations: Invalid or empty main distance matrix.")
                return language_embeddings_df # Return what we have so far
            
            consolidated_dist_matrix = language_distance_matrices
            languages_list_all = consolidated_dist_matrix.index.tolist()

            # Create a consolidated language_pca_df for visualizations
            lang_pca_df_all = pd.DataFrame({
                'Language': language_embeddings_df['Language'].unique(), # All unique languages across all profs
                'Label': language_embeddings_df['Language'].unique(),
                 # For consolidated, we need a single embedding per language. 
                 # This requires re-calculating mean embeddings across all proficiencies if not already done.
                 # The initial language_embeddings_df might be by proficiency. We need one without it.
                'Embedding': [language_embeddings_df[language_embeddings_df['Language'] == lang]['Embedding'].mean(axis=0) 
                              if 'Proficiency' in language_embeddings_df.columns else 
                              language_embeddings_df[language_embeddings_df['Language'] == lang]['Embedding'].iloc[0]
                             for lang in language_embeddings_df['Language'].unique()]
            })
            lang_pca_df_all.drop_duplicates(subset=['Language'], inplace=True) # Ensure unique languages

            if len(lang_pca_df_all) < 2:
                print("    Skipping consolidated visualizations: Less than 2 unique languages after aggregation.")
                return language_embeddings_df

            # 1. LanguageEmbeddingDistance Visualizations (Consolidated)
            dendrogram_path_all = os.path.join(visualizations_dir, "dendrograms", "language_level", "LanguageEmbeddingDistance", f"{base_filename}_language_dendrogram.png")
            try: plot_language_dendrogram(consolidated_dist_matrix, languages_list_all, dendrogram_path_all, f"Language Clustering - {api_model}")
            except Exception as e: print(f"Error generating consolidated dendrogram: {e}")
            heatmap_path_all = os.path.join(visualizations_dir, "heatmaps", "language_level", "LanguageEmbeddingDistance", f"{base_filename}_language_heatmap.html")
            try: plot_interactive_heatmap(consolidated_dist_matrix, heatmap_path_all, f"Language Distance Heatmap - {api_model}")
            except Exception as e: print(f"Error generating consolidated heatmap: {e}")

            pca_dir_all = os.path.join(visualizations_dir, "pca", "language_level", "LanguageEmbeddingDistance")
            try:
                plot_pca(lang_pca_df_all, os.path.join(pca_dir_all, f"{base_filename}_language_pca_2d"), f"Language PCA - LanguageEmbeddingDistance - {api_model}", 'Language', True, dimensions=2)
                plot_pca(lang_pca_df_all, os.path.join(pca_dir_all, f"{base_filename}_language_pca_3d"), f"3D Language PCA - LanguageEmbeddingDistance - {api_model}", 'Language', True, dimensions=3)
            except Exception as e: print(f"Error generating consolidated PCA (LanguageEmbeddingDistance): {e}")

            tsne_dir_all = os.path.join(visualizations_dir, "tsne", "language_level", "LanguageEmbeddingDistance")
            try:
                plot_tsne(lang_pca_df_all, os.path.join(tsne_dir_all, f"{base_filename}_language_tsne_2d.html"), f"Language t-SNE - LanguageEmbeddingDistance - {api_model}", 'Language', True, original_distance_matrix=consolidated_dist_matrix, dimensions=2)
                plot_tsne(lang_pca_df_all, os.path.join(tsne_dir_all, f"{base_filename}_language_tsne_3d.html"), f"3D Language t-SNE - LanguageEmbeddingDistance - {api_model}", 'Language', True, original_distance_matrix=consolidated_dist_matrix, dimensions=3)
            except Exception as e: print(f"Error generating consolidated t-SNE (LanguageEmbeddingDistance): {e}")
            
            umap_dir_all = os.path.join(visualizations_dir, "umap", "language_level", "LanguageEmbeddingDistance")
            try:
                plot_umap(lang_pca_df_all, os.path.join(umap_dir_all, f"{base_filename}_language_umap_2d.html"), f"Language UMAP - LanguageEmbeddingDistance - {api_model}", 'Language', True, original_distance_matrix=consolidated_dist_matrix, dimensions=2)
                plot_umap(lang_pca_df_all, os.path.join(umap_dir_all, f"{base_filename}_language_umap_3d.html"), f"3D Language UMAP - LanguageEmbeddingDistance - {api_model}", 'Language', True, original_distance_matrix=consolidated_dist_matrix, dimensions=3)
            except Exception as e: print(f"Error generating consolidated UMAP (LanguageEmbeddingDistance): {e}")

            # 2. AvgNounDistance Visualizations (Consolidated)
            if os.path.exists(consolidated_lang_comp_path):
                try:
                    all_comp_df = pd.read_csv(consolidated_lang_comp_path)
                    if 'AvgNounDistance' in all_comp_df.columns:
                        avg_noun_dist_matrix_all = pd.pivot_table(all_comp_df, values='AvgNounDistance', index='Language1', columns='Language2', fill_value=0)
                        all_langs_in_comp = sorted(list(set(all_comp_df['Language1']) | set(all_comp_df['Language2'])))
                        avg_noun_dist_matrix_all = avg_noun_dist_matrix_all.reindex(index=all_langs_in_comp, columns=all_langs_in_comp, fill_value=0)
                        for lang1_idx, lang1_val in enumerate(all_langs_in_comp):
                            for lang2_idx, lang2_val in enumerate(all_langs_in_comp):
                                if lang1_idx < lang2_idx:
                                    val = avg_noun_dist_matrix_all.loc[lang1_val, lang2_val]
                                    if pd.isna(val) or val == 0:
                                        val_rev = avg_noun_dist_matrix_all.loc[lang2_val, lang1_val]
                                        if not pd.isna(val_rev):
                                            avg_noun_dist_matrix_all.loc[lang1_val, lang2_val] = val_rev
                                        avg_noun_dist_matrix_all.loc[lang2_val, lang1_val] = avg_noun_dist_matrix_all.loc[lang1_val, lang2_val]
                        np.fill_diagonal(avg_noun_dist_matrix_all.values, 0)
                        
                        if not avg_noun_dist_matrix_all.empty and len(all_langs_in_comp) >=2:
                            heatmap_avg_noun_path_all = os.path.join(visualizations_dir, "heatmaps", "language_level", "AvgNounDistance", f"{base_filename}_avg_noun_dist_heatmap.html")
                            plot_interactive_heatmap(avg_noun_dist_matrix_all, heatmap_avg_noun_path_all, f"Average Noun Distance Heatmap - {api_model}")

                            from sklearn.manifold import MDS # Import here as it's specific to this block
                            mds_2d_avg_noun_all = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
                            coords_2d_avg_noun_all = mds_2d_avg_noun_all.fit_transform(avg_noun_dist_matrix_all.values)
                            mds_df_2d_avg_noun_all = pd.DataFrame(coords_2d_avg_noun_all, columns=['x', 'y'], index=all_langs_in_comp)
                            mds_df_2d_avg_noun_all['Language'] = mds_df_2d_avg_noun_all.index
                            mds_df_2d_avg_noun_all['Label'] = mds_df_2d_avg_noun_all.index

                            mds_3d_avg_noun_all = MDS(n_components=3, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
                            coords_3d_avg_noun_all = mds_3d_avg_noun_all.fit_transform(avg_noun_dist_matrix_all.values)
                            mds_df_3d_avg_noun_all = pd.DataFrame(coords_3d_avg_noun_all, columns=['x', 'y', 'z'], index=all_langs_in_comp)
                            mds_df_3d_avg_noun_all['Language'] = mds_df_3d_avg_noun_all.index
                            mds_df_3d_avg_noun_all['Label'] = mds_df_3d_avg_noun_all.index

                            pca_avg_noun_dir_all = os.path.join(visualizations_dir, "pca", "language_level", "AvgNounDistance")
                            if len(mds_df_2d_avg_noun_all) >= 2:
                                plot_pca(mds_df_2d_avg_noun_all, os.path.join(pca_avg_noun_dir_all, f"{base_filename}_avg_noun_pca_2d"), f"Language PCA - AvgNounDistance (MDS) - {api_model}", 'Language', True, dimensions=2, use_existing_coords=True)
                            if len(mds_df_3d_avg_noun_all) >= 2:
                                plot_pca(mds_df_3d_avg_noun_all, os.path.join(pca_avg_noun_dir_all, f"{base_filename}_avg_noun_pca_3d"), f"3D Language PCA - AvgNounDistance (MDS) - {api_model}", 'Language', True, dimensions=3, use_existing_coords=True)

                except Exception as e: print(f"Error generating consolidated AvgNounDistance visualizations: {e}")

            # 3. Combined Language-Family PCA (Consolidated)
            if lang_to_family_map:
                try:
                    all_family_embeddings = []
                    all_lang_to_family_dict = {}
                    family_counts_all = {}

                    for lang_pca_entry_all in lang_pca_df_all['Language']:
                        lang_code_all = lang_pca_entry_all.split(' # ')[0] if ' # ' in lang_pca_entry_all else lang_pca_entry_all
                        family_all = lang_to_family_map.get(lang_code_all, "Unknown")
                        all_lang_to_family_dict[lang_pca_entry_all] = family_all
                        family_counts_all[family_all] = family_counts_all.get(family_all, 0) + 1
                    
                    for family_name_all, count_all in family_counts_all.items():
                        if count_all > 0:
                            family_langs_all = [l for l in lang_pca_df_all['Language'] if all_lang_to_family_dict.get(l) == family_name_all]
                            if not family_langs_all: continue

                            embeddings_for_family_all = np.vstack([
                                lang_pca_df_all[lang_pca_df_all['Language'] == lang_val]['Embedding'].values[0]
                                for lang_val in family_langs_all
                            ])
                            avg_embedding_all = np.mean(embeddings_for_family_all, axis=0)
                            all_family_embeddings.append({
                                'LanguageFamily': family_name_all,
                                'Label': family_name_all,
                                'Embedding': avg_embedding_all
                            })
                    all_family_df_for_plot = pd.DataFrame(all_family_embeddings)
                    if not all_family_df_for_plot.empty and len(all_family_df_for_plot) >=1:
                        combined_pca_path_all = os.path.join(visualizations_dir, "pca", "language_level", f"{base_filename}_language_family_combined_pca.html")
                        plot_combined_family_language_pca(lang_pca_df_all, all_family_df_for_plot, combined_pca_path_all, f"Languages and Families PCA - {api_model}", all_lang_to_family_dict)
                except Exception as e: print(f"Error generating consolidated combined family-language PCA: {e}")

    except Exception as e_outer_lang_level:
        print(f"Major error in language-level analysis and visualization: {e_outer_lang_level}")
        traceback.print_exc()
        return pd.DataFrame() # Return empty df on major error

    print("--- Finished Comprehensive Language-Level Analysis and Visualization ---")
    return language_embeddings_df

def run_family_level_analysis_and_visualization(
    language_embeddings_df, # Should be the one from all proficiencies / overall
    lang_to_family_map,
    provider_folder, # Used for stat report output path construction
    base_filename,
    api_model,
    embeddings_dir,
    distances_dir,
    visualizations_dir
):
    """
    Orchestrates the full suite of language family-level analysis, including:
    - Calculating language family embeddings and distances from language embeddings.
    - Generating language family comparison CSVs and statistical reports.
    - Producing all language family-level visualizations (dendrograms, heatmaps, PCA, t-SNE, UMAP).

    Args:
        language_embeddings_df (pd.DataFrame): DataFrame with language-level embeddings (aggregated across proficiencies).
        lang_to_family_map (dict): Mapping of language codes to family names.
        provider_folder (str): Base output directory for the current analysis run.
        base_filename (str): Base name for output files.
        api_model (str): Name of the API model used.
        embeddings_dir (str): Directory to save family-level embeddings.
        distances_dir (str): Directory to save family-level distances.
        visualizations_dir (str): Directory to save family-level visualizations.

    Returns:
        tuple: (family_embeddings_df, family_distances_df) or (None, None) if an error occurs.
    """
    print("\n--- Starting Comprehensive Language Family-Level Analysis and Visualization ---")
    
    if language_embeddings_df is None or language_embeddings_df.empty:
        print("Error: No language_embeddings_df provided for family analysis. Skipping.")
        return None, None
    if not lang_to_family_map:
        print("Error: No lang_to_family_map provided. Skipping family analysis.")
        return None, None

    family_embeddings_output_dir = os.path.join(embeddings_dir, "language_family_level")
    family_distances_output_dir = os.path.join(distances_dir, "language_family_level")
    # Ensure these specific output directories exist before calling the calculation function
    os.makedirs(family_embeddings_output_dir, exist_ok=True)
    os.makedirs(family_distances_output_dir, exist_ok=True)

    family_analysis_prefix = f"global_{base_filename}_families"
    
    family_embeddings_df, family_distances_df = calculate_language_family_embeddings_and_distances(
        language_embeddings_df, 
        lang_to_family_map, 
        family_embeddings_output_dir, # Corrected to pass the specific directory
        family_distances_output_dir,  # Corrected to pass the specific directory
        base_filename_prefix=family_analysis_prefix
    )

    if family_embeddings_df is None or family_embeddings_df.empty:
        print("No family embeddings were generated. Skipping family visualizations.")
        print("--- Finished Language Family-Level Analysis and Visualization (early exit) ---")
        return family_embeddings_df, family_distances_df # Return what we have

    print(f"\nGenerating language family visualizations...")

    # For visualizations, we need to handle if embeddings/distances are per-proficiency or global.
    # The family_embeddings_df from calculate_language_family_embeddings_and_distances might have a 'Proficiency' column.
    has_prof_in_family_df = 'Proficiency' in family_embeddings_df.columns and family_embeddings_df['Proficiency'].nunique() > 1

    # Import visualization functions locally if not already at module level
    from lr_analysis_functions.visualization import plot_language_dendrogram, plot_interactive_heatmap, plot_pca, plot_tsne, plot_umap, plot_combined_family_language_pca
    from lr_analysis_functions.utils import sanitize_name

    if has_prof_in_family_df:
        print("  Generating family visualizations per proficiency level.")
        for prof, prof_family_embed_df in family_embeddings_df.groupby('Proficiency'):
            prof_safe = sanitize_name(str(prof))
            prof_fam_base_filename = f"{family_analysis_prefix}_{prof_safe}"
            families_in_prof = prof_family_embed_df['LanguageFamily'].unique()

            if len(families_in_prof) < 2:
                print(f"    Skipping family visualizations for proficiency '{prof}': Not enough families ({len(families_in_prof)}).")
                continue

            # Create distance matrix for this proficiency's families if family_distances_df exists and has proficiency
            prof_family_dist_matrix = pd.DataFrame()
            if family_distances_df is not None and not family_distances_df.empty and 'Proficiency' in family_distances_df.columns:
                prof_dist_subset_df = family_distances_df[family_distances_df['Proficiency'] == prof]
                if not prof_dist_subset_df.empty:
                    prof_family_dist_matrix = pd.pivot_table(prof_dist_subset_df, values='Distance', index='Family1', columns='Family2', fill_value=0)
                    # Symmetrize
                    all_fams = sorted(list(set(prof_dist_subset_df['Family1']) | set(prof_dist_subset_df['Family2'])))
                    prof_family_dist_matrix = prof_family_dist_matrix.reindex(index=all_fams, columns=all_fams, fill_value=0)
                    for f1_idx, f1_val in enumerate(all_fams):
                        for f2_idx, f2_val in enumerate(all_fams):
                            if f1_idx < f2_idx:
                                val = prof_family_dist_matrix.loc[f1_val, f2_val]
                                if pd.isna(val) or val == 0:
                                     val_rev = prof_family_dist_matrix.loc[f2_val, f1_val]
                                     if not pd.isna(val_rev): prof_family_dist_matrix.loc[f1_val, f2_val] = val_rev
                                prof_family_dist_matrix.loc[f2_val, f1_val] = prof_family_dist_matrix.loc[f1_val, f2_val]
                            prof_family_dist_matrix.loc[f2_val, f1_val] = prof_family_dist_matrix.loc[f1_val, f2_val]
                    np.fill_diagonal(prof_family_dist_matrix.values, 0)
            
            if not prof_family_dist_matrix.empty:
                dendro_path = os.path.join(visualizations_dir, "dendrograms", "language_family_level", "LanguageEmbeddingDistance", f"{prof_fam_base_filename}_family_dendrogram.png")
                try: plot_language_dendrogram(prof_family_dist_matrix, families_in_prof, dendro_path, f"Language Family Clustering ({prof}) - {api_model}")
                except Exception as e: print(f"Err fam dendro prof '{prof}': {e}")
                heatmap_path = os.path.join(visualizations_dir, "heatmaps", "language_family_level", "LanguageEmbeddingDistance", f"{prof_fam_base_filename}_family_heatmap.html")
                try: plot_interactive_heatmap(prof_family_dist_matrix, heatmap_path, f"Language Family Distance Heatmap ({prof}) - {api_model}")
                except Exception as e: print(f"Err fam heatmap prof '{prof}': {e}")

            # PCA for families (per proficiency)
            fam_pca_dir = os.path.join(visualizations_dir, "pca", "language_family_level")
            try:
                plot_pca(prof_family_embed_df, os.path.join(fam_pca_dir, f"{prof_fam_base_filename}_family_pca_2d"), f"Language Family PCA ({prof}) - {api_model}", 'LanguageFamily', True, dimensions=2)
                plot_pca(prof_family_embed_df, os.path.join(fam_pca_dir, f"{prof_fam_base_filename}_family_pca_3d"), f"3D Language Family PCA ({prof}) - {api_model}", 'LanguageFamily', True, dimensions=3)
            except Exception as e: print(f"Err fam PCA prof '{prof}': {e}")
    else:
        print("  Generating consolidated family visualizations.")
        all_families_list = family_embeddings_df['LanguageFamily'].unique()
        if len(all_families_list) < 2:
            print("    Skipping consolidated family visualizations: Not enough families.")
        else:
            global_family_dist_matrix = pd.DataFrame()
            if family_distances_df is not None and not family_distances_df.empty:
                # Assuming if no proficiency in family_embeddings_df, then family_distances_df is also global
                global_family_dist_matrix = pd.pivot_table(family_distances_df, values='Distance', index='Family1', columns='Family2', fill_value=0)
                all_fams_global = sorted(list(set(family_distances_df['Family1']) | set(family_distances_df['Family2'])))
                global_family_dist_matrix = global_family_dist_matrix.reindex(index=all_fams_global, columns=all_fams_global, fill_value=0)
                for f1_idx, f1_val in enumerate(all_fams_global):
                    for f2_idx, f2_val in enumerate(all_fams_global):
                        if f1_idx < f2_idx:
                            val = global_family_dist_matrix.loc[f1_val, f2_val]
                            if pd.isna(val) or val == 0:
                                 val_rev = global_family_dist_matrix.loc[f2_val, f1_val]
                                 if not pd.isna(val_rev): global_family_dist_matrix.loc[f1_val, f2_val] = val_rev
                            global_family_dist_matrix.loc[f2_val, f1_val] = global_family_dist_matrix.loc[f1_val, f2_val]
                        global_family_dist_matrix.loc[f2_val, f1_val] = global_family_dist_matrix.loc[f1_val, f2_val]
                np.fill_diagonal(global_family_dist_matrix.values, 0)

            if not global_family_dist_matrix.empty:
                dendro_path_global = os.path.join(visualizations_dir, "dendrograms", "language_family_level", "LanguageEmbeddingDistance", f"{family_analysis_prefix}_family_dendrogram.png")
                try: plot_language_dendrogram(global_family_dist_matrix, all_families_list, dendro_path_global, f"Language Family Clustering - {api_model}")
                except Exception as e: print(f"Err global fam dendro: {e}")
                heatmap_path_global = os.path.join(visualizations_dir, "heatmaps", "language_family_level", "LanguageEmbeddingDistance", f"{family_analysis_prefix}_family_heatmap.html")
                try: plot_interactive_heatmap(global_family_dist_matrix, heatmap_path_global, f"Language Family Distance Heatmap - {api_model}")
                except Exception as e: print(f"Err global fam heatmap: {e}")

            # PCA for all families (global)
            fam_pca_dir_global = os.path.join(visualizations_dir, "pca", "language_family_level")
            try:
                plot_pca(family_embeddings_df, os.path.join(fam_pca_dir_global, f"{family_analysis_prefix}_family_pca_2d"), f"Language Family PCA - {api_model}", 'LanguageFamily', True, dimensions=2)
                plot_pca(family_embeddings_df, os.path.join(fam_pca_dir_global, f"{family_analysis_prefix}_family_pca_3d"), f"3D Language Family PCA - {api_model}", 'LanguageFamily', True, dimensions=3)
            except Exception as e: print(f"Err global fam PCA: {e}")

            # Combined Language-Family PCA (Global)
            # We need the global language_embeddings_df for this, which is the input to this function.
            if lang_to_family_map and not language_embeddings_df.empty:
                try:
                    # This global_lang_to_family_dict is for the overall language_embeddings_df
                    global_lang_to_family_dict = { 
                        (l_name if isinstance(l_name, str) else str(l_name)): 
                        lang_to_family_map.get((l_name.split(' # ')[0] if ' # ' in l_name else l_name), "Unknown") 
                        for l_name in language_embeddings_df['Language'].unique()
                    }
                    combined_pca_global_path = os.path.join(visualizations_dir, "pca", "language_family_level", f"{family_analysis_prefix}_combined_family_language_pca.html")
                    plot_combined_family_language_pca(language_embeddings_df, family_embeddings_df, combined_pca_global_path, f"Global Languages and Families PCA - {api_model}", global_lang_to_family_dict)
                except Exception as e: print(f"Err global combined PCA: {e}")

    # Global t-SNE and UMAP for language families (using family_embeddings_df)
    if not family_embeddings_df.empty and len(family_embeddings_df) >= 2:
        print("\n  Generating global language-family-level t-SNE and UMAP visualizations...")
        family_viz_df_global = family_embeddings_df.copy()
        family_color_by_global = 'LanguageFamily'
        if 'Proficiency' in family_viz_df_global.columns and family_viz_df_global['Proficiency'].nunique() > 1:
            family_viz_df_global['PlotLabel'] = family_viz_df_global.apply(lambda row: f"{row['LanguageFamily']} ({row.get('Proficiency', '')})", axis=1)
            family_color_by_global = 'Proficiency' # Or keep as LanguageFamily if preferred
        else:
            family_viz_df_global['PlotLabel'] = family_viz_df_global['LanguageFamily']
        family_viz_df_global['Label'] = family_viz_df_global['PlotLabel']

        # Create distance matrix for Kruskal Stress if possible (using family_distances_df)
        family_orig_dist_df_global = None
        if family_distances_df is not None and not family_distances_df.empty:
            try:
                # If proficiency specific, use the first proficiency for stress calc example or make it more complex
                # For now, if global, use it directly
                if not ('Proficiency' in family_distances_df.columns and family_distances_df['Proficiency'].nunique() > 1):
                    temp_stress_dist_df = pd.pivot_table(family_distances_df, values='Distance', index='Family1', columns='Family2', fill_value=0)
                    all_fams_stress = sorted(list(set(family_distances_df['Family1']) | set(family_distances_df['Family2'])))
                    temp_stress_dist_df = temp_stress_dist_df.reindex(index=all_fams_stress, columns=all_fams_stress, fill_value=0)
                    # Symmetrize for pdist
                    for r in range(len(all_fams_stress)):
                        for c in range(r + 1, len(all_fams_stress)):
                            val1 = temp_stress_dist_df.iloc[r,c]
                            val2 = temp_stress_dist_df.iloc[c,r]
                            if pd.isna(val1) or val1 == 0: temp_stress_dist_df.iloc[r,c] = val2
                            if pd.isna(val2) or val2 == 0: temp_stress_dist_df.iloc[c,r] = val1
                    np.fill_diagonal(temp_stress_dist_df.values, 0)
                    if not temp_stress_dist_df.empty and temp_stress_dist_df.shape[0] == len(family_viz_df_global['Label'].unique()):
                        family_orig_dist_df_global = temp_stress_dist_df.reindex(index=family_viz_df_global['Label'].unique(), columns=family_viz_df_global['Label'].unique())
            except Exception as e_fam_stress_prep: print(f"Error preparing family distance matrix for Kruskal stress: {e_fam_stress_prep}")

        tsne_fam_path = os.path.join(visualizations_dir, "tsne", "language_family_level", "LanguageEmbeddingDistance", f"global_{base_filename}_language_family_tsne.html")
        try: plot_tsne(family_viz_df_global, tsne_fam_path, f"Global Language Family Semantic Space (t-SNE) - {api_model}", family_color_by_global, True, family_orig_dist_df_global, 2)
        except Exception as e: print(f"Error global fam t-SNE 2D: {e}")
        tsne_fam_3d_path = os.path.join(visualizations_dir, "tsne", "language_family_level", "LanguageEmbeddingDistance", f"global_{base_filename}_language_family_tsne_3d.html")
        try: plot_tsne(family_viz_df_global, tsne_fam_3d_path, f"Global Language Family Semantic Space (t-SNE 3D) - {api_model}", family_color_by_global, True, family_orig_dist_df_global, 3)
        except Exception as e: print(f"Error global fam t-SNE 3D: {e}")

        umap_fam_path = os.path.join(visualizations_dir, "umap", "language_family_level", "LanguageEmbeddingDistance", f"global_{base_filename}_language_family_umap.html")
        try: plot_umap(family_viz_df_global, umap_fam_path, f"Global Language Family UMAP - {api_model}", family_color_by_global, True, family_orig_dist_df_global, 2)
        except Exception as e: print(f"Error global fam UMAP 2D: {e}")
        umap_fam_3d_path = os.path.join(visualizations_dir, "umap", "language_family_level", "LanguageEmbeddingDistance", f"global_{base_filename}_language_family_umap_3d.html")
        try: plot_umap(family_viz_df_global, umap_fam_3d_path, f"Global Language Family 3D UMAP - {api_model}", family_color_by_global, True, family_orig_dist_df_global, 3)
        except Exception as e: print(f"Error global fam UMAP 3D: {e}")

    print("--- Finished Language Family-Level Analysis and Visualization ---")
    return family_embeddings_df, family_distances_df
