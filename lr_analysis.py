"""
Language Representation Analysis

This is the main entry point for the language representation analysis tool.
It imports functionality from the lr_analysis_functions modules.
"""

import csv
import embedding_models
import glob
import json
import os
import pandas as pd
import sys
import time
import multiprocessing as mp
import traceback
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stderr
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE  # Add the TSNE import
from sklearn.manifold import MDS

# Import functionality from our modules
from lr_analysis_functions import (
    # Data loading
    load_data, load_noun_categories, load_language_families, load_grammatical_gender, 
    load_adjective_prompts, find_csv_files, list_api_providers, list_api_models,
    
    # Distance metrics
    compute_pairwise_distances, filter_distances, compute_jaccard_distances,
    
    # Visualization
    plot_tsne, plot_umap, plot_pca, plot_language_dendrogram, plot_interactive_heatmap,
    plot_combined_family_language_pca, plot_average_noun_differences, generate_standard_dim_reduction_plots,
    
    # Statistical analysis
    calculate_cohens_d, analyze_noun_type_distances, calculate_category_percentages,
    analyze_gender_language_distances, analyze_distance_metric_correlation,
    analyze_noun_level_comparisons,
    
    # Language processing
    calculate_language_embeddings, compute_language_distances, calculate_language_family_embeddings_and_distances,
    generate_language_level_comparisons, generate_language_comparisons_df,
    run_language_level_analysis_and_visualization, run_family_level_analysis_and_visualization, # Added import
    
    # Utils
    sanitize_prompt, sanitize_category, sanitize_name, detect_optimal_device,
    create_directory_structure, create_category_directories,
    
    # Comprehensive analysis
    generate_combined_analysis, _generate_and_save_noun_level_distances_csv, _save_category_percentages_to_csv,
    process_and_save_consolidated_analysis
)


def analysis_pipeline(selected_file, model_name, api_provider, api_model, temperature, delimiter=","):
    """
    Main analysis pipeline for language representation studies.
    
    Handles all processing steps:
    - Data loading and embedding generation
    - Distance computation and dimensionality reduction
    - Visualization production (t-SNE, UMAP, PCA)
    - Statistical analysis
    - Result organization by proficiency level, noun category, and prompt
    
    Args:
        selected_file: Path to the input CSV file
        model_name: Name of the embedding model to use
        api_provider: Name of the API provider
        api_model: Name of the API model
        temperature: Temperature setting used for generation
        delimiter: CSV delimiter character (default: ",")
    
    Returns:
        None. Results are saved to the output directory structure.
    """
    # Get mapping of embedding model names to abbreviations
    model_abbreviations = embedding_models.get_embedding_model_abbreviations()
    
    # Generate analysis folder based on embedding model name
    embedding_model_short = os.path.basename(model_name).replace("/", "_")
    embedding_base_folder = f"embedding_analysis/{embedding_model_short}"
    os.makedirs(embedding_base_folder, exist_ok=True)

    # Get base file name from the input CSV to make the output folder unique
    base_filename = os.path.basename(selected_file).replace('.csv', '')
    
    # Get model abbreviation if it exists in our mapping
    model_abbr = ""
    for full_model_name, abbr in model_abbreviations.items():
        if full_model_name in model_name or os.path.basename(full_model_name) in model_name:
            model_abbr = abbr
            break
    
    # If we found an abbreviation, prepend it to the base filename
    if model_abbr:
        # Produce modified base filename with the embedding model abbreviation
        modified_base_filename = f"{model_abbr}-{base_filename}"
    else:
        # If no matching abbreviation is found, use the original filename
        modified_base_filename = base_filename
        print(f"Warning: No abbreviation found for embedding model {model_name}")

    # Produce provider-specific output folder structure
    provider_folder = f"{embedding_base_folder}/{api_provider}/{api_model}/{temperature}/{modified_base_filename}"
    os.makedirs(provider_folder, exist_ok=True)

    # Define output directory structure
    embeddings_dir = os.path.join(provider_folder, "01_embeddings")
    distances_dir = os.path.join(provider_folder, "02_distances")
    visualizations_dir = os.path.join(provider_folder, "03_visualizations")
    stats_dir = os.path.join(provider_folder, "04_statistical_reports")

    # Create directory structure
    create_directory_structure(provider_folder, embeddings_dir, distances_dir, visualizations_dir, stats_dir)

    print(f"\n{'='*80}")
    print(f"STARTING ANALYSIS PIPELINE")
    print(f"File: {selected_file}")
    print(f"Embedding model: {model_name}")
    print(f"API provider: {api_provider}")
    print(f"API model: {api_model}")
    print(f"Temperature: {temperature}")
    print(f"Output directory: {provider_folder}")
    print(f"{'='*80}\n")
    
    # Load necessary data
    print("1. Loading noun categories...")
    try:
        noun_categories_map = load_noun_categories()
        print(f"Loaded {len(noun_categories_map)} noun categories")
    except Exception as e:
        print(f"Warning: Could not load noun categories: {e}")
        noun_categories_map = {}

    print("1b. Loading grammatical gender mapping...")
    try:
        lang_to_gender_map = load_grammatical_gender()
        print(f"Loaded {len(lang_to_gender_map)} language gender mappings.")
    except Exception as e:
        print(f"Warning: Could not load grammatical gender mapping: {e}")
        lang_to_gender_map = {}
    
    print("1c. Loading language family mapping...")
    try:
        lang_to_family_map = load_language_families()
        print(f"Loaded {len(lang_to_family_map)} language family mappings.")
    except Exception as e:
        print(f"Warning: Could not load language family mapping: {e}")
        lang_to_family_map = {}
    
    # Load data
    print("2. Loading data...")
    start_time = time.time()
    df = load_data(selected_file, delimiter, noun_categories_map)
    if df.empty:
        print(f"Error: No valid data found in file {selected_file}")
        return
    print(f"Loaded {len(df)} rows with adjectives. Time: {time.time() - start_time:.2f}s")
    
    # Build embeddings
    print("\n3. Building embeddings...")
    df_with_embeddings = embedding_models.build_embeddings(df, model_name)
    if df_with_embeddings is None:
        print("Error building embeddings. Analysis pipeline stopped.")
        return
    
    # Save embeddings for future use (across all proficiency levels)
    embeddings_file_all_prof = os.path.join(embeddings_dir, f"all_proficiencies_{base_filename}.pkl")
    print(f"\nSaving embeddings to {embeddings_file_all_prof}")
    df_with_embeddings.to_pickle(embeddings_file_all_prof)
    print(f"Embeddings saved successfully.")
    
    # Process each proficiency level
    proficiency_levels = sorted(df_with_embeddings['Proficiency'].unique())
    print(f"\nProcessing {len(proficiency_levels)} proficiency levels: {', '.join(proficiency_levels)}")
    
    # Initialize master DataFrame for comprehensive analysis
    all_comprehensive_analyses = []
    
    # Process each proficiency level
    processed_prof_count = 0
    successful_prof_count = 0
    
    # The rest of the analysis pipeline would go here,
    # but has been moved to the modules in lr_analysis_functions
    
    # For brevity, we'll stop at this point in the main file, as the
    # detailed analysis code is now in our modular functions
    
    # Process each proficiency level
    for proficiency in proficiency_levels:
        prof_df = df_with_embeddings[df_with_embeddings['Proficiency'] == proficiency].copy()
        processed_prof_count += 1
        
        if prof_df.empty:
            print(f"No data for proficiency level '{proficiency}', skipping")
            continue
            
        # Generate proficiency-specific folder only if we have data
        prof_folder = f"{provider_folder}/{proficiency}"
        os.makedirs(prof_folder, exist_ok=True)
        
        print(f"\n{'*'*80}")
        print(f"PROCESSING PROFICIENCY LEVEL: {proficiency} ({processed_prof_count}/{len(proficiency_levels)})")
        print(f"Data rows: {len(prof_df)}")
        print(f"Output directory: {prof_folder}")
        print(f"{'*'*80}\n")
        
        try:
            # Process each prompt category separately
            prompt_categories = sorted(prof_df['Prompt'].unique())
            processed_prompt_count = 0
            for prompt in prompt_categories:
                processed_prompt_count += 1
                print(f"\nProcessing prompt: '{prompt}' ({processed_prompt_count}/{len(prompt_categories)})")
                
                # Process the prompt data
                process_prompt_data(prof_df, prompt, proficiency, provider_folder, base_filename, api_model,
                                embeddings_dir, distances_dir, visualizations_dir, stats_dir, all_comprehensive_analyses,
                                lang_to_gender_map, lang_to_family_map)
            
            # Save embeddings for this proficiency (all prompts and categories)
            prof_embeddings_filename = f"{base_filename}_{proficiency}_embeddings.pkl"
            prof_embeddings_file_path = os.path.join(embeddings_dir, "proficiency_specific", prof_embeddings_filename)
            # Ensure the specific directory exists (it should from above)
            os.makedirs(os.path.dirname(prof_embeddings_file_path), exist_ok=True)
            print(f"Saving {proficiency} embeddings to {prof_embeddings_file_path}")
            prof_df.to_pickle(prof_embeddings_file_path)
            
            successful_prof_count += 1
            print(f"\nSuccessfully completed processing for proficiency level '{proficiency}'")
            
        except Exception as e:
            print(f"\n!!! ERROR processing proficiency level '{proficiency}': {e}")
            traceback.print_exc()
            print(f"Continuing with next proficiency level...")
    
    print(f"\n{'='*40}")
    print(f"PROFICIENCY LEVEL PROCESSING SUMMARY:")
    print(f"  Total levels: {len(proficiency_levels)}")
    print(f"  Successfully processed: {successful_prof_count}")
    print(f"  Failed: {len(proficiency_levels) - successful_prof_count}")
    print(f"{'='*40}\n")
    
    # Generate a consolidated analysis file across all proficiency levels
    if all_comprehensive_analyses:
        print("\nGenerating consolidated analysis file across all proficiency levels...")
        consolidated_df = process_and_save_consolidated_analysis(
            all_comprehensive_analyses,
            base_filename,
            distances_dir
        )
        
        # Run analysis on the consolidated data if it was successfully created
        if not consolidated_df.empty:
            stats_prefix_consolidated_gender = os.path.join(stats_dir, "gender_language_analysis", f"all_proficiencies_consolidated_{base_filename}")
            
            # Ensure the gender analysis directory exists
            consolidated_gender_dir = os.path.dirname(stats_prefix_consolidated_gender)
            if not os.path.exists(consolidated_gender_dir):
                print(f"Creating consolidated gender language analysis directory: {consolidated_gender_dir}")
                os.makedirs(consolidated_gender_dir, exist_ok=True)
            
            print(f"\nGenerating consolidated gender language analysis across all proficiency levels...")
            
            # Run the analysis with verification (aggregated analysis for all languages)
            gender_results = analyze_gender_language_distances(consolidated_df.copy(), lang_to_gender_map, stats_prefix_consolidated_gender)
            
            # Now perform per-language gender analysis for the consolidated data
            if 'Language1' in consolidated_df.columns and 'Language2' in consolidated_df.columns:
                # Get unique languages from both Language1 and Language2 columns
                languages = set(consolidated_df['Language1'].unique()) | set(consolidated_df['Language2'].unique())
                print(f"Performing per-language gender language analysis for consolidated data with {len(languages)} unique languages")
                
                # Create a directory for language-specific gender language analyses
                language_specific_gender_dir = os.path.join(stats_dir, "gender_language_analysis", "per_language")
                os.makedirs(language_specific_gender_dir, exist_ok=True)
                
                # Process each language
                for language in sorted(languages):
                    try:
                        # Create language-specific prefix for files
                        lang_sanitized = sanitize_name(language)
                        lang_gender_prefix = os.path.join(language_specific_gender_dir, f"all_proficiencies_consolidated_{base_filename}_{lang_sanitized}")
                        
                        # Perform gender analysis for this language
                        analyze_gender_language_distances(consolidated_df.copy(), lang_to_gender_map, lang_gender_prefix, language=language)
                    except Exception as e_lang:
                        print(f"Error processing gender analysis for language '{language}' in consolidated data: {e_lang}")
                        traceback.print_exc()
            
            # Verify the main output file - the detailed stats text file is most important
            gender_lang_stats_file = f"{stats_prefix_consolidated_gender}_gender_lang_stats.txt"
            if os.path.exists(gender_lang_stats_file):
                print(f"Confirmed: Consolidated gender language analysis saved to: {gender_lang_stats_file}")
                with open(gender_lang_stats_file, 'r') as f:
                    # Verify content has the expected format (ANOVA tables, etc.)
                    content = f.read(500)  # Just check first 500 chars
                    if "Gender Language Analysis" in content and "Grammatical Gender Category Counts" in content:
                        print("Content verification: Gender language analysis file has expected format")
                    else:
                        print("Warning: Gender language analysis file might not have expected content format")
            else:
                print(f"Warning: Consolidated gender language analysis file not found at: {gender_lang_stats_file}")
                
                # Try to recreate the analysis with more explicit error handling
                try:
                    print("Attempting to recreate the gender language analysis...")
                    os.makedirs(os.path.dirname(stats_prefix_consolidated_gender), exist_ok=True)
                    analyze_gender_language_distances(consolidated_df.copy(), lang_to_gender_map, stats_prefix_consolidated_gender)
                    
                    if os.path.exists(gender_lang_stats_file):
                        print(f"Successfully recreated gender language analysis file at: {gender_lang_stats_file}")
                    else:
                        print(f"Failed to recreate gender language analysis file at: {gender_lang_stats_file}")
                except Exception as e_retry:
                    print(f"Error recreating gender language analysis: {e_retry}")
                    traceback.print_exc()
                
            # Call correlation analysis for the final consolidated_df
            stats_prefix_consolidated_corr = os.path.join(stats_dir, "correlations", f"all_proficiencies_consolidated_{base_filename}")
            # Ensure the base 'correlations' directory exists
            os.makedirs(os.path.join(stats_dir, "correlations"), exist_ok=True) 
            
            # Perform aggregated correlation analysis (all languages)
            analyze_distance_metric_correlation(consolidated_df.copy(), stats_prefix_consolidated_corr)
            
            # Now perform per-language correlation analysis for the consolidated data
            if 'Language1' in consolidated_df.columns and 'Language2' in consolidated_df.columns:
                # Get unique languages from both Language1 and Language2 columns
                languages = set(consolidated_df['Language1'].unique()) | set(consolidated_df['Language2'].unique())
                print(f"Performing per-language correlation analysis for consolidated data with {len(languages)} unique languages")
                
                # Create a directory for language-specific correlation analyses
                language_specific_corr_dir = os.path.join(stats_dir, "correlations", "per_language")
                os.makedirs(language_specific_corr_dir, exist_ok=True)
                
                # Process each language
                for language in sorted(languages):
                    try:
                        # Create language-specific prefix for files
                        lang_sanitized = sanitize_name(language)
                        lang_corr_prefix = os.path.join(language_specific_corr_dir, f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_all_categories_{lang_sanitized}")
                        
                        # Calculate correlations for this language
                        analyze_distance_metric_correlation(consolidated_df.copy(), lang_corr_prefix, language=language)
                    except Exception as e_lang:
                        print(f"Error processing correlation for language '{language}' in consolidated data: {e_lang}")
                        traceback.print_exc()

    else: # This is if all_comprehensive_analyses is empty
        print("\nWarning: No comprehensive analyses were generated, cannot produce consolidated file or run consolidated stats")
    
    # ///////////////////////////////////////////////////////////////////////////
    # Language-Level Analysis and Visualization (using the new centralized function)
    # ///////////////////////////////////////////////////////////////////////////
    # The run_language_level_analysis_and_visualization function now handles 
    # calculating language_embeddings_df internally and all associated outputs.
    language_embeddings_df = run_language_level_analysis_and_visualization(
        df_with_embeddings=df_with_embeddings,
        provider_folder=provider_folder, 
        base_filename=base_filename,
        api_model=api_model,
        embeddings_dir=embeddings_dir,
        distances_dir=distances_dir,
        visualizations_dir=visualizations_dir,
        lang_to_family_map=lang_to_family_map # Pass this for combined PCA plots inside the function
    )
    
    # ///////////////////////////////////////////////////////////////////////////
    # Language Family Analysis & Visualization (using the new centralized function)
    # ///////////////////////////////////////////////////////////////////////////
    if language_embeddings_df is not None and not language_embeddings_df.empty:
        run_family_level_analysis_and_visualization(
            language_embeddings_df=language_embeddings_df, # This is the df returned by the language-level function
            lang_to_family_map=lang_to_family_map,
            provider_folder=provider_folder,
            base_filename=base_filename,
            api_model=api_model,
            embeddings_dir=embeddings_dir,
            distances_dir=distances_dir,
            visualizations_dir=visualizations_dir
        )
    else:
        print("Skipping language family analysis as language_embeddings_df is empty or None.")

    # The original traceback.print_exc() for the outer try-except block for language_embeddings_df calculation
    # was here. It's better handled if run_language_level_analysis_and_visualization returns None or an empty df on error.
    # The outer try-except in analysis_pipeline handles broader errors.
    # try:
        # ... original language and family processing ...
    # except Exception as e:
    #     print(f"Error in language embedding calculation or subsequent family analysis: {e}")
    #     traceback.print_exc() # This was the original location
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS PIPELINE COMPLETE")
    print(f"All results saved to {provider_folder}")
    print(f"{'='*80}\n")
    
    # Call clear_model_cache from the embedding_models module at the end of main() or when needed
    embedding_models.clear_model_cache()
    
    return


def process_prompt_data(prof_df, prompt, proficiency, provider_folder, base_filename, api_model,
                     embeddings_dir, distances_dir, visualizations_dir, stats_dir, all_comprehensive_analyses,
                     lang_to_gender_map, lang_to_family_map):
    """
    Process data for a specific prompt and proficiency level.
    This is the main function for handling a specific subset of the data.
    """
    prompt_df = prof_df[prof_df['Prompt'] == prompt].copy()
    
    if prompt_df.empty:
        print(f"  No data for prompt '{prompt}', skipping")
        return
        
    # Check if the prompt has any non-empty noun categories before creating the folder
    empty = True
    for noun_category in prompt_df['NounCategory'].unique():
        category_df = prompt_df[prompt_df['NounCategory'] == noun_category].copy()
        if not category_df.empty and len(category_df) >= 2:
            empty = False
            break
    
    if empty:
        print(f"  No usable data for prompt '{prompt}', skipping folder creation")
        return
        
    # Generate prompt-specific folder
    prompt_folder = f"{provider_folder}/{proficiency}/{sanitize_prompt(prompt)}"
    os.makedirs(prompt_folder, exist_ok=True)
    
    print(f"  Data rows: {len(prompt_df)}")
    
    # Process each noun category separately
    noun_categories_in_data = sorted(prompt_df['NounCategory'].unique())
    print(f"\nFound {len(noun_categories_in_data)} noun categories in prompt '{prompt}': {', '.join(sorted(noun_categories_in_data))}")
    processed_category_count = 0
    
    # Create a list to collect noun category dataframes for combined visualization
    noun_category_dataframes = []
    
    # Also create a list to collect comprehensive analyses for cross-distance scatter plots
    category_comprehensive_dfs = []
    
    for noun_category in noun_categories_in_data:
        processed_category_count += 1
        print(f"\nProcessing category {processed_category_count}/{len(noun_categories_in_data)}: '{noun_category}'")
        category_df = prompt_df[prompt_df['NounCategory'] == noun_category].copy()
        
        if category_df.empty:
            continue
            
        # Save category df for later combined visualization
        noun_category_dataframes.append(category_df)
            
        # Define detailed output paths for this specific category
        detail_path_segment = os.path.join(proficiency, sanitize_prompt(prompt), sanitize_category(noun_category))
        
        # Create required directories for this category
        create_category_directories(embeddings_dir, distances_dir, visualizations_dir, stats_dir, detail_path_segment)
        
        current_embeddings_detail_dir = os.path.join(embeddings_dir, "details", detail_path_segment)
        current_distances_noun_level_detail_dir = os.path.join(distances_dir, "noun_level_distances", "details", detail_path_segment)
        current_distances_comprehensive_detail_dir = os.path.join(distances_dir, "comprehensive_analysis", "details", detail_path_segment)
        current_distances_lang_comp_detail_dir = os.path.join(distances_dir, "noun_level_language_comparisons", "details", detail_path_segment)
        current_visualizations_tsne_detail_dir = os.path.join(visualizations_dir, "tsne", "details", detail_path_segment)
        current_visualizations_umap_detail_dir = os.path.join(visualizations_dir, "umap", "details", detail_path_segment)
        current_visualizations_pca_detail_dir = os.path.join(visualizations_dir, "pca", "details", detail_path_segment)
        current_stats_noun_type_detail_dir = os.path.join(stats_dir, "noun_type_analysis", "details", detail_path_segment)
        current_stats_category_perc_detail_dir = os.path.join(stats_dir, "category_percentage_analysis", "details", detail_path_segment)
        current_stats_gender_detail_dir = os.path.join(stats_dir, "gender_language_analysis", "details", detail_path_segment)
        current_stats_corr_detail_dir = os.path.join(stats_dir, "correlations", "details", detail_path_segment)

        print(f"\nProcessing noun category: '{noun_category}'")
        print(f"Data rows: {len(category_df)}")
        
        # Only proceed with analysis if we have enough data (at least 2 rows)
        if len(category_df) < 2:
            print(f"Not enough data for category '{noun_category}', need at least 2 rows")
            continue
        
        # Save category-specific embeddings
        category_embeddings_filename = f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_{sanitize_category(noun_category)}_embeddings.pkl"
        category_embeddings_file_path = os.path.join(current_embeddings_detail_dir, category_embeddings_filename)
        category_df.to_pickle(category_embeddings_file_path)
        
        # Generate distances CSVs
        filtered_df_for_comprehensive, jaccard_df_for_comprehensive = _generate_and_save_noun_level_distances_csv(
            category_df.copy(), # Pass a copy to be safe
            base_filename, 
            proficiency, 
            prompt, 
            noun_category, 
            current_distances_noun_level_detail_dir
        )

        distances_df = compute_pairwise_distances(category_df) # This is full cosine, potentially for t-SNE/UMAP
        
        # Generate standard dimensionality reduction visualizations for this category
        dim_red_plot_base_filename = f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_{sanitize_category(noun_category)}"
        dim_red_plot_title_core = f"{proficiency} - {prompt} - {noun_category}"
        
        generate_standard_dim_reduction_plots(
            df=category_df,
            base_filename_prefix=dim_red_plot_base_filename,
            title_core=dim_red_plot_title_core,
            color_by='Language',
            original_distance_matrix=distances_df, # Pass the full distance matrix for Kruskal stress if needed
            tsne_dir=current_visualizations_tsne_detail_dir,
            umap_dir=current_visualizations_umap_detail_dir,
            pca_dir=current_visualizations_pca_detail_dir,
            api_model_name=api_model
        )

        # Generate language comparisons
        lang_comp_filename = f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_{sanitize_category(noun_category)}_language_comparisons.csv"
        lang_comp_file_path = os.path.join(current_distances_lang_comp_detail_dir, lang_comp_filename)
        lang_comp_df = generate_language_comparisons_df(category_df, lang_comp_file_path)
        
        # Produce comprehensive analysis incorporating all the data
        comprehensive_filename = f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_{sanitize_category(noun_category)}_comprehensive.csv"
        comprehensive_file_path = os.path.join(current_distances_comprehensive_detail_dir, comprehensive_filename)
        try:
            comprehensive_df = generate_combined_analysis(category_df, filtered_df_for_comprehensive, jaccard_df_for_comprehensive, comprehensive_file_path)
            
            if not comprehensive_df.empty:
                all_comprehensive_analyses.append(comprehensive_df)
                
                # Also save for later scatter plot generation
                category_comprehensive_dfs.append(comprehensive_df)
                
                # Generate statistical analyses
                try:
                    # Noun type analysis
                    stats_prefix_cat = os.path.join(current_stats_noun_type_detail_dir, f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_{sanitize_category(noun_category)}")
                    print(f"\nGenerating noun type statistical analysis for '{noun_category}'...")
                    # Perform aggregated noun type analysis (all languages together)
                    analyze_noun_type_distances(comprehensive_df.copy(), output_file_path_prefix=stats_prefix_cat, expect_single_category=True)
                    
                    # Now perform per-language noun type analysis
                    if 'Language1' in comprehensive_df.columns and 'Language2' in comprehensive_df.columns:
                        # Get unique languages from both Language1 and Language2 columns
                        languages = set(comprehensive_df['Language1'].unique()) | set(comprehensive_df['Language2'].unique())
                        print(f"Performing per-language noun type analysis for {len(languages)} unique languages")
                        
                        # Create a directory for language-specific noun type analyses
                        language_specific_noun_type_dir = os.path.join(current_stats_noun_type_detail_dir, "per_language")
                        os.makedirs(language_specific_noun_type_dir, exist_ok=True)
                        
                        # Process each language
                        for language in sorted(languages):
                            try:
                                # Create language-specific prefix for files
                                lang_sanitized = sanitize_name(language)
                                lang_noun_type_prefix = os.path.join(language_specific_noun_type_dir, f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_{sanitize_category(noun_category)}_{lang_sanitized}")
                                
                                # Perform noun type analysis for this language
                                analyze_noun_type_distances(comprehensive_df.copy(), output_file_path_prefix=lang_noun_type_prefix, expect_single_category=True, language=language)
                            except Exception as e_lang:
                                print(f"Error processing noun type analysis for language '{language}': {e_lang}")
                                traceback.print_exc()
                    
                    # Gender language analysis
                    stats_prefix_gender_cat = os.path.join(current_stats_gender_detail_dir, f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_{sanitize_category(noun_category)}")
                    # Perform aggregated gender language analysis (all languages together)
                    analyze_gender_language_distances(comprehensive_df.copy(), lang_to_gender_map, stats_prefix_gender_cat)
                    
                    # Now perform per-language gender language analysis
                    if 'Language1' in comprehensive_df.columns and 'Language2' in comprehensive_df.columns:
                        # Get unique languages from both Language1 and Language2 columns
                        languages = set(comprehensive_df['Language1'].unique()) | set(comprehensive_df['Language2'].unique())
                        print(f"Performing per-language gender language analysis for {len(languages)} unique languages")
                        
                        # Create a directory for language-specific gender language analyses
                        language_specific_gender_dir = os.path.join(current_stats_gender_detail_dir, "per_language")
                        os.makedirs(language_specific_gender_dir, exist_ok=True)
                        
                        # Process each language
                        for language in sorted(languages):
                            try:
                                # Create language-specific prefix for files
                                lang_sanitized = sanitize_name(language)
                                lang_gender_prefix = os.path.join(language_specific_gender_dir, f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_{sanitize_category(noun_category)}_{lang_sanitized}")
                                
                                # Perform gender language analysis for this language
                                analyze_gender_language_distances(comprehensive_df.copy(), lang_to_gender_map, lang_gender_prefix, language=language)
                            except Exception as e_lang:
                                print(f"Error processing gender language analysis for language '{language}': {e_lang}")
                                traceback.print_exc()
                    
                    # Category percentages
                    if 'NounCategory' in comprehensive_df.columns:
                        print(f"\nGenerating category percentage analysis for '{noun_category}'...")
                        
                        # Ensure the comprehensive df has all required data
                        if not comprehensive_df['NounCategory'].isna().any():
                            # Calculate aggregated category statistics (all languages together)
                            category_percentages = calculate_category_percentages(comprehensive_df)
                            
                            # Create filename and path
                            percentages_filename = f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_{sanitize_category(noun_category)}_category_percentages.csv"
                            percentages_file_path = os.path.join(current_stats_category_perc_detail_dir, percentages_filename)
                            
                            # Verify the structure of the percentages dictionary
                            if category_percentages:
                                # Save the category percentages to CSV
                                success = _save_category_percentages_to_csv(category_percentages, percentages_file_path)
                                if success:
                                    print(f"Successfully saved aggregated category percentages for '{noun_category}' to: {percentages_file_path}")
                                    
                                    # Verify the file was created
                                    if not os.path.exists(percentages_file_path):
                                        print(f"Warning: Category percentages file not found at: {percentages_file_path}")
                                else:
                                    print(f"Failed to save category percentages for '{noun_category}' to: {percentages_file_path}")
                            else:
                                print(f"Warning: No category percentages data generated for '{noun_category}'")
                            
                            # Now perform per-language analysis
                            if 'Language1' in comprehensive_df.columns and 'Language2' in comprehensive_df.columns:
                                # Get unique languages from both Language1 and Language2 columns
                                languages = set(comprehensive_df['Language1'].unique()) | set(comprehensive_df['Language2'].unique())
                                print(f"Found {len(languages)} unique languages for per-language analysis")
                                
                                # Create a directory for language-specific analyses
                                language_specific_dir = os.path.join(current_stats_category_perc_detail_dir, "per_language")
                                os.makedirs(language_specific_dir, exist_ok=True)
                                
                                # Process each language
                                for language in sorted(languages):
                                    try:
                                        # Calculate statistics for this language
                                        lang_percentages = calculate_category_percentages(comprehensive_df, language=language)
                                        
                                        if lang_percentages:
                                            # Create language-specific filename
                                            lang_sanitized = sanitize_name(language)
                                            lang_percentages_filename = f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_{sanitize_category(noun_category)}_{lang_sanitized}_category_percentages.csv"
                                            lang_percentages_file_path = os.path.join(language_specific_dir, lang_percentages_filename)
                                            
                                            # Save the language-specific percentages
                                            success = _save_category_percentages_to_csv(lang_percentages, lang_percentages_file_path)
                                            if success:
                                                print(f"Saved category percentages for language '{language}' to: {lang_percentages_file_path}")
                                            else:
                                                print(f"Failed to save category percentages for language '{language}'")
                                    except Exception as e_lang:
                                        print(f"Error processing language '{language}': {e_lang}")
                                        traceback.print_exc()
                        else:
                            print(f"Warning: NounCategory column contains NaN values for '{noun_category}' - skipping category percentage analysis")
                    
                    # Correlation analysis
                    stats_prefix_corr_cat = os.path.join(current_stats_corr_detail_dir, f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_{sanitize_category(noun_category)}")
                    # Perform aggregated correlation analysis (all languages together)
                    analyze_distance_metric_correlation(comprehensive_df.copy(), stats_prefix_corr_cat)
                    
                    # Now perform per-language correlation analysis
                    if 'Language1' in comprehensive_df.columns and 'Language2' in comprehensive_df.columns:
                        # Get unique languages from both Language1 and Language2 columns
                        languages = set(comprehensive_df['Language1'].unique()) | set(comprehensive_df['Language2'].unique())
                        print(f"Performing per-language correlation analysis for {len(languages)} unique languages")
                        
                        # Create a directory for language-specific correlation analyses
                        language_specific_corr_dir = os.path.join(current_stats_corr_detail_dir, "per_language")
                        os.makedirs(language_specific_corr_dir, exist_ok=True)
                        
                        # Process each language
                        for language in sorted(languages):
                            try:
                                # Create language-specific prefix for files
                                lang_sanitized = sanitize_name(language)
                                lang_corr_prefix = os.path.join(language_specific_corr_dir, f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_all_categories_{lang_sanitized}")
                                
                                # Calculate correlations for this language
                                analyze_distance_metric_correlation(comprehensive_df.copy(), lang_corr_prefix, language=language)
                            except Exception as e_lang:
                                print(f"Error processing correlation for language '{language}' in consolidated data: {e_lang}")
                                traceback.print_exc()
                    
                    # Create a new directory for noun-level evaluations
                    current_stats_noun_level_dir = os.path.join(stats_dir, "noun_level_evaluations", "details", detail_path_segment)
                    os.makedirs(current_stats_noun_level_dir, exist_ok=True)
                    
                    # Perform noun-level evaluations
                    stats_prefix_noun_level = os.path.join(current_stats_noun_level_dir, f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_{sanitize_category(noun_category)}")
                    print(f"\nGenerating noun-level evaluations for '{noun_category}'...")
                    analyze_noun_level_comparisons(comprehensive_df.copy(), stats_prefix_noun_level, lang_to_family_map)
                    
                except Exception as e_stats:
                    print(f"Error during statistical analysis for '{noun_category}': {e_stats}")
                    traceback.print_exc()
                
        except Exception as e:
            print(f"Warning: Error generating comprehensive analysis: {e}")
            traceback.print_exc()
            print(f"Continuing with analysis...")

    # Prepare for creating cross-distance scatter plots
    scatter_plots_dir = os.path.join(visualizations_dir, "scatter_plots", "cross_distance_correlations", proficiency, sanitize_prompt(prompt))
    os.makedirs(scatter_plots_dir, exist_ok=True)
    
    # Also create cosine_jaccard_correlations subfolder
    cosine_jaccard_dir = os.path.join(visualizations_dir, "scatter_plots", "cosine_jaccard_correlations", proficiency, sanitize_prompt(prompt))
    os.makedirs(cosine_jaccard_dir, exist_ok=True)
    
    # Generate scatter plots comparing cosine vs jaccard distances
    if category_comprehensive_dfs:
        combined_comp_df = pd.concat(category_comprehensive_dfs, ignore_index=True)
        
        # Generate overall scatter plot (all categories)
        from lr_analysis_functions.visualization import plot_cross_distance_scatter
        
        # Create a scatter plot with all categories
        all_cat_scatter_path = os.path.join(cosine_jaccard_dir, f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_all_categories_distance_scatter.html")
        try:
            plot_cross_distance_scatter(
                combined_comp_df, 
                all_cat_scatter_path,
                f"Cosine vs Jaccard Distance - All Categories - {api_model} - {proficiency} - {prompt}",
                color_by='NounCategory',
                x_dist='CosineDistance',
                y_dist='JaccardDistance'
            )
            print(f"Cross-distance scatter plot for all categories saved to: {all_cat_scatter_path}")
            
            # Create an additional all_categories global scatter plot
            all_cat_global_scatter_path = os.path.join(
                visualizations_dir, "scatter_plots", "cross_distance_correlations", "all_categories",
                f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_all_categories_language_distance_scatter.html"
            )
            
            # For this plot, we'll use one distance type and color by language pairs
            combined_comp_df['LanguagePair'] = combined_comp_df['Language1'] + '-' + combined_comp_df['Language2']
            plot_cross_distance_scatter(
                combined_comp_df,
                all_cat_global_scatter_path,
                f"Language Distance Comparison - All Categories - {api_model} - {proficiency} - {prompt}",
                color_by='LanguagePair',
                x_dist='CosineDistance',
                y_dist=None,  # Only use x-axis for distance
                plot_type='bar'  # Use bar chart for language comparison
            )
            print(f"Language distance scatter/bar plot for all categories saved to: {all_cat_global_scatter_path}")
        except Exception as e:
            print(f"Error generating cross-distance scatter plot: {e}")
            traceback.print_exc()
        
        # Create scatter plots for each category
        for category in combined_comp_df['NounCategory'].unique():
            category_df = combined_comp_df[combined_comp_df['NounCategory'] == category]
            if len(category_df) < 3:  # Need at least a few points for a meaningful scatter
                continue
                
            cat_scatter_path = os.path.join(cosine_jaccard_dir, f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_{sanitize_category(category)}_distance_scatter.html")
            try:
                plot_cross_distance_scatter(
                    category_df, 
                    cat_scatter_path,
                    f"Cosine vs Jaccard Distance - {category} - {api_model} - {proficiency} - {prompt}",
                    color_by='Noun',
                    x_dist='CosineDistance',
                    y_dist='JaccardDistance'
                )
                print(f"Cross-distance scatter plot for {category} saved to: {cat_scatter_path}")
            except Exception as e:
                print(f"Error generating cross-distance scatter plot for {category}: {e}")
                traceback.print_exc()
                
            # Create a second scatter plot comparing distances between languages for this category
            lang_scatter_path = os.path.join(scatter_plots_dir, f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_{sanitize_category(category)}_language_distance_scatter.html")
            try:
                # For this plot, we'll just use one distance type but color by language pairs
                category_df['LanguagePair'] = category_df['Language1'] + '-' + category_df['Language2']
                plot_cross_distance_scatter(
                    category_df, 
                    lang_scatter_path,
                    f"Language Distance Comparison - {category} - {api_model} - {proficiency} - {prompt}",
                    color_by='LanguagePair',
                    x_dist='CosineDistance',
                    y_dist=None,  # Only use x-axis for distance
                    plot_type='bar'  # Use bar chart for language comparison
                )
                print(f"Language distance scatter/bar plot for {category} saved to: {lang_scatter_path}")
            except Exception as e:
                print(f"Error generating language distance scatter/bar plot for {category}: {e}")
                traceback.print_exc()
    
    # Generate noun category language matrix plots for this prompt
    noun_cat_matrix_dir = os.path.join(visualizations_dir, "noun_category_matrix", proficiency, sanitize_prompt(prompt))
    os.makedirs(noun_cat_matrix_dir, exist_ok=True)
    
    # Generate plots using the combined comprehensive data
    if category_comprehensive_dfs:
        combined_comp_df = pd.concat(category_comprehensive_dfs, ignore_index=True)
        
        try:
            print(f"\nGenerating noun category language matrix plots for prompt '{prompt}'...")
            from lr_analysis_functions.visualization import plot_noun_category_language_matrix
            
            # Generate plots from the combined data
            plot_noun_category_language_matrix(
                combined_comp_df, 
                noun_cat_matrix_dir, 
                base_filename, 
                api_model, 
                proficiency_level=proficiency,
                language_family_map=lang_to_family_map
            )
            print(f"Noun category language matrix plots generated for '{prompt}'")
        except Exception as e:
            print(f"Error generating noun category language matrix plots: {e}")
            traceback.print_exc()

    # After all individual noun categories, generate combined visualizations showing all categories
    # (This will display noun categories in the same space for comparison)
    print("\nGenerating combined visualizations for all noun categories...")
    
    # First, create directories for this combined visualization
    combined_categories_path = os.path.join(visualizations_dir, "pca", "noun_categories_combined")
    os.makedirs(combined_categories_path, exist_ok=True)
    
    # Process all_categories case for PCA (comparing all noun categories in same space)
    if len(noun_category_dataframes) > 0:
        # Create a combined dataframe with representatives from each category
        # Label with noun category for coloring
        combined_category_df = pd.concat(noun_category_dataframes, axis=0)
        
        # Generate a PCA plot showing all categories in the same space
        combined_pca_path = os.path.join(combined_categories_path, 
                                       f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_all_categories_combined.png")
        try:
            # Plot with NounCategory as the color
            plot_pca(combined_category_df, combined_pca_path,
                   f"PCA - All Noun Categories - {api_model} - {proficiency} - {prompt}",
                   color_by='NounCategory', interactive=True)
            print(f"Combined PCA for all noun categories saved to: {combined_pca_path}")
        except Exception as e:
            print(f"Error generating combined PCA for all noun categories: {e}")
            traceback.print_exc()
    
    # Process the "all_categories" case for this prompt
    print("\nProcessing all noun categories combined as 'all_categories'...")
    
    # Define detail path for all_categories
    all_categories_path_segment = os.path.join(proficiency, sanitize_prompt(prompt), "all_categories")
    
    # Create all_categories directories
    create_category_directories(embeddings_dir, distances_dir, visualizations_dir, stats_dir, all_categories_path_segment)
    
    current_embeddings_all_cat_dir = os.path.join(embeddings_dir, "details", all_categories_path_segment)
    current_distances_noun_level_all_cat_dir = os.path.join(distances_dir, "noun_level_distances", "details", all_categories_path_segment)
    current_distances_comprehensive_all_cat_dir = os.path.join(distances_dir, "comprehensive_analysis", "details", all_categories_path_segment)
    current_distances_lang_comp_all_cat_dir = os.path.join(distances_dir, "noun_level_language_comparisons", "details", all_categories_path_segment)
    current_visualizations_tsne_all_cat_dir = os.path.join(visualizations_dir, "tsne", "details", all_categories_path_segment)
    current_visualizations_umap_all_cat_dir = os.path.join(visualizations_dir, "umap", "details", all_categories_path_segment)
    current_visualizations_pca_all_cat_dir = os.path.join(visualizations_dir, "pca", "details", all_categories_path_segment)
    current_stats_noun_type_all_cat_dir = os.path.join(stats_dir, "noun_type_analysis", "details", all_categories_path_segment)
    current_stats_category_perc_all_cat_dir = os.path.join(stats_dir, "category_percentage_analysis", "details", all_categories_path_segment)
    current_stats_gender_all_cat_dir = os.path.join(stats_dir, "gender_language_analysis", "details", all_categories_path_segment)
    current_stats_corr_all_cat_dir = os.path.join(stats_dir, "correlations", "details", all_categories_path_segment)
    
    # Create all stats directories
    os.makedirs(current_stats_noun_type_all_cat_dir, exist_ok=True)
    os.makedirs(current_stats_category_perc_all_cat_dir, exist_ok=True)
    os.makedirs(current_stats_gender_all_cat_dir, exist_ok=True)
    os.makedirs(current_stats_corr_all_cat_dir, exist_ok=True)
    
    # Also ensure the main-level all_categories directory exists
    all_categories_main_dir = os.path.join(stats_dir, "noun_type_analysis", "all_categories")
    os.makedirs(all_categories_main_dir, exist_ok=True)
    
    # Only proceed if we have enough data
    if len(prompt_df) >= 2:
        # Save all-categories embeddings
        all_categories_embeddings_filename = f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_all_categories_embeddings.pkl"
        all_categories_embeddings_file_path = os.path.join(current_embeddings_all_cat_dir, all_categories_embeddings_filename)
        prompt_df.to_pickle(all_categories_embeddings_file_path)
        
        # Generate noun-level distances for all categories
        all_categories_filtered_df, all_categories_jaccard_df = _generate_and_save_noun_level_distances_csv(
            prompt_df.copy(),
            base_filename,
            proficiency,
            prompt,
            "all_categories",
            current_distances_noun_level_all_cat_dir
        )
        
        # Compute full distance matrix for visualization
        all_categories_distances_df = compute_pairwise_distances(prompt_df)
        
        # Language comparisons for all categories
        all_categories_lang_comp_filename = f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_all_categories_language_comparisons.csv"
        all_categories_lang_comp_path = os.path.join(current_distances_lang_comp_all_cat_dir, all_categories_lang_comp_filename)
        all_categories_lang_comp_df = generate_language_comparisons_df(prompt_df, all_categories_lang_comp_path)
        
        # Comprehensive analysis for all categories
        all_categories_comprehensive_filename = f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_all_categories_comprehensive.csv"
        all_categories_comprehensive_path = os.path.join(current_distances_comprehensive_all_cat_dir, all_categories_comprehensive_filename)
        try:
            all_categories_comprehensive_df = generate_combined_analysis(
                prompt_df, 
                all_categories_filtered_df, 
                all_categories_jaccard_df, 
                all_categories_comprehensive_path
            )
            
            if not all_categories_comprehensive_df.empty:
                all_comprehensive_analyses.append(all_categories_comprehensive_df)
                
                # Statistical analyses for all_categories
                try:
                    # Noun type analysis
                    stats_prefix_all_cat = os.path.join(current_stats_noun_type_all_cat_dir, 
                                                       f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_all_categories")
                    print(f"\nGenerating noun type statistical analysis for all_categories...")
                    
                    # Double-check that the directory exists
                    noun_type_dir = os.path.dirname(stats_prefix_all_cat)
                    if not os.path.exists(noun_type_dir):
                        print(f"Creating noun type analysis directory: {noun_type_dir}")
                        os.makedirs(noun_type_dir, exist_ok=True)
                    
                    # Run noun type analysis
                    noun_type_results = analyze_noun_type_distances(all_categories_comprehensive_df.copy(), output_file_path_prefix=stats_prefix_all_cat)
                    
                    # Verify output files
                    noun_type_effect_sizes = f"{stats_prefix_all_cat}_effect_sizes.csv"
                    noun_type_anova_cosine = f"{stats_prefix_all_cat}_anova_cosine.csv"
                    noun_type_anova_jaccard = f"{stats_prefix_all_cat}_anova_jaccard.csv"
                    noun_type_summary = f"{stats_prefix_all_cat}_noun_type_analysis_summary.txt"
                    
                    files_to_check = [
                        (noun_type_effect_sizes, "Effect sizes CSV"),
                        (noun_type_anova_cosine, "ANOVA Cosine CSV"),
                        (noun_type_anova_jaccard, "ANOVA Jaccard CSV"),
                        (noun_type_summary, "Noun type analysis summary")
                    ]
                    
                    for file_path, file_desc in files_to_check:
                        if os.path.exists(file_path):
                            print(f"Confirmed: {file_desc} exists at: {file_path}")
                            
                            # Copy to the main-level all_categories directory if it exists
                            if file_path == noun_type_summary and os.path.exists(all_categories_main_dir):
                                main_summary = os.path.join(all_categories_main_dir, 
                                                         f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_all_categories_noun_type_analysis_summary.txt")
                                try:
                                    # Copy file instead of symlink to avoid broken links on different systems
                                    import shutil
                                    shutil.copy2(file_path, main_summary)
                                    print(f"Copied summary to main all_categories directory: {main_summary}")
                                except Exception as e_copy:
                                    print(f"Warning: Could not copy to main directory: {e_copy}")
                        else:
                            print(f"Warning: {file_desc} file not found at: {file_path}")
                    
                    print(f"Noun type analysis for all_categories completed")
                    
                    # Add per-language noun type analysis for all_categories
                    if 'Language1' in all_categories_comprehensive_df.columns and 'Language2' in all_categories_comprehensive_df.columns:
                        # Get unique languages from both Language1 and Language2 columns
                        languages = set(all_categories_comprehensive_df['Language1'].unique()) | set(all_categories_comprehensive_df['Language2'].unique())
                        print(f"Performing per-language noun type analysis for all_categories with {len(languages)} unique languages")
                        
                        # Create a directory for language-specific noun type analyses
                        language_specific_noun_type_dir = os.path.join(current_stats_noun_type_all_cat_dir, "per_language")
                        os.makedirs(language_specific_noun_type_dir, exist_ok=True)
                        
                        # Process each language
                        for language in sorted(languages):
                            try:
                                # Create language-specific prefix for files
                                lang_sanitized = sanitize_name(language)
                                lang_noun_type_prefix = os.path.join(language_specific_noun_type_dir, 
                                                               f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_all_categories_{lang_sanitized}")
                                
                                # Perform noun type analysis for this language
                                noun_type_lang_results = analyze_noun_type_distances(all_categories_comprehensive_df.copy(), 
                                                                          output_file_path_prefix=lang_noun_type_prefix, 
                                                                          language=language)
                                
                                # Verify the key output files
                                lang_effect_sizes = f"{lang_noun_type_prefix}_effect_sizes.csv"
                                lang_single_summary = f"{lang_noun_type_prefix}_single_category_summary.csv"
                                lang_summary = f"{lang_noun_type_prefix}_noun_type_analysis_summary.txt"
                                
                                for file_path in [lang_effect_sizes, lang_single_summary, lang_summary]:
                                    if os.path.exists(file_path):
                                        print(f"  Created: {os.path.basename(file_path)} for language '{language}'")
                            except Exception as e_lang:
                                print(f"Error processing noun type analysis for language '{language}' in all_categories: {e_lang}")
                                traceback.print_exc()
                    
                    # Gender language analysis
                    stats_prefix_gender_all_cat = os.path.join(current_stats_gender_all_cat_dir,
                                                             f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_all_categories")
                    print(f"\nGenerating gender language analysis for all_categories...")
                    
                    # Double-check that the directory exists
                    gender_dir = os.path.dirname(stats_prefix_gender_all_cat)
                    if not os.path.exists(gender_dir):
                        print(f"Creating gender language analysis directory: {gender_dir}")
                        os.makedirs(gender_dir, exist_ok=True)
                    
                    # Run aggregated gender language analysis (all languages)
                    gender_result = analyze_gender_language_distances(all_categories_comprehensive_df.copy(), lang_to_gender_map, stats_prefix_gender_all_cat)
                    
                    # Now perform per-language gender language analysis for all_categories
                    if 'Language1' in all_categories_comprehensive_df.columns and 'Language2' in all_categories_comprehensive_df.columns:
                        # Get unique languages from both Language1 and Language2 columns
                        languages = set(all_categories_comprehensive_df['Language1'].unique()) | set(all_categories_comprehensive_df['Language2'].unique())
                        print(f"Performing per-language gender language analysis for all_categories with {len(languages)} unique languages")
                        
                        # Create a directory for language-specific gender language analyses
                        language_specific_gender_dir = os.path.join(current_stats_gender_all_cat_dir, "per_language")
                        os.makedirs(language_specific_gender_dir, exist_ok=True)
                        
                        # Process each language
                        for language in sorted(languages):
                            try:
                                # Create language-specific prefix for files
                                lang_sanitized = sanitize_name(language)
                                lang_gender_prefix = os.path.join(language_specific_gender_dir, f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_all_categories_{lang_sanitized}")
                                
                                # Perform gender language analysis for this language
                                analyze_gender_language_distances(all_categories_comprehensive_df.copy(), lang_to_gender_map, lang_gender_prefix, language=language)
                            except Exception as e_lang:
                                print(f"Error processing gender language analysis for language '{language}' in all_categories: {e_lang}")
                                traceback.print_exc()
                    
                    # Verify output files
                    gender_lang_stats = f"{stats_prefix_gender_all_cat}_gender_lang_stats.txt"
                    gender_averages = f"{stats_prefix_gender_all_cat}_gender_averages.csv"
                    gender_effects = f"{stats_prefix_gender_all_cat}_gender_effects.csv"
                    
                    if os.path.exists(gender_lang_stats):
                        print(f"Confirmed: Gender language stats exists at: {gender_lang_stats}")
                    else:
                        print(f"Warning: Gender language stats file not found at: {gender_lang_stats}")
                        
                    if os.path.exists(gender_averages):
                        print(f"Confirmed: Gender averages CSV exists at: {gender_averages}")
                    else:
                        print(f"Warning: Gender averages CSV not found at: {gender_averages}")
                        
                    if os.path.exists(gender_effects):
                        print(f"Confirmed: Gender effects CSV exists at: {gender_effects}")
                    else:
                        print(f"Warning: Gender effects CSV not found at: {gender_effects}")
                    
                    print(f"Gender language analysis for all_categories completed")
                    
                    # Correlation analysis
                    stats_prefix_corr_all_cat = os.path.join(current_stats_corr_all_cat_dir,
                                                           f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_all_categories")
                    print(f"\nGenerating correlation analysis for all_categories...")
                    
                    # Double-check that the directory exists
                    corr_dir = os.path.dirname(stats_prefix_corr_all_cat)
                    if not os.path.exists(corr_dir):
                        print(f"Creating correlations directory: {corr_dir}")
                        os.makedirs(corr_dir, exist_ok=True)
                    
                    # Run aggregated correlation analysis (all languages)
                    corr_result = analyze_distance_metric_correlation(all_categories_comprehensive_df.copy(), stats_prefix_corr_all_cat)
                    
                    # Verify output files
                    corr_csv = f"{stats_prefix_corr_all_cat}_correlation.csv"
                    corr_plot = f"{stats_prefix_corr_all_cat}_correlation_plot.png"
                    
                    if os.path.exists(corr_csv):
                        print(f"Confirmed: Correlation CSV exists at: {corr_csv}")
                    else:
                        print(f"Warning: Correlation CSV file not found at: {corr_csv}")
                        
                    if os.path.exists(corr_plot):
                        print(f"Confirmed: Correlation plot exists at: {corr_plot}")
                    else:
                        print(f"Warning: Correlation plot not found at: {corr_plot}")
                    
                    # Now perform per-language correlation analysis for all_categories
                    if 'Language1' in all_categories_comprehensive_df.columns and 'Language2' in all_categories_comprehensive_df.columns:
                        # Get unique languages from both Language1 and Language2 columns
                        languages = set(all_categories_comprehensive_df['Language1'].unique()) | set(all_categories_comprehensive_df['Language2'].unique())
                        print(f"Performing per-language correlation analysis for all_categories with {len(languages)} unique languages")
                        
                        # Create a directory for language-specific correlation analyses
                        language_specific_corr_dir = os.path.join(current_stats_corr_all_cat_dir, "per_language")
                        os.makedirs(language_specific_corr_dir, exist_ok=True)
                        
                        # Process each language
                        for language in sorted(languages):
                            try:
                                # Create language-specific prefix for files
                                lang_sanitized = sanitize_name(language)
                                lang_corr_prefix = os.path.join(language_specific_corr_dir, f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_all_categories_{lang_sanitized}")
                                
                                # Calculate correlations for this language
                                analyze_distance_metric_correlation(all_categories_comprehensive_df.copy(), lang_corr_prefix, language=language)
                            except Exception as e_lang:
                                print(f"Error processing correlation for language '{language}' in all_categories: {e_lang}")
                                traceback.print_exc()
                        
                    print(f"Correlation analysis for all_categories completed")
                    
                    # Category percentages
                    if 'NounCategory' in all_categories_comprehensive_df.columns:
                        print(f"\nGenerating category percentage analysis for all_categories...")
                        
                        # Ensure the comprehensive df has all required data
                        if not all_categories_comprehensive_df['NounCategory'].isna().any():
                            # Calculate aggregated category statistics (all languages together)
                            all_categories_percentages = calculate_category_percentages(all_categories_comprehensive_df)
                            
                            # Create filename and path
                            all_categories_percentages_filename = f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_all_categories_category_percentages.csv"
                            all_categories_percentages_path = os.path.join(current_stats_category_perc_all_cat_dir, all_categories_percentages_filename)
                            
                            # Verify the structure of the percentages dictionary
                            if all_categories_percentages:
                                # Save the category percentages to CSV
                                success = _save_category_percentages_to_csv(all_categories_percentages, all_categories_percentages_path)
                                if success:
                                    print(f"Successfully saved aggregated category percentages for all_categories to: {all_categories_percentages_path}")
                                    
                                    # Verify the file was created
                                    if not os.path.exists(all_categories_percentages_path):
                                        print(f"Warning: Category percentages file not found at: {all_categories_percentages_path}")
                                else:
                                    print(f"Failed to save category percentages for all_categories to: {all_categories_percentages_path}")
                            else:
                                print("Warning: No category percentages data generated for all_categories")
                            
                            # Now perform per-language analysis
                            if 'Language1' in all_categories_comprehensive_df.columns and 'Language2' in all_categories_comprehensive_df.columns:
                                # Get unique languages from both Language1 and Language2 columns
                                languages = set(all_categories_comprehensive_df['Language1'].unique()) | set(all_categories_comprehensive_df['Language2'].unique())
                                print(f"Found {len(languages)} unique languages for per-language analysis")
                                
                                # Create a directory for language-specific analyses
                                language_specific_dir = os.path.join(current_stats_category_perc_all_cat_dir, "per_language")
                                os.makedirs(language_specific_dir, exist_ok=True)
                                
                                # Process each language
                                for language in sorted(languages):
                                    try:
                                        # Calculate statistics for this language
                                        lang_percentages = calculate_category_percentages(all_categories_comprehensive_df, language=language)
                                        
                                        if lang_percentages:
                                            # Create language-specific filename
                                            lang_sanitized = sanitize_name(language)
                                            lang_percentages_filename = f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_all_categories_{lang_sanitized}_category_percentages.csv"
                                            lang_percentages_file_path = os.path.join(language_specific_dir, lang_percentages_filename)
                                            
                                            # Save the language-specific percentages
                                            success = _save_category_percentages_to_csv(lang_percentages, lang_percentages_file_path)
                                            if success:
                                                print(f"Saved all_categories percentages for language '{language}' to: {lang_percentages_file_path}")
                                            else:
                                                print(f"Failed to save all_categories percentages for language '{language}'")
                                    except Exception as e_lang:
                                        print(f"Error processing language '{language}' for all_categories: {e_lang}")
                                        traceback.print_exc()
                        else:
                            print("Warning: NounCategory column contains NaN values - skipping category percentage analysis")
                            
                    # Create a directory for noun-level evaluations for all_categories
                    current_stats_noun_level_all_cat_dir = os.path.join(stats_dir, "noun_level_evaluations", "details", all_categories_path_segment)
                    os.makedirs(current_stats_noun_level_all_cat_dir, exist_ok=True)
                    
                    # Perform noun-level evaluations for all_categories
                    stats_prefix_noun_level_all_cat = os.path.join(current_stats_noun_level_all_cat_dir, f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_all_categories")
                    print(f"\nGenerating noun-level evaluations for all_categories...")
                    analyze_noun_level_comparisons(all_categories_comprehensive_df.copy(), stats_prefix_noun_level_all_cat, lang_to_family_map)
                    
                    # Create a global noun-level analysis directory
                    global_noun_level_dir = os.path.join(stats_dir, "noun_level_evaluations", "all_categories")
                    os.makedirs(global_noun_level_dir, exist_ok=True)
                    
                    # Generate a combined noun-level analysis across all categories
                    global_noun_level_path = os.path.join(global_noun_level_dir, f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_noun_level_global_analysis")
                    print(f"\nGenerating global noun-level evaluations across all categories...")
                    analyze_noun_level_comparisons(all_categories_comprehensive_df.copy(), global_noun_level_path, lang_to_family_map)
                    
                except Exception as e_stats:
                    print(f"Error during statistical analysis for all_categories: {e_stats}")
                    traceback.print_exc()
        except Exception as e_comp:
            print(f"Error generating comprehensive analysis for all_categories: {e_comp}")
            traceback.print_exc()
        
        # Generate visualizations for all categories
        all_cat_dim_red_plot_base_filename = f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_all_categories"
        all_cat_dim_red_plot_title_core = f"{proficiency} - {prompt} - All Categories"

        generate_standard_dim_reduction_plots(
            df=prompt_df, # Use the full prompt_df for "all_categories"
            base_filename_prefix=all_cat_dim_red_plot_base_filename,
            title_core=all_cat_dim_red_plot_title_core,
            color_by='NounCategory', # Color by NounCategory for the combined plot
            original_distance_matrix=all_categories_distances_df,
            tsne_dir=current_visualizations_tsne_all_cat_dir,
            umap_dir=current_visualizations_umap_all_cat_dir,
            pca_dir=current_visualizations_pca_all_cat_dir,
            api_model_name=api_model
        )
            
        # Generate average noun difference visualization
        try:
            print(f"\nGenerating average noun difference visualization for all_categories...")
            
            # Create a directory for average noun difference visualizations if it doesn't exist
            avg_noun_diff_dir = os.path.join(visualizations_dir, "average_noun_differences")
            os.makedirs(avg_noun_diff_dir, exist_ok=True)
            
            # Create a specific directory for all_categories
            all_cat_avg_noun_diff_dir = os.path.join(avg_noun_diff_dir, "all_categories")
            os.makedirs(all_cat_avg_noun_diff_dir, exist_ok=True)
            
            # Create a proficiency/prompt specific directory
            prof_prompt_dir = os.path.join(all_cat_avg_noun_diff_dir, proficiency, sanitize_prompt(prompt))
            os.makedirs(prof_prompt_dir, exist_ok=True)
            
            # Check if we have comprehensive analysis data available
            if all_categories_comprehensive_df is not None and not all_categories_comprehensive_df.empty:
                # Generate visualization for cosine distance
                cosine_filename = f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_all_categories_noun_differences_cosine.html"
                cosine_output_path = os.path.join(prof_prompt_dir, cosine_filename)
                
                plot_average_noun_differences(
                    all_categories_comprehensive_df,
                    cosine_output_path,
                    f"Average Noun Differences by Category (Cosine) - {api_model} - {proficiency} - {prompt}",
                    distance_metric='CosineDistance',
                    interactive=True
                )
                
                # Generate visualization for Jaccard distance if available
                if 'JaccardDistance' in all_categories_comprehensive_df.columns:
                    jaccard_filename = f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_all_categories_noun_differences_jaccard.html"
                    jaccard_output_path = os.path.join(prof_prompt_dir, jaccard_filename)
                    
                    plot_average_noun_differences(
                        all_categories_comprehensive_df,
                        jaccard_output_path,
                        f"Average Noun Differences by Category (Jaccard) - {api_model} - {proficiency} - {prompt}",
                        distance_metric='JaccardDistance',
                        interactive=True
                    )
                    
                # Generate per-language visualizations if language columns are available
                if 'Language1' in all_categories_comprehensive_df.columns and 'Language2' in all_categories_comprehensive_df.columns:
                    # Get unique languages
                    languages = set(all_categories_comprehensive_df['Language1'].unique()) | set(all_categories_comprehensive_df['Language2'].unique())
                    print(f"Generating per-language average noun differences for {len(languages)} languages")
                    
                    # Create a per-language directory
                    per_lang_dir = os.path.join(prof_prompt_dir, "per_language")
                    os.makedirs(per_lang_dir, exist_ok=True)
                    
                    # Process each language
                    for language in sorted(languages):
                        try:
                            # Create language-specific filename
                            lang_sanitized = sanitize_name(language)
                            lang_cosine_filename = f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_all_categories_{lang_sanitized}_noun_differences_cosine.html"
                            lang_cosine_path = os.path.join(per_lang_dir, lang_cosine_filename)
                            
                            # Generate cosine visualization for this language
                            plot_average_noun_differences(
                                all_categories_comprehensive_df,
                                lang_cosine_path,
                                f"Average Noun Differences by Category (Cosine) - {language} - {proficiency} - {prompt}",
                                distance_metric='CosineDistance',
                                filter_language=language,
                                interactive=True
                            )
                            
                            # Generate Jaccard visualization if available
                            if 'JaccardDistance' in all_categories_comprehensive_df.columns:
                                lang_jaccard_filename = f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_all_categories_{lang_sanitized}_noun_differences_jaccard.html"
                                lang_jaccard_path = os.path.join(per_lang_dir, lang_jaccard_filename)
                                
                                plot_average_noun_differences(
                                    all_categories_comprehensive_df,
                                    lang_jaccard_path,
                                    f"Average Noun Differences by Category (Jaccard) - {language} - {proficiency} - {prompt}",
                                    distance_metric='JaccardDistance',
                                    filter_language=language,
                                    interactive=True
                                )
                        except Exception as e_lang:
                            print(f"Error generating average noun differences for language '{language}': {e_lang}")
                            traceback.print_exc()
                
                print(f"Average noun differences visualizations completed for all_categories")
            else:
                print(f"Warning: No comprehensive analysis data available for average noun difference visualization")
        except Exception as e_noun_diff:
            print(f"Warning: Error generating average noun differences visualization: {e_noun_diff}")
            traceback.print_exc()
    else:
        print(f"Not enough data for all_categories analysis in proficiency '{proficiency}', prompt '{prompt}'")

    # Add dendrogram generation for noun categories
    print(f"\nGenerating dendrograms for noun categories...")
    
    # Process each noun category separately for dendrograms
    for noun_category in noun_categories_in_data + ["all_categories"]:
        # Skip if no data for this category
        if noun_category == "all_categories":
            category_df = prompt_df  # All categories combined
            category_name = "all_categories"
        else:
            category_df = prompt_df[prompt_df['NounCategory'] == noun_category]
            category_name = noun_category
        
        if category_df.empty or len(category_df) < 2:
            continue
        
        # Calculate distance matrix for the noun category
        try:
            # Compute pairwise distances between languages for this noun category
            embeddings = np.vstack(category_df['Embedding'])
            languages = category_df['Language'].tolist()
            
            # Skip if not enough unique languages
            if len(set(languages)) < 2:
                continue
                
            # Create distance matrix
            distances = 1 - cosine_similarity(embeddings)
            distance_matrix = pd.DataFrame(distances, index=languages, columns=languages)
            
            # Generate dendrogram
            dendrogram_filename = f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_{sanitize_category(category_name)}_noun_dendrogram.png"
            dendrogram_file_path = os.path.join(visualizations_dir, "dendrograms", dendrogram_filename)
            
            plot_language_dendrogram(
                distance_matrix,
                languages,
                dendrogram_file_path,
                f"Noun Category: {category_name} - {proficiency} - {prompt} - {api_model}"
            )
            print(f"  Dendrogram generated for noun category '{category_name}'")
            
        except Exception as e:
            print(f"  Error generating dendrogram for noun category '{category_name}': {e}")
            traceback.print_exc()

    # Noun category heatmap generation has been removed to keep only language-level and language-family-level heatmaps
    
    # Generate language-level visualization if we have the data
    # We'll use available functions to calculate language embeddings from the current dataset
    try:
        print("\nGenerating language-level visualizations...")
        # Calculate language embeddings for the current prompt_df
        temp_lang_embeddings_df = calculate_language_embeddings(prompt_df)
        
        if not temp_lang_embeddings_df.empty and len(temp_lang_embeddings_df) >= 2:
            # Prepare labels for visualization
            temp_lang_embeddings_df['PlotLabel'] = temp_lang_embeddings_df['Language']
            temp_lang_embeddings_df['Label'] = temp_lang_embeddings_df['Language']
            
            # Generate PCA plot
            lang_pca_path = os.path.join(
                visualizations_dir, 
                "pca", 
                "language_level", 
                f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_language_pca.html"
            )
            
            try:
                # Generate 2D PCA
                plot_pca(
                    temp_lang_embeddings_df, 
                    lang_pca_path,
                    f"Language-Level PCA - {proficiency} - {prompt} - {api_model}",
                    color_by='Language', 
                    interactive=True
                )
                print(f"Language-level 2D PCA visualization saved to {lang_pca_path}")
                
                # Generate 3D PCA
                lang_pca_3d_path = os.path.join(
                    visualizations_dir, 
                    "pca", 
                    "language_level", 
                    f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_language_pca_3d.html"
                )
                plot_pca(
                    temp_lang_embeddings_df, 
                    lang_pca_3d_path,
                    f"3D Language-Level PCA - {proficiency} - {prompt} - {api_model}",
                    color_by='Language', 
                    interactive=True,
                    dimensions=3
                )
                print(f"Language-level 3D PCA visualization saved to {lang_pca_3d_path}")
            except Exception as e_pca:
                print(f"Error generating language-level PCA: {e_pca}")
                traceback.print_exc()
                
            # Generate t-SNE visualization
            lang_tsne_path = os.path.join(
                visualizations_dir, 
                "tsne", 
                "language_level", 
                "LanguageEmbeddingDistance",
                f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_language_tsne.html"
            )
            
            try:
                # Generate 2D t-SNE
                plot_tsne(
                    temp_lang_embeddings_df, 
                    lang_tsne_path,
                    f"Language-Level t-SNE - {proficiency} - {prompt} - {api_model}",
                    color_by='Language', 
                    interactive=True
                )
                print(f"Language-level 2D t-SNE visualization saved to {lang_tsne_path}")
                
                # Generate 3D t-SNE
                lang_tsne_3d_path = os.path.join(
                    visualizations_dir, 
                    "tsne", 
                    "language_level", 
                    "LanguageEmbeddingDistance",
                    f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_language_tsne_3d.html"
                )
                plot_tsne(
                    temp_lang_embeddings_df, 
                    lang_tsne_3d_path,
                    f"3D Language-Level t-SNE - {proficiency} - {prompt} - {api_model}",
                    color_by='Language', 
                    interactive=True,
                    dimensions=3
                )
                print(f"Language-level 3D t-SNE visualization saved to {lang_tsne_3d_path}")
            except Exception as e_tsne:
                print(f"Error generating language-level t-SNE: {e_tsne}")
                traceback.print_exc()
                
            # Generate UMAP visualization
            lang_umap_path = os.path.join(
                visualizations_dir, 
                "umap", 
                "language_level", 
                "LanguageEmbeddingDistance",
                f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_language_umap.html"
            )
            
            try:
                # Generate 2D UMAP
                plot_umap(
                    temp_lang_embeddings_df, 
                    lang_umap_path,
                    f"Language-Level UMAP - {proficiency} - {prompt} - {api_model}",
                    color_by='Language', 
                    interactive=True
                )
                print(f"Language-level 2D UMAP visualization saved to {lang_umap_path}")
                
                # Generate 3D UMAP
                lang_umap_3d_path = os.path.join(
                    visualizations_dir, 
                    "umap", 
                    "language_level", 
                    "LanguageEmbeddingDistance",
                    f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_language_umap_3d.html"
                )
                plot_umap(
                    temp_lang_embeddings_df, 
                    lang_umap_3d_path,
                    f"3D Language-Level UMAP - {proficiency} - {prompt} - {api_model}",
                    color_by='Language', 
                    interactive=True,
                    dimensions=3
                )
                print(f"Language-level 3D UMAP visualization saved to {lang_umap_3d_path}")
            except Exception as e_umap:
                print(f"Error generating language-level UMAP: {e_umap}")
                traceback.print_exc()
        else:
            print("Insufficient data for language-level visualizations")
    except Exception as e_lang_viz:
        print(f"Error in language-level visualization section: {e_lang_viz}")
        traceback.print_exc()
        # Note: Removed erroneous nested block to fix indentation and syntax errors

    # Now add prompt-scoped language family visualizations
    print("\nGenerating prompt-scoped language family visualizations...")
    try:
        # Deep copy to avoid modifying the original
        temp_fam_df = temp_lang_embeddings_df.copy(deep=True) if 'temp_lang_embeddings_df' in locals() else pd.DataFrame()
        
        if temp_fam_df.empty:
            raise ValueError("No language embeddings available for prompt-scoped family visualizations")

        # Extract language code (handles formats like 'EN # English')
        temp_fam_df['LanguageCode'] = temp_fam_df['Language'].apply(
            lambda x: x.split(' # ')[0] if isinstance(x, str) and ' # ' in x else x
        )
        
        # Map to language families
        if lang_to_family_map:
            temp_fam_df['LanguageFamily'] = temp_fam_df['LanguageCode'].map(lang_to_family_map)
            temp_fam_df.dropna(subset=['LanguageFamily'], inplace=True)
            
            if len(temp_fam_df) >= 2:
                family_groups = []
                for fam_name, fam_group in temp_fam_df.groupby('LanguageFamily'):
                    if len(fam_group) > 0:
                        family_embeddings = np.vstack(fam_group['Embedding'])
                        avg_embedding = np.mean(family_embeddings, axis=0)
                        family_groups.append({
                            'LanguageFamily': fam_name,
                            'Embedding': avg_embedding,
                            'Label': fam_name,
                            'PlotLabel': fam_name
                        })
                
                prompt_family_df = pd.DataFrame(family_groups)
                
                if len(prompt_family_df) >= 2:
                    # Generate family PCA
                    fam_pca_path = os.path.join(
                        visualizations_dir, 
                        "pca", 
                        "language_family_level", 
                        f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_family_pca.html"
                    )
                    plot_pca(
                        prompt_family_df, 
                        fam_pca_path,
                        f"Language Family PCA - {proficiency} - {prompt} - {api_model}",
                        color_by='LanguageFamily', 
                        interactive=True
                    )
                    print(f"Prompt-scoped language family PCA saved to {fam_pca_path}")
                        
                # Combined family-language PCA
                combined_pca_path = os.path.join(
                            visualizations_dir, 
                    "pca",
                            "language_family_level", 
                            f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_combined_family_language_pca.html"
                        )
                        
                prompt_lang_to_family_dict = {(
                    lang if isinstance(lang, str) else str(lang)):
                    lang_to_family_map.get((lang.split(' # ')[0] if ' # ' in lang else lang), "Unknown")
                    for lang in temp_lang_embeddings_df['Language']}

                plot_combined_family_language_pca(
                    temp_lang_embeddings_df,
                            prompt_family_df, 
                    combined_pca_path,
                    f"Language Families and Languages PCA - {proficiency} - {prompt} - {api_model}",
                    family_mapping=prompt_lang_to_family_dict
                )
                print(f"Combined prompt-scoped family and language PCA saved to: {combined_pca_path}")
            else:
                print("Insufficient data for prompt-scoped language family visualizations (not enough families)")
        else:
            print("Skipping prompt-scoped language family visualizations (lang_to_family_map not available)")
    except Exception as e_fam:
        print(f"Error preparing prompt-scoped language family visualizations: {e_fam}")
    # traceback.print_exc() # Removed this line
    
    # Collect comprehensive analysis dataframes


def main():
    """
    Main entry point for the language representation analysis tool.
    Guides the user through provider, model, and file selection before running analysis.
    """
    # Step 1: Select API provider
    api_providers = list_api_providers()
    if not api_providers:
        print("No API providers found in 'api_generations' folder.")
        return
    
    print("Available API providers:")
    for i, provider in enumerate(api_providers, 1):
        print(f"  {i}. {provider}")
    
    try:
        provider_idx = int(input("Enter the number of the API provider to analyze: ")) - 1
        if provider_idx < 0 or provider_idx >= len(api_providers):
            print("Invalid selection. Exiting.")
            return
    except ValueError:
        print("Invalid input. Please enter a number.")
        return
    
    api_provider = api_providers[provider_idx]
    
    # Step 2: Select API model
    provider_path = os.path.join("api_generations", api_provider)
    api_models = list_api_models(provider_path)
    
    if not api_models:
        print(f"No models found for provider '{api_provider}'.")
        return
    
    print(f"\nAvailable models for {api_provider}:")
    for i, model in enumerate(api_models, 1):
        print(f"  {i}. {model}")
    
    try:
        model_idx = int(input("Enter the number of the model to analyze: ")) - 1
        if model_idx < 0 or model_idx >= len(api_models):
            print("Invalid selection. Exiting.")
            return
    except ValueError:
        print("Invalid input. Please enter a number.")
        return
    
    api_model = api_models[model_idx]
    model_path = os.path.join(provider_path, api_model)

    # Step 3: Select CSV file to analyze
    print(f"\nLooking for CSV files in: {model_path}")
    
    current_path = model_path
    file_selected = False
    
    while not file_selected:
        results = find_csv_files(current_path)
        csv_files = results['files']
        directories = results['directories']
        
        print(f"\nCurrent directory: {current_path}")
        
        if not csv_files and not directories:
            print("No CSV files or directories found in this location.")
            if current_path != model_path:
                current_path = os.path.dirname(current_path)
                continue
            else:
                print("No CSV files found in model directory. Exiting.")
                return
        
        options = []
        
        if current_path != model_path:
            print("  0. <- Go back to parent directory")
            options.append({"type": "nav", "action": "back"})
        
        for i, directory in enumerate(directories, start=len(options)):
            print(f"  {i}. [DIR] {directory}/")
            options.append({"type": "dir", "path": os.path.join(current_path, directory)})
        
        for i, csv_file in enumerate(csv_files, start=len(options)):
            print(f"  {i}. {csv_file}")
            options.append({"type": "file", "path": os.path.join(current_path, csv_file)})
        
        try:
            choice = int(input("\nEnter the number of your selection: "))
            if choice < 0 or choice >= len(options):
                print("Invalid selection. Please try again.")
                continue
            
            selected = options[choice]
            
            if selected["type"] == "nav" and selected["action"] == "back":
                current_path = os.path.dirname(current_path)
            elif selected["type"] == "dir":
                current_path = selected["path"]
            else:  # selected["type"] == "file"
                selected_file_path = selected["path"]
                selected_csv_filename = os.path.basename(selected_file_path)
                file_selected = True
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue
        except KeyboardInterrupt:
            print("\nSelection cancelled. Exiting.")
            return
    
    # Parse temperature from the filename (e.g., "XG3M-BB-1T-33L-5P-40N-11AP.csv")
    base_filename = os.path.basename(selected_csv_filename)
    temperature_str_from_filename = "unknown_temp"
    try:
        filename_parts = os.path.splitext(base_filename)[0].split('-')
        if len(filename_parts) > 2 and filename_parts[2].endswith('T') and filename_parts[2][:-1].isdigit():
            temperature_str_from_filename = filename_parts[2]
            print(f"Parsed temperature from filename: {temperature_str_from_filename}")
        else:
            print(f"Warning: Could not parse temperature from filename '{base_filename}'. Using default: '{temperature_str_from_filename}'.")
    except Exception as e:
        print(f"Warning: Error parsing temperature from filename '{base_filename}': {e}. Using default: '{temperature_str_from_filename}'.")

    # Step 4: Choose embedding model
    embedding_model_selected = False
    
    while not embedding_model_selected:
        embedding_models_list = embedding_models.load_embedding_models()
        print("\nAvailable embedding models:")
        print("Note: Some models may not be compatible with your environment.")
        print("Recommended models: sentence-transformers/* (options 6-10)")
        print("─" * 60)
        for i, model in enumerate(embedding_models_list, 1):
            status = ""
            if "jinaai/jina-embeddings" in model:
                status = " (uses jina-embeddings-python package, will be installed automatically)"
            elif "nomic-ai/" in model:
                try:
                    import einops
                    status = ""
                except ImportError:
                    status = " (requires einops package - not installed)"
            print(f"  {i}. {model}{status}")
        print("─" * 60)
        
        try:
            model_idx = int(input("Enter the number of the embedding model to use: ")) - 1
            if model_idx < 0 or model_idx >= len(embedding_models_list):
                print("Invalid selection. Please try again.")
                continue
                
            model_name = embedding_models_list[model_idx]
            print(f"\nSelected model: {model_name}")
            
            # Provide guidance for specific models
            if "jinaai/jina-embeddings" in model_name:
                print("Warning: Jina models require custom modules that may not be available.")
                print("This model may not work in your environment.")
            elif "nomic-ai/" in model_name:
                try:
                    import einops
                except ImportError:
                    print("Warning: Nomic models require the 'einops' package which is not installed.")
                    print("You can install it with: pip install einops")
                    continue
            
            # Ask about delimiter
            delimiter = input("\nEnter the CSV delimiter (leave empty for comma): ") or ","
            
            # Run the analysis pipeline
            print(f"\nStarting analysis with model: {model_name}")
            
            # The analysis_pipeline function will return None if the model fails
            analysis_pipeline(selected_file_path, model_name, api_provider, api_model, temperature_str_from_filename, delimiter)
            
            # Ask if the user wants to try another model
            retry = input("\nWould you like to try a different model? (y/n): ").lower().strip()
            if retry != 'y':
                embedding_model_selected = True
                print("Analysis completed.")
                
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue
        except KeyboardInterrupt:
            print("\nAnalysis interrupted by user.")
            return
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            retry = input("Would you like to try a different model? (y/n): ").lower().strip()
            if retry != 'y':
                print("Exiting analysis.")
                return


if __name__ == "__main__":
    # Set multiprocessing start method for consistency, especially on macOS/Windows
    try:
        mp.set_start_method('spawn', force=True)
        print("INFO: Multiprocessing start method set to 'spawn'.")
    except RuntimeError as e:
        print(f"INFO: Could not set multiprocessing start method to spawn (may be already set or not applicable): {e}")
        pass

    try:
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user.")
    finally:
        # Clear model cache before exiting to free memory
        embedding_models.clear_model_cache()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Suppress ResourceTracker errors that occur during process termination
        with open(os.devnull, 'w') as devnull:
            with redirect_stderr(devnull):
                # Force any remaining multiprocessing cleanup to happen silently
                if hasattr(mp, 'util') and hasattr(mp.util, '_exit_function'):
                    mp.util._exit_function() 