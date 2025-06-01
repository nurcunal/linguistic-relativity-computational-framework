"""
Language Representation Analysis

This is the main entry point for the language representation analysis tool.
It imports functionality from the lr_analysis_functions modules.
"""

import csv
from lr_analysis_functions import embedding_models
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

# Import functionality from our modules
from lr_analysis_functions import (
    # Data loading
    load_data, load_noun_categories, load_language_families, load_grammatical_gender, 
    load_adjective_prompts, find_csv_files, list_api_providers, list_api_models,
    
    # Distance metrics
    compute_pairwise_distances, filter_distances, compute_jaccard_distances,
    
    # Visualization
    plot_tsne, plot_umap, plot_pca, plot_language_dendrogram, plot_interactive_heatmap,
    
    # Statistical analysis
    calculate_cohens_d, analyze_noun_type_distances, calculate_category_percentages,
    analyze_gender_language_distances, analyze_distance_metric_correlation,
    
    # Language processing
    calculate_language_embeddings, compute_language_distances, calculate_language_family_embeddings_and_distances,
    generate_language_level_comparisons, generate_language_comparisons_df,
    
    # Utils
    sanitize_prompt, sanitize_category, sanitize_name, detect_optimal_device,
    
    # Comprehensive analysis
    generate_combined_analysis, _generate_and_save_noun_level_distances_csv
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
    snapshot_dir = os.path.join(provider_folder, "00_source_data_snapshot")
    embeddings_dir = os.path.join(provider_folder, "01_embeddings")
    distances_dir = os.path.join(provider_folder, "02_distances")
    visualizations_dir = os.path.join(provider_folder, "03_visualizations")
    stats_dir = os.path.join(provider_folder, "04_statistical_reports")

    # Create directory structure
    create_directory_structure(provider_folder, snapshot_dir, embeddings_dir, distances_dir, visualizations_dir, stats_dir)

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
            
        # Generate proficiency-specific folder
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
                                lang_to_gender_map)
            
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
    
    # Generate a single consolidated comprehensive analysis file across all proficiency levels
    if all_comprehensive_analyses:
        consolidated_df = pd.concat(all_comprehensive_analyses, ignore_index=True)
        
        # Add deduplication step to remove duplicate rows
        consolidated_df = consolidated_df.drop_duplicates(subset=[
            "Language1", "Language2", "NounCategory", "Noun", "Proficiency", "Prompt"
        ])
        
        consolidated_df.sort_values(
            by=["Language1", "Language2", "NounCategory", "Noun", "Proficiency", "Prompt"],
            ascending=[True, True, True, True, True, True],
            inplace=True
        )
        consolidated_filename = f"all_proficiencies_consolidated_{base_filename}.csv"
        consolidated_file_path = os.path.join(distances_dir, "comprehensive_analysis", consolidated_filename)
        consolidated_df.to_csv(consolidated_file_path, index=False)
        print(f"Generated consolidated analysis file with {len(consolidated_df)} rows: {consolidated_file_path}")

        # Run analysis on the consolidated data
        stats_prefix_consolidated_gender = os.path.join(stats_dir, "gender_language_analysis", f"all_proficiencies_consolidated_{base_filename}")
        analyze_gender_language_distances(consolidated_df.copy(), lang_to_gender_map, stats_prefix_consolidated_gender)

        # Call correlation analysis for the final consolidated_df
        stats_prefix_consolidated_corr = os.path.join(stats_dir, "correlations", f"all_proficiencies_consolidated_{base_filename}")
        # Ensure the base 'correlations' directory exists
        os.makedirs(os.path.join(stats_dir, "correlations"), exist_ok=True) 
        analyze_distance_metric_correlation(consolidated_df.copy(), stats_prefix_consolidated_corr)

    else: # This is if all_comprehensive_analyses is empty
        print("\nWarning: No comprehensive analyses were generated, cannot produce consolidated file or run consolidated stats")
    
    # Calculate language embeddings across all proficiency levels
    print("\nCalculating language-level embeddings...")
    try:
        language_embeddings_df = calculate_language_embeddings(df_with_embeddings)
        
        lang_embeddings_filename = f"language_level_all_proficiencies_{base_filename}.pkl"
        lang_embeddings_file_path = os.path.join(embeddings_dir, "language_level", lang_embeddings_filename)
        language_embeddings_df.to_pickle(lang_embeddings_file_path)
        print(f"Language embeddings saved to {lang_embeddings_file_path}")
        
        language_dist_filename = f"all_proficiencies_lang_distances_{base_filename}.csv"
        language_dist_file_path = os.path.join(distances_dir, "language_level", language_dist_filename)
        language_distances = compute_language_distances(language_embeddings_df, language_dist_file_path)
        
        language_comparisons_distances_dir = os.path.join(distances_dir, "language_level")
        language_comparisons_visualizations_dir = os.path.join(visualizations_dir, "scatter_plots")
        os.makedirs(os.path.join(language_comparisons_distances_dir, "proficiency_specific"), exist_ok=True)
        os.makedirs(language_comparisons_visualizations_dir, exist_ok=True)

        language_comparisons = generate_language_level_comparisons(
            df_with_embeddings, 
            provider_folder, 
            base_filename,
            api_model,
            language_comparisons_distances_dir, 
            language_comparisons_visualizations_dir
        )
        
        # Language Family Analysis
        family_embeddings_output_dir = os.path.join(embeddings_dir, "language_family_level")
        family_distances_output_dir = os.path.join(distances_dir, "language_family_level")
        family_analysis_prefix = f"{base_filename}_families"
        
        family_embeddings_df, family_distances_df = calculate_language_family_embeddings_and_distances(
            language_embeddings_df, 
            lang_to_family_map, 
            family_embeddings_output_dir,
            family_distances_output_dir,
            base_filename_prefix=family_analysis_prefix # This prefix is for the saved files
        )
        
        # Visualizations for language families would go here
        # Omitted for brevity

    except Exception as e:
        print(f"Error in language embedding calculation: {e}")
        traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS PIPELINE COMPLETE")
    print(f"All results saved to {provider_folder}")
    print(f"{'='*80}\n")
    
    # Call clear_model_cache from the embedding_models module at the end of main() or when needed
    embedding_models.clear_model_cache()
    
    return


def process_prompt_data(prof_df, prompt, proficiency, provider_folder, base_filename, api_model,
                     embeddings_dir, distances_dir, visualizations_dir, stats_dir, all_comprehensive_analyses,
                     lang_to_gender_map):
    """
    Process data for a specific prompt within a proficiency level.
    This is a helper function to make the main pipeline more modular.
    """
    prompt_df = prof_df[prof_df['Prompt'] == prompt].copy()
    
    if prompt_df.empty:
        print(f"  No data for prompt '{prompt}', skipping")
        return
        
    # Generate prompt-specific folder
    prompt_folder = f"{provider_folder}/{proficiency}/{sanitize_prompt(prompt)}"
    os.makedirs(prompt_folder, exist_ok=True)
    
    print(f"  Data rows: {len(prompt_df)}")
    
    # Process each noun category separately
    noun_categories_in_data = sorted(prompt_df['NounCategory'].unique())
    print(f"\nFound {len(noun_categories_in_data)} noun categories in prompt '{prompt}': {', '.join(sorted(noun_categories_in_data))}")
    processed_category_count = 0
    
    for noun_category in noun_categories_in_data:
        processed_category_count += 1
        print(f"\nProcessing category {processed_category_count}/{len(noun_categories_in_data)}: '{noun_category}'")
        category_df = prompt_df[prompt_df['NounCategory'] == noun_category].copy()
        
        if category_df.empty:
            continue
            
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
        
        # Generate visualizations for this category
        # t-SNE
        print(f"  Generating t-SNE for category '{noun_category}' with {len(category_df)} rows...")
        tsne_filename_base = f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_{sanitize_category(noun_category)}_tsne"
        tsne_file_png_path = os.path.join(current_visualizations_tsne_detail_dir, f"{tsne_filename_base}.png")
        try:
            plot_tsne(category_df, tsne_file_png_path, 
                    f"Semantic Space - {api_model} - {proficiency} - {prompt} - {noun_category}", 
                    color_by='Language', interactive=True, 
                    original_distance_matrix=distances_df)
        except Exception as e:
            print(f"  Warning: Error producing t-SNE visualization for '{noun_category}': {e}")
            traceback.print_exc()
            print(f"  Continuing with analysis...")
        
        # UMAP
        print(f"  Generating UMAP for category '{noun_category}' with {len(category_df)} rows...")
        umap_filename_base = f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_{sanitize_category(noun_category)}_umap"
        umap_file_png_path = os.path.join(current_visualizations_umap_detail_dir, f"{umap_filename_base}.png")
        try:
            plot_umap(category_df, umap_file_png_path, 
                    f"UMAP - {api_model} - {proficiency} - {prompt} - {noun_category}", 
                    color_by='Language', interactive=True, 
                    original_distance_matrix=distances_df)
        except Exception as e:
            print(f"  Warning: Error producing UMAP visualization for '{noun_category}': {e}")
            traceback.print_exc()
            print(f"  Continuing with analysis...")
        
        # PCA
        print(f"  Generating PCA for category '{noun_category}' with {len(category_df)} rows...")
        pca_filename_base = f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_{sanitize_category(noun_category)}_pca"
        pca_file_png_path = os.path.join(current_visualizations_pca_detail_dir, f"{pca_filename_base}.png")
        try:
            plot_pca(category_df, pca_file_png_path,
                   f"PCA - {api_model} - {proficiency} - {prompt} - {noun_category}",
                   color_by='Language', interactive=True)
        except Exception as e:
            print(f"Warning: Error producing PCA visualization for {noun_category}: {e}")
            print(f"Continuing with analysis...")

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
                # Generate statistical analyses
                try:
                    # Noun type analysis
                    stats_prefix_cat = os.path.join(current_stats_noun_type_detail_dir, f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_{sanitize_category(noun_category)}")
                    print(f"\nGenerating noun type statistical analysis for '{noun_category}'...")
                    analyze_noun_type_distances(comprehensive_df.copy(), output_file_path_prefix=stats_prefix_cat, expect_single_category=True)
                    
                    # Gender language analysis
                    stats_prefix_gender_cat = os.path.join(current_stats_gender_detail_dir, f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_{sanitize_category(noun_category)}")
                    analyze_gender_language_distances(comprehensive_df.copy(), lang_to_gender_map, stats_prefix_gender_cat)
                    
                    # Correlation analysis
                    stats_prefix_corr_cat = os.path.join(current_stats_corr_detail_dir, f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_{sanitize_category(noun_category)}")
                    analyze_distance_metric_correlation(comprehensive_df.copy(), stats_prefix_corr_cat)
                    
                    # Category percentages
                    if 'NounCategory' in comprehensive_df.columns:
                        print(f"\nGenerating category percentage analysis for '{noun_category}'...")
                        category_percentages = calculate_category_percentages(comprehensive_df)
                        percentages_filename = f"{base_filename}_{proficiency}_{sanitize_prompt(prompt)}_{sanitize_category(noun_category)}_category_percentages.csv"
                        percentages_file_path = os.path.join(current_stats_category_perc_detail_dir, percentages_filename)
                        
                        # Helper function to save category percentages in the correct format
                        _save_category_percentages_to_csv(category_percentages, percentages_file_path)
                except Exception as e_stats:
                    print(f"Error during statistical analysis for '{noun_category}': {e_stats}")
                    traceback.print_exc()
                
        except Exception as e:
            print(f"Warning: Error generating comprehensive analysis: {e}")
            traceback.print_exc()
            print(f"Continuing with analysis...")


def create_directory_structure(provider_folder, snapshot_dir, embeddings_dir, distances_dir, visualizations_dir, stats_dir):
    """
    Creates the necessary directory structure for the analysis pipeline.
    """
    # Produce all necessary base subdirectories
    os.makedirs(snapshot_dir, exist_ok=True) # For user to manually copy source CSV
    os.makedirs(os.path.join(embeddings_dir, "proficiency_specific"), exist_ok=True)
    os.makedirs(os.path.join(embeddings_dir, "details"), exist_ok=True)
    os.makedirs(os.path.join(embeddings_dir, "language_level"), exist_ok=True)
    os.makedirs(os.path.join(embeddings_dir, "language_family_level"), exist_ok=True)

    os.makedirs(os.path.join(distances_dir, "comprehensive_analysis", "proficiency_specific"), exist_ok=True)
    os.makedirs(os.path.join(distances_dir, "comprehensive_analysis", "details"), exist_ok=True)
    os.makedirs(os.path.join(distances_dir, "noun_level_distances", "details"), exist_ok=True)
    os.makedirs(os.path.join(distances_dir, "language_level", "proficiency_specific"), exist_ok=True)
    os.makedirs(os.path.join(distances_dir, "noun_level_language_comparisons", "details"), exist_ok=True)
    os.makedirs(os.path.join(distances_dir, "language_family_level"), exist_ok=True)

    os.makedirs(os.path.join(visualizations_dir, "tsne", "details"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "umap", "details"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "pca", "details"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "heatmaps"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "dendrograms"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "scatter_plots"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "umap", "language_level"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "tsne", "language_level"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "umap", "language_family_level"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "tsne", "language_family_level"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "heatmaps", "language_family_level"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "dendrograms", "language_family_level"), exist_ok=True)

    # Create deeper statistical report directories to ensure they exist
    print("Ensuring all statistical report directories exist...")
    os.makedirs(os.path.join(stats_dir, "noun_type_analysis", "details"), exist_ok=True)
    os.makedirs(os.path.join(stats_dir, "category_percentage_analysis", "details"), exist_ok=True)
    os.makedirs(os.path.join(stats_dir, "gender_language_analysis", "details"), exist_ok=True)
    os.makedirs(os.path.join(stats_dir, "correlations", "details"), exist_ok=True)


def create_category_directories(embeddings_dir, distances_dir, visualizations_dir, stats_dir, detail_path_segment):
    """
    Creates the detailed directory structure needed for a specific category.
    """
    # Create required statistics directories
    os.makedirs(os.path.join(stats_dir, "noun_type_analysis", "details", detail_path_segment), exist_ok=True)
    os.makedirs(os.path.join(stats_dir, "category_percentage_analysis", "details", detail_path_segment), exist_ok=True)
    os.makedirs(os.path.join(stats_dir, "gender_language_analysis", "details", detail_path_segment), exist_ok=True)
    os.makedirs(os.path.join(stats_dir, "correlations", "details", detail_path_segment), exist_ok=True)
    
    # Create required data directories
    os.makedirs(os.path.join(embeddings_dir, "details", detail_path_segment), exist_ok=True)
    os.makedirs(os.path.join(distances_dir, "noun_level_distances", "details", detail_path_segment), exist_ok=True)
    os.makedirs(os.path.join(distances_dir, "noun_level_language_comparisons", "details", detail_path_segment), exist_ok=True)
    os.makedirs(os.path.join(distances_dir, "comprehensive_analysis", "details", detail_path_segment), exist_ok=True)
    
    # Create required visualization directories
    os.makedirs(os.path.join(visualizations_dir, "tsne", "details", detail_path_segment), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "umap", "details", detail_path_segment), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "pca", "details", detail_path_segment), exist_ok=True)


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
                    continue_anyway = input("Continue anyway? (y/n): ").lower().strip()
                    if continue_anyway != 'y':
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