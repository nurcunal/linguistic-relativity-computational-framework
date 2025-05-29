"""
Language Representation Analysis Functions Package

This package contains various functions for analyzing language representations,
calculating distances between languages, and visualizing the results.
"""

__version__ = '1.0.0'

# Import key functions from each module to make them directly accessible
# from the package root

# Import from visualization module
from .visualization import (
    plot_tsne,
    plot_umap,
    plot_pca,
    plot_language_dendrogram,
    plot_interactive_heatmap,
    plot_cross_distance_scatter,
    plot_noun_category_language_matrix,
    plot_combined_family_language_pca,
    plot_average_noun_differences,
    generate_standard_dim_reduction_plots
)

# Import from distance_metrics module
from .distance_metrics import (
    compute_pairwise_distances,
    filter_distances,
    compute_jaccard_distances,
    calculate_kruskal_stress
)

# Import from data_loading module
from .data_loading import (
    load_data,
    load_adjective_prompts,
    load_noun_categories,
    load_language_families,
    load_grammatical_gender,
    find_csv_files,
    list_api_providers,
    list_api_models
)

# Import from statistical_analysis module
from .statistical_analysis import (
    calculate_cohens_d,
    analyze_noun_type_distances,
    calculate_category_percentages,
    analyze_gender_language_distances,
    analyze_distance_metric_correlation,
    analyze_noun_level_comparisons,
    analyze_language_level_statistics,
    analyze_language_family_statistics
)

# Import from language_processing module
from .language_processing import (
    calculate_language_embeddings,
    compute_language_distances,
    calculate_language_family_embeddings_and_distances,
    generate_language_level_comparisons,
    generate_language_comparisons_df,
    run_language_level_analysis_and_visualization,
    run_family_level_analysis_and_visualization
)

# Import from utils module
from .utils import (
    sanitize_prompt,
    sanitize_category,
    sanitize_name,
    detect_optimal_device,
    create_directory_structure,
    create_category_directories
)

# Import from comprehensive_analysis module
from .comprehensive_analysis import (
    _generate_and_save_noun_level_distances_csv,
    generate_combined_analysis,
    process_and_save_consolidated_analysis
)

# Add a helper function for the main script
def _save_category_percentages_to_csv(percentages, output_file):
    """
    Helper function to save category percentages dictionary to CSV format.
    
    Args:
        percentages (dict): Dictionary of category percentages
        output_file (str): Path to save the CSV file
    """
    import pandas as pd
    import os
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Check if the input contains the expected structure
    if not percentages:
        print(f"Warning: Invalid or empty percentages data structure for {output_file}")
        return False
    
    try:
        # Check if we have stats_df in the dictionary
        if 'stats_df' in percentages:
            # Get the stats DataFrame
            stats_df = percentages["stats_df"]
            
            # Check if DataFrame is valid
            if stats_df.empty:
                print(f"Warning: Empty stats DataFrame for {output_file}")
                return False
            
            # Define column descriptions
            column_descriptions = {
                'NounCategory': 'NounCategory',
                'LanguageEmbeddingDistance_mean': 'LanguageEmbeddingDistance_mean (calculated as 1-cos_sim between aggregated language-level embeddings, higher values indicate greater semantic dissimilarity between languages)',
                'LanguageEmbeddingDistance_min': 'LanguageEmbeddingDistance_min',
                'LanguageEmbeddingDistance_max': 'LanguageEmbeddingDistance_max',
                'LanguageEmbeddingDistance_count': 'LanguageEmbeddingDistance_count',
                'LanguageEmbeddingDistance_std': 'LanguageEmbeddingDistance_std',
                'LanguageEmbeddingDistance_percentile': 'LanguageEmbeddingDistance_percentile',
                'CosineDistance_mean': 'CosineDistance_mean (calculated as 1-cos_sim between noun-level embeddings, measures semantic distance between individual noun representations)',
                'CosineDistance_min': 'CosineDistance_min',
                'CosineDistance_max': 'CosineDistance_max',
                'CosineDistance_count': 'CosineDistance_count',
                'CosineDistance_std': 'CosineDistance_std',
                'CosineDistance_percentile': 'CosineDistance_percentile',
                'JaccardDistance_mean': 'JaccardDistance_mean (calculated as 1-(intersection/union) of adjective sets, measures dissimilarity between adjective choices)',
                'JaccardDistance_min': 'JaccardDistance_min',
                'JaccardDistance_max': 'JaccardDistance_max',
                'JaccardDistance_count': 'JaccardDistance_count',
                'JaccardDistance_std': 'JaccardDistance_std',
                'JaccardDistance_percentile': 'JaccardDistance_percentile',
                'combined_uniqueness_score': 'combined_uniqueness_score (average of available distance percentiles, higher values indicate greater cross-linguistic uniqueness)'
            }
            
            # Save the columns that exist in stats_df with renamed headers
            columns_to_save = []
            renamed_columns = {}
            
            # Filter to only include columns that exist in stats_df
            for col in stats_df.columns:
                if col in column_descriptions:
                    columns_to_save.append(col)
                    renamed_columns[col] = column_descriptions[col]
            
            # Make a copy to avoid modifying the original
            output_df = stats_df[columns_to_save].copy()
            
            # Write to CSV with renamed columns
            output_df.to_csv(output_file, header=renamed_columns.values(), index=False)
            print(f"Category statistics saved to: {output_file}")
            
            # Also save a summary file with basic stats
            summary_file = output_file.replace('.csv', '_summary.txt')
            with open(summary_file, 'w') as f:
                if 'summary' in percentages:
                    f.write(percentages['summary'])
                else:
                    f.write(f"Statistics for {len(stats_df)} categories\n")
                    for _, row in stats_df.iterrows():
                        cat = row['NounCategory']
                        f.write(f"  {cat}: ")
                        if 'LanguageEmbeddingDistance_mean' in row:
                            f.write(f"LanguageEmbeddingDistance mean={row['LanguageEmbeddingDistance_mean']:.4f}, ")
                        if 'CosineDistance_mean' in row:
                            f.write(f"CosineDistance mean={row['CosineDistance_mean']:.4f}, ")
                        if 'JaccardDistance_mean' in row:
                            f.write(f"JaccardDistance mean={row['JaccardDistance_mean']:.4f}, ")
                        f.write(f"count={row.get('CosineDistance_count', 0) or row.get('JaccardDistance_count', 0) or row.get('LanguageEmbeddingDistance_count', 0)}\n")
            
            return True
        else:
            # Handle the case where stats_df isn't directly in the dictionary
            # This might happen when category_percentages is called directly
            
            # Check if we have the 'categories' key instead
            if 'categories' in percentages:
                # Convert the nested dictionary to a DataFrame
                records = []
                for cat_name, cat_stats in percentages['categories'].items():
                    record = {'NounCategory': cat_name}
                    record.update(cat_stats)
                    records.append(record)
                
                if records:
                    cat_df = pd.DataFrame(records)
                    
                    # Define column descriptions
                    column_descriptions = {
                        'NounCategory': 'NounCategory',
                        'LanguageEmbeddingDistance_mean': 'LanguageEmbeddingDistance_mean (calculated as 1-cos_sim between aggregated language-level embeddings, higher values indicate greater semantic dissimilarity between languages)',
                        'LanguageEmbeddingDistance_min': 'LanguageEmbeddingDistance_min',
                        'LanguageEmbeddingDistance_max': 'LanguageEmbeddingDistance_max',
                        'LanguageEmbeddingDistance_count': 'LanguageEmbeddingDistance_count',
                        'LanguageEmbeddingDistance_std': 'LanguageEmbeddingDistance_std',
                        'LanguageEmbeddingDistance_percentile': 'LanguageEmbeddingDistance_percentile',
                        'CosineDistance_mean': 'CosineDistance_mean (calculated as 1-cos_sim between noun-level embeddings, measures semantic distance between individual noun representations)',
                        'CosineDistance_min': 'CosineDistance_min',
                        'CosineDistance_max': 'CosineDistance_max',
                        'CosineDistance_count': 'CosineDistance_count',
                        'CosineDistance_std': 'CosineDistance_std',
                        'CosineDistance_percentile': 'CosineDistance_percentile',
                        'JaccardDistance_mean': 'JaccardDistance_mean (calculated as 1-(intersection/union) of adjective sets, measures dissimilarity between adjective choices)',
                        'JaccardDistance_min': 'JaccardDistance_min',
                        'JaccardDistance_max': 'JaccardDistance_max',
                        'JaccardDistance_count': 'JaccardDistance_count',
                        'JaccardDistance_std': 'JaccardDistance_std',
                        'JaccardDistance_percentile': 'JaccardDistance_percentile',
                        'combined_uniqueness_score': 'combined_uniqueness_score (average of available distance percentiles, higher values indicate greater cross-linguistic uniqueness)'
                    }
                    
                    # Save the columns that exist in cat_df with renamed headers
                    columns_to_save = []
                    renamed_columns = {}
                    
                    # Filter to only include columns that exist in cat_df
                    for col in cat_df.columns:
                        if col in column_descriptions:
                            columns_to_save.append(col)
                            renamed_columns[col] = column_descriptions[col]
                    
                    # Make a copy to avoid modifying the original
                    output_df = cat_df[columns_to_save].copy()
                    
                    # Write to CSV with renamed columns
                    output_df.to_csv(output_file, header=renamed_columns.values(), index=False)
                    print(f"Category statistics saved to: {output_file}")
                    
                    # Also save a summary file
                    summary_file = output_file.replace('.csv', '_summary.txt')
                    with open(summary_file, 'w') as f:
                        if 'summary' in percentages:
                            f.write(percentages['summary'])
                        else:
                            f.write(f"Statistics for {len(records)} categories\n")
                            for record in records:
                                cat = record['NounCategory']
                                f.write(f"  {cat}: ")
                                if 'LanguageEmbeddingDistance_mean' in record:
                                    f.write(f"LanguageEmbeddingDistance mean={record['LanguageEmbeddingDistance_mean']:.4f}, ")
                                if 'CosineDistance_mean' in record:
                                    f.write(f"CosineDistance mean={record['CosineDistance_mean']:.4f}, ")
                                if 'JaccardDistance_mean' in record:
                                    f.write(f"JaccardDistance mean={record['JaccardDistance_mean']:.4f}, ")
                                f.write(f"count={record.get('CosineDistance_count', 0) or record.get('JaccardDistance_count', 0) or record.get('LanguageEmbeddingDistance_count', 0)}\n")
                    
                    return True
                else:
                    print(f"Warning: No category records found for {output_file}")
                    return False
            else:
                print(f"Warning: No stats_df or categories found in percentages data for {output_file}")
                return False
    except Exception as e:
        print(f"Error saving category percentages to {output_file}: {e}")
        import traceback
        traceback.print_exc()
        return False

__all__ = [
    # Data loading
    'load_data', 'load_noun_categories', 'load_language_families', 'load_grammatical_gender', 
    'load_adjective_prompts', 'find_csv_files', 'list_api_providers', 'list_api_models',
    
    # Distance metrics
    'compute_pairwise_distances', 'filter_distances', 'compute_jaccard_distances',
    
    # Visualization
    'plot_tsne', 'plot_umap', 'plot_pca', 'plot_language_dendrogram', 'plot_interactive_heatmap',
    'plot_combined_family_language_pca', 'plot_cross_distance_scatter', 
    'plot_noun_category_language_matrix', 'plot_average_noun_differences',
    'generate_standard_dim_reduction_plots',
    
    # Statistical analysis
    'calculate_cohens_d', 'analyze_noun_type_distances', 'calculate_category_percentages',
    'analyze_gender_language_distances', 'analyze_distance_metric_correlation',
    'analyze_noun_level_comparisons', 'analyze_language_level_statistics',
    'analyze_language_family_statistics',
    
    # Language processing
    'calculate_language_embeddings', 'compute_language_distances', 'calculate_language_family_embeddings_and_distances',
    'generate_language_level_comparisons', 'generate_language_comparisons_df',
    'run_language_level_analysis_and_visualization', 'run_family_level_analysis_and_visualization',
    
    # Utils
    'sanitize_prompt', 'sanitize_category', 'sanitize_name', 'detect_optimal_device',
    'create_directory_structure', 'create_category_directories',
    
    # Comprehensive analysis
    'generate_combined_analysis', '_generate_and_save_noun_level_distances_csv', '_save_category_percentages_to_csv',
    'process_and_save_consolidated_analysis'
]
