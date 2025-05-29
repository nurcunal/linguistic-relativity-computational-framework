"""
Utility functions for the Language Representation Analysis package.

This module contains various utility functions for string manipulation, 
device detection, and other general-purpose utilities.
"""

import os
import re
import torch
import platform
import pandas as pd
import hashlib


def sanitize_prompt(prompt):
    """
    Converts a prompt to lowercase and replaces spaces and slashes with underscores.
    
    Args:
        prompt (str): The prompt string to sanitize.
        
    Returns:
        str: Sanitized prompt safe for use in file/directory names.
    """
    return prompt.lower().replace(" ", "_").replace("/", "_")


def sanitize_category(category):
    """
    Converts a category name to lowercase and replaces spaces and slashes with underscores.
    
    Args:
        category (str): The category string to sanitize.
        
    Returns:
        str: Sanitized category name safe for use in file/directory names.
    """
    return category.lower().replace(" ", "_").replace("/", "_")


def sanitize_name(name):
    """
    Sanitizes a name for use in filenames by replacing non-alphanumeric characters.
    
    Args:
        name (str): The name string to sanitize.
        
    Returns:
        str: Sanitized name string safe for use in filenames.
    """
    if not name:
        return "unnamed"
    
    return re.sub(r'[^\w\-_\.]', '_', name)


def detect_optimal_device():
    """
    Detects the best available computing device.
    Prioritizes CUDA, then MPS (Mac GPU), then falls back to CPU.
    
    Returns:
        str: 'cuda', 'mps', or 'cpu'
    """
    device = 'cpu'  # Default fallback
    
    try:
        # Check if running on Apple Silicon Mac
        is_mac = platform.system() == 'Darwin'
        is_arm64 = platform.machine() == 'arm64'
        is_apple_silicon = is_mac and is_arm64
        
        if is_apple_silicon:
            print("Detected Apple Silicon Mac - checking for Metal GPU acceleration")
        
        # Check for CUDA (NVIDIA GPU)
        if torch.cuda.is_available():
            try:
                # Test CUDA with a small tensor operation to verify it works
                test_tensor = torch.ones(2, 2, device="cuda")
                result = test_tensor + test_tensor
                del result  # Clean up
                torch.cuda.empty_cache()
                
                device = 'cuda'
                print(f"CUDA GPU detected and verified - will use GPU acceleration")
                return device
            except Exception as cuda_error:
                print(f"CUDA available but test failed: {cuda_error}")
                print(f"Checking for Apple Silicon GPU acceleration...")
        
        # Check for MPS (Apple Silicon GPU)
        if is_apple_silicon and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                # More extensive MPS test with multiple operations
                print("Testing Apple Silicon GPU (MPS) acceleration...")
                mps_device = torch.device('mps')
                
                # Test 1: Simple addition
                test_tensor = torch.ones(2, 2)
                test_tensor = test_tensor.to(mps_device)
                result = test_tensor + test_tensor
                
                # Test 2: Matrix multiplication
                test_matrix = torch.rand(10, 10, device=mps_device)
                result = torch.matmul(test_matrix, test_matrix)
                
                # Cleanup
                del test_tensor, test_matrix, result
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                
                device = 'mps'
                print(f"Apple Silicon GPU acceleration (MPS) thoroughly verified - will use native GPU acceleration")
                return device
            except Exception as mps_error:
                print(f"Apple Silicon GPU (MPS) test failed: {mps_error}")
                print(f"Using CPU acceleration instead")
        
        # If we get here, use CPU
        if is_apple_silicon:
            print("Native Metal acceleration not available - using CPU on Apple Silicon")
        else:
            print(f"No GPU acceleration detected, using CPU")
        
    except ImportError:
        print(f"PyTorch not properly installed, using CPU")
    
    return device


def create_directory_structure(provider_folder, embeddings_dir, distances_dir, visualizations_dir, stats_dir):
    """
    Creates the necessary directory structure for the analysis pipeline.
    """
    # Produce all necessary base subdirectories
    os.makedirs(os.path.join(embeddings_dir, "proficiency_specific"), exist_ok=True)
    os.makedirs(os.path.join(embeddings_dir, "details"), exist_ok=True)
    os.makedirs(os.path.join(embeddings_dir, "language_level"), exist_ok=True)
    os.makedirs(os.path.join(embeddings_dir, "language_family_level"), exist_ok=True)

    # Remove unused proficiency_specific subfolder under comprehensive_analysis
    os.makedirs(os.path.join(distances_dir, "comprehensive_analysis", "details"), exist_ok=True)
    os.makedirs(os.path.join(distances_dir, "noun_level_distances", "details"), exist_ok=True)
    os.makedirs(os.path.join(distances_dir, "language_level", "proficiency_specific"), exist_ok=True)
    os.makedirs(os.path.join(distances_dir, "noun_level_language_comparisons", "details"), exist_ok=True)
    os.makedirs(os.path.join(distances_dir, "language_family_level"), exist_ok=True)

    # Visualization directories - per category details
    os.makedirs(os.path.join(visualizations_dir, "tsne", "details"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "tsne", "2d"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "tsne", "3d"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "umap", "details"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "umap", "2d"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "umap", "3d"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "pca", "details"), exist_ok=True)
    
    # Global visualization directories
    os.makedirs(os.path.join(visualizations_dir, "heatmaps"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "dendrograms"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "scatter_plots"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "scatter_plots", "cross_distance_correlations"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "scatter_plots", "cross_distance_correlations", "all_categories"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "scatter_plots", "cosine_jaccard_correlations"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "noun_category_matrix"), exist_ok=True)
    
    # Language-level visualization directories
    print("Creating language-level visualization directories...")
    
    # Create directories for UMAP language level visualizations
    os.makedirs(os.path.join(visualizations_dir, "umap", "language_level"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "umap", "language_level", "LanguageEmbeddingDistance"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "umap", "language_level", "AvgNounDistance"), exist_ok=True)
    
    # Create directories for T-SNE language level visualizations
    os.makedirs(os.path.join(visualizations_dir, "tsne", "language_level"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "tsne", "language_level", "LanguageEmbeddingDistance"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "tsne", "language_level", "LanguageEmbeddingDistance", "2d"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "tsne", "language_level", "LanguageEmbeddingDistance", "3d"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "tsne", "language_level", "AvgNounDistance"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "tsne", "language_level", "AvgNounDistance", "2d"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "tsne", "language_level", "AvgNounDistance", "3d"), exist_ok=True)
    
    # Create directories for PCA language level visualizations
    os.makedirs(os.path.join(visualizations_dir, "pca", "language_level"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "pca", "language_level", "LanguageEmbeddingDistance"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "pca", "language_level", "AvgNounDistance"), exist_ok=True)
    
    # Create directories for dendrogram language level visualizations with distance metric subfolders
    os.makedirs(os.path.join(visualizations_dir, "dendrograms", "language_level"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "dendrograms", "language_level", "LanguageEmbeddingDistance"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "dendrograms", "language_level", "AvgNounDistance"), exist_ok=True)
    
    # Create directories for heatmap language level visualizations with distance metric subfolders
    os.makedirs(os.path.join(visualizations_dir, "heatmaps", "language_level"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "heatmaps", "language_level", "LanguageEmbeddingDistance"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "heatmaps", "language_level", "AvgNounDistance"), exist_ok=True)
    
    # Language family level visualization directories with distance metric subfolders
    print("Creating language-family-level visualization directories...")
    
    # Create directories for UMAP language family level visualizations
    os.makedirs(os.path.join(visualizations_dir, "umap", "language_family_level"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "umap", "language_family_level", "LanguageEmbeddingDistance"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "umap", "language_family_level", "AvgNounDistance"), exist_ok=True)
    
    # Create directories for T-SNE language family level visualizations
    os.makedirs(os.path.join(visualizations_dir, "tsne", "language_family_level"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "tsne", "language_family_level", "LanguageEmbeddingDistance"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "tsne", "language_family_level", "LanguageEmbeddingDistance", "2d"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "tsne", "language_family_level", "LanguageEmbeddingDistance", "3d"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "tsne", "language_family_level", "AvgNounDistance"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "tsne", "language_family_level", "AvgNounDistance", "2d"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "tsne", "language_family_level", "AvgNounDistance", "3d"), exist_ok=True)
    
    # Create directories for PCA language family level visualizations
    os.makedirs(os.path.join(visualizations_dir, "pca", "language_family_level"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "pca", "language_family_level", "LanguageEmbeddingDistance"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "pca", "language_family_level", "AvgNounDistance"), exist_ok=True)
    
    # Create directories for heatmap language family level visualizations
    os.makedirs(os.path.join(visualizations_dir, "heatmaps", "language_family_level"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "heatmaps", "language_family_level", "LanguageEmbeddingDistance"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "heatmaps", "language_family_level", "AvgNounDistance"), exist_ok=True)
    
    # Create directories for dendrogram language family level visualizations
    os.makedirs(os.path.join(visualizations_dir, "dendrograms", "language_family_level"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "dendrograms", "language_family_level", "LanguageEmbeddingDistance"), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "dendrograms", "language_family_level", "AvgNounDistance"), exist_ok=True)
    
    # Make sure a specific directory for the "all_categories" visualization exists
    print("Creating \'all_categories\' visualization directories...")
    all_cat_base = os.path.join(visualizations_dir, "all_categories")
    os.makedirs(all_cat_base, exist_ok=True)
    os.makedirs(os.path.join(all_cat_base, "tsne"), exist_ok=True)
    os.makedirs(os.path.join(all_cat_base, "umap"), exist_ok=True)
    os.makedirs(os.path.join(all_cat_base, "pca"), exist_ok=True)
    
    # Create directory for average noun differences
    print("Creating average noun differences visualization directories...")
    avg_noun_diff_dir = os.path.join(visualizations_dir, "average_noun_differences")
    os.makedirs(avg_noun_diff_dir, exist_ok=True)
    os.makedirs(os.path.join(avg_noun_diff_dir, "all_categories"), exist_ok=True)
    
    # Directories for combined visualizations
    print("Creating combined visualization directories...")
    os.makedirs(os.path.join(visualizations_dir, "pca", "noun_categories_combined"), exist_ok=True)

    # Create deeper statistical report directories to ensure they exist
    print("Ensuring all statistical report directories exist...")
    os.makedirs(os.path.join(stats_dir, "noun_type_analysis", "details"), exist_ok=True)
    os.makedirs(os.path.join(stats_dir, "category_percentage_analysis", "details"), exist_ok=True)
    os.makedirs(os.path.join(stats_dir, "gender_language_analysis", "details"), exist_ok=True)
    os.makedirs(os.path.join(stats_dir, "correlations", "details"), exist_ok=True)
    
    # Create language level statistics directories with separate subdirectories for each distance metric
    print("Creating language-level statistics directories...")
    lang_stats_base_dir = os.path.join(stats_dir, "language_level_statistics")
    os.makedirs(os.path.join(lang_stats_base_dir, "LanguageEmbeddingDistance"), exist_ok=True)
    os.makedirs(os.path.join(lang_stats_base_dir, "LanguageEmbeddingDistance", "proficiency_specific"), exist_ok=True)
    os.makedirs(os.path.join(lang_stats_base_dir, "AvgNounDistance"), exist_ok=True)
    os.makedirs(os.path.join(lang_stats_base_dir, "AvgNounDistance", "proficiency_specific"), exist_ok=True)
    os.makedirs(os.path.join(lang_stats_base_dir, "AvgJaccardDistance"), exist_ok=True)
    os.makedirs(os.path.join(lang_stats_base_dir, "AvgJaccardDistance", "proficiency_specific"), exist_ok=True)
    
    # Create language family statistics directories with separate subdirectories for each distance metric
    print("Creating language-family-level statistics directories...")
    family_stats_base_dir = os.path.join(stats_dir, "language_family_statistics")
    os.makedirs(os.path.join(family_stats_base_dir, "LanguageEmbeddingDistance"), exist_ok=True)
    os.makedirs(os.path.join(family_stats_base_dir, "LanguageEmbeddingDistance", "proficiency_specific"), exist_ok=True)
    os.makedirs(os.path.join(family_stats_base_dir, "AvgNounDistance"), exist_ok=True)
    os.makedirs(os.path.join(family_stats_base_dir, "AvgNounDistance", "proficiency_specific"), exist_ok=True)
    
    # Create directories for noun-level evaluations
    print("Creating noun-level evaluation directories...")
    os.makedirs(os.path.join(stats_dir, "noun_level_evaluations"), exist_ok=True)
    os.makedirs(os.path.join(stats_dir, "noun_level_evaluations", "details"), exist_ok=True)
    os.makedirs(os.path.join(stats_dir, "noun_level_evaluations", "all_categories"), exist_ok=True)
    
    # Create per-language directories
    print("Creating per-language directories for statistical analyses...")
    os.makedirs(os.path.join(stats_dir, "category_percentage_analysis", "per_language"), exist_ok=True)
    os.makedirs(os.path.join(stats_dir, "correlations", "per_language"), exist_ok=True)
    os.makedirs(os.path.join(stats_dir, "gender_language_analysis", "per_language"), exist_ok=True)
    os.makedirs(os.path.join(stats_dir, "noun_level_evaluations", "per_language"), exist_ok=True)


def create_category_directories(embeddings_dir, distances_dir, visualizations_dir, stats_dir, detail_path_segment):
    """
    Creates the detailed directory structure needed for a specific category.
    """
    # Create required statistics directories
    os.makedirs(os.path.join(stats_dir, "noun_type_analysis", "details", detail_path_segment), exist_ok=True)
    os.makedirs(os.path.join(stats_dir, "category_percentage_analysis", "details", detail_path_segment), exist_ok=True)
    os.makedirs(os.path.join(stats_dir, "gender_language_analysis", "details", detail_path_segment), exist_ok=True)
    os.makedirs(os.path.join(stats_dir, "correlations", "details", detail_path_segment), exist_ok=True)
    os.makedirs(os.path.join(stats_dir, "noun_level_evaluations", "details", detail_path_segment), exist_ok=True)
    
    # Create per-language directories for category percentage analysis
    os.makedirs(os.path.join(stats_dir, "category_percentage_analysis", "details", detail_path_segment, "per_language"), exist_ok=True)
    
    # Create per-language directories for correlation analysis
    os.makedirs(os.path.join(stats_dir, "correlations", "details", detail_path_segment, "per_language"), exist_ok=True)
    
    # Create per-language directories for gender language analysis
    os.makedirs(os.path.join(stats_dir, "gender_language_analysis", "details", detail_path_segment, "per_language"), exist_ok=True)
    
    # Create per-language directories for noun-level evaluations
    os.makedirs(os.path.join(stats_dir, "noun_level_evaluations", "details", detail_path_segment, "per_language"), exist_ok=True)
    
    # Create required data directories
    os.makedirs(os.path.join(embeddings_dir, "details", detail_path_segment), exist_ok=True)
    os.makedirs(os.path.join(distances_dir, "noun_level_distances", "details", detail_path_segment), exist_ok=True)
    os.makedirs(os.path.join(distances_dir, "noun_level_language_comparisons", "details", detail_path_segment), exist_ok=True)
    os.makedirs(os.path.join(distances_dir, "comprehensive_analysis", "details", detail_path_segment), exist_ok=True)
    
    # Create required visualization directories
    os.makedirs(os.path.join(visualizations_dir, "tsne", "details", detail_path_segment), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "umap", "details", detail_path_segment), exist_ok=True)
    os.makedirs(os.path.join(visualizations_dir, "pca", "details", detail_path_segment), exist_ok=True)
