import os
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
import umap

# Utility functions
def find_embedding_models(embedding_dir="embedding_analysis"):
    """Find all embedding models that have been used"""
    embedding_models = set()
    
    for config_dir in os.listdir(embedding_dir):
        config_path = os.path.join(embedding_dir, config_dir)
        if not os.path.isdir(config_path):
            continue
            
        # Check for embedder info files
        embedder_info_path = os.path.join(config_path, "embedder_info.json")
        if os.path.exists(embedder_info_path):
            with open(embedder_info_path, "r") as f:
                info = json.load(f)
                if "model_name" in info:
                    embedding_models.add(info["model_name"])
    
    return list(embedding_models)

def find_config_results_by_embedder(embedding_dir="embedding_analysis", llm_config=None):
    """Find results for a specific LLM config across different embedders"""
    results_by_embedder = {}
    
    for config_dir in os.listdir(embedding_dir):
        config_path = os.path.join(embedding_dir, config_dir)
        if not os.path.isdir(config_path):
            continue
            
        # Skip if we're looking for a specific LLM config
        if llm_config and not config_dir.startswith(llm_config):
            continue
            
        # Check for embedder info
        embedder_info_path = os.path.join(config_path, "embedder_info.json")
        if os.path.exists(embedder_info_path):
            with open(embedder_info_path, "r") as f:
                info = json.load(f)
                embedder = info.get("model_name", "unknown")
                
                # Store config and paths
                if embedder not in results_by_embedder:
                    results_by_embedder[embedder] = []
                    
                results_by_embedder[embedder].append({
                    "config_id": config_dir,
                    "path": config_path,
                    "embedder": embedder
                })
    
    return results_by_embedder

def load_language_distances(config_path):
    """Load language distance matrix for a specific configuration"""
    file_path = os.path.join(config_path, "language_distances.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        # Convert to matrix format if necessary
        if "Language1" in df.columns and "Language2" in df.columns:
            # Convert from long format to matrix
            languages = sorted(list(set(df["Language1"]).union(set(df["Language2"]))))
            matrix = np.zeros((len(languages), len(languages)))
            
            for _, row in df.iterrows():
                i = languages.index(row["Language1"])
                j = languages.index(row["Language2"])
                matrix[i, j] = matrix[j, i] = row["Distance"]
                
            return {"matrix": matrix, "languages": languages}
        return df
    return None

def load_noun_embeddings(config_path):
    """Load noun embeddings for a specific configuration"""
    file_path = os.path.join(config_path, "noun_embeddings.csv")
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

# Comparison metrics
def compute_matrix_correlation(matrix1, matrix2):
    """Compute correlation between two distance matrices"""
    # Flatten upper triangular parts
    flat1 = matrix1[np.triu_indices_from(matrix1, k=1)]
    flat2 = matrix2[np.triu_indices_from(matrix2, k=1)]
    
    # Compute Pearson correlation
    r, p = stats.pearsonr(flat1, flat2)
    return {"correlation": r, "p_value": p}

def compute_kendall_tau_correlation(matrix1, matrix2):
    """Compute rank correlation between distance matrices"""
    # Flatten upper triangular parts
    flat1 = matrix1[np.triu_indices_from(matrix1, k=1)]
    flat2 = matrix2[np.triu_indices_from(matrix2, k=1)]
    
    # Compute Kendall's Tau
    tau, p = stats.kendalltau(flat1, flat2)
    return {"tau": tau, "p_value": p}

# Visualization functions
def plot_embedder_correlation_heatmap(corr_matrix, embedders, title="Embedder Correlation"):
    """Plot heatmap of correlations between embedders"""
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="viridis", 
                xticklabels=embedders, yticklabels=embedders)
    plt.title(title)
    plt.tight_layout()
    return fig

def plot_embedder_comparison_scatter(matrix1, matrix2, embedder1, embedder2):
    """Plot scatter comparison of distance values between two embedders"""
    # Flatten upper triangular parts
    flat1 = matrix1[np.triu_indices_from(matrix1, k=1)]
    flat2 = matrix2[np.triu_indices_from(matrix2, k=1)]
    
    r, p = stats.pearsonr(flat1, flat2)
    
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(flat1, flat2, alpha=0.5)
    plt.xlabel(f"{embedder1} Distances")
    plt.ylabel(f"{embedder2} Distances")
    plt.title(f"Distance Comparison\nr = {r:.3f}, p = {p:.3e}")
    
    # Add regression line
    m, b = np.polyfit(flat1, flat2, 1)
    plt.plot(flat1, m*flat1 + b, 'r-')
    
    # Add identity line
    min_val = min(flat1.min(), flat2.min())
    max_val = max(flat1.max(), flat2.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3)
    
    plt.tight_layout()
    return fig

def compare_language_clusters(matrix1, matrix2, languages, embedder1, embedder2):
    """Compare language clusters between two embedders using UMAP"""
    # Compute UMAP for both matrices
    reducer = umap.UMAP(metric="precomputed")
    embedding1 = reducer.fit_transform(matrix1)
    embedding2 = reducer.fit_transform(matrix2)
    
    # Create side-by-side plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot first embedder
    for i, lang in enumerate(languages):
        ax1.scatter(embedding1[i, 0], embedding1[i, 1])
        ax1.text(embedding1[i, 0], embedding1[i, 1], lang)
    ax1.set_title(f"{embedder1} Language Clustering")
    
    # Plot second embedder
    for i, lang in enumerate(languages):
        ax2.scatter(embedding2[i, 0], embedding2[i, 1])
        ax2.text(embedding2[i, 0], embedding2[i, 1], lang)
    ax2.set_title(f"{embedder2} Language Clustering")
    
    plt.tight_layout()
    return fig

# Main analysis functions
def analyze_embedder_consistency_for_config(llm_config, output_dir):
    """Analyze consistency across embedders for a specific LLM configuration"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Find results for this config across embedders
    results_by_embedder = find_config_results_by_embedder(llm_config=llm_config)
    
    if not results_by_embedder:
        print(f"No results found for LLM config: {llm_config}")
        return
    
    embedders = list(results_by_embedder.keys())
    print(f"Found {len(embedders)} embedders for {llm_config}: {', '.join(embedders)}")
    
    # Load language distances for each embedder
    language_distances = {}
    for embedder, configs in results_by_embedder.items():
        # Just use the first config for each embedder for now
        config = configs[0]
        language_distances[embedder] = load_language_distances(config["path"])
    
    # Compute correlation matrix between embedders
    n_embedders = len(embedders)
    corr_matrix = np.zeros((n_embedders, n_embedders))
    kendall_matrix = np.zeros((n_embedders, n_embedders))
    
    # Fill in identity diagonal
    np.fill_diagonal(corr_matrix, 1.0)
    np.fill_diagonal(kendall_matrix, 1.0)
    
    comparison_results = []
    
    for i, emb1 in enumerate(embedders):
        for j in range(i+1, n_embedders):
            emb2 = embedders[j]
            
            if language_distances[emb1] is None or language_distances[emb2] is None:
                continue
                
            # Get matrices
            matrix1 = language_distances[emb1]["matrix"]
            matrix2 = language_distances[emb2]["matrix"]
            languages = language_distances[emb1]["languages"]
            
            # Compute correlations
            pearson = compute_matrix_correlation(matrix1, matrix2)
            kendall = compute_kendall_tau_correlation(matrix1, matrix2)
            
            # Store results
            corr_matrix[i, j] = corr_matrix[j, i] = pearson["correlation"]
            kendall_matrix[i, j] = kendall_matrix[j, i] = kendall["tau"]
            
            comparison_results.append({
                "embedder1": emb1,
                "embedder2": emb2,
                "pearson_r": pearson["correlation"],
                "pearson_p": pearson["p_value"],
                "kendall_tau": kendall["tau"],
                "kendall_p": kendall["p_value"]
            })
            
            # Generate scatter plot
            scatter_fig = plot_embedder_comparison_scatter(
                matrix1, matrix2, emb1, emb2
            )
            scatter_fig.savefig(os.path.join(
                output_dir, f"scatter_{emb1}_vs_{emb2}.png"
            ))
            
            # Generate cluster comparison
            cluster_fig = compare_language_clusters(
                matrix1, matrix2, languages, emb1, emb2
            )
            cluster_fig.savefig(os.path.join(
                output_dir, f"clusters_{emb1}_vs_{emb2}.png"
            ))
    
    # Generate correlation heatmap
    heatmap_fig = plot_embedder_correlation_heatmap(
        corr_matrix, embedders, title=f"Pearson Correlation Between Embedders ({llm_config})"
    )
    heatmap_fig.savefig(os.path.join(output_dir, "embedder_correlation_heatmap.png"))
    
    # Generate Kendall heatmap
    kendall_fig = plot_embedder_correlation_heatmap(
        kendall_matrix, embedders, title=f"Kendall's Tau Between Embedders ({llm_config})"
    )
    kendall_fig.savefig(os.path.join(output_dir, "embedder_kendall_heatmap.png"))
    
    # Save comparison results
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df.to_csv(os.path.join(output_dir, "embedder_comparisons.csv"), index=False)
    
    # Save correlation matrices
    np.save(os.path.join(output_dir, "pearson_matrix.npy"), corr_matrix)
    np.save(os.path.join(output_dir, "kendall_matrix.npy"), kendall_matrix)
    
    # Generate summary
    summary = {
        "llm_config": llm_config,
        "embedders": embedders,
        "mean_pearson": np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]),
        "std_pearson": np.std(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]),
        "mean_kendall": np.mean(kendall_matrix[np.triu_indices_from(kendall_matrix, k=1)]),
        "std_kendall": np.std(kendall_matrix[np.triu_indices_from(kendall_matrix, k=1)])
    }
    
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Analysis complete for {llm_config}. Results saved to: {output_dir}")
    return summary

def analyze_all_embedders(embedding_dir="embedding_analysis", output_dir="analysis_results/embedder_comparison"):
    """Run analysis across all embedders and LLM configurations"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all unique LLM configurations
    llm_configs = set()
    for config_dir in os.listdir(embedding_dir):
        if os.path.isdir(os.path.join(embedding_dir, config_dir)):
            # Extract LLM config prefix (e.g., GG2-TN-2T-24L-5P-144N-1AP)
            # Assuming config_dir follows a pattern like: GG2-TN-2T-24L-5P-144N-1AP_mpnet
            parts = config_dir.split("_")
            if len(parts) > 0:
                llm_configs.add(parts[0])
    
    print(f"Found {len(llm_configs)} LLM configurations")
    
    all_summaries = []
    for llm_config in llm_configs:
        llm_output_dir = os.path.join(output_dir, llm_config)
        summary = analyze_embedder_consistency_for_config(llm_config, llm_output_dir)
        if summary:
            all_summaries.append(summary)
    
    # Create overall summary
    if all_summaries:
        overall_df = pd.DataFrame(all_summaries)
        overall_df.to_csv(os.path.join(output_dir, "all_config_summaries.csv"), index=False)
        
        # Only create visualizations if we have enough data
        if len(overall_df) > 1:  # Need at least 2 rows for boxplots
            # Plot distribution of correlations
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=overall_df, x="mean_pearson")
            plt.title("Distribution of Mean Pearson Correlations Across Configurations")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "pearson_distribution.png"))
            
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=overall_df, x="mean_kendall")
            plt.title("Distribution of Mean Kendall's Tau Across Configurations")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "kendall_distribution.png"))
        else:
            print("Not enough data for boxplot visualizations (need at least 2 LLM configurations).")
    
    print(f"Overall analysis complete. Results saved to: {output_dir}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare results across embedding models")
    parser.add_argument("--llm_config", default=None, 
                       help="Specific LLM configuration to analyze")
    parser.add_argument("--output", default="analysis_results/embedder_comparison", 
                       help="Output directory for analysis results")
    args = parser.parse_args()
    
    if args.llm_config:
        # Analyze a specific LLM configuration
        output_dir = os.path.join(args.output, args.llm_config)
        analyze_embedder_consistency_for_config(args.llm_config, output_dir)
    else:
        # Analyze all configurations
        analyze_all_embedders(output_dir=args.output)

if __name__ == "__main__":
    main() 