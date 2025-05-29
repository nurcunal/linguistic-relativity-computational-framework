"""
Visualization functions for Language Representation Analysis.

This module contains functions for generating various visualizations of
language representation analysis results, including dimensionality reduction
plots (t-SNE, UMAP, PCA), dendrograms, and heatmaps.
"""

import matplotlib.pyplot as plt
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import os
import shutil
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import umap
from scipy.spatial.distance import squareform
import traceback

from .distance_metrics import calculate_kruskal_stress
from lr_analysis_functions.utils import sanitize_name


def plot_tsne(df, plot_filename="semantic_space.png", title="Semantic Space (t-SNE)", color_by=None, interactive=False, original_distance_matrix=None, dimensions=2):
    """
    Generates a t-SNE plot of embeddings and optionally calculates Kruskal's Stress.
    
    Args:
        df (pd.DataFrame): DataFrame with embeddings and metadata (must have 'Embedding' and 'Label' columns)
        plot_filename (str): Path to save the plot image
        title (str): Title for the plot
        color_by (str, optional): Column name for coloring points (e.g., 'Language', 'NounCategory')
        interactive (bool): Whether to generate an interactive Plotly plot instead of static Matplotlib
        original_distance_matrix (pd.DataFrame, optional): Pairwise distance matrix for Kruskal's Stress calculation
        dimensions (int): Number of dimensions for the t-SNE projection (2 or 3)
        
    Returns:
        matplotlib.figure.Figure or plotly.graph_objects.Figure: The generated figure object
    """
    try:
        # Validate dimensions parameter
        if dimensions not in [2, 3]:
            print(f"Warning: dimensions must be 2 or 3. Defaulting to 2.")
            dimensions = 2
            
        print(f"Starting t-SNE visualization for {plot_filename} in {dimensions}D")
        print(f"  DataFrame has {len(df)} rows, columns: {list(df.columns)}")
        
        # Check if 'all_categories' is in the title
        is_all_categories = 'all categories' in title.lower() or 'all_categories' in title.lower()
        if is_all_categories:
            print(f"  Processing all_categories visualization")
        
        if len(df) < 2:
            print(f"  Warning: Not enough data to plot t-SNE visualization (need at least 2 rows, got {len(df)})")
            return plt.figure() if not interactive else go.Figure()
            
        embeddings = np.vstack(df['Embedding'])
        print(f"  Successfully stacked {len(embeddings)} embeddings with shape {embeddings.shape}")
        labels = df['Label'].tolist()
        
        if embeddings.shape[0] < 2:
            print(f"  Warning: Not enough embedding rows for t-SNE (need at least 2, got {embeddings.shape[0]})")
            return plt.figure() if not interactive else go.Figure()
            
        # Generate t-SNE projection
        print(f"  Generating {dimensions}D t-SNE projection for {embeddings.shape[0]} embeddings...")
        # Determine number of cores to use
        n_tsne_jobs = min(12, max(1, (os.cpu_count() - 4) if os.cpu_count() is not None else 1))
        if n_tsne_jobs > 1:
            print(f"  TSNE will attempt to use {n_tsne_jobs} cores.")

        # Use appropriate perplexity based on dataset size (perplexity should be smaller than n/3)
        perplexity = min(30, embeddings.shape[0] - 1)
        tsne = TSNE(
            n_components=dimensions, 
            random_state=42, 
            perplexity=perplexity, 
            n_jobs=n_tsne_jobs
        )
        
        emb_reduced = tsne.fit_transform(embeddings)
        print(f"  t-SNE projection complete")
        
        # Calculate Kruskal's Stress if original distances are provided
        if original_distance_matrix is not None and dimensions == 2:
            if original_distance_matrix.shape[0] == embeddings.shape[0] and original_distance_matrix.shape[1] == embeddings.shape[0]:
                try:
                    # Ensure the original_distance_matrix is in the right order
                    # For simplicity, we assume labels order matches the original_distance_matrix
                    stress = calculate_kruskal_stress(original_distance_matrix.values, emb_reduced)
                    print(f"Kruskal's Stress for t-SNE ('{title}'): {stress:.4f}")
                except Exception as e_stress:
                    print(f"Could not calculate Kruskal's stress: {e_stress}")
            else:
                print("Warning: Shape of original_distance_matrix does not match embeddings for Kruskal's stress calculation.")

        # Extract color information if specified
        if color_by and color_by in df.columns:
            # Get the categorical values
            categorical_values = df[color_by].tolist()
            
            # For categorical string values, map to integers for coloring
            if categorical_values and isinstance(categorical_values[0], str):
                # Get unique categories and map to integers
                unique_categories = sorted(set(categorical_values))
                category_to_int = {cat: i for i, cat in enumerate(unique_categories)}
                color_values = [category_to_int[cat] for cat in categorical_values]
                
                # Save the mapping for the legend
                category_mapping = category_to_int
                
                # Print categories for debugging
                if is_all_categories:
                    print(f"  Found {len(unique_categories)} unique categories for coloring: {unique_categories}")
            else:
                # For numeric values, use directly
                color_values = categorical_values
                category_mapping = None
        else:
            color_values = np.zeros(len(labels))
            category_mapping = None
        
        # Determine if this is a language-level or language-family visualization that needs labels
        is_language_level_plot = "Language-Level" in title or "Language Family" in title
        
        if interactive:
            # Generate an interactive Plotly figure
            try:
                if color_by:
                    # Prepare data for Plotly with all the information
                    if dimensions == 2:
                        plot_df = pd.DataFrame({
                            'x': emb_reduced[:, 0],
                            'y': emb_reduced[:, 1],
                            'label': labels,
                            'color_category': df[color_by].tolist() if color_by else None,
                            'color_value': color_values
                        })
                    else:  # 3D
                        plot_df = pd.DataFrame({
                            'x': emb_reduced[:, 0],
                            'y': emb_reduced[:, 1],
                            'z': emb_reduced[:, 2],
                            'label': labels,
                            'color_category': df[color_by].tolist() if color_by else None,
                            'color_value': color_values
                        })
                    
                    # Add all available columns from the original dataframe for tooltips
                    for col in df.columns:
                        if col not in ['Embedding'] and col not in plot_df.columns:
                            plot_df[col] = df[col].tolist()
                    
                    # Generate interactive scatter plot (2D or 3D)
                    if dimensions == 2:
                        fig = px.scatter(
                            plot_df, x='x', y='y', color='color_category',
                            hover_data={col: True for col in plot_df.columns if col not in ['x', 'y', 'color_value', 'color_category']},
                            title=title
                        )
                    else:  # 3D
                        fig = px.scatter_3d(
                            plot_df, x='x', y='y', z='z', color='color_category',
                            hover_data={col: True for col in plot_df.columns if col not in ['x', 'y', 'z', 'color_value', 'color_category']},
                            title=title
                        )
                    
                    # Add text labels for language-level or language-family plots
                    if is_language_level_plot and dimensions == 2:
                        fig.add_trace(go.Scatter(
                            x=plot_df['x'],
                            y=plot_df['y'],
                            mode='text',
                            text=labels,
                            textposition='top center',
                            textfont=dict(size=9, color='black'),
                            showlegend=False
                        ))
                    # In 3D, don't add text labels as they can be confusing
                else:
                    if dimensions == 2:
                        fig = px.scatter(
                            x=emb_reduced[:, 0], y=emb_reduced[:, 1],
                            hover_name=labels,
                            title=title
                        )
                        
                        # Add text labels for language-level or language-family plots
                        if is_language_level_plot:
                            fig.add_trace(go.Scatter(
                                x=emb_reduced[:, 0],
                                y=emb_reduced[:, 1],
                                mode='text',
                                text=labels,
                                textposition='top center',
                                textfont=dict(size=9, color='black'),
                                showlegend=False
                            ))
                    else:  # 3D
                        fig = px.scatter_3d(
                            x=emb_reduced[:, 0], y=emb_reduced[:, 1], z=emb_reduced[:, 2],
                            hover_name=labels,
                            title=title
                        )
                
                # Update layout
                if dimensions == 2:
                    fig.update_layout(
                        xaxis_title="Semantic Distance % (t-SNE)",
                        yaxis_title="Semantic Distance % (t-SNE)",
                        legend_title=color_by if color_by else "",
                        width=900, height=700,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    # Add dropdown filter for 2D plots when coloring by categories
                    if color_by and color_by in df.columns:
                        unique_categories = sorted(df[color_by].unique())
                        if len(unique_categories) > 1:
                            buttons = []
                            # First button shows all traces
                            buttons.append(dict(label='All', method='update', args=[{'visible': [True]*len(fig.data)}]))
                            
                            # Create a button for each category
                            for category in unique_categories:
                                # Set visibility array - True for traces that match this category, False for others
                                vis = []
                                for trace in fig.data:
                                    # Check if this is a labels trace (which should remain visible)
                                    if trace.mode == 'text' and not trace.showlegend:
                                        vis.append(True)
                                    else:
                                        # For data traces, show only if they match the category
                                        vis.append(str(trace.name) == str(category))
                                buttons.append(dict(label=str(category), method='update', args=[{'visible': vis}]))
                            
                            # Add the dropdown menu to the layout
                            fig.update_layout(
                                updatemenus=[dict(
                                    active=0,
                                    buttons=buttons,
                                    x=1.02,
                                    y=1,
                                    xanchor='left',
                                    yanchor='top',
                                    title=f"Filter by {color_by}"
                                )]
                            )
                else:  # 3D
                    fig.update_layout(
                        scene=dict(
                            xaxis_title="Semantic Distance % (t-SNE Dim 1)",
                            yaxis_title="Semantic Distance % (t-SNE Dim 2)",
                            zaxis_title="Semantic Distance % (t-SNE Dim 3)",
                            # Allow unlimited zoom with no constraints
                            camera=dict(
                                eye=dict(x=1.5, y=1.5, z=1.5),  # Default camera position
                                projection=dict(type="perspective")
                            ),
                            dragmode="turntable"
                        ),
                        legend_title=color_by if color_by else "",
                        width=1000, height=800,
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    
                    # Configure additional options for interactive HTML output
                    if plot_filename.endswith('.html'):
                        config = {
                            'scrollZoom': True,  # Enable scroll wheel zoom
                            'displayModeBar': True,  # Always show the mode bar
                            'modeBarButtonsToAdd': ['resetCameraDefault3d'],  # Add reset camera button
                            'showAxisDragHandles': True
                        }
                    else:
                        config = {'staticPlot': False}
                    
                    # Add dropdown filter for 3D plots when coloring by categories
                    if color_by and color_by in df.columns:
                        unique_categories = sorted(df[color_by].unique())
                        if len(unique_categories) > 1:
                            buttons = []
                            # First button shows all traces
                            buttons.append(dict(label='All', method='update', args=[{'visible': [True]*len(fig.data)}]))
                            
                            # Create a button for each category
                            for category in unique_categories:
                                # Set visibility array - True for traces that match this category, False for others
                                vis = []
                                for trace in fig.data:
                                    # For data traces, show only if they match the category
                                    vis.append(str(trace.name) == str(category))
                                buttons.append(dict(label=str(category), method='update', args=[{'visible': vis}]))
                            
                            # Add the dropdown menu to the layout
                            fig.update_layout(
                                updatemenus=[dict(
                                    active=0,
                                    buttons=buttons,
                                    x=1.02,
                                    y=1,
                                    xanchor='left',
                                    yanchor='top',
                                    title=f"Filter by {color_by}"
                                )]
                            )
                
                # Ensure directory exists
                output_dir = os.path.dirname(plot_filename)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                
                # Determine base path for files (without extension)
                base_path = os.path.splitext(plot_filename)[0]
                html_filename = base_path + ".html"
                png_filename = base_path + ".png"
                
                # Adjust output path for proper 2D/3D directory structure
                dir_parts = output_dir.split(os.sep)
                if 'tsne' in dir_parts:
                    # Get the index of 'tsne' in the path
                    tsne_idx = dir_parts.index('tsne')
                    
                    # Get path components up to and including 'tsne'
                    tsne_base_path = os.sep.join(dir_parts[:tsne_idx+1])
                    
                    # Check if we already have a dimension path component
                    if len(dir_parts) > tsne_idx + 1 and dir_parts[tsne_idx+1] in ['2d', '3d']:
                        # Already in a dimension directory, we're good
                        pass
                    else:
                        # Create proper dimension path
                        dim_dir = f"{dimensions}d"
                        dim_path = os.path.join(tsne_base_path, dim_dir)
                        
                        # Get filename without directory
                        filename = os.path.basename(plot_filename)
                        
                        # Construct new paths with proper dimension directory
                        new_path = os.path.join(dim_path, filename)
                        new_base_path = os.path.splitext(new_path)[0]
                        html_filename = new_base_path + ".html"
                        png_filename = new_base_path + ".png"
                        
                        # Make sure the dimension directory exists
                        os.makedirs(dim_path, exist_ok=True)
                
                fig.write_html(html_filename)
                print(f"Interactive {dimensions}D t-SNE plot saved to: {html_filename}")
                
                # Also save as PNG for static use
                try:
                    # Check if kaleido is installed for image export
                    try:
                        import kaleido
                        kaleido_available = True
                    except ImportError:
                        kaleido_available = False
                    
                    if kaleido_available:
                        fig.write_image(png_filename, scale=2)
                        print(f"Static {dimensions}D t-SNE plot also saved to: {png_filename}")
                    else:
                        print(f"Warning: Could not save static PNG image with Plotly - kaleido package is required")
                        print(f"To install kaleido run: pip install kaleido")
                        print(f"Interactive HTML version is still available at: {html_filename}")
                        
                        # Always generate a fallback static visualization with matplotlib
                        print("Generating fallback static visualization with matplotlib...")
                        # Create a matplotlib figure
                        if dimensions == 2:
                            plt.figure(figsize=(12, 10))
                            
                            # Handle color information
                            if color_by and color_by in df.columns:
                                categories = df[color_by].tolist()
                                unique_categories = sorted(set(categories))
                                category_to_int = {cat: i for i, cat in enumerate(unique_categories)}
                                colors = [category_to_int[cat] for cat in categories]
                                
                                scatter = plt.scatter(emb_reduced[:, 0], emb_reduced[:, 1], c=colors, cmap='tab10', alpha=0.7)
                                
                                # Add legend
                                handles = []
                                for cat, idx in category_to_int.items():
                                    handle = plt.Line2D(
                                        [0], [0], marker='o', color='w',
                                        markerfacecolor=plt.cm.tab10(idx/len(category_to_int)),
                                        markersize=10
                                    )
                                    handles.append(handle)
                                plt.legend(handles, unique_categories, title=color_by, loc='best', bbox_to_anchor=(1.05, 1))
                            else:
                                plt.scatter(emb_reduced[:, 0], emb_reduced[:, 1], alpha=0.7)
                            
                            # Add text labels for language-level plots
                            if is_language_level_plot:
                                for i, label in enumerate(labels):
                                    plt.annotate(
                                        label,
                                        (emb_reduced[i, 0], emb_reduced[i, 1]),
                                        textcoords="offset points",
                                        xytext=(0, 5),
                                        ha='center',
                                        fontsize=8
                                    )
                            
                            plt.title(title)
                            plt.xlabel("Semantic Distance % (t-SNE)")
                            plt.ylabel("Semantic Distance % (t-SNE)")
                            plt.tight_layout()
                        else:  # 3D
                            fig_mpl = plt.figure(figsize=(12, 10))
                            ax = fig_mpl.add_subplot(111, projection='3d')
                            
                            # Handle color information
                            if color_by and color_by in df.columns:
                                categories = df[color_by].tolist()
                                unique_categories = sorted(set(categories))
                                category_to_int = {cat: i for i, cat in enumerate(unique_categories)}
                                colors = [category_to_int[cat] for cat in categories]
                                
                                scatter = ax.scatter(emb_reduced[:, 0], emb_reduced[:, 1], emb_reduced[:, 2], 
                                                    c=colors, cmap='tab10', alpha=0.7)
                                
                                # Add legend
                                handles = []
                                for cat, idx in category_to_int.items():
                                    handle = plt.Line2D(
                                        [0], [0], marker='o', color='w',
                                        markerfacecolor=plt.cm.tab10(idx/len(category_to_int)),
                                        markersize=10
                                    )
                                    handles.append(handle)
                                ax.legend(handles, unique_categories, title=color_by, loc='best')
                            else:
                                ax.scatter(emb_reduced[:, 0], emb_reduced[:, 1], emb_reduced[:, 2], alpha=0.7)
                            
                            ax.set_title(title)
                            ax.set_xlabel("Semantic Distance % (t-SNE Dim 1)")
                            ax.set_ylabel("Semantic Distance % (t-SNE Dim 2)")
                            ax.set_zlabel("Semantic Distance % (t-SNE Dim 3)")
                            plt.tight_layout()
                        
                        # Always save the figure
                        plt.savefig(png_filename, dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"Successfully generated fallback {dimensions}D t-SNE visualization at: {png_filename}")
                except Exception as img_err:
                    print(f"Warning: Could not save static PNG image: {img_err}")
                    print(f"Interactive HTML version is still available")
            except Exception as plotly_err:
                print(f"Error generating interactive plot: {plotly_err}")
                # Fall back to matplotlib
                interactive = False
                # If falling back to matplotlib, fig will be reassigned
        
        if not interactive:  # Use Matplotlib for static plots
            # Generate a static Matplotlib plot
            if dimensions == 2:
                fig_mpl = plt.figure(figsize=(10, 8))
                scatter = plt.scatter(emb_reduced[:, 0], emb_reduced[:, 1], c=color_values, cmap="tab10", alpha=0.7, edgecolors='k')
                
                plt.title(title)
                plt.xlabel("Semantic Distance % (t-SNE)")
                plt.ylabel("Semantic Distance % (t-SNE)")
                
                # Add text labels for language-level or language-family plots
                if is_language_level_plot:
                    for i, label in enumerate(labels):
                        plt.annotate(
                            label,
                            (emb_reduced[i, 0], emb_reduced[i, 1]),
                            textcoords="offset points",
                            xytext=(0, 5),
                            ha='center',
                            fontsize=8
                        )
                
                # Add legend for categorical data
                if category_mapping:
                    handles = []
                    legend_labels = []  # Renamed to avoid conflict with outer scope 'labels'
                    for cat, idx in category_mapping.items():
                        handle = plt.Line2D(
                            [0], [0], marker='o', color='w', 
                            markerfacecolor=plt.cm.tab10(idx/len(category_mapping)), 
                            markersize=10
                        )
                        handles.append(handle)
                        legend_labels.append(cat)
                    plt.legend(handles, legend_labels, title=color_by, loc='best', bbox_to_anchor=(1.05, 1))
                # Add colorbar for numeric data
                elif color_by and isinstance(color_values[0], (int, float, np.number)):
                    plt.colorbar(scatter, label=color_by)
            else:  # 3D
                fig_mpl = plt.figure(figsize=(10, 8))
                ax = fig_mpl.add_subplot(111, projection='3d')
                scatter = ax.scatter(emb_reduced[:, 0], emb_reduced[:, 1], emb_reduced[:, 2], 
                                     c=color_values, cmap="tab10", alpha=0.7)
                
                ax.set_title(title)
                ax.set_xlabel("Semantic Distance % (t-SNE Dim 1)")
                ax.set_ylabel("Semantic Distance % (t-SNE Dim 2)")
                ax.set_zlabel("Semantic Distance % (t-SNE Dim 3)")
                
                # Add legend for categorical data
                if category_mapping:
                    handles = []
                    legend_labels = []  # Renamed to avoid conflict with outer scope 'labels'
                    for cat, idx in category_mapping.items():
                        handle = plt.Line2D(
                            [0], [0], marker='o', color='w', 
                            markerfacecolor=plt.cm.tab10(idx/len(category_mapping)), 
                            markersize=10
                        )
                        handles.append(handle)
                        legend_labels.append(cat)
                    ax.legend(handles, legend_labels, title=color_by, loc='best')
                # Add colorbar for numeric data
                elif color_by and isinstance(color_values[0], (int, float, np.number)):
                    plt.colorbar(scatter, ax=ax, label=color_by)
            
            # Ensure directory exists
            output_dir = os.path.dirname(plot_filename)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                
            # Adjust output path for proper 2D/3D directory structure
            dir_parts = output_dir.split(os.sep)
            if 'tsne' in dir_parts:
                # Get the index of 'tsne' in the path
                tsne_idx = dir_parts.index('tsne')
                
                # Get path components up to and including 'tsne'
                tsne_base_path = os.sep.join(dir_parts[:tsne_idx+1])
                
                # Check if we already have a dimension path component
                if len(dir_parts) > tsne_idx + 1 and dir_parts[tsne_idx+1] in ['2d', '3d']:
                    # Already in a dimension directory, we're good
                    pass
                else:
                    # Create proper dimension path
                    dim_dir = f"{dimensions}d"
                    dim_path = os.path.join(tsne_base_path, dim_dir)
                    
                    # Get filename without directory
                    filename = os.path.basename(plot_filename)
                    
                    # Construct new paths with proper dimension directory
                    plot_filename = os.path.join(dim_path, filename)
                    
                    # Make sure the dimension directory exists
                    os.makedirs(dim_path, exist_ok=True)
                
            print(f"{dimensions}D t-SNE plot generated: '{plot_filename}'")
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close(fig_mpl)  # Close the figure to free memory
            return fig_mpl
        
        # Return the Plotly figure if interactive mode was successful
        return fig
    
    except Exception as e:
        print(f"Error during t-SNE visualization: {e}")
        import traceback
        traceback.print_exc()
        # Return empty figure to allow pipeline to continue
        return plt.figure() if not interactive else go.Figure()


def plot_umap(df, plot_filename="umap_visualization.png", title="UMAP Visualization", color_by=None, interactive=False, original_distance_matrix=None, dimensions=2):
    """
    Generates a UMAP plot of embeddings with optional Kruskal's Stress calculation.
    UMAP often preserves global structure better than t-SNE.
    
    Args:
        df (pd.DataFrame): DataFrame with embeddings and metadata
        plot_filename (str): Path to save the plot image
        title (str): Title for the plot
        color_by (str, optional): Column name for coloring points
        interactive (bool): Whether to generate an interactive Plotly plot
        original_distance_matrix (pd.DataFrame, optional): Pairwise distance matrix for Kruskal's Stress
        dimensions (int): Number of dimensions for the UMAP projection (2 or 3)
        
    Returns:
        matplotlib.figure.Figure or plotly.graph_objects.Figure: The generated figure object
    """
    if dimensions not in [2, 3]:
        print(f"Warning: dimensions must be 2 or 3. Defaulting to 2.")
        dimensions = 2
        
    if len(df) < 2:
        print("Not enough data to plot.")
        return plt.figure() if not interactive else go.Figure()
        
    embeddings = np.vstack(df['Embedding'])
    labels = df['Label'].tolist()
    
    if embeddings.shape[0] < 2:
        print("Not enough rows for UMAP.")
        return plt.figure() if not interactive else go.Figure()
    
    # Generate UMAP projection
    n_neighbors = min(30, embeddings.shape[0] - 1)
    min_dist = 0.1
    
    try:
        # Adjust parameters based on data size
        if embeddings.shape[0] > 100:
            n_neighbors = 30
        elif embeddings.shape[0] > 50:
            n_neighbors = 15
        else:
            n_neighbors = 5
        
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=dimensions,
            random_state=42,
            n_jobs=-1  # Add n_jobs parameter
        )
        
        emb_reduced = reducer.fit_transform(embeddings)
        print(f"UMAP reduction complete with n_neighbors={n_neighbors}, min_dist={min_dist}, dimensions={dimensions}")

        # Calculate Kruskal's Stress if original distances are provided
        if original_distance_matrix is not None and dimensions == 2:
            if original_distance_matrix.shape[0] == embeddings.shape[0] and original_distance_matrix.shape[1] == embeddings.shape[0]:
                try:
                    stress = calculate_kruskal_stress(original_distance_matrix.values, emb_reduced)
                    print(f"Kruskal's Stress for UMAP ('{title}'): {stress:.4f}")
                except Exception as e_stress:
                    print(f"Could not calculate Kruskal's stress for UMAP: {e_stress}")
            else:
                print("Warning: Shape of original_distance_matrix does not match embeddings for UMAP Kruskal's stress calculation.")

    except Exception as e:
        print(f"Error during UMAP reduction: {e}")
        return plt.figure() if not interactive else go.Figure()
    
    # Extract color information if specified
    if color_by and color_by in df.columns:
        # Get the categorical values
        categorical_values = df[color_by].tolist()
        
        # For categorical string values, map to integers for coloring
        if categorical_values and isinstance(categorical_values[0], str):
            # Get unique categories and map to integers
            unique_categories = sorted(set(categorical_values))
            category_to_int = {cat: i for i, cat in enumerate(unique_categories)}
            color_values = [category_to_int[cat] for cat in categorical_values]
            
            # Save the mapping for the legend
            category_mapping = category_to_int
        else:
            # For numeric values, use directly
            color_values = categorical_values
            category_mapping = None
    else:
        color_values = np.zeros(len(labels))
        category_mapping = None
        
    # Determine if this is a language-level or language-family visualization that needs labels
    is_language_level_plot = "Language-Level" in title or "Language Family" in title
    
    if interactive:
        # Generate an interactive Plotly figure
        try:
            if color_by:
                # Prepare data for Plotly with all the information
                if dimensions == 2:
                    plot_df = pd.DataFrame({
                        'x': emb_reduced[:, 0],
                        'y': emb_reduced[:, 1],
                        'label': labels,
                        'color_category': df[color_by].tolist() if color_by else None,
                        'color_value': color_values
                    })
                else:  # 3D
                    plot_df = pd.DataFrame({
                        'x': emb_reduced[:, 0],
                        'y': emb_reduced[:, 1],
                        'z': emb_reduced[:, 2],
                        'label': labels,
                        'color_category': df[color_by].tolist() if color_by else None,
                        'color_value': color_values
                    })
                
                # Add all available columns from the original dataframe for tooltips
                for col in df.columns:
                    if col not in ['Embedding'] and col not in plot_df.columns:
                        plot_df[col] = df[col].tolist()
                
                # Generate interactive scatter plot
                if dimensions == 2:
                    fig = px.scatter(
                        plot_df, x='x', y='y', color='color_category',
                        hover_data={col: True for col in plot_df.columns if col not in ['x', 'y', 'color_value', 'color_category']},
                        title=title
                    )
                    
                    # Add text labels for language-level or language-family plots
                    if is_language_level_plot:
                        fig.add_trace(go.Scatter(
                            x=plot_df['x'],
                            y=plot_df['y'],
                            mode='text',
                            text=labels,
                            textposition='top center',
                            textfont=dict(size=9, color='black'),
                            showlegend=False
                        ))
                else:  # 3D
                    fig = px.scatter_3d(
                        plot_df, x='x', y='y', z='z', color='color_category',
                        hover_data={col: True for col in plot_df.columns if col not in ['x', 'y', 'z', 'color_value', 'color_category']},
                        title=title
                    )
            else:
                if dimensions == 2:
                    fig = px.scatter(
                        x=emb_reduced[:, 0], y=emb_reduced[:, 1],
                        hover_name=labels,
                        title=title
                    )
                    
                    # Add text labels for language-level or language-family plots
                    if is_language_level_plot:
                        fig.add_trace(go.Scatter(
                            x=emb_reduced[:, 0],
                            y=emb_reduced[:, 1],
                            mode='text',
                            text=labels,
                            textposition='top center',
                            textfont=dict(size=9, color='black'),
                            showlegend=False
                        ))
                else:  # 3D
                    fig = px.scatter_3d(
                        x=emb_reduced[:, 0], y=emb_reduced[:, 1], z=emb_reduced[:, 2],
                        hover_name=labels,
                        title=title
                    )
            
            # Update layout with more meaningful axis labels
            if dimensions == 2:
                fig.update_layout(
                    xaxis_title="Semantic Distance % (UMAP)",
                    yaxis_title="Semantic Distance % (UMAP)",
                    legend_title=color_by if color_by else "",
                    width=900, height=700,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                # Add dropdown filter for 2D UMAP plots when coloring by categories
                if color_by and color_by in df.columns:
                    unique_categories = sorted(df[color_by].unique())
                    if len(unique_categories) > 1:
                        buttons = []
                        # First button shows all traces
                        buttons.append(dict(label='All', method='update', args=[{'visible': [True]*len(fig.data)}]))
                        
                        # Create a button for each category
                        for category in unique_categories:
                            # Set visibility array - True for traces that match this category, False for others
                            vis = []
                            for trace in fig.data:
                                # Check if this is a labels trace (which should remain visible)
                                if trace.mode == 'text' and not trace.showlegend:
                                    vis.append(True)
                                else:
                                    # For data traces, show only if they match the category
                                    vis.append(str(trace.name) == str(category))
                            buttons.append(dict(label=str(category), method='update', args=[{'visible': vis}]))
                        
                        # Add the dropdown menu to the layout
                        fig.update_layout(
                            updatemenus=[dict(
                                active=0,
                                buttons=buttons,
                                x=1.02,
                                y=1,
                                xanchor='left',
                                yanchor='top',
                                title=f"Filter by {color_by}"
                            )]
                        )
            else:  # 3D
                fig.update_layout(
                    scene=dict(
                        xaxis_title="Semantic Distance % (UMAP Dim 1)",
                        yaxis_title="Semantic Distance % (UMAP Dim 2)",
                        zaxis_title="Semantic Distance % (UMAP Dim 3)",
                        # Allow unlimited zoom with no constraints
                        camera=dict(
                            eye=dict(x=1.5, y=1.5, z=1.5),  # Default camera position
                            projection=dict(type="perspective")
                        ),
                        dragmode="turntable"
                    ),
                    legend_title=color_by if color_by else "",
                    width=1000, height=800,
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                
                # Configure additional options for interactive HTML output
                if plot_filename.endswith('.html'):
                    config = {
                        'scrollZoom': True,  # Enable scroll wheel zoom
                        'displayModeBar': True,  # Always show the mode bar
                        'modeBarButtonsToAdd': ['resetCameraDefault3d'],  # Add reset camera button
                        'showAxisDragHandles': True
                    }
                else:
                    config = {'staticPlot': False}
                
                # Add dropdown filter for 3D plots when coloring by categories
                if color_by and color_by in df.columns:
                    unique_categories = sorted(df[color_by].unique())
                    if len(unique_categories) > 1:
                        buttons = []
                        # First button shows all traces
                        buttons.append(dict(label='All', method='update', args=[{'visible': [True]*len(fig.data)}]))
                        
                        # Create a button for each category
                        for category in unique_categories:
                            # Set visibility array - True for traces that match this category, False for others
                            vis = []
                            for trace in fig.data:
                                # We need to check the trace name to see if it matches the category
                                vis.append(str(trace.name) == str(category))
                            buttons.append(dict(label=str(category), method='update', args=[{'visible': vis}]))
                        
                        # Add the dropdown menu to the layout
                        fig.update_layout(
                            updatemenus=[dict(
                                active=0,
                                buttons=buttons,
                                x=1.02,
                                y=1,
                                xanchor='left',
                                yanchor='top',
                                title=f"Filter by {color_by}"
                            )]
                        )
            
            # Save as HTML for interactivity
            base_path = os.path.splitext(plot_filename)[0]
            html_filename = base_path + ".html"
            png_filename = base_path + ".png"
            
            # Adjust output path for proper 2D/3D directory structure
            output_dir = os.path.dirname(plot_filename)
            dir_parts = output_dir.split(os.sep)
            if 'umap' in dir_parts:
                # Get the index of 'umap' in the path
                umap_idx = dir_parts.index('umap')
                
                # Get path components up to and including 'umap'
                umap_base_path = os.sep.join(dir_parts[:umap_idx+1])
                
                # Check if we already have a dimension path component
                if len(dir_parts) > umap_idx + 1 and dir_parts[umap_idx+1] in ['2d', '3d']:
                    # Already in a dimension directory, we're good
                    pass
                else:
                    # Create proper dimension path
                    dim_dir = f"{dimensions}d"
                    dim_path = os.path.join(umap_base_path, dim_dir)
                    
                    # Get filename without directory
                    filename = os.path.basename(plot_filename)
                    
                    # Construct new paths with proper dimension directory
                    new_path = os.path.join(dim_path, filename)
                    new_base_path = os.path.splitext(new_path)[0]
                    html_filename = new_base_path + ".html"
                    png_filename = new_base_path + ".png"
                    
                    # Make sure the dimension directory exists
                    os.makedirs(dim_path, exist_ok=True)
            
            fig.write_html(html_filename)
            print(f"Interactive {dimensions}D UMAP plot saved to: {html_filename}")
            
            # Also save as PNG for static use
            try:
                # Check if kaleido is installed for image export
                try:
                    import kaleido
                    kaleido_available = True
                except ImportError:
                    kaleido_available = False
                
                if kaleido_available:
                    fig.write_image(png_filename, scale=2)
                    print(f"Static {dimensions}D UMAP plot also saved to: {png_filename}")
                else:
                    print(f"Warning: Could not save static PNG image with Plotly - kaleido package is required")
                    print(f"To install kaleido run: pip install kaleido")
                    print(f"Interactive HTML version is still available at: {html_filename}")
                    
                    # Always generate a fallback static visualization with matplotlib
                    print(f"Generating fallback static {dimensions}D UMAP visualization with matplotlib...")
                    if dimensions == 2:
                        # Create a matplotlib figure
                        plt.figure(figsize=(12, 10))
                        
                        # Handle color information
                        if color_by and color_by in df.columns:
                            categories = df[color_by].tolist()
                            unique_categories = sorted(set(categories))
                            category_to_int = {cat: i for i, cat in enumerate(unique_categories)}
                            colors = [category_to_int[cat] for cat in categories]
                            
                            scatter = plt.scatter(emb_reduced[:, 0], emb_reduced[:, 1], c=colors, cmap='tab10', alpha=0.7)
                            
                            # Add legend
                            handles = []
                            for cat, idx in category_to_int.items():
                                handle = plt.Line2D(
                                    [0], [0], marker='o', color='w',
                                    markerfacecolor=plt.cm.tab10(idx/len(category_to_int)),
                                    markersize=10
                                )
                                handles.append(handle)
                            plt.legend(handles, unique_categories, title=color_by, loc='best', bbox_to_anchor=(1.05, 1))
                        else:
                            plt.scatter(emb_reduced[:, 0], emb_reduced[:, 1], alpha=0.7)
                        
                        # Add text labels for language-level plots
                        if is_language_level_plot:
                            for i, label in enumerate(labels):
                                plt.annotate(
                                    label,
                                    (emb_reduced[i, 0], emb_reduced[i, 1]),
                                    textcoords="offset points",
                                    xytext=(0, 5),
                                    ha='center',
                                    fontsize=8
                                )
                        
                        plt.title(title)
                        plt.xlabel("Semantic Distance % (UMAP)")
                        plt.ylabel("Semantic Distance % (UMAP)")
                        plt.tight_layout()
                    else:  # 3D
                        # Create a matplotlib figure with 3D axis
                        fig_mpl = plt.figure(figsize=(12, 10))
                        ax = fig_mpl.add_subplot(111, projection='3d')
                        
                        # Handle color information
                        if color_by and color_by in df.columns:
                            categories = df[color_by].tolist()
                            unique_categories = sorted(set(categories))
                            category_to_int = {cat: i for i, cat in enumerate(unique_categories)}
                            colors = [category_to_int[cat] for cat in categories]
                            
                            scatter = ax.scatter(emb_reduced[:, 0], emb_reduced[:, 1], emb_reduced[:, 2], 
                                               c=colors, cmap='tab10', alpha=0.7)
                            
                            # Add legend
                            handles = []
                            for cat, idx in category_to_int.items():
                                handle = plt.Line2D(
                                    [0], [0], marker='o', color='w',
                                    markerfacecolor=plt.cm.tab10(idx/len(category_to_int)),
                                    markersize=10
                                )
                                handles.append(handle)
                            ax.legend(handles, unique_categories, title=color_by, loc='best')
                        else:
                            ax.scatter(emb_reduced[:, 0], emb_reduced[:, 1], emb_reduced[:, 2], alpha=0.7)
                        
                        ax.set_title(title)
                        ax.set_xlabel("Semantic Distance % (UMAP Dim 1)")
                        ax.set_ylabel("Semantic Distance % (UMAP Dim 2)")
                        ax.set_zlabel("Semantic Distance % (UMAP Dim 3)")
                        plt.tight_layout()
                    
                    # Save the matplotlib version
                    plt.savefig(png_filename, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Successfully generated fallback {dimensions}D UMAP visualization at: {png_filename}")
            except Exception as img_err:
                print(f"Warning: Could not save static PNG image: {img_err}")
                print(f"Interactive HTML version is still available")
        except Exception as plotly_err:
            print(f"Error generating interactive plot: {plotly_err}")
            # Fall back to matplotlib
            interactive = False
        
        
    if not interactive:  # Use Matplotlib for static plots
        # Generate a static Matplotlib plot
        if dimensions == 2:
            fig_mpl = plt.figure(figsize=(10, 8))
            scatter = plt.scatter(emb_reduced[:, 0], emb_reduced[:, 1], c=color_values, cmap="tab10", alpha=0.7, edgecolors='k')
            
            plt.title(title)
            plt.xlabel("Semantic Distance % (UMAP)")
            plt.ylabel("Semantic Distance % (UMAP)")
            
            # Add text labels for language-level or language-family plots
            if is_language_level_plot:
                for i, label in enumerate(labels):
                    plt.annotate(
                        label,
                        (emb_reduced[i, 0], emb_reduced[i, 1]),
                        textcoords="offset points",
                        xytext=(0, 5),
                        ha='center',
                        fontsize=8
                    )
            
            # Add legend for categorical data
            if category_mapping:
                handles = []
                labels_legend = []
                for cat, idx in category_mapping.items():
                    handle = plt.Line2D(
                        [0], [0], marker='o', color='w', 
                        markerfacecolor=plt.cm.tab10(idx/len(category_mapping)), 
                        markersize=10
                    )
                    handles.append(handle)
                    labels_legend.append(cat)
                plt.legend(handles, labels_legend, title=color_by, loc='best', bbox_to_anchor=(1.05, 1))
            # Add colorbar for numeric data
            elif color_by and isinstance(color_values[0], (int, float, np.number)):
                plt.colorbar(scatter, label=color_by)
        else:  # 3D
            fig_mpl = plt.figure(figsize=(10, 8))
            ax = fig_mpl.add_subplot(111, projection='3d')
            scatter = ax.scatter(emb_reduced[:, 0], emb_reduced[:, 1], emb_reduced[:, 2], 
                               c=color_values, cmap="tab10", alpha=0.7)
            
            ax.set_title(title)
            ax.set_xlabel("Semantic Distance % (UMAP Dim 1)")
            ax.set_ylabel("Semantic Distance % (UMAP Dim 2)")
            ax.set_zlabel("Semantic Distance % (UMAP Dim 3)")
            
            # Add legend for categorical data
            if category_mapping:
                handles = []
                labels_legend = []
                for cat, idx in category_mapping.items():
                    handle = plt.Line2D(
                        [0], [0], marker='o', color='w', 
                        markerfacecolor=plt.cm.tab10(idx/len(category_mapping)), 
                        markersize=10
                    )
                    handles.append(handle)
                    labels_legend.append(cat)
                ax.legend(handles, labels_legend, title=color_by, loc='best')
            # Add colorbar for numeric data
            elif color_by and isinstance(color_values[0], (int, float, np.number)):
                plt.colorbar(scatter, ax=ax, label=color_by)
        
        print(f"{dimensions}D UMAP plot generated: '{plot_filename}'")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close(fig_mpl)  # Close the figure to free memory
        return fig_mpl
    
    # Return the Plotly figure if interactive mode was successful
    return fig


def plot_language_dendrogram(distance_matrix, languages, plot_filename="language_dendrogram.png", title="Language Clustering Dendrogram"):
    """
    Generates a hierarchical clustering dendrogram of languages based on distances.
    
    Args:
        distance_matrix (np.ndarray or pd.DataFrame): Square distance matrix between languages
        languages (list): List of language names corresponding to the matrix rows/columns
        plot_filename (str): Path to save the plot image
        title (str): Title for the plot
        
    Returns:
        matplotlib.figure.Figure: The generated figure object
    """
    try:
        if len(languages) < 2:
            print(f"Not enough languages ({len(languages)}) to produce a dendrogram for '{title}'.")
            return plt.figure()
        
        # Ensure we have a numpy array
        if isinstance(distance_matrix, pd.DataFrame):
            dist_array = distance_matrix.values
        else:
            dist_array = np.asarray(distance_matrix)

        n_langs = len(languages)
        if dist_array.shape != (n_langs, n_langs):
            print(f"Error: Distance matrix shape {dist_array.shape} doesn't match number of languages {n_langs} for '{title}'.")
            return plt.figure()
        
        # Check for NaN/inf values which can cause issues with linkage
        if np.any(np.isnan(dist_array)) or np.any(np.isinf(dist_array)):
            print(f"Warning: NaN or Inf values found in distance matrix for dendrogram title='{title}'. Attempting to fill them.")
            # Replace NaNs/Infs. A large value or mean/median could be used.
            # Using a value larger than any other actual finite distance.
            if np.all(np.isnan(dist_array)): # All values are NaN
                print(f"Error: All values in distance matrix are NaN for '{title}'. Cannot compute dendrogram.")
                return plt.figure()

            max_finite_dist = np.nanmax(dist_array[np.isfinite(dist_array)])
            
            # Determine a fill_value. If max_finite_dist is nan (e.g. all values were nan/inf), use a default large value.
            if pd.isna(max_finite_dist) or not np.isfinite(max_finite_dist):
                fill_value = 10.0 # Default large value if no finite distances exist
            else:
                fill_value = max_finite_dist * 2 if max_finite_dist > 0 else 1.0 # Handle case where max_finite_dist might be 0

            print(f"  Filling NaNs/Infs with: {fill_value}")
            dist_array = np.nan_to_num(dist_array, nan=fill_value, posinf=fill_value, neginf=0) # neginf to 0 for distances (should not happen)

        # Convert the square distance matrix to a condensed distance matrix (1D array)
        # linkage expects condensed 1D array (upper triangle)
        # Ensure the matrix is symmetric before converting, or handle potential issues.
        # For simplicity, we assume it should be symmetric. squareform will check.
        try:
            condensed_dist = squareform(dist_array, force='tovector', checks=True)
        except ValueError as sve:
            print(f"Error converting distance matrix to condensed form for '{title}': {sve}. Matrix might not be a valid distance matrix.")
            # Attempt to make it symmetric if it's a common issue like floating point inaccuracies
            # Forcing symmetry: (A + A.T) / 2
            if dist_array.shape[0] == dist_array.shape[1]: # only if square
                print("  Attempting to force symmetry on the distance matrix.")
                dist_array_sym = (dist_array + dist_array.T) / 2.0
                np.fill_diagonal(dist_array_sym, 0) # Diagonals must be zero
                try:
                    condensed_dist = squareform(dist_array_sym, force='tovector', checks=True)
                    print("  Successfully converted to condensed form after forcing symmetry.")
                except ValueError as sve_sym:
                    print(f"  Still could not convert to condensed form after forcing symmetry for '{title}': {sve_sym}")
                    return plt.figure()
            else:
                return plt.figure()


        if condensed_dist.ndim != 1 or len(condensed_dist) < 1 :
             print(f"Warning: Condensed distance matrix is not valid for linkage for dendrogram title='{title}'. Shape: {condensed_dist.shape}")
             return plt.figure()

        Z = linkage(condensed_dist, method='average')
        
        # Generate dendrogram
        fig, ax = plt.subplots(figsize=(max(12, n_langs * 0.5), 8)) # Adjust width based on number of languages
        
        # Set color threshold to get some distinction in the clusters
        color_threshold = 0.7 * np.max(Z[:, 2]) if Z.shape[0] > 0 else 0
        
        dendrogram(
            Z,
            labels=languages,
            leaf_rotation=90,
            leaf_font_size=10,
            color_threshold=color_threshold,
            ax=ax
        )
        
        plt.title(title)
        plt.xlabel('Languages' if "Family" not in title else "Language Families") # Adjust label
        plt.ylabel('Semantic Distance %')
        plt.tight_layout()
        
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Dendrogram saved to: {plot_filename}")
        
        plt.close(fig)
        return fig
    
    except Exception as e:
        print(f"Error during dendrogram generation for '{title}': {e}")
        import traceback
        traceback.print_exc()
        return plt.figure()


def plot_interactive_heatmap(matrix, output_file, title="Distance Heatmap"):
    """
    Creates an interactive heatmap using Plotly.
    
    Args:
        matrix (pd.DataFrame): Distance matrix
        output_file (str): Path to save the HTML file
        title (str): Title for the heatmap
    """
    fig = go.Figure()
    
    # Get labels
    labels = matrix.index.tolist()
    
    # Create heatmap
    fig.add_trace(go.Heatmap(
        z=matrix.values,
        x=labels,
        y=labels,
        colorscale='Viridis',
        colorbar=dict(title="Semantic Distance %"),
        hovertemplate='%{y} - %{x}: %{z:.2f} semantic distance (%{z:.1%})<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        width=800, 
        height=800,
        margin=dict(t=50, l=50, r=50, b=50)
    )
    
    # Add dropdown filter for selecting specific labels
    if len(labels) > 3:  # Only add filter if we have enough labels to make it useful
        buttons = []
        # First button shows all data
        buttons.append(dict(label='All', method='update', args=[{'visible': [True]}]))
        
        # Create a button for each label to filter
        for label in labels:
            # Create a custom z array that shows only the row/column for this label
            z_filtered = np.zeros_like(matrix.values)
            label_idx = labels.index(label)
            
            # Make this label's row and column visible, all others zero
            z_filtered[label_idx, :] = matrix.values[label_idx, :]
            z_filtered[:, label_idx] = matrix.values[:, label_idx]
            
            # Create button to update the heatmap
            buttons.append(dict(
                label=str(label),
                method='update',
                args=[{
                    'z': [z_filtered],
                    'visible': [True]
                }]
            ))
        
        # Add the dropdown menu to the layout
        fig.update_layout(
            updatemenus=[dict(
                active=0,
                buttons=buttons,
                x=1.02,
                y=1,
                xanchor='left',
                yanchor='top',
                title="Filter by label"
            )]
        )
    
    # Save as HTML file
    fig.write_html(output_file)
    
    # Save static PNG version
    try:
        # Try with kaleido if available
        try:
            import kaleido
            png_file = os.path.splitext(output_file)[0] + ".png"
            fig.write_image(png_file, width=800, height=800, scale=2)
            print(f"Static heatmap PNG saved to: {png_file}")
        except ImportError:
            # Fallback to matplotlib if kaleido is not available
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Create a matplotlib version of the heatmap
            plt.figure(figsize=(10, 8))
            plt.imshow(matrix.values, cmap='viridis')
            plt.colorbar(label='Semantic Distance %')
            plt.xticks(np.arange(len(labels)), labels, rotation=90)
            plt.yticks(np.arange(len(labels)), labels)
            plt.title(title)
            plt.tight_layout()
            
            # Save the matplotlib version
            png_file = os.path.splitext(output_file)[0] + ".png"
            plt.savefig(png_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Static heatmap PNG saved to: {png_file} (matplotlib fallback)")
    except Exception as e:
        print(f"Warning: Could not save static PNG for heatmap: {e}")
    
    return fig


def plot_cross_distance_scatter(df, output_file, title="Distance Correlation", color_by=None, 
                              x_dist='CosineDistance', y_dist='JaccardDistance', plot_type='scatter'):
    """
    Creates a scatter plot comparing different distance metrics (e.g., Cosine vs Jaccard)
    or a bar chart for single distance comparison.
    
    Args:
        df (pd.DataFrame): DataFrame with distance data and metadata
        output_file (str): Path to save the HTML file
        title (str): Title for the plot
        color_by (str, optional): Column name for coloring points
        x_dist (str): Column name for x-axis distance metric
        y_dist (str, optional): Column name for y-axis distance metric (None for bar chart)
        plot_type (str): 'scatter' or 'bar' for type of visualization
    
    Returns:
        plotly.graph_objects.Figure: The generated figure
    """
    if df.empty:
        print("Warning: Empty DataFrame, cannot create cross-distance plot")
        return go.Figure()
    
    # Validate inputs
    if x_dist not in df.columns:
        print(f"Warning: {x_dist} column not found in DataFrame")
        return go.Figure()
    
    if plot_type == 'scatter' and y_dist is not None and y_dist not in df.columns:
        print(f"Warning: {y_dist} column not found in DataFrame")
        return go.Figure()
    
    # Create appropriate plot based on type
    fig = go.Figure()
    
    if plot_type == 'scatter' and y_dist is not None:
        # Scatter plot comparing two distance metrics
        if color_by and color_by in df.columns:
            # Use color for categorical data
            fig = px.scatter(
                df, x=x_dist, y=y_dist, 
                color=color_by,
                labels={
                    x_dist: f"Semantic Distance % ({x_dist.replace('Distance', '')})",
                    y_dist: f"Semantic Distance % ({y_dist.replace('Distance', '')})"
                },
                title=title,
                hover_data=['Noun', 'Language1', 'Language2', 'NounCategory'] if all(col in df.columns for col in ['Noun', 'Language1', 'Language2', 'NounCategory']) else None
            )
            
            # Add correlation trendline and annotation
            try:
                from scipy import stats
                x_values = df[x_dist].dropna()
                y_values = df[y_dist].dropna()
                
                # Filter for rows where both metrics are available
                mask = df[x_dist].notna() & df[y_dist].notna()
                if mask.sum() >= 2:  # Need at least 2 points for correlation
                    x_values = df.loc[mask, x_dist]
                    y_values = df.loc[mask, y_dist]
                    
                    r_value, p_value = stats.pearsonr(x_values, y_values)
                    
                    # Add regression line
                    slope, intercept, _, _, _ = stats.linregress(x_values, y_values)
                    x_range = np.linspace(x_values.min(), x_values.max(), 100)
                    y_range = slope * x_range + intercept
                    
                    fig.add_trace(go.Scatter(
                        x=x_range, y=y_range, 
                        mode='lines', 
                        line=dict(color='red', dash='dash'),
                        name=f'r = {r_value:.3f}, p = {p_value:.3f}'
                    ))
                    
                    # Add text annotation with correlation statistics
                    fig.add_annotation(
                        x=0.95, y=0.05,
                        xref="paper", yref="paper",
                        text=f"Pearson r = {r_value:.3f}<br>p-value = {p_value:.3f}<br>n = {len(x_values)}",
                        showarrow=False,
                        align="right",
                        bgcolor="white",
                        bordercolor="black",
                        borderwidth=1
                    )
            except Exception as e:
                print(f"Warning: Could not calculate correlation: {e}")
        
        else:
            # Basic scatter plot without color
            fig = px.scatter(
                df, x=x_dist, y=y_dist,
                labels={
                    x_dist: f"Semantic Distance % ({x_dist.replace('Distance', '')})",
                    y_dist: f"Semantic Distance % ({y_dist.replace('Distance', '')})"
                },
                title=title
            )
        
        # Update hover template to show percentages
        fig.update_traces(
            hovertemplate='%{x:.2f} (%{x:.1%}), %{y:.2f} (%{y:.1%})<extra></extra>'
        )
    
    elif plot_type == 'bar':
        # Bar chart for single distance metric comparison
        # If y_dist is None, we make a bar chart of x_dist values
        
        # Group data as needed for the bar chart
        if color_by and color_by in df.columns:
            # For language pairs or other categorical comparisons
            grouped_df = df.groupby(color_by)[x_dist].agg(['mean', 'std']).reset_index()
            
            # Create bar chart with error bars
            fig = px.bar(
                grouped_df,
                x=color_by,
                y='mean',
                error_y='std',
                labels={
                    'mean': f'Semantic Distance % ({x_dist.replace("Distance", "")})',
                    color_by: color_by
                },
                title=title
            )
            
            # Update hover template to show percentages
            fig.update_traces(
                hovertemplate='%{y:.2f} (%{y:.1%})<extra></extra>'
            )
            
            # Add individual data points as a strip/swarm plot overlay
            if len(df) <= 100:  # Only add points for smaller datasets
                fig.add_trace(go.Box(
                    x=df[color_by],
                    y=df[x_dist],
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=0,
                    marker=dict(color='black', size=4),
                    line=dict(color='rgba(0,0,0,0)'),
                    fillcolor='rgba(0,0,0,0)',
                    name='Individual Points',
                    hovertemplate='%{y:.2f} (%{y:.1%})<extra></extra>'
                ))
        else:
            # If no color_by column, create a simple histogram
            fig = px.histogram(
                df, x=x_dist,
                title=title,
                labels={x_dist: f"Semantic Distance % ({x_dist.replace('Distance', '')})"} 
            )
            
            # No need to update hover template for histogram
    
    # Update layout
    fig.update_layout(
        width=900, 
        height=700,
        margin=dict(t=50, l=50, r=50, b=50),
        legend_title_text=color_by if color_by else '',
        template='plotly_white'
    )
    
    # Save the result
    try:
        fig.write_html(output_file)
        print(f"Cross-distance plot saved to: {output_file}")
    except Exception as e:
        print(f"Error saving plot to {output_file}: {e}")
    
    return fig


def plot_noun_category_language_matrix(comprehensive_df, output_dir, base_filename, api_model, proficiency_level=None, language_family_map=None):
    """
    Generates scatter plots where languages are on x and y axes and noun categories are plotted as points.
    
    Args:
        comprehensive_df (pd.DataFrame): DataFrame with language pair data
        output_dir (str): Directory to save the plots
        base_filename (str): Base name for output files
        api_model (str): Name of the API model
        proficiency_level (str, optional): Proficiency level if applicable
        language_family_map (dict, optional): Dictionary mapping languages to language families
    """
    print("Generating noun category language matrix plots...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter out any rows with missing data
    df = comprehensive_df.dropna(subset=['Language1', 'Language2', 'CosineDistance', 'NounCategory'])
    
    if len(df) == 0:
        print("No valid data for noun category language matrix plots")
        return
    
    # Get unique languages and noun categories
    languages = sorted(set(df['Language1'].unique()) | set(df['Language2'].unique()))
    noun_categories = sorted(df['NounCategory'].unique())
    
    if len(languages) < 2:
        print("Need at least 2 languages for language matrix plots")
        return
        
    print(f"Found {len(languages)} languages and {len(noun_categories)} noun categories")
    
    # Create a 2D matrix of all language pairs
    language_pairs = []
    for lang1 in languages:
        for lang2 in languages:
            if lang1 != lang2:  # Skip self-comparison
                language_pairs.append((lang1, lang2))
    
    # Set up proficiency suffix for filenames
    prof_suffix = f"_{proficiency_level}" if proficiency_level else ""
    
    # 1. Create plots showing all categories between each language pair
    for lang1, lang2 in language_pairs:
        # Get data for this language pair
        pair_data = df[((df['Language1'] == lang1) & (df['Language2'] == lang2)) | 
                       ((df['Language1'] == lang2) & (df['Language2'] == lang1))].copy()
        
        if len(pair_data) == 0:
            continue
            
        # Standardize so lang1 is always Language1 for consistent plotting
        pair_data.loc[pair_data['Language1'] == lang2, ['Language1', 'Language2']] = \
            pair_data.loc[pair_data['Language1'] == lang2, ['Language2', 'Language1']].values
        
        # Calculate mean distance per category
        category_distances = pair_data.groupby('NounCategory')['CosineDistance'].mean().reset_index()
        
        if len(category_distances) == 0:
            continue
            
        # Create interactive plot
        try:
            # Create x-positions for each category (spread the points horizontally)
            category_distances['x_position'] = np.linspace(0.1, 0.9, len(category_distances))
            
            fig = px.scatter(
                category_distances, 
                x='x_position',  # Now use the proper position values
                y='CosineDistance',
                color='NounCategory',
                text='NounCategory',  # Add category labels to the points
                title=f"Noun Category Distances Between {lang1} and {lang2}",
                labels={'CosineDistance': 'Semantic Distance %', 'x_position': 'Category Position'},
                hover_data={'NounCategory': True, 'CosineDistance': ':.4f'}
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title=f"Noun Categories ({lang1} - {lang2})",
                yaxis_title="Semantic Distance %",
                width=800, height=600,
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis=dict(showticklabels=False)  # Hide x-axis labels since we use the colors and hover
            )
            
            # Add dropdown filter for categories
            if len(category_distances) > 1:
                buttons = []
                # First button shows all categories
                buttons.append(dict(label='All Categories', method='update', args=[{'visible': [True]}]))
                
                # Add button for each category
                for category in category_distances['NounCategory'].unique():
                    # Create a visibility array that only shows this category
                    category_df = category_distances[category_distances['NounCategory'] == category]
                    fig_data = go.Scatter(
                        x=category_df['x_position'],
                        y=category_df['CosineDistance'],
                        mode='markers+text',
                        marker=dict(color=px.colors.qualitative.Plotly[category_distances['NounCategory'].unique().tolist().index(category) % len(px.colors.qualitative.Plotly)]),
                        text=category_df['NounCategory'],
                        name=category
                    )
                    buttons.append(dict(
                        label=category,
                        method='update',
                        args=[{'visible': [i == category_distances['NounCategory'].unique().tolist().index(category) + 1 for i in range(len(category_distances['NounCategory'].unique()))]}]
                    ))
                
                # Add the dropdown menu to the layout
                fig.update_layout(
                    updatemenus=[dict(
                        active=0,
                        buttons=buttons,
                        x=1.02,
                        y=1,
                        xanchor='left',
                        yanchor='top',
                        title="Filter by Category"
                    )]
                )
            
            # Update hover template to show percentages
            fig.update_traces(
                hovertemplate='%{y:.2f} (%{y:.1%})<extra></extra>',
                textposition='top center',
                textfont=dict(size=10)
            )
            
            # Save plot with sanitized language names
            plot_filename = f"{base_filename}{prof_suffix}_{sanitize_name(lang1)}_{sanitize_name(lang2)}_category_matrix.html"
            plot_path = os.path.join(output_dir, plot_filename)
            fig.write_html(plot_path)
            print(f"Saved language pair noun category plot to: {plot_path}")
            
            # Generate static PNG version
            try:
                # Try with kaleido if available
                try:
                    import kaleido
                    png_path = os.path.splitext(plot_path)[0] + ".png"
                    fig.write_image(png_path, width=800, height=600, scale=2)
                    print(f"Static PNG saved to: {png_path}")
                except ImportError:
                    # Fallback to matplotlib
                    import matplotlib.pyplot as plt
                    
                    # Create a scatter plot with matplotlib - one point per category
                    plt.figure(figsize=(10, 8))
                    
                    # Get unique categories for consistent colors
                    categories = category_distances['NounCategory'].unique()
                    colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
                    color_map = dict(zip(categories, colors))
                    
                    # Plot each category as a point with labels
                    for i, (_, row) in enumerate(category_distances.iterrows()):
                        cat = row['NounCategory']
                        x_pos = i / (len(category_distances) - 1 if len(category_distances) > 1 else 1)  # Spread points
                        plt.scatter(x_pos, row['CosineDistance'], color=color_map[cat], s=100, label=cat)
                        plt.text(x_pos, row['CosineDistance'] + 0.01, cat, ha='center', fontsize=9)
                    
                    # Add a title and labels
                    plt.title(f"Noun Category Distances Between {lang1} and {lang2}")
                    plt.xlabel(f"Noun Categories ({lang1} - {lang2})")
                    plt.ylabel("Semantic Distance %")
                    
                    # Remove x-ticks as we have text labels
                    plt.xticks([])
                    
                    # Add a legend with unique categories (if not too many)
                    if len(categories) <= 10:
                        handles, labels = plt.gca().get_legend_handles_labels()
                        by_label = dict(zip(labels, handles))
                        plt.legend(by_label.values(), by_label.keys(), 
                                  loc='best', bbox_to_anchor=(1.05, 1))
                    
                    plt.tight_layout()
                    
                    # Save the matplotlib version
                    png_path = os.path.splitext(plot_path)[0] + ".png"
                    plt.savefig(png_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Static PNG saved to: {png_path} (matplotlib fallback)")
            except Exception as e:
                print(f"Warning: Could not save static PNG for category matrix: {e}")
        except Exception as e:
            print(f"Error creating noun category plot for {lang1}-{lang2}: {e}")
    
    # 2. Create plots showing noun distribution by language family
    if language_family_map:
        print("Generating language family noun category matrix plots...")
        
        # Map languages to families
        df['Family1'] = df['Language1'].map(language_family_map)
        df['Family2'] = df['Language2'].map(language_family_map)
        
        # Filter out rows with missing family mapping
        df = df.dropna(subset=['Family1', 'Family2'])
        
        # Get unique families
        families = sorted(set(df['Family1'].unique()) | set(df['Family2'].unique()))
        
        if len(families) < 2:
            print("Need at least 2 language families for family matrix plots")
            return
            
        print(f"Found {len(families)} language families")
        
        # Create plots for each family pair
        for fam1 in families:
            for fam2 in families:
                if fam1 >= fam2:  # Skip self-comparison and duplicate pairs (we want each pair only once)
                    continue
                
                # Get data for this family pair (both directions)
                family_data = df[((df['Family1'] == fam1) & (df['Family2'] == fam2)) | 
                                 ((df['Family1'] == fam2) & (df['Family2'] == fam1))].copy()
                
                if len(family_data) == 0:
                    continue
                    
                # Standardize so fam1 is always Family1 for consistent plotting
                family_data.loc[family_data['Family1'] == fam2, ['Family1', 'Family2', 'Language1', 'Language2']] = \
                    family_data.loc[family_data['Family1'] == fam2, ['Family2', 'Family1', 'Language2', 'Language1']].values
                
                # Calculate mean distance per category
                category_family_distances = family_data.groupby('NounCategory')['CosineDistance'].mean().reset_index()
                
                if len(category_family_distances) == 0:
                    continue
                    
                # Create interactive plot
                try:
                    # Create x-positions for each category (spread the points horizontally)
                    category_family_distances['x_position'] = np.linspace(0.1, 0.9, len(category_family_distances))
                    
                    fig = px.scatter(
                        category_family_distances, 
                        x='x_position',  # Now use the proper position values
                        y='CosineDistance',
                        color='NounCategory',
                        text='NounCategory',  # Add category labels to the points
                        title=f"Noun Category Distances Between {fam1} and {fam2} Language Families",
                        labels={'CosineDistance': 'Semantic Distance %', 'x_position': 'Category Position'},
                        hover_data={'NounCategory': True, 'CosineDistance': ':.4f'}
                    )
                    
                    # Update layout
                    fig.update_layout(
                        xaxis_title=f"Noun Categories ({fam1} - {fam2})",
                        yaxis_title="Semantic Distance %",
                        width=800, height=600,
                        margin=dict(l=20, r=20, t=40, b=20),
                        xaxis=dict(showticklabels=False)  # Hide x-axis labels since we use the colors and hover
                    )
                    
                    # Add dropdown filter for categories
                    if len(category_family_distances) > 1:
                        buttons = []
                        # First button shows all categories
                        buttons.append(dict(label='All Categories', method='update', args=[{'visible': [True]}]))
                        
                        # Add button for each category
                        for category in category_family_distances['NounCategory'].unique():
                            # Create a visibility array that only shows this category
                            category_df = category_family_distances[category_family_distances['NounCategory'] == category]
                            fig_data = go.Scatter(
                                x=category_df['x_position'],
                                y=category_df['CosineDistance'],
                                mode='markers+text',
                                marker=dict(color=px.colors.qualitative.Plotly[category_family_distances['NounCategory'].unique().tolist().index(category) % len(px.colors.qualitative.Plotly)]),
                                text=category_df['NounCategory'],
                                name=category
                            )
                            buttons.append(dict(
                                label=category,
                                method='update',
                                args=[{'visible': [i == category_family_distances['NounCategory'].unique().tolist().index(category) + 1 for i in range(len(category_family_distances['NounCategory'].unique()))]}]
                            ))
                        
                        # Add the dropdown menu to the layout
                        fig.update_layout(
                            updatemenus=[dict(
                                active=0,
                                buttons=buttons,
                                x=1.02,
                                y=1,
                                xanchor='left',
                                yanchor='top',
                                title="Filter by Category"
                            )]
                        )
                    
                    # Update hover template to show percentages
                    fig.update_traces(
                        hovertemplate='%{y:.2f} (%{y:.1%})<extra></extra>',
                        textposition='top center',
                        textfont=dict(size=10)
                    )
                    
                    # Save plot with sanitized family names
                    plot_filename = f"{base_filename}{prof_suffix}_family_{sanitize_name(fam1)}_{sanitize_name(fam2)}_category_matrix.html"
                    plot_path = os.path.join(output_dir, plot_filename)
                    fig.write_html(plot_path)
                    print(f"Saved language family noun category plot to: {plot_path}")
                    
                    # Generate static PNG version
                    try:
                        # Try with kaleido if available
                        try:
                            import kaleido
                            png_path = os.path.splitext(plot_path)[0] + ".png"
                            fig.write_image(png_path, width=800, height=600, scale=2)
                            print(f"Static PNG saved to: {png_path}")
                        except ImportError:
                            # Fallback to matplotlib
                            import matplotlib.pyplot as plt
                            
                            # Create a scatter plot with matplotlib - one point per category
                            plt.figure(figsize=(10, 8))
                            
                            # Get unique categories for consistent colors
                            categories = category_family_distances['NounCategory'].unique()
                            colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
                            color_map = dict(zip(categories, colors))
                            
                            # Plot each category as a point with labels
                            for i, (_, row) in enumerate(category_family_distances.iterrows()):
                                cat = row['NounCategory']
                                x_pos = i / (len(category_family_distances) - 1 if len(category_family_distances) > 1 else 1)  # Spread points
                                plt.scatter(x_pos, row['CosineDistance'], color=color_map[cat], s=100, label=cat)
                                plt.text(x_pos, row['CosineDistance'] + 0.01, cat, ha='center', fontsize=9)
                            
                            # Add a title and labels
                            plt.title(f"Noun Category Distances Between {fam1} and {fam2} Language Families")
                            plt.xlabel(f"Noun Categories ({fam1} - {fam2})")
                            plt.ylabel("Semantic Distance %")
                            
                            # Remove x-ticks as we have text labels
                            plt.xticks([])
                            
                            # Add a legend with unique categories (if not too many)
                            if len(categories) <= 10:
                                handles, labels = plt.gca().get_legend_handles_labels()
                                by_label = dict(zip(labels, handles))
                                plt.legend(by_label.values(), by_label.keys(), 
                                          loc='best', bbox_to_anchor=(1.05, 1))
                            
                            plt.tight_layout()
                            
                            # Save the matplotlib version
                            png_path = os.path.splitext(plot_path)[0] + ".png"
                            plt.savefig(png_path, dpi=300, bbox_inches='tight')
                            plt.close()
                            print(f"Static PNG saved to: {png_path} (matplotlib fallback)")
                    except Exception as e:
                        print(f"Warning: Could not save static PNG for family category matrix: {e}")
                except Exception as e:
                    print(f"Error creating noun category plot for {fam1}-{fam2}: {e}")
    
    # 3. Create a matrix map of each noun category across different language pairs
    try:
        # Create a directory for category-specific plots
        category_dir = os.path.join(output_dir, "by_category")
        os.makedirs(category_dir, exist_ok=True)
        
        for category in noun_categories:
            # Get data for this category
            cat_data = df[df['NounCategory'] == category].copy()
            
            if len(cat_data) < len(languages):
                continue  # Not enough data for this category
                
            # Create a matrix of distances
            matrix_data = []
            for lang1 in languages:
                for lang2 in languages:
                    if lang1 != lang2:  # Skip self-comparison
                        # Get data for this language pair (both directions)
                        pair_data = cat_data[((cat_data['Language1'] == lang1) & (cat_data['Language2'] == lang2)) | 
                                           ((cat_data['Language1'] == lang2) & (cat_data['Language2'] == lang1))]
                        
                        if len(pair_data) > 0:
                            # Calculate mean distance
                            mean_dist = pair_data['CosineDistance'].mean()
                            matrix_data.append({
                                'Language1': lang1,
                                'Language2': lang2,
                                'Distance': mean_dist,
                                'NounCategory': category
                            })
            
            # Create DataFrame from matrix data
            if not matrix_data:
                continue
                
            matrix_df = pd.DataFrame(matrix_data)
            
            # Create heatmap
            try:
                # Create pivot table
                pivot_df = matrix_df.pivot(index='Language1', columns='Language2', values='Distance')
                
                # Create figure
                fig = px.imshow(
                    pivot_df,
                    labels=dict(x="Language", y="Language", color="Distance"),
                    x=pivot_df.columns.tolist(),
                    y=pivot_df.index.tolist(),
                    color_continuous_scale='Viridis',
                    title=f"Language Distance Matrix for Noun Category: {category}"
                )
                
                # Add dropdown filter for languages
                if len(pivot_df.columns) > 3:
                    buttons = []
                    # First button shows all languages
                    buttons.append(dict(label='All Languages', method='update', args=[{'visible': [True]}]))
                    
                    # Add button for each language
                    for language in pivot_df.columns:
                        # Create a custom data array that highlights only this language's row and column
                        language_idx = pivot_df.columns.get_loc(language)
                        z_filtered = np.zeros_like(pivot_df.values)
                        z_filtered[language_idx, :] = pivot_df.values[language_idx, :]
                        z_filtered[:, language_idx] = pivot_df.values[:, language_idx]
                        
                        buttons.append(dict(
                            label=language,
                            method='update',
                            args=[{
                                'z': [z_filtered],
                                'visible': [True]
                            }]
                        ))
                    
                    # Add the dropdown menu to the layout
                    fig.update_layout(
                        updatemenus=[dict(
                            active=0,
                            buttons=buttons,
                            x=1.02,
                            y=1,
                            xanchor='left',
                            yanchor='top',
                            title="Filter by Language"
                        )]
                    )
                
                # Update layout
                fig.update_layout(
                    width=900, height=700,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                
                # Save plot with sanitized category name
                plot_filename = f"{base_filename}{prof_suffix}_{sanitize_name(category)}_language_matrix.html"
                plot_path = os.path.join(category_dir, plot_filename)
                fig.write_html(plot_path)
                print(f"Saved language matrix plot for category {category} to: {plot_path}")
                
                # Generate static PNG version
                try:
                    # Try with kaleido if available
                    try:
                        import kaleido
                        png_path = os.path.splitext(plot_path)[0] + ".png"
                        fig.write_image(png_path, width=900, height=700, scale=2)
                        print(f"Static PNG saved to: {png_path}")
                    except ImportError:
                        # Fallback to matplotlib
                        import matplotlib.pyplot as plt
                        import seaborn as sns
                        
                        # Create a simplified heatmap version with matplotlib/seaborn
                        plt.figure(figsize=(12, 10))
                        
                        # Use pivot_df for the heatmap
                        try:
                            sns.heatmap(pivot_df, annot=False, cmap='viridis', 
                                       cbar_kws={'label': 'Distance'})
                        except ImportError:  # If seaborn isn't available
                            plt.imshow(pivot_df.values, cmap='viridis')
                            plt.colorbar(label='Distance')
                            plt.xticks(range(len(pivot_df.columns)), pivot_df.columns, rotation=90)
                            plt.yticks(range(len(pivot_df.index)), pivot_df.index)
                        
                        plt.title(f"Language Distance Matrix for Noun Category: {category}")
                        plt.tight_layout()
                        
                        # Save the matplotlib version
                        png_path = os.path.splitext(plot_path)[0] + ".png"
                        plt.savefig(png_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"Static PNG saved to: {png_path} (matplotlib fallback)")
                except Exception as e:
                    print(f"Warning: Could not save static PNG for language matrix: {e}")
            except Exception as e:
                print(f"Error creating language matrix plot for category {category}: {e}")
    except Exception as e:
        print(f"Error generating category-specific language matrix plots: {e}")
        
    print("Completed noun category language matrix plots generation")


def plot_combined_family_language_pca(language_df, family_df, plot_filename, title="Combined Language and Family PCA", family_mapping=None, interactive=True):
    """
    Generates a combined PCA visualization showing both language families and the individual languages within them.
    This creates a hierarchical visualization where users can see language families as clusters,
    and zoom in to see individual languages within each family.
    
    Args:
        language_df (pd.DataFrame): DataFrame with language embeddings (must have 'Language' and 'Embedding' columns)
        family_df (pd.DataFrame): DataFrame with language family embeddings (must have 'LanguageFamily' and 'Embedding' columns)
        plot_filename (str): Path to save the plot
        title (str): Title for the plot
        family_mapping (dict, optional): Dictionary mapping languages to their families
        interactive (bool): Whether to generate an interactive Plotly plot (default: True)
        
    Returns:
        plotly.graph_objects.Figure: The generated figure object
    """
    if len(language_df) < 2 or len(family_df) < 1:
        print("Not enough data to plot combined family-language PCA.")
        return go.Figure()
    
    # Extract embeddings
    language_embeddings = np.vstack(language_df['Embedding'])
    family_embeddings = np.vstack(family_df['Embedding'])
    
    # Combine all embeddings for PCA
    all_embeddings = np.vstack([family_embeddings, language_embeddings])
    
    # Generate PCA projection
    pca = PCA(n_components=2, random_state=42)
    all_emb_2d = pca.fit_transform(all_embeddings)
    
    # Split the PCA results back into family and language parts
    family_emb_2d = all_emb_2d[:len(family_embeddings)]
    language_emb_2d = all_emb_2d[len(family_embeddings):]
    
    # Calculate explained variance for labeling
    explained_variance = pca.explained_variance_ratio_
    explained_variance_str = ", ".join([f"{var:.1%}" for var in explained_variance])
    
    # Create DataFrames for plotting
    if family_mapping is None:
        # If no family mapping provided but we have family info in language_df
        if 'LanguageFamily' in language_df.columns:
            family_mapping = dict(zip(language_df['Language'], language_df['LanguageFamily']))
        else:
            # Extract from language codes if available (assuming format like "EN # English")
            family_mapping = {}
            for _, row in language_df.iterrows():
                lang = row['Language']
                # Try to find matching family in family_df
                matching_families = [f for f in family_df['LanguageFamily'] if lang.startswith(f) or f in lang]
                if matching_families:
                    family_mapping[lang] = matching_families[0]
                else:
                    # Default to "Unknown" if no match
                    family_mapping[lang] = "Unknown"
    
    # Create plotting dataframes with all necessary information
    # Families dataframe
    families_plot_df = pd.DataFrame({
        'x': family_emb_2d[:, 0],
        'y': family_emb_2d[:, 1],
        'name': family_df['LanguageFamily'].tolist(),
        'type': ['Family'] * len(family_df),
        'size': [15] * len(family_df)  # Larger markers for families
    })
    
    # Languages dataframe
    languages_plot_df = pd.DataFrame({
        'x': language_emb_2d[:, 0],
        'y': language_emb_2d[:, 1],
        'name': language_df['Language'].tolist(),
        'type': ['Language'] * len(language_df),
        'size': [10] * len(language_df),  # Smaller markers for languages
        'family': [family_mapping.get(lang, 'Unknown') for lang in language_df['Language']]
    })
    
    # Combine the dataframes
    combined_df = pd.concat([families_plot_df, languages_plot_df], ignore_index=True)
    
    # Create the interactive plot
    fig = go.Figure()
    
    # Add family points (larger markers)
    family_colors = {}
    for i, family in enumerate(family_df['LanguageFamily']):
        # Assign a color to each family
        family_colors[family] = px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
        
        # Add family point
        family_row = families_plot_df[families_plot_df['name'] == family].iloc[0]
        fig.add_trace(go.Scatter(
            x=[family_row['x']],
            y=[family_row['y']],
            mode='markers',
            marker=dict(
                size=20,
                color=family_colors[family],
                line=dict(width=2, color='black')
            ),
            name=f"Family: {family}",
            legendgroup=family,
            hoverinfo='name'
        ))
    
    # Add language points (smaller markers)
    for family in sorted(languages_plot_df['family'].unique()):
        family_langs = languages_plot_df[languages_plot_df['family'] == family]
        
        # Use the same color as the family, but with transparency
        color = family_colors.get(family, 'grey')
        
        # Add language points for this family
        fig.add_trace(go.Scatter(
            x=family_langs['x'],
            y=family_langs['y'],
            mode='markers',
            marker=dict(
                size=10,
                color=color,
                opacity=0.7,
                line=dict(width=1, color='black')
            ),
            text=family_langs['name'],
            hoverinfo='text',
            name=f"Languages in {family}",
            legendgroup=family,
            showlegend=False
        ))
    
    # Update layout for the plot
    fig.update_layout(
        title=f"{title} (Explained Variance: {explained_variance_str})",
        xaxis_title=f"Semantic Distance % - PC1 ({explained_variance[0]:.1%})",
        yaxis_title=f"Semantic Distance % - PC2 ({explained_variance[1]:.1%})",
        width=900, height=700,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            groupclick="toggleitem"  # Allow toggling family groups
        )
    )
    
    # Save the figure
    html_filename = os.path.splitext(plot_filename)[0] + ".html"
    fig.write_html(html_filename)
    print(f"Combined family-language PCA saved to: {html_filename}")
    
    # Save static PNG version
    try:
        # Try with kaleido if available
        try:
            import kaleido
            png_filename = os.path.splitext(plot_filename)[0] + ".png"
            fig.write_image(png_filename, width=950, height=700, scale=2)
            print(f"Static combined family-language PCA PNG saved to: {png_filename}")
        except ImportError:
            # Fallback to matplotlib
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Create a simplified plot with matplotlib
            plt.figure(figsize=(12, 10))
            
            # Plot family points (larger markers)
            for family in family_df['LanguageFamily'].unique():
                family_data = family_emb_2d[family_df['LanguageFamily'] == family]
                plt.scatter(
                    family_data[:, 0], family_data[:, 1],
                    marker='o', s=200, label=f"Family: {family}", 
                    edgecolor='k', linewidth=2
                )
            
            # Plot language points (smaller markers)
            if family_mapping:
                for family in sorted(set(family_mapping.values())):
                    # Get languages in this family
                    family_langs = [lang for lang, fam in family_mapping.items() if fam == family]
                    
                    # Get their indices in the language dataframe
                    lang_indices = [i for i, lang in enumerate(language_df['Language']) if lang in family_langs]
                    
                    if lang_indices:
                        # Extract their embeddings
                        lang_data = language_emb_2d[lang_indices]
                        plt.scatter(
                            lang_data[:, 0], lang_data[:, 1],
                            marker='o', s=80, alpha=0.6, edgecolor='k', linewidth=1
                        )
            
            plt.title(f"{title} (Explained Variance: {explained_variance_str})")
            plt.xlabel(f"PC1 ({explained_variance[0]:.1%})")
            plt.ylabel(f"PC2 ({explained_variance[1]:.1%})")
            plt.legend()
            plt.tight_layout()
            
            # Save the matplotlib version
            png_filename = os.path.splitext(plot_filename)[0] + ".png"
            plt.savefig(png_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Static combined family-language PCA PNG saved to: {png_filename} (matplotlib fallback)")
    except Exception as e:
        print(f"Warning: Could not save static PNG for combined family-language PCA: {e}")
        
    return fig


def plot_average_noun_differences(comprehensive_df, output_file, title="Average Noun Differences by Category", 
                                interactive=True, distance_metric='CosineDistance', filter_language=None):
    """
    Creates a visualization showing the average differences between language pairs for each noun,
    grouped by noun category.
    
    Args:
        comprehensive_df (pd.DataFrame): DataFrame with comprehensive analysis data
        output_file (str): Path to save the output visualization
        title (str): Title for the visualization
        interactive (bool): Whether to generate an interactive Plotly visualization (True) or static Matplotlib (False)
        distance_metric (str): Which distance metric to use ('CosineDistance' or 'JaccardDistance')
        filter_language (str, optional): Filter to only show data for a specific language
        
    Returns:
        None
    """
    if comprehensive_df.empty:
        print("Error: Empty dataframe provided to plot_average_noun_differences")
        return
    
    # Verify required columns are present
    required_columns = ['NounCategory', 'Noun', distance_metric, 'Language1', 'Language2']
    if not all(col in comprehensive_df.columns for col in required_columns):
        print(f"Error: Missing required columns for average noun difference plot. Required: {required_columns}")
        print(f"Available columns: {comprehensive_df.columns.tolist()}")
        return
    
    # Filter data if language is specified
    if filter_language is not None:
        language_df = comprehensive_df[(comprehensive_df['Language1'] == filter_language) | 
                                     (comprehensive_df['Language2'] == filter_language)]
        
        if language_df.empty:
            print(f"Warning: No data found for language '{filter_language}' in average noun difference analysis")
            return
        
        print(f"Analyzing average noun differences for language: '{filter_language}' ({len(language_df)} rows)")
        comprehensive_df = language_df
    
    # Group by NounCategory and Noun to calculate average distances between language pairs
    try:
        # Calculate average distance between language pairs for each noun
        noun_avg_distances = comprehensive_df.groupby(['NounCategory', 'Noun'])[distance_metric].mean().reset_index()
        
        # Calculate overall category averages for reference
        category_avg_distances = noun_avg_distances.groupby('NounCategory')[distance_metric].mean().reset_index()
        
        # Sort by category and distance for better visualization
        noun_avg_distances = noun_avg_distances.sort_values(['NounCategory', distance_metric])
        category_avg_distances = category_avg_distances.sort_values(distance_metric, ascending=False)
        
        # Count nouns per category for reference
        noun_counts = noun_avg_distances.groupby('NounCategory')['Noun'].nunique().reset_index()
        noun_counts.columns = ['NounCategory', 'NounCount']
        
        # Merge count information into category averages
        category_avg_distances = pd.merge(category_avg_distances, noun_counts, on='NounCategory')
        
        # Create the visualization
        if interactive:
            # Create an interactive Plotly figure
            
            # Prepare data for plotting with category information
            plot_df = noun_avg_distances.copy()
            
            # First create the detailed noun-level plot
            fig = px.box(
                plot_df, 
                x='NounCategory', 
                y=distance_metric,
                color='NounCategory',
                title=title,
                labels={
                    'NounCategory': 'Noun Category',
                    distance_metric: f"Semantic Distance % ({distance_metric.replace('Distance', '')})"
                },
                points='all',  # Show all individual points
                hover_data=['Noun']  # Show noun names on hover
            )
            
            # Add dropdown filter for categories
            if 'NounCategory' in plot_df.columns and len(plot_df['NounCategory'].unique()) > 1:
                buttons = []
                # First button shows all categories
                buttons.append(dict(label='All Categories', method='update', args=[{'visible': [True] * len(fig.data)}]))
                
                # Add a button for each category
                for i, category in enumerate(sorted(plot_df['NounCategory'].unique())):
                    # Set visibility array - True for this category, False for others
                    vis = []
                    for j, trace_category in enumerate(sorted(plot_df['NounCategory'].unique())):
                        vis.append(trace_category == category)
                    
                    buttons.append(dict(
                        label=category,
                        method='update',
                        args=[{'visible': vis}]
                    ))
                
                # Add the dropdown menu to the layout
                fig.update_layout(
                    updatemenus=[dict(
                        active=0,
                        buttons=buttons,
                        x=1.02,
                        y=1,
                        xanchor='left',
                        yanchor='top',
                        title="Filter by Category"
                    )]
                )
            
            # Update hover template to show percentages
            fig.update_traces(
                hovertemplate='%{y:.3f} (%{y:.1%})<br>Noun: %{customdata[0]}<extra>%{x}</extra>'
            )
            
            # Add category average lines
            for idx, row in category_avg_distances.iterrows():
                category = row['NounCategory']
                avg_value = row[distance_metric]
                
                # Add a horizontal line for the category average
                fig.add_shape(
                    type="line",
                    x0=idx-0.4,  # Adjust these values based on your visualization
                    x1=idx+0.4,
                    y0=avg_value,
                    y1=avg_value,
                    line=dict(color="black", width=2, dash="dash"),
                    xref="x",
                    yref="y"
                )
                
                # Add annotation with average value and count at the BOTTOM of the chart
                fig.add_annotation(
                    x=idx,
                    y=avg_value,
                    text=f"Avg: {avg_value:.3f} (n={row['NounCount']})",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    ax=0,
                    ay=30,  # Positive value to move downward
                    yanchor="top"  # Anchor text at top so it goes below the line
                )
            
            # Update layout for better readability
            fig.update_layout(
                xaxis_title="Noun Category",
                yaxis_title=f"Semantic Distance % ({distance_metric.replace('Distance', '')})",
                width=900,
                height=600,
                showlegend=True
            )
            
            # Ensure directory exists
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # Save as interactive HTML
            fig.write_html(output_file)
            print(f"Interactive average noun differences plot saved to: {output_file}")
            
            # Also save a static PNG version
            png_file = os.path.splitext(output_file)[0] + '.png'
            fig.write_image(png_file, width=900, height=600, scale=2)
            print(f"Static average noun differences plot saved to: {png_file}")
            
            return fig
            
        else:
            # Create a static matplotlib figure
            plt.figure(figsize=(12, 8))
            
            # Get unique categories and set up colors
            categories = category_avg_distances['NounCategory'].tolist()
            colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
            
            # Map categories to colors
            category_colors = {cat: colors[i] for i, cat in enumerate(categories)}
            
            # Plot individual noun points by category
            for i, category in enumerate(categories):
                category_data = noun_avg_distances[noun_avg_distances['NounCategory'] == category]
                plt.scatter(
                    [i] * len(category_data),  # x-coordinates (jittered)
                    category_data[distance_metric],
                    color=category_colors[category],
                    alpha=0.7,
                    label=category if i == 0 else ""  # Only add to legend once
                )
                
                # Add category average line
                cat_avg = category_avg_distances[category_avg_distances['NounCategory'] == category][distance_metric].values[0]
                cat_count = category_avg_distances[category_avg_distances['NounCategory'] == category]['NounCount'].values[0]
                plt.plot(
                    [i-0.3, i+0.3],
                    [cat_avg, cat_avg],
                    color='black',
                    linestyle='--'
                )
                
                # Add text with average and count AT THE BOTTOM
                plt.text(
                    i,
                    cat_avg,
                    f"Avg: {cat_avg:.3f}\n(n={cat_count})",
                    horizontalalignment='center',
                    verticalalignment='top',  # Changed to 'top' so text goes below
                    fontsize=9
                )
            
            # Customize the plot
            plt.xlabel('Noun Category')
            plt.ylabel(f"Semantic Distance % ({distance_metric.replace('Distance', '')})")
            plt.title(title)
            plt.xticks(range(len(categories)), categories, rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save the figure
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Average noun differences plot saved to: {output_file}")
            
    except Exception as e:
        print(f"Error generating average noun differences plot: {e}")
        import traceback
        traceback.print_exc()


def plot_pca(df, plot_filename="pca_visualization.png", title="PCA Visualization", color_by=None, interactive=False, dimensions=2):
    """
    Generates a PCA plot of embeddings and returns the figure.
    PCA is simpler and faster than t-SNE/UMAP but may not capture non-linear relationships.
    
    Args:
        df (pd.DataFrame): DataFrame with embeddings and metadata (must have 'Embedding' and 'Label' columns)
        plot_filename (str): Path to save the plot image
        title (str): Title for the plot
        color_by (str, optional): Column name for coloring points (e.g., 'Language', 'NounCategory')
        interactive (bool): Whether to generate an interactive Plotly plot
        dimensions (int): Number of dimensions for the PCA projection (2 or 3)
        
    Returns:
        matplotlib.figure.Figure or plotly.graph_objects.Figure: The generated figure
    """
    if dimensions not in [2, 3]:
        print(f"Warning: dimensions must be 2 or 3. Defaulting to 2.")
        dimensions = 2
        
    if len(df) < 2:
        print("Not enough data to plot.")
        return plt.figure() if not interactive else go.Figure()
        
    embeddings = np.vstack(df['Embedding'])
    labels = df['Label'].tolist()
    
    if embeddings.shape[0] < 2:
        print("Not enough rows for PCA.")
        return plt.figure() if not interactive else go.Figure()
    
    # Generate PCA projection
    # Use appropriate n_components based on requested dimensions
    n_components = dimensions  # Number of components to keep
    pca = PCA(n_components=n_components)
    emb_reduced = pca.fit_transform(embeddings)
    explained_variance = pca.explained_variance_ratio_
    explained_variance_str = ", ".join([f"PC{i+1}: {var:.1%}" for i, var in enumerate(explained_variance)])
    print(f"PCA explained variance: {explained_variance_str}")
    
    # Extract color information if specified
    if color_by and color_by in df.columns:
        # Get the categorical values
        categorical_values = df[color_by].tolist()
        
        # For categorical string values, map to integers for coloring
        if categorical_values and isinstance(categorical_values[0], str):
            # Get unique categories and map to integers
            unique_categories = sorted(set(categorical_values))
            category_to_int = {cat: i for i, cat in enumerate(unique_categories)}
            color_values = [category_to_int[cat] for cat in categorical_values]
            
            # Save the mapping for the legend
            category_mapping = category_to_int
        else:
            # For numeric values, use directly
            color_values = categorical_values
            category_mapping = None
    else:
        color_values = np.zeros(len(labels))
        category_mapping = None
    
    # Determine if this is a language-level or language-family visualization that needs labels
    is_language_level_plot = "Language-Level" in title or "Language Family" in title
    
    if interactive:
        # Generate an interactive Plotly figure
        try:
            if color_by:
                # Prepare data for Plotly with all the information
                if dimensions == 2:
                    plot_df = pd.DataFrame({
                        'x': emb_reduced[:, 0],
                        'y': emb_reduced[:, 1],
                        'label': labels,
                        'color_category': df[color_by].tolist() if color_by else None,
                        'color_value': color_values
                    })
                else:  # 3D
                    plot_df = pd.DataFrame({
                        'x': emb_reduced[:, 0],
                        'y': emb_reduced[:, 1],
                        'z': emb_reduced[:, 2],
                        'label': labels,
                        'color_category': df[color_by].tolist() if color_by else None,
                        'color_value': color_values
                    })
                
                # Add all available columns from the original dataframe for tooltips
                for col in df.columns:
                    if col not in ['Embedding'] and col not in plot_df.columns:
                        plot_df[col] = df[col].tolist()
                
                # Generate interactive scatter plot
                if dimensions == 2:
                    fig = px.scatter(
                        plot_df, x='x', y='y', color='color_category',
                        hover_data={col: True for col in plot_df.columns if col not in ['x', 'y', 'color_value', 'color_category']},
                        title=f"{title} (Explained Variance: {explained_variance_str})"
                    )
                    
                    # Add text labels for language-level or language-family plots
                    if is_language_level_plot:
                        fig.add_trace(go.Scatter(
                            x=plot_df['x'],
                            y=plot_df['y'],
                            mode='text',
                            text=labels,
                            textposition='top center',
                            textfont=dict(size=9, color='black'),
                            showlegend=False
                        ))
                else:  # 3D
                    fig = px.scatter_3d(
                        plot_df, x='x', y='y', z='z', color='color_category',
                        hover_data={col: True for col in plot_df.columns if col not in ['x', 'y', 'z', 'color_value', 'color_category']},
                        title=f"{title} (Explained Variance: {explained_variance_str})"
                    )
            else:
                if dimensions == 2:
                    fig = px.scatter(
                        x=emb_reduced[:, 0], y=emb_reduced[:, 1],
                        hover_name=labels,
                        title=f"{title} (Explained Variance: {explained_variance_str})"
                    )
                    
                    # Add text labels for language-level or language-family plots
                    if is_language_level_plot:
                        fig.add_trace(go.Scatter(
                            x=emb_reduced[:, 0],
                            y=emb_reduced[:, 1],
                            mode='text',
                            text=labels,
                            textposition='top center',
                            textfont=dict(size=9, color='black'),
                            showlegend=False
                        ))
                else:  # 3D
                    fig = px.scatter_3d(
                        x=emb_reduced[:, 0], y=emb_reduced[:, 1], z=emb_reduced[:, 2],
                        hover_name=labels,
                        title=f"{title} (Explained Variance: {explained_variance_str})"
                    )
            
            # Update layout
            if dimensions == 2:
                fig.update_layout(
                    xaxis_title=f"Semantic Distance % - PC1 ({explained_variance[0]:.1%})",
                    yaxis_title=f"Semantic Distance % - PC2 ({explained_variance[1]:.1%})",
                    legend_title=color_by if color_by else "",
                    width=900, height=700,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                
                # Add dropdown filter for 2D plots when coloring by categories
                if color_by and color_by in df.columns:
                    unique_categories = sorted(df[color_by].unique())
                    if len(unique_categories) > 1:
                        buttons = []
                        # First button shows all traces
                        buttons.append(dict(label='All', method='update', args=[{'visible': [True]*len(fig.data)}]))
                        
                        # Create a button for each category
                        for category in unique_categories:
                            # Set visibility array - True for traces that match this category, False for others
                            vis = []
                            for trace in fig.data:
                                # Check if this is a labels trace (which should remain visible)
                                if trace.mode == 'text' and not trace.showlegend:
                                    vis.append(True)
                                else:
                                    # For data traces, show only if they match the category
                                    vis.append(str(trace.name) == str(category))
                            buttons.append(dict(label=str(category), method='update', args=[{'visible': vis}]))
                        
                        # Add the dropdown menu to the layout
                        fig.update_layout(
                            updatemenus=[dict(
                                active=0,
                                buttons=buttons,
                                x=1.02,
                                y=1,
                                xanchor='left',
                                yanchor='top',
                                title=f"Filter by {color_by}"
                            )]
                        )
            else:  # 3D
                fig.update_layout(
                    scene=dict(
                        xaxis_title=f"Semantic Distance % - PC1 ({explained_variance[0]:.1%})",
                        yaxis_title=f"Semantic Distance % - PC2 ({explained_variance[1]:.1%})",
                        zaxis_title=f"Semantic Distance % - PC3 ({explained_variance[2]:.1%})",
                        # Allow unlimited zoom with no constraints
                        camera=dict(
                            eye=dict(x=1.5, y=1.5, z=1.5),  # Default camera position
                            projection=dict(type="perspective")
                        ),
                        dragmode="turntable"
                    ),
                    legend_title=color_by if color_by else "",
                    width=1000, height=800,
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                
                # Configure additional options for interactive HTML output
                if plot_filename.endswith('.html'):
                    config = {
                        'scrollZoom': True,  # Enable scroll wheel zoom
                        'displayModeBar': True,  # Always show the mode bar
                        'modeBarButtonsToAdd': ['resetCameraDefault3d'],  # Add reset camera button
                        'showAxisDragHandles': True
                    }
                else:
                    config = {'staticPlot': False}
                
                # Add dropdown filter for 3D plots when coloring by categories
                if color_by and color_by in df.columns:
                    unique_categories = sorted(df[color_by].unique())
                    if len(unique_categories) > 1:
                        buttons = []
                        # First button shows all traces
                        buttons.append(dict(label='All', method='update', args=[{'visible': [True]*len(fig.data)}]))
                        
                        # Create a button for each category
                        for category in unique_categories:
                            # Set visibility array - True for traces that match this category, False for others
                            vis = []
                            for trace in fig.data:
                                # For data traces, show only if they match the category
                                vis.append(str(trace.name) == str(category))
                            buttons.append(dict(label=str(category), method='update', args=[{'visible': vis}]))
                        
                        # Add the dropdown menu to the layout
                        fig.update_layout(
                            updatemenus=[dict(
                                active=0,
                                buttons=buttons,
                                x=1.02,
                                y=1,
                                xanchor='left',
                                yanchor='top',
                                title=f"Filter by {color_by}"
                            )]
                        )
            
            # Ensure directory exists
            output_dir = os.path.dirname(plot_filename)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                
            # Save as HTML for interactivity
            html_filename = plot_filename
            if not html_filename.endswith('.html'):
                html_filename = os.path.splitext(plot_filename)[0] + ".html"
                
            fig.write_html(html_filename)
            print(f"Interactive {dimensions}D PCA plot saved to: {html_filename}")
            
            # Also save as PNG for static use
            try:
                png_filename = os.path.splitext(html_filename)[0] + ".png"
                
                # Check if kaleido is installed for image export
                try:
                    import kaleido
                    kaleido_available = True
                except ImportError:
                    kaleido_available = False
                
                if kaleido_available:
                    fig.write_image(png_filename, scale=2)
                    print(f"Static {dimensions}D PCA plot also saved to: {png_filename}")
                else:
                    print(f"Warning: Could not save static PNG image with Plotly - kaleido package is required")
                    print(f"To install kaleido run: pip install kaleido")
                    print(f"Interactive HTML version is still available at: {html_filename}")
                    
                    # Always generate a fallback static visualization with matplotlib
                    fallback_png_plot(df, emb_reduced, png_filename, title, color_by, labels, 
                                      category_mapping, is_language_level_plot, dimensions, explained_variance)
            except Exception as img_err:
                print(f"Warning: Could not save static PNG image: {img_err}")
                print(f"Interactive HTML version is still available")
            
            return fig
            
        except Exception as plotly_err:
            print(f"Error generating interactive plot: {plotly_err}")
            # Fall back to matplotlib
            interactive = False
    
    # Use Matplotlib for static plots if interactive failed or not requested
    if not interactive:
        # Generate a static Matplotlib plot
        if dimensions == 2:
            fig_mpl = plt.figure(figsize=(10, 8))
            scatter = plt.scatter(emb_reduced[:, 0], emb_reduced[:, 1], c=color_values, cmap="tab10", alpha=0.7, edgecolors='k')
            
            plt.title(f"{title} (Explained Variance: {explained_variance_str})")
            plt.xlabel(f"Semantic Distance % - PC1 ({explained_variance[0]:.1%})")
            plt.ylabel(f"Semantic Distance % - PC2 ({explained_variance[1]:.1%})")
            
            # Add text labels for language-level or language-family plots
            if is_language_level_plot:
                for i, label in enumerate(labels):
                    plt.annotate(
                        label,
                        (emb_reduced[i, 0], emb_reduced[i, 1]),
                        textcoords="offset points",
                        xytext=(0, 5),
                        ha='center',
                        fontsize=8
                    )
            
            # Add legend for categorical data
            if category_mapping:
                handles = []
                labels_legend = []
                for cat, idx in category_mapping.items():
                    handle = plt.Line2D(
                        [0], [0], marker='o', color='w', 
                        markerfacecolor=plt.cm.tab10(idx/len(category_mapping)), 
                        markersize=10
                    )
                    handles.append(handle)
                    labels_legend.append(cat)
                plt.legend(handles, labels_legend, title=color_by, loc='best', bbox_to_anchor=(1.05, 1))
            # Add colorbar for numeric data
            elif color_by and isinstance(color_values[0], (int, float, np.number)):
                plt.colorbar(scatter, label=color_by)
        else:  # 3D
            fig_mpl = plt.figure(figsize=(10, 8))
            ax = fig_mpl.add_subplot(111, projection='3d')
            scatter = ax.scatter(emb_reduced[:, 0], emb_reduced[:, 1], emb_reduced[:, 2], 
                               c=color_values, cmap="tab10", alpha=0.7)
            
            ax.set_title(f"{title} (Explained Variance: {explained_variance_str})")
            ax.set_xlabel(f"Semantic Distance % - PC1 ({explained_variance[0]:.1%})")
            ax.set_ylabel(f"Semantic Distance % - PC2 ({explained_variance[1]:.1%})")
            ax.set_zlabel(f"Semantic Distance % - PC3 ({explained_variance[2]:.1%})")
            
            # Add legend for categorical data
            if category_mapping:
                handles = []
                labels_legend = []
                for cat, idx in category_mapping.items():
                    handle = plt.Line2D(
                        [0], [0], marker='o', color='w', 
                        markerfacecolor=plt.cm.tab10(idx/len(category_mapping)), 
                        markersize=10
                    )
                    handles.append(handle)
                    labels_legend.append(cat)
                ax.legend(handles, labels_legend, title=color_by, loc='best')
            # Add colorbar for numeric data
            elif color_by and isinstance(color_values[0], (int, float, np.number)):
                plt.colorbar(scatter, ax=ax, label=color_by)
        
        # Ensure directory exists
        output_dir = os.path.dirname(plot_filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        print(f"{dimensions}D PCA plot generated: '{plot_filename}'")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close(fig_mpl)  # Close the figure to free memory
        return fig_mpl
    
    # Should not reach here, but just in case
    return plt.figure() if not interactive else go.Figure()


def fallback_png_plot(df, emb_reduced, png_filename, title, color_by, labels, 
                     category_mapping, is_language_level_plot, dimensions, explained_variance):
    """Helper function to create a matplotlib fallback plot when kaleido is not available"""
    explained_variance_str = ", ".join([f"PC{i+1}: {var:.1%}" for i, var in enumerate(explained_variance)])
    
    if dimensions == 2:
        # Create a matplotlib figure
        plt.figure(figsize=(12, 10))
        
        # Handle color information
        if color_by and color_by in df.columns:
            categories = df[color_by].tolist()
            unique_categories = sorted(set(categories))
            category_to_int = {cat: i for i, cat in enumerate(unique_categories)}
            colors = [category_to_int[cat] for cat in categories]
            
            scatter = plt.scatter(emb_reduced[:, 0], emb_reduced[:, 1], c=colors, cmap='tab10', alpha=0.7)
            
            # Add legend
            handles = []
            for cat, idx in category_to_int.items():
                handle = plt.Line2D(
                    [0], [0], marker='o', color='w',
                    markerfacecolor=plt.cm.tab10(idx/len(category_to_int)),
                    markersize=10
                )
                handles.append(handle)
            plt.legend(handles, unique_categories, title=color_by, loc='best', bbox_to_anchor=(1.05, 1))
        else:
            plt.scatter(emb_reduced[:, 0], emb_reduced[:, 1], alpha=0.7)
        
        # Add text labels for language-level plots
        if is_language_level_plot:
            for i, label in enumerate(labels):
                plt.annotate(
                    label,
                    (emb_reduced[i, 0], emb_reduced[i, 1]),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha='center',
                    fontsize=8
                )
        
        plt.title(f"{title} (Explained Variance: {explained_variance_str})")
        plt.xlabel(f"Semantic Distance % - PC1 ({explained_variance[0]:.1%})")
        plt.ylabel(f"Semantic Distance % - PC2 ({explained_variance[1]:.1%})")
        plt.tight_layout()
    else:  # 3D
        # Create a matplotlib figure with 3D axis
        fig_mpl = plt.figure(figsize=(12, 10))
        ax = fig_mpl.add_subplot(111, projection='3d')
        
        # Handle color information
        if color_by and color_by in df.columns:
            categories = df[color_by].tolist()
            unique_categories = sorted(set(categories))
            category_to_int = {cat: i for i, cat in enumerate(unique_categories)}
            colors = [category_to_int[cat] for cat in categories]
            
            scatter = ax.scatter(emb_reduced[:, 0], emb_reduced[:, 1], emb_reduced[:, 2], 
                               c=colors, cmap='tab10', alpha=0.7)
            
            # Add legend
            handles = []
            for cat, idx in category_to_int.items():
                handle = plt.Line2D(
                    [0], [0], marker='o', color='w',
                    markerfacecolor=plt.cm.tab10(idx/len(category_to_int)),
                    markersize=10
                )
                handles.append(handle)
            ax.legend(handles, unique_categories, title=color_by, loc='best')
        else:
            ax.scatter(emb_reduced[:, 0], emb_reduced[:, 1], emb_reduced[:, 2], alpha=0.7)
        
        ax.set_title(f"{title} (Explained Variance: {explained_variance_str})")
        ax.set_xlabel(f"Semantic Distance % - PC1 ({explained_variance[0]:.1%})")
        ax.set_ylabel(f"Semantic Distance % - PC2 ({explained_variance[1]:.1%})")
        ax.set_zlabel(f"Semantic Distance % - PC3 ({explained_variance[2]:.1%})")
        plt.tight_layout()
    
    # Save the matplotlib version
    plt.savefig(png_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Successfully generated fallback {dimensions}D PCA visualization at: {png_filename}")


def plot_combined_family_language_pca(lang_df, family_df, plot_filename, title, family_mapping=None, interactive=True):
    """
    Creates a PCA plot with both language and language family points.
    
    Args:
        lang_df (pd.DataFrame): DataFrame with language embeddings
        family_df (pd.DataFrame): DataFrame with language family embeddings
        plot_filename (str): Path to save the plot
        title (str): Title for the plot
        family_mapping (dict): Mapping from language names to family names
        interactive (bool): Whether to generate an interactive Plotly plot
        
    Returns:
        plotly.graph_objects.Figure or matplotlib.figure.Figure: The generated figure
    """
    if len(lang_df) < 1 or len(family_df) < 1:
        print("Not enough data to plot combined family-language visualization")
        return plt.figure() if not interactive else go.Figure()
    
    # Extract embeddings
    lang_embeddings = np.vstack(lang_df['Embedding'])
    family_embeddings = np.vstack(family_df['Embedding'])
    
    # Combine embeddings for PCA
    combined_embeddings = np.vstack([lang_embeddings, family_embeddings])
    
    # Perform PCA
    pca = PCA(n_components=2)
    combined_reduced = pca.fit_transform(combined_embeddings)
    explained_variance = pca.explained_variance_ratio_
    explained_variance_str = ", ".join([f"PC{i+1}: {var:.1%}" for i, var in enumerate(explained_variance)])
    
    # Split back into language and family embeddings
    lang_reduced = combined_reduced[:len(lang_embeddings)]
    family_reduced = combined_reduced[len(lang_embeddings):]
    
    if interactive:
        # Use Plotly for interactive visualization
        fig = go.Figure()
        
        # Add language points
        for i, lang in enumerate(lang_df['Language']):
            lang_label = lang_df['Label'].iloc[i] if 'Label' in lang_df.columns else lang
            family = family_mapping.get(lang, "Unknown") if family_mapping else "Unknown"
            
            # Use a consistent color scheme based on language family
            fig.add_trace(go.Scatter(
                x=[lang_reduced[i, 0]],
                y=[lang_reduced[i, 1]],
                mode='markers',
                marker=dict(size=10),
                name=f"Language: {lang_label}",
                legendgroup=family,
                hovertext=f"Language: {lang_label}<br>Family: {family}"
            ))
        
        # Add family points with larger markers
        for i, fam in enumerate(family_df['LanguageFamily']):
            fam_label = family_df['Label'].iloc[i] if 'Label' in family_df.columns else fam
            
            fig.add_trace(go.Scatter(
                x=[family_reduced[i, 0]],
                y=[family_reduced[i, 1]],
                mode='markers+text',
                marker=dict(size=15, symbol='star', line=dict(width=2, color='black')),
                text=fam_label,
                textposition="top center",
                name=f"Family: {fam_label}",
                legendgroup=fam,
                hovertext=f"Language Family: {fam_label}"
            ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title=f"Semantic Distance % - PC1 ({explained_variance[0]:.1%})",
            yaxis_title=f"Semantic Distance % - PC2 ({explained_variance[1]:.1%})",
            height=800, width=1000,
            legend=dict(
                title="Type",
                orientation="h", 
                y=1.02, x=0.5, xanchor="center", yanchor="bottom"
            ),
            margin=dict(l=20, r=20, b=20, t=60),
        )
        
        # Add dropdown filter to select specific languages or families
        # First, get all unique languages and families
        all_entities = []
        for trace in fig.data:
            if trace.name and 'Language: ' in trace.name:
                all_entities.append(trace.name.replace('Language: ', ''))
            elif trace.name and 'Family: ' in trace.name:
                all_entities.append(trace.name.replace('Family: ', ''))
        
        if all_entities:
            unique_entities = sorted(set(all_entities))
            buttons = []
            
            # First button shows all traces
            buttons.append(dict(label='All', method='update', args=[{'visible': [True]*len(fig.data)}]))
            
            # Add a button to show only families
            family_vis = []
            for trace in fig.data:
                family_vis.append(True if trace.name and 'Family: ' in trace.name else False)
            buttons.append(dict(label='All Families', method='update', args=[{'visible': family_vis}]))
            
            # Add a button to show only languages
            language_vis = []
            for trace in fig.data:
                language_vis.append(True if trace.name and 'Language: ' in trace.name else False)
            buttons.append(dict(label='All Languages', method='update', args=[{'visible': language_vis}]))
            
            # Add buttons for each language family
            if family_mapping:
                unique_families = sorted(set(family_mapping.values()))
                for family in unique_families:
                    family_filtered_vis = []
                    for trace in fig.data:
                        # Show traces that are this family or languages belonging to this family
                        if trace.name and 'Family: ' in trace.name and family in trace.name:
                            family_filtered_vis.append(True)
                        elif trace.name and 'Language: ' in trace.name:
                            lang = trace.name.replace('Language: ', '')
                            if family_mapping.get(lang) == family:
                                family_filtered_vis.append(True)
                            else:
                                family_filtered_vis.append(False)
                        else:
                            family_filtered_vis.append(False)
                    buttons.append(dict(label=f"Family: {family}", method='update', args=[{'visible': family_filtered_vis}]))
            
            # Add buttons for individual languages
            lang_names = [name.replace('Language: ', '') for name in [trace.name for trace in fig.data if trace.name and 'Language: ' in trace.name]]
            for lang in sorted(set(lang_names)):
                lang_vis = []
                for trace in fig.data:
                    if trace.name and 'Language: ' in trace.name and lang in trace.name:
                        lang_vis.append(True)
                    elif trace.name and 'Family: ' in trace.name:
                        # Show the corresponding family
                        family = family_mapping.get(lang) if family_mapping else None
                        if family and family in trace.name:
                            lang_vis.append(True)
                        else:
                            lang_vis.append(False)
                    else:
                        lang_vis.append(False)
                buttons.append(dict(label=f"Language: {lang}", method='update', args=[{'visible': lang_vis}]))
            
            # Add the dropdown menu to the layout
            fig.update_layout(
                updatemenus=[dict(
                    active=0,
                    buttons=buttons,
                    x=1.02,
                    y=1,
                    xanchor='left',
                    yanchor='top',
                    title="Filter by"
                )]
            )
        
        # Save as HTML file
        os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
        fig.write_html(plot_filename)
        print(f"Combined family and language PCA visualization saved to: {plot_filename}")
        
        # Save as PNG if kaleido is available
        try:
            png_filename = os.path.splitext(plot_filename)[0] + ".png"
            try:
                import kaleido
                fig.write_image(png_filename, scale=2)
                print(f"Static PNG version saved to: {png_filename}")
            except ImportError:
                print("Warning: kaleido package not available for PNG export")
                # Save with matplotlib instead
                plt.figure(figsize=(12, 10))
                
                # Plot languages
                for i, lang in enumerate(lang_df['Language']):
                    lang_label = lang_df['Label'].iloc[i] if 'Label' in lang_df.columns else lang
                    plt.scatter(lang_reduced[i, 0], lang_reduced[i, 1], s=50, label=f"Lang: {lang_label}")
                    plt.text(lang_reduced[i, 0], lang_reduced[i, 1], lang_label, fontsize=8)
                
                # Plot families
                for i, fam in enumerate(family_df['LanguageFamily']):
                    fam_label = family_df['Label'].iloc[i] if 'Label' in family_df.columns else fam
                    plt.scatter(family_reduced[i, 0], family_reduced[i, 1], s=100, marker='*', label=f"Fam: {fam_label}")
                    plt.text(family_reduced[i, 0], family_reduced[i, 1], fam_label, fontsize=10, fontweight='bold')
                
                plt.title(title)
                plt.xlabel(f"Semantic Distance % - PC1 ({explained_variance[0]:.1%})")
                plt.ylabel(f"Semantic Distance % - PC2 ({explained_variance[1]:.1%})")
                plt.tight_layout()
                plt.savefig(png_filename, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Static PNG fallback version saved to: {png_filename}")
        except Exception as e:
            print(f"Warning: Error saving PNG version: {e}")
        
        return fig
    else:
        # Use Matplotlib for static visualization
        fig = plt.figure(figsize=(12, 10))
        
        # Plot languages
        for i, lang in enumerate(lang_df['Language']):
            lang_label = lang_df['Label'].iloc[i] if 'Label' in lang_df.columns else lang
            family = family_mapping.get(lang, "Unknown") if family_mapping else "Unknown"
            
            plt.scatter(lang_reduced[i, 0], lang_reduced[i, 1], s=50, label=f"Lang: {lang_label}")
            plt.text(lang_reduced[i, 0], lang_reduced[i, 1], lang_label, fontsize=8)
        
        # Plot families
        for i, fam in enumerate(family_df['LanguageFamily']):
            fam_label = family_df['Label'].iloc[i] if 'Label' in family_df.columns else fam
            
            plt.scatter(family_reduced[i, 0], family_reduced[i, 1], s=100, marker='*', label=f"Fam: {fam_label}")
            plt.text(family_reduced[i, 0], family_reduced[i, 1], fam_label, fontsize=10, fontweight='bold')
        
        plt.title(title)
        plt.xlabel(f"Semantic Distance % - PC1 ({explained_variance[0]:.1%})")
        plt.ylabel(f"Semantic Distance % - PC2 ({explained_variance[1]:.1%})")
        plt.tight_layout()
        
        # Save figure
        os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Combined family and language PCA visualization saved to: {plot_filename}")
        
        # Save HTML version if possible
        try:
            from plotly.subplots import make_subplots
            html_filename = os.path.splitext(plot_filename)[0] + ".html"
            
            # Create a simple Plotly version
            plotly_fig = go.Figure()
            
            # Add language points
            for i, lang in enumerate(lang_df['Language']):
                lang_label = lang_df['Label'].iloc[i] if 'Label' in lang_df.columns else lang
                family = family_mapping.get(lang, "Unknown") if family_mapping else "Unknown"
                
                plotly_fig.add_trace(go.Scatter(
                    x=[lang_reduced[i, 0]],
                    y=[lang_reduced[i, 1]],
                    mode='markers+text',
                    marker=dict(size=10),
                    text=lang_label,
                    textposition="top center",
                    name=f"Language: {lang_label}",
                    hovertext=f"Language: {lang_label}<br>Family: {family}"
                ))
            
            # Add family points
            for i, fam in enumerate(family_df['LanguageFamily']):
                fam_label = family_df['Label'].iloc[i] if 'Label' in family_df.columns else fam
                
                plotly_fig.add_trace(go.Scatter(
                    x=[family_reduced[i, 0]],
                    y=[family_reduced[i, 1]],
                    mode='markers+text',
                    marker=dict(size=15, symbol='star'),
                    text=fam_label,
                    textposition="top center",
                    name=f"Family: {fam_label}",
                    hovertext=f"Language Family: {fam_label}"
                ))
            
            plotly_fig.update_layout(
                title=title,
                xaxis_title=f"Semantic Distance % - PC1 ({explained_variance[0]:.1%})",
                yaxis_title=f"Semantic Distance % - PC2 ({explained_variance[1]:.1%})"
            )
            
            plotly_fig.write_html(html_filename)
            print(f"Interactive HTML version saved to: {html_filename}")
        except Exception as e:
            print(f"Warning: Could not create interactive HTML version: {e}")
        
        plt.close(fig)
        return fig


def interactive_html_correlation_plot(df, output_file, distance_x='CosineDistance', distance_y='JaccardDistance', 
                         hover_data=None, color_by=None, title=None, language=None):
    """
    Creates an interactive Plotly scatter plot showing correlation between two distance metrics.
    
    Args:
        df (pd.DataFrame): DataFrame with correlation data
        output_file (str): Path to save the HTML plot
        distance_x (str): Column name for x-axis data (default: 'CosineDistance')
        distance_y (str): Column name for y-axis data (default: 'JaccardDistance')
        hover_data (list): Additional columns to show in hover tooltip
        color_by (str): Column to color points by
        title (str): Plot title
        language (str): Optional language name for title customization
        
    Returns:
        plotly.graph_objects.Figure: The Plotly figure object
    """
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        import plotly.io as pio
        from scipy import stats
        import numpy as np
        import pandas as pd
        import os
        
        # Validate inputs
        if df is None or df.empty:
            print(f"Error: Empty DataFrame provided to interactive_html_correlation_plot")
            return None
            
        # Ensure required columns exist
        if distance_x not in df.columns:
            print(f"Error: Column '{distance_x}' not found in DataFrame")
            return None
            
        if distance_y not in df.columns:
            print(f"Error: Column '{distance_y}' not found in DataFrame")
            return None
        
        # Create a clean dataframe without NaNs in the key columns
        df_clean = df.dropna(subset=[distance_x, distance_y])
        
        if df_clean.empty:
            print(f"Error: No valid data points after dropping NaN values")
            return None
        
        # If color_by is specified but not in columns, reset it to None
        if color_by and color_by not in df_clean.columns:
            print(f"Warning: Column '{color_by}' specified for coloring not found, using default colors")
            color_by = None
            
        # Set up hover data
        hover_cols = []
        if hover_data:
            for col in hover_data:
                if col in df_clean.columns:
                    hover_cols.append(col)
                else:
                    print(f"Warning: Hover column '{col}' not found in DataFrame")
        
        # Add noun to hover_cols if it exists and not already there
        if 'Noun' in df_clean.columns and 'Noun' not in hover_cols:
            hover_cols.append('Noun')
            
        # Add NounCategory to hover_cols if it exists and not already there
        if 'NounCategory' in df_clean.columns and 'NounCategory' not in hover_cols:
            hover_cols.append('NounCategory')
        
        # Calculate correlations
        try:
            # Pearson correlation
            pearson_r, pearson_p = stats.pearsonr(df_clean[distance_x], df_clean[distance_y])
            
            # Spearman correlation
            spearman_r, spearman_p = stats.spearmanr(df_clean[distance_x], df_clean[distance_y])
        except Exception as e_corr:
            print(f"Error calculating correlations: {e_corr}")
            pearson_r = pearson_p = spearman_r = spearman_p = float('nan')
        
        # Create the scatter plot
        fig = px.scatter(
            df_clean,
            x=distance_x,
            y=distance_y,
            color=color_by,
            hover_data=hover_cols,
            opacity=0.7,
            title=title or f"Correlation between {distance_x} and {distance_y}" + (f" for {language}" if language else ""),
            labels={
                distance_x: f"Semantic Distance % ({distance_x})",
                distance_y: f"Semantic Distance % ({distance_y})"
            },
            template="plotly_white"
        )
        
        # Add correlation stats as an annotation
        correlation_text = f"<b>Statistics:</b><br>Pearson r: {pearson_r:.3f} (p = {pearson_p:.3f})<br>Spearman ρ: {spearman_r:.3f} (p = {spearman_p:.3f})<br>n = {len(df_clean)}"
        
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text=correlation_text,
            showarrow=False,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            opacity=0.8,
            align="left",
            font=dict(size=12)
        )
        
        # Update layout for better appearance
        fig.update_layout(
            width=900,
            height=700,
            legend_title=color_by if color_by else "",
            margin=dict(l=30, r=30, t=70, b=30)
        )
        
        # Save the figure if output path provided
        if output_file:
            # Ensure directory exists
            try:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                
                # Save as HTML
                pio.write_html(fig, output_file, auto_open=False)
                print(f"Interactive correlation plot saved to: {output_file}")
            except Exception as e_save:
                print(f"Error saving interactive plot to {output_file}: {e_save}")
        
        return fig
    except Exception as e:
        print(f"Error generating interactive correlation plot: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_standard_dim_reduction_plots(
    df, 
    base_filename_prefix, 
    title_core, 
    color_by, 
    original_distance_matrix, 
    tsne_dir, 
    umap_dir, 
    pca_dir,
    api_model_name="Unknown Model"
):
    """
    Generates a standard set of dimensionality reduction plots (t-SNE, UMAP, PCA).

    Args:
        df (pd.DataFrame): DataFrame with embeddings and metadata.
        base_filename_prefix (str): Base prefix for output filenames (e.g., f"{base_filename}_{proficiency}_{prompt}_{category}").
        title_core (str): Core part of the plot titles (e.g., f"{proficiency} - {prompt} - {category}").
        color_by (str): Column name to color points by (e.g., 'Language').
        original_distance_matrix (pd.DataFrame, optional): Pairwise distance matrix for Kruskal's Stress.
        tsne_dir (str): Output directory for t-SNE plots.
        umap_dir (str): Output directory for UMAP plots.
        pca_dir (str): Output directory for PCA plots.
        api_model_name (str): Name of the API model, for inclusion in titles.
    """
    if df.empty or len(df) < 2:
        print(f"  Skipping dimensionality reduction plots for '{title_core}': Insufficient data (rows: {len(df)}).")
        return

    print(f"  Generating dimensionality reduction plots for '{title_core}' with {len(df)} data points...")

    # Ensure output directories exist
    os.makedirs(tsne_dir, exist_ok=True)
    os.makedirs(umap_dir, exist_ok=True)
    os.makedirs(pca_dir, exist_ok=True)

    # t-SNE Plots (2D and 3D)
    try:
        print(f"    Generating t-SNE plots...")
        tsne_base = os.path.join(tsne_dir, f"{base_filename_prefix}_tsne")
        plot_tsne(df, f"{tsne_base}.html", 
                  f"Semantic Space (t-SNE) - {api_model_name} - {title_core}", 
                  color_by=color_by, interactive=True, 
                  original_distance_matrix=original_distance_matrix, dimensions=2)
        plot_tsne(df, f"{tsne_base}_3d.html", 
                  f"3D Semantic Space (t-SNE) - {api_model_name} - {title_core}", 
                  color_by=color_by, interactive=True, 
                  original_distance_matrix=original_distance_matrix, dimensions=3) # original_distance_matrix is only used for 2D Kruskal Stress by plot_tsne
    except Exception as e:
        print(f"    Warning: Error producing t-SNE visualizations for '{title_core}': {e}")
        traceback.print_exc()

    # UMAP Plots (2D and 3D)
    try:
        print(f"    Generating UMAP plots...")
        umap_base = os.path.join(umap_dir, f"{base_filename_prefix}_umap")
        plot_umap(df, f"{umap_base}.html", 
                  f"UMAP Visualization - {api_model_name} - {title_core}", 
                  color_by=color_by, interactive=True, 
                  original_distance_matrix=original_distance_matrix, dimensions=2)
        plot_umap(df, f"{umap_base}_3d.html", 
                  f"3D UMAP Visualization - {api_model_name} - {title_core}", 
                  color_by=color_by, interactive=True, 
                  original_distance_matrix=original_distance_matrix, dimensions=3)
    except Exception as e:
        print(f"    Warning: Error producing UMAP visualizations for '{title_core}': {e}")
        traceback.print_exc()

    # PCA Plots (2D and 3D)
    try:
        print(f"    Generating PCA plots...")
        pca_base = os.path.join(pca_dir, f"{base_filename_prefix}_pca")
        plot_pca(df, f"{pca_base}.html", 
                 f"PCA Visualization - {api_model_name} - {title_core}", 
                 color_by=color_by, interactive=True, dimensions=2)
        plot_pca(df, f"{pca_base}_3d.html", 
                 f"3D PCA Visualization - {api_model_name} - {title_core}", 
                 color_by=color_by, interactive=True, dimensions=3)
    except Exception as e:
        print(f"    Warning: Error producing PCA visualizations for '{title_core}': {e}")
        traceback.print_exc()
