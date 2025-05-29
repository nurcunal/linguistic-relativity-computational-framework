"""
Distance metric calculation functions for Language Representation Analysis.

This module contains functions for computing various distance measures between
embeddings and adjective sets, including cosine distances, Jaccard distances,
and helper functions for parallel processing of distance calculations.
"""

import numpy as np
import pandas as pd
import time
import os
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform


def compute_pairwise_distances(df):
    """
    Computes pairwise cosine distances between embeddings.
    Uses multiprocessing for larger datasets to improve performance.
    
    Args:
        df (pd.DataFrame): DataFrame containing embeddings to compare.
        
    Returns:
        pd.DataFrame: Matrix of pairwise distances between embeddings.
    """
    print(f"Computing pairwise distances between {len(df)} embeddings...")
    start_time = time.time()
    
    embeddings = np.vstack(df['Embedding'])
    
    # For larger datasets, use multiprocessing
    if len(df) > 100:
        print(f"Using optimized matrix operations for {len(df)}x{len(df)} ({len(df)**2} comparisons)")
        
        # Set up parallel processing with limited cores
        # Always leave 4 cores available for the user
        n_cores = min(12, max(1, (os.cpu_count() - 4) if os.cpu_count() is not None else 1))
        print(f"Using {n_cores} CPU cores for distance computation (leaving at least 4 cores free if possible, up to 12 max)")
        
        # We'll use the parallel_cosine_similarity function if we're in a situation
        # where it would be beneficial (very large matrices)
        try:
            from joblib import Parallel, delayed
            
            def parallel_cosine_similarity_joblib(X_normalized_data, n_samples_data, batch_size_data, n_cores_data):
                """
                Compute cosine similarity with parallel processing using joblib.
                
                Args:
                    X_normalized_data: Normalized embeddings matrix.
                    n_samples_data: Number of samples in the data.
                    batch_size_data: Batch size for parallelization.
                    n_cores_data: Number of cores to use.
                    
                Returns:
                    np.array: Distance matrix (1 - similarity).
                """
                n_batches = int(np.ceil(n_samples_data / batch_size_data))
                results_joblib = Parallel(n_jobs=n_cores_data)(
                    delayed(lambda i_batch: X_normalized_data[i_batch*batch_size_data:min((i_batch+1)*batch_size_data, n_samples_data)] @ X_normalized_data.T)(i_batch)
                    for i_batch in range(n_batches)
                )
                similarities_joblib = np.vstack(results_joblib)
                return 1 - similarities_joblib  # Convert to distances
            
            # Lowered threshold for using parallel processing with joblib
            # Use if joblib is available and dataset size is moderate and multiple cores are set up.
            if len(df) > 200 and n_cores > 1:
                print(f"Using joblib parallel_cosine_similarity for {len(df)} embeddings.")
                # Pre-normalize vectors for faster cosine similarity
                norms = np.sqrt(np.sum(embeddings ** 2, axis=1))
                # Handle potential division by zero if a vector is all zeros
                norms[norms == 0] = 1e-9  # Avoid division by zero, replace 0 norm with a tiny number
                X_normalized_emb = embeddings / norms[:, np.newaxis]
                
                dist = parallel_cosine_similarity_joblib(X_normalized_emb, embeddings.shape[0], batch_size_data=1000, n_cores_data=n_cores)
            else:
                # For medium-sized matrices or if joblib not preferred, the built-in function is often fast enough
                # and might use BLAS/OpenMP based parallelism implicitly.
                print(f"Using sklearn.metrics.pairwise.cosine_similarity for {len(df)} embeddings.")
                sim = cosine_similarity(embeddings)
                dist = 1 - sim
        except ImportError:
            # Fall back to standard approach if joblib not available
            print("  Note: joblib not available, using standard computation")
            sim = cosine_similarity(embeddings)
            dist = 1 - sim
    else:
        # Small datasets - standard approach
        sim = cosine_similarity(embeddings)
        dist = 1 - sim
    
    labels = df['Label']
    dist_df = pd.DataFrame(dist, index=labels, columns=labels)
    
    print(f"Distance computation complete. Time: {time.time() - start_time:.2f}s")
    print(f"Matrix shape: {dist_df.shape}")
    
    return dist_df.clip(lower=0).round(6)


def _process_filter_distances_chunk(args):
    """
    Helper function for parallel processing in filter_distances. Processes a chunk of rows.
    
    Args:
        args (tuple): Contains chunk_start_i, chunk_end_i, n, row_info_list, dist_values_np_array.
            chunk_start_i (int): Start index for this chunk.
            chunk_end_i (int): End index for this chunk.
            n (int): Total number of rows.
            row_info_list (list): List of dictionaries with row information.
            dist_values_np_array (np.array): Distance matrix.
    
    Returns:
        list: List of tuples containing (label1, label2, distance).
    """
    chunk_start_i, chunk_end_i, n, row_info_list, dist_values_np_array = args
    local_records = []
    # print(f"Worker processing chunk: i from {chunk_start_i} to {chunk_end_i-1}")  # Optional: for debugging
    
    for i in range(chunk_start_i, chunk_end_i):
        row_i_info = row_info_list[i]
        for j in range(i + 1, n):
            row_j_info = row_info_list[j]
            if row_i_info['noun'] != row_j_info['noun']:
                continue
            dist_val = dist_values_np_array[i, j]
            local_records.append((row_i_info['label'], row_j_info['label'], dist_val))
    
    return local_records


def filter_distances(df, dist_df, output_csv=None):
    """
    Filters pairwise distances so that only rows with the same noun are compared.
    Rows differing in language are allowed if other fields match.
    
    Args:
        df (pd.DataFrame): Input DataFrame with embeddings and metadata.
        dist_df (pd.DataFrame): Distance matrix between all embeddings.
        output_csv (str, optional): Path to save filtered results.
        
    Returns:
        pd.DataFrame: Filtered DataFrame with only relevant pairwise distances.
    """
    n = len(df)
    
    if n < 2 or dist_df.empty:
        print("Warning: Not enough data to filter distances")
        return pd.DataFrame(columns=["Label1", "Label2", "Distance"])
    
    # Prepare row information for filtering
    row_info = []
    print(f"Preparing row info for filtering {n} embeddings...")
    for i in range(n):
        row_info.append({
            'label': df.iloc[i]['Label'],
            'noun': df.iloc[i]['Noun'],
            'lang': df.iloc[i]['Language'],
            'prof': df.iloc[i]['Proficiency'],
            'prompt': df.iloc[i]['Prompt']
        })
    
    # Set up parallel processing
    dist_values_np = dist_df.values
    n_cores = min(12, max(1, (os.cpu_count() - 4) if os.cpu_count() is not None else 1))
    print(f"Filtering distances between {n} embeddings using up to {n_cores} CPU cores...")

    records = []
    # Use parallel processing for larger datasets if multiple cores configured
    if n > 100 and n_cores > 1:
        # Determine chunk size for better parallelization
        num_chunks = n_cores * 2  # Generate more chunks than cores to keep them busy
        chunk_size = (n + num_chunks - 1) // num_chunks  # Ceiling division
        if chunk_size == 0:
            chunk_size = 1  # Prevent chunk_size from being 0 if n is small

        # Generate tasks for parallel processing
        tasks = []
        for i_chunk_start in range(0, n, chunk_size):
            i_chunk_end = min(i_chunk_start + chunk_size, n)
            if i_chunk_start < i_chunk_end:  # Ensure there are rows in the chunk
                tasks.append((i_chunk_start, i_chunk_end, n, row_info, dist_values_np))
        
        if not tasks:
            print("No tasks generated for parallel processing in filter_distances, falling back to single core.")
        else:
            print(f"Using ProcessPoolExecutor with {n_cores} workers for {len(tasks)} chunks in filter_distances (chunk_size ~{chunk_size}).")
            try:
                with ProcessPoolExecutor(max_workers=n_cores) as executor:
                    results = executor.map(_process_filter_distances_chunk, tasks)
                    for res_list in results:
                        if res_list:
                            records.extend(res_list)
            except Exception as e_exec:
                print(f"Error during ProcessPoolExecutor in filter_distances: {e_exec}")
                print("Falling back to single-core processing for filter_distances.")
                # Fallback to single core if executor fails
                for i in range(n):  # Original single-core logic
                    row_i_info = row_info[i]
                    for j in range(i + 1, n):
                        if row_i_info['noun'] != row_info[j]['noun']:
                            continue
                        dist_val = dist_values_np[i, j]
                        records.append((row_i_info['label'], row_info[j]['label'], dist_val))
    
    # If parallel processing failed or produced no records, or if it was skipped
    if not records:
        if not (n > 100 and n_cores > 1 and tasks):  # Check if we intended to run parallel processing
            print("Using single-core processing for filter_distances (small dataset, 1 core, or no parallel tasks generated).")
            for i in range(n):
                row_i_info = row_info[i]
                for j in range(i + 1, n):
                    if row_i_info['noun'] != row_info[j]['noun']:
                        continue
                    dist_val = dist_values_np[i, j]
                    records.append((row_i_info['label'], row_info[j]['label'], dist_val))
    
    # Convert results to DataFrame
    filtered_df = pd.DataFrame(records, columns=["Label1", "Label2", "Distance"])
    
    # If no records found, return empty DataFrame
    if filtered_df.empty:
        print("Warning: No valid filtered distances found")
    else:
        # Sort results
        filtered_df.sort_values(by="Distance", inplace=True)
        
        # Save to CSV if output path provided
        if output_csv:
            filtered_df.to_csv(output_csv, index=False)
            print(f"Saved {len(filtered_df)} filtered distances to '{output_csv}'")
    
    return filtered_df


def _process_jaccard_distances_chunk(args):
    """Helper function for parallel processing in compute_jaccard_distances, processes a chunk of rows."""
    chunk_start_i, chunk_end_i, n, row_info_list = args
    local_records = []
    for i in range(chunk_start_i, chunk_end_i):
        row_i_info = row_info_list[i]
        for j in range(i + 1, n):
            row_j_info = row_info_list[j]
            if (row_i_info["noun"] == row_j_info["noun"] and
                    row_i_info["prof"] == row_j_info["prof"] and
                    row_i_info["prompt"] == row_j_info["prompt"]):
                if row_i_info["lang"] != row_j_info["lang"]:
                    set_i = row_i_info["adjs"]
                    set_j = row_j_info["adjs"]
                    union = set_i.union(set_j)
                    
                    if not union: # Skip empty adjective sets
                        continue
                        
                    intersection = set_i.intersection(set_j)
                    jaccard_dist = 1.0 - (len(intersection) / len(union))
                    local_records.append((row_i_info["label"], row_j_info["label"], jaccard_dist))
    return local_records


def compute_jaccard_distances(df, output_csv=None):
    """
    Computes Jaccard distances between adjective sets for rows with the same noun,
    proficiency, and prompt but different languages.
    Optionally saves the results to a CSV file if output_csv is provided.
    Returns the DataFrame with Jaccard distances.
    """
    print(f"Computing Jaccard distances between adjectives...")
    # records = [] # Initialized later
    n = len(df)
    
    if n < 2:
        print("Warning: Not enough data to compute Jaccard distances")
        return pd.DataFrame(columns=["Label1", "Label2", "JaccardDistance"])

    row_info = []
    print(f"Preparing row info for Jaccard distances ({n} rows)...")
    for i in range(n):
        adjs = df.iloc[i]["Adjectives"]
        if isinstance(adjs, str):
            adjs = adjs.split(";")
        elif not isinstance(adjs, list):
            adjs = []
            
        row_info.append({
            "label": df.iloc[i]["Label"],
            "noun": df.iloc[i]["Noun"],
            "lang": df.iloc[i]["Language"],
            "prof": df.iloc[i]["Proficiency"],
            "prompt": df.iloc[i]["Prompt"],
            "adjs": set(adjs)
        })
    
    n_cores = min(12, max(1, (os.cpu_count() - 4) if os.cpu_count() is not None else 1))
    print(f"Computing Jaccard distances using up to {n_cores} CPU cores...")

    records = []
    if n > 100 and n_cores > 1: 
        num_chunks = n_cores * 2 
        chunk_size = (n + num_chunks - 1) // num_chunks
        if chunk_size == 0 : chunk_size = 1

        tasks = []
        for i_chunk_start in range(0, n, chunk_size):
            i_chunk_end = min(i_chunk_start + chunk_size, n)
            if i_chunk_start < i_chunk_end:
                tasks.append((i_chunk_start, i_chunk_end, n, row_info))

        if not tasks:
            print("No tasks generated for parallel processing in compute_jaccard_distances, falling back to single core.")
        else:
            print(f"Using ProcessPoolExecutor with {n_cores} workers for {len(tasks)} chunks in compute_jaccard_distances (chunk_size ~{chunk_size}).")
            try:
                with ProcessPoolExecutor(max_workers=n_cores) as executor:
                    results = executor.map(_process_jaccard_distances_chunk, tasks)
                    for res_list in results:
                        if res_list:
                            records.extend(res_list)
            except Exception as e_exec:
                print(f"Error during ProcessPoolExecutor in compute_jaccard_distances: {e_exec}")
                print("Falling back to single-core processing for compute_jaccard_distances.")
                # Fallback logic explicitly for this case
                for i in range(n):
                    row_i_info = row_info[i]
                    for j in range(i + 1, n):
                        row_j_info = row_info[j]  # DEFINED HERE
                        if (row_i_info["noun"] == row_j_info["noun"] and
                                row_i_info["prof"] == row_j_info["prof"] and
                                row_i_info["prompt"] == row_j_info["prompt"]):
                            if row_i_info["lang"] != row_j_info["lang"]:
                                set_i = row_i_info["adjs"]
                                set_j = row_j_info["adjs"]
                                union = set_i.union(set_j)
                                if not union: continue
                                intersection = set_i.intersection(set_j)
                                jaccard_dist = 1.0 - (len(intersection) / len(union))
                                records.append((row_i_info["label"], row_j_info["label"], jaccard_dist))
    
    if not records: # If parallel processing failed/skipped and produced no records
         if not (n > 100 and n_cores > 1 and tasks): # check if we intended to run parallel processing
            print("Using single-core processing for compute_jaccard_distances (small dataset, 1 core, or no parallel tasks generated).")
            for i in range(n):
                row_i_info = row_info[i]
                for j in range(i + 1, n):
                    row_j_info = row_info[j]  # DEFINED HERE
                    if (row_i_info["noun"] == row_j_info["noun"] and
                            row_i_info["prof"] == row_j_info["prof"] and
                            row_i_info["prompt"] == row_j_info["prompt"]):
                        if row_i_info["lang"] != row_j_info["lang"]:
                            set_i = row_i_info["adjs"]
                            set_j = row_j_info["adjs"]
                            union = set_i.union(set_j)
                            if not union: continue
                            intersection = set_i.intersection(set_j)
                            jaccard_dist = 1.0 - (len(intersection) / len(union))
                            records.append((row_i_info["label"], row_j_info["label"], jaccard_dist))
    
    jaccard_df = pd.DataFrame(records, columns=["Label1", "Label2", "JaccardDistance"])
    
    # If no records found, return empty DataFrame
    if jaccard_df.empty:
        print("Warning: No valid Jaccard distances found")
    else:
        # Sort results
        jaccard_df.sort_values(by="JaccardDistance", inplace=True)
        
        # Save to CSV only if output path is explicitly provided
        if output_csv:
            jaccard_df.to_csv(output_csv, index=False)
            print(f"Saved {len(jaccard_df)} Jaccard distances to '{output_csv}'")
    
    return jaccard_df


def calculate_kruskal_stress(original_distances, reduced_coordinates):
    """
    Calculates Kruskal's Stress to measure how well a dimensionality reduction preserves distances.
    Lower values indicate better preservation of the original structure.
    
    Args:
        original_distances (np.ndarray): Original distance matrix
        reduced_coordinates (np.ndarray): Coordinates in the reduced space (e.g., from t-SNE or UMAP)
        
    Returns:
        float: Kruskal's stress value (normalized between 0 and 1)
    """
    try:
        if original_distances.shape[0] != reduced_coordinates.shape[0]:
            raise ValueError(f"Mismatch between original_distances ({original_distances.shape}) and reduced_coordinates ({reduced_coordinates.shape})")
            
        # Calculate distances in the reduced (e.g., 2D) space
        reduced_distances = squareform(pdist(reduced_coordinates))
        
        # If the matrices are symmetric (square), focus only on upper triangular part to avoid duplicate calculations
        if original_distances.shape[0] == original_distances.shape[1]:
            original_dist_flat = original_distances[np.triu_indices_from(original_distances, k=1)]
            reduced_dist_flat = reduced_distances[np.triu_indices_from(reduced_distances, k=1)]
        else:
            # If not symmetric, flatten everything
            original_dist_flat = original_distances.flatten()
            reduced_dist_flat = reduced_distances.flatten()
        
        # Classic Kruskal's Stress formula
        numerator = np.sum((reduced_dist_flat - original_dist_flat) ** 2)
        denominator = np.sum(original_dist_flat ** 2)
        
        # Handle potential division by zero
        if denominator == 0:
            return 1.0  # Worst possible value
            
        stress = np.sqrt(numerator / denominator)
        
        # Interpretation guidelines:
        # 0.20+ = poor representation (original distances not preserved)
        # 0.10-0.20 = fair
        # 0.05-0.10 = good
        # 0.00-0.05 = excellent (high fidelity to original distances)
        
        return stress
    
    except Exception as e:
        print(f"Error calculating Kruskal's stress: {e}")
        import traceback
        traceback.print_exc()
        return 1.0  # Return worst possible value on error
