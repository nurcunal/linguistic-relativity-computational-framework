"""
Comprehensive analysis functions for Language Representation Analysis.

This module contains functions for generating comprehensive analysis results
by combining different types of analysis data, such as noun-level distances,
Jaccard distances, and more detailed analysis.
"""

import os
import numpy as np
import pandas as pd


def _parse_label_for_noun_level_distances(label_str):
    """
    Parses a label string to extract noun, language, proficiency, prompt, and category.
    
    Args:
        label_str (str): A label string in the format "noun, language, proficiency, prompt, category"
        
    Returns:
        tuple: (noun, language, proficiency, prompt, category)
    """
    try:
        parts = label_str.split(", ")
        if len(parts) >= 5:
            # Format: "noun, language, proficiency, prompt, category"
            return tuple(parts[:5])
        elif len(parts) == 4:
            # Format with missing category: "noun, language, proficiency, prompt"
            return tuple(parts + ["Unknown"])
        else:
            # Default for malformed labels
            print(f"Warning: Malformed label string: '{label_str}'")
            placeholder = "unknown"
            return (
                parts[0] if len(parts) > 0 else placeholder,
                parts[1] if len(parts) > 1 else placeholder,
                parts[2] if len(parts) > 2 else placeholder,
                parts[3] if len(parts) > 3 else placeholder,
                "Unknown"
            )
    except Exception as e:
        print(f"Error parsing label string '{label_str}': {e}")
        return ("error", "error", "error", "error", "error")


def _generate_and_save_noun_level_distances_csv(category_df_arg, base_filename_arg, proficiency_arg, prompt_arg, noun_category_arg, output_dir_arg):
    """
    Generates and saves noun-level cosine and Jaccard distances.
    
    Args:
        category_df_arg (pd.DataFrame): DataFrame with embeddings and metadata
        base_filename_arg (str): Base filename for the output file
        proficiency_arg (str): Proficiency level string
        prompt_arg (str): Prompt string
        noun_category_arg (str): Noun category string
        output_dir_arg (str): Directory to save the output CSV
        
    Returns:
        tuple: (filtered_cosine_df, jaccard_df) DataFrames or None if processing fails
    """
    if category_df_arg.empty or len(category_df_arg) < 2:
        print(f"Skipping noun-level distance file for {proficiency_arg}/{prompt_arg}/{noun_category_arg} due to insufficient data.")
        return

    print(f"Processing noun-level distances for: {proficiency_arg} - {prompt_arg} - {noun_category_arg}")

    # Import the necessary functions from their new locations
    from lr_analysis_functions.distance_metrics import compute_pairwise_distances, filter_distances, compute_jaccard_distances

    # 1. Compute pairwise cosine distances (full matrix for the category_df_arg)
    all_cosine_distances_df = compute_pairwise_distances(category_df_arg)
    if all_cosine_distances_df.empty:
        print(f"  Warning: No cosine distances computed for {proficiency_arg}/{prompt_arg}/{noun_category_arg}")
        return

    # 2. Filter cosine distances (same noun, different language)
    #    filter_distances returns Label1, Label2, Distance (where Distance is cosine)
    filtered_cosine_df = filter_distances(category_df_arg, all_cosine_distances_df, output_csv=None)
    if filtered_cosine_df.empty:
        print(f"  Warning: No filtered cosine distances for {proficiency_arg}/{prompt_arg}/{noun_category_arg}")
        # We might still have Jaccard, so don't return yet unless Jaccard also fails

    # 3. Compute Jaccard distances
    #    compute_jaccard_distances returns Label1, Label2, JaccardDistance
    jaccard_df = compute_jaccard_distances(category_df_arg, output_csv=None)
    if jaccard_df.empty:
        print(f"  Warning: No Jaccard distances computed for {proficiency_arg}/{prompt_arg}/{noun_category_arg}")
        if filtered_cosine_df.empty:  # If both are empty, then return
            return

    # 4. Merge Cosine and Jaccard distances
    # Rename cosine's distance column before merge
    if not filtered_cosine_df.empty:
        filtered_cosine_df = filtered_cosine_df.rename(columns={'Distance': 'CosineDistance'})
        if not jaccard_df.empty:
            merged_df = pd.merge(filtered_cosine_df, jaccard_df, on=['Label1', 'Label2'], how='outer')
        else:
            merged_df = filtered_cosine_df
            merged_df['JaccardDistance'] = np.nan  # Add Jaccard column if it doesn't exist
    elif not jaccard_df.empty:  # Only Jaccard exists
        merged_df = jaccard_df
        merged_df['CosineDistance'] = np.nan  # Add Cosine column
    else:  # Both were empty after all
        print(f"  No data to merge for noun-level distances for {proficiency_arg}/{prompt_arg}/{noun_category_arg}")
        return

    # 5. Parse Labels and Generate Final DataFrame
    output_records = []
    for _, row in merged_df.iterrows():
        # Correctly unpack all 5 values returned by _parse_label_for_noun_level_distances
        noun1, lang1, _prof1, _prompt1, _cat1 = _parse_label_for_noun_level_distances(row['Label1'])
        noun2, lang2, _prof2, _prompt2, _cat2 = _parse_label_for_noun_level_distances(row['Label2'])

        if noun1 != noun2:  # Should not happen due to filter_distances logic, but as a safeguard
            print(f"Warning: Mismatched nouns in merged data: {noun1} vs {noun2} for labels {row['Label1']} | {row['Label2']}")
            continue
        
        output_records.append({
            'Language1': lang1,
            'Language2': lang2,
            'Noun': noun1,  # Noun is the same for both labels in a pair
            'CosineDistance': row.get('CosineDistance', np.nan),
            'JaccardDistance': row.get('JaccardDistance', np.nan)
        })
    
    if not output_records:
        print(f"  No records to save after parsing for {proficiency_arg}/{prompt_arg}/{noun_category_arg}")
        return

    final_noun_level_df = pd.DataFrame(output_records)
    # Sort for consistency, although not strictly required by user request for these specific columns
    final_noun_level_df.sort_values(by=['Noun', 'Language1', 'Language2'], inplace=True)

    # 6. Save the Final DataFrame
    from lr_analysis_functions.utils import sanitize_prompt, sanitize_category
    output_filename = f"{base_filename_arg}_{proficiency_arg}_{sanitize_prompt(prompt_arg)}_{sanitize_category(noun_category_arg)}_noun_level_distances.csv"
    output_filepath = os.path.join(output_dir_arg, output_filename)
    
    try:
        final_noun_level_df.to_csv(output_filepath, index=False)
        print(f"  Successfully saved noun-level distances to: {output_filepath} ({len(final_noun_level_df)} rows)")
    except Exception as e_save:
        print(f"  Error saving noun-level distances to {output_filepath}: {e_save}")
    
    # Return the computed dataframes for potential reuse
    return filtered_cosine_df, jaccard_df


def generate_combined_analysis(df, filtered_distances_df, jaccard_distances_df, output_file=None):
    """
    Generates a comprehensive analysis combining noun-level distances, Jaccard distances, and language-level metrics.
    
    Args:
        df (pd.DataFrame): Original DataFrame with embeddings
        filtered_distances_df (pd.DataFrame): Filtered cosine distances for noun-level analysis
        jaccard_distances_df (pd.DataFrame): Jaccard distances for adjective similarity analysis
        output_file (str): Optional path to save the output CSV
        
    Returns:
        pd.DataFrame: Comprehensive analysis DataFrame
    """
    import pandas as pd
    import numpy as np
    import os
    
    print("Generating comprehensive analysis...")
    if filtered_distances_df is None or filtered_distances_df.empty:
        print("Warning: No filtered distances provided for comprehensive analysis")
        return pd.DataFrame()
    
    try:
        # If jaccard_distances_df is None or empty, create an empty DataFrame with correct columns
        if jaccard_distances_df is None or jaccard_distances_df.empty:
            jaccard_distances_df = pd.DataFrame(columns=['Label1', 'Label2', 'JaccardDistance'])
            print("Warning: No Jaccard distances available, continuing without them")
        
        # Merge the distance DataFrames on Label1 and Label2
        merged_df = pd.merge(filtered_distances_df, jaccard_distances_df, on=['Label1', 'Label2'], how='left')
        if 'JaccardDistance' not in merged_df.columns:
            merged_df['JaccardDistance'] = np.nan
        
        # Create output records for the comprehensive analysis
        output_records = []
        
        # Parse labels and extract relevant information
        for _, row in merged_df.iterrows():
            # Parse the labels to get noun, languages, etc.
            try:
                label1_parts = row["Label1"].split(", ")
                label2_parts = row["Label2"].split(", ")
                
                if len(label1_parts) >= 5 and len(label2_parts) >= 5:
                    noun = label1_parts[0]  # Same for both labels
                    lang1 = label1_parts[1]
                    lang2 = label2_parts[1]
                    proficiency = label1_parts[2]  # Same for both labels
                    prompt = label1_parts[3]  # Same for both labels
                    noun_category = label1_parts[4]  # Same for both labels
                    
                    # Find adjectives for these languages and this noun
                    adj1_rows = df[
                        (df["Noun"] == noun) & 
                        (df["Language"] == lang1) & 
                        (df["Proficiency"] == proficiency) & 
                        (df["Prompt"] == prompt)
                    ]
                    
                    adj2_rows = df[
                        (df["Noun"] == noun) & 
                        (df["Language"] == lang2) & 
                        (df["Proficiency"] == proficiency) & 
                        (df["Prompt"] == prompt)
                    ]
                    
                    # Extract and format adjectives
                    adj1_list = []
                    adj2_list = []
                    
                    if not adj1_rows.empty and 'Adjectives' in adj1_rows.columns:
                        adj1 = adj1_rows.iloc[0]['Adjectives']
                        if isinstance(adj1, list):
                            adj1_list = adj1
                        elif isinstance(adj1, str):
                            adj1_list = adj1.split(';')
                    
                    if not adj2_rows.empty and 'Adjectives' in adj2_rows.columns:
                        adj2 = adj2_rows.iloc[0]['Adjectives']
                        if isinstance(adj2, list):
                            adj2_list = adj2
                        elif isinstance(adj2, str):
                            adj2_list = adj2.split(';')
                    
                    # Create a record with all information
                    record = {
                        'Noun': noun,
                        'Language1': lang1,
                        'Language2': lang2,
                        'NounCategory': noun_category,
                        'Proficiency': proficiency,
                        'Prompt': prompt,
                        'CosineDistance': row.get('CosineDistance', np.nan),
                        'JaccardDistance': row.get('JaccardDistance', np.nan),
                        'Adjectives1': ';'.join(adj1_list) if adj1_list else '',
                        'Adjectives2': ';'.join(adj2_list) if adj2_list else ''
                    }
                    
                    # Try to add language-level embedding distance if we have it
                    # First, check if we have a way to get language embeddings
                    has_language_embeddings = False
                    
                    # Try to compute language-level embedding distance
                    lang_emb_dist = np.nan
                    
                    # Option 1: Check if we have language-level embedding rows in the original dataset
                    lang1_rows = df[df["Language"] == lang1]
                    lang2_rows = df[df["Language"] == lang2]
                    
                    if not lang1_rows.empty and not lang2_rows.empty and 'Embedding' in lang1_rows.columns and 'Embedding' in lang2_rows.columns:
                        # We have embeddings for both languages, compute their average embeddings
                        try:
                            # Get all embeddings for each language
                            lang1_embeddings = np.vstack(lang1_rows['Embedding'].dropna())
                            lang2_embeddings = np.vstack(lang2_rows['Embedding'].dropna())
                            
                            # Compute average embeddings for each language
                            if len(lang1_embeddings) > 0 and len(lang2_embeddings) > 0:
                                lang1_avg_embedding = np.mean(lang1_embeddings, axis=0)
                                lang2_avg_embedding = np.mean(lang2_embeddings, axis=0)
                                
                                # Compute language embedding distance (1 - cosine similarity)
                                norm1 = np.linalg.norm(lang1_avg_embedding)
                                norm2 = np.linalg.norm(lang2_avg_embedding)
                                
                                if norm1 > 0 and norm2 > 0:
                                    cosine_sim = np.dot(lang1_avg_embedding, lang2_avg_embedding) / (norm1 * norm2)
                                    lang_emb_dist = 1.0 - cosine_sim
                                    has_language_embeddings = True
                        except Exception as e_emb:
                            print(f"Error computing language embedding distance: {e_emb}")
                    
                    # Option 2: Check if AvgNounDistance is available as a proxy for language embedding distance
                    # This would be added here if it's available from a language comparison dataset
                    
                    # Add language embedding distance if we were able to compute it
                    if has_language_embeddings:
                        record['LanguageEmbeddingDistance'] = lang_emb_dist
                    
                    # Add AvgNounDistance if we computed it
                    avg_noun_distance = np.nan
                    
                    # Try to compute average noun distance based on CosineDistance values for this language pair
                    try:
                        # Get all rows for this language pair
                        lang_pair_rows = merged_df[
                            ((merged_df['Label1'].str.contains(f", {lang1}, ")) & 
                             (merged_df['Label2'].str.contains(f", {lang2}, "))) |
                            ((merged_df['Label1'].str.contains(f", {lang2}, ")) & 
                             (merged_df['Label2'].str.contains(f", {lang1}, ")))
                        ]
                        
                        if not lang_pair_rows.empty and 'CosineDistance' in lang_pair_rows.columns:
                            avg_noun_distance = lang_pair_rows['CosineDistance'].mean()
                    except Exception as e_avg:
                        print(f"Error computing average noun distance: {e_avg}")
                    
                    # Add AvgNounDistance
                    record['AvgNounDistance'] = avg_noun_distance
                    
                    # Add record to output
                    output_records.append(record)
            except Exception as e:
                print(f"Error parsing labels: {e}")
                continue
        
        # Create DataFrame from records
        output_df = pd.DataFrame(output_records)
        
        # If both LanguageEmbeddingDistance and AvgNounDistance are not present, 
        # use AvgNounDistance to fill LanguageEmbeddingDistance
        if (('LanguageEmbeddingDistance' not in output_df.columns or 
             output_df['LanguageEmbeddingDistance'].isna().all()) and
            'AvgNounDistance' in output_df.columns and 
            not output_df['AvgNounDistance'].isna().all()):
            
            print("Using AvgNounDistance as proxy for LanguageEmbeddingDistance")
            output_df['LanguageEmbeddingDistance'] = output_df['AvgNounDistance']
        
        # Sort the DataFrame for better readability
        sort_cols = ['NounCategory', 'Noun', 'Language1', 'Language2']
        if 'Proficiency' in output_df.columns:
            sort_cols.insert(3, 'Proficiency')
        if 'Prompt' in output_df.columns:
            sort_cols.insert(4, 'Prompt')
        
        output_df = output_df.sort_values(by=sort_cols)
        
        # Save to CSV if path provided
        if output_file and not output_df.empty:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            output_df.to_csv(output_file, index=False)
            print(f"Comprehensive analysis saved to: {output_file} ({len(output_df)} rows)")
        
        return output_df
    except Exception as e:
        print(f"Error generating comprehensive analysis: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def process_and_save_consolidated_analysis(all_comprehensive_analyses, base_filename, distances_dir):
    """
    Processes a list of comprehensive analysis DataFrames, consolidates them,
    and saves the result to a CSV file.

    Args:
        all_comprehensive_analyses (list): List of pandas DataFrames from individual analyses.
        base_filename (str): Base filename for the output consolidated CSV.
        distances_dir (str): Base directory for saving distance-related files.

    Returns:
        pd.DataFrame: The consolidated and processed DataFrame, or an empty DataFrame if an error occurs
                      or no data is available.
    """
    if not all_comprehensive_analyses:
        print("\nWarning: No comprehensive analyses were generated, cannot produce consolidated file.")
        return pd.DataFrame()

    print("\nGenerating consolidated analysis file across all proficiency levels...")
    try:
        consolidated_df = pd.concat(all_comprehensive_analyses, ignore_index=True)

        # Remove duplicates if any
        # Ensure all key columns for duplicate checking exist
        subset_cols = ['Language1', 'Language2', 'NounCategory', 'Noun', 'Proficiency', 'Prompt']
        existing_subset_cols = [col for col in subset_cols if col in consolidated_df.columns]
        if not existing_subset_cols:
            print("Warning: Key columns for duplicate checking are missing in consolidated data. Skipping duplicate removal.")
        else:
            consolidated_df = consolidated_df.drop_duplicates(subset=existing_subset_cols)

        # Sort by language, category, noun, proficiency for better readability
        # Ensure all key columns for sorting exist
        sort_by_cols = ["Language1", "Language2", "NounCategory", "Noun", "Proficiency", "Prompt"]
        existing_sort_cols = [col for col in sort_by_cols if col in consolidated_df.columns]
        if existing_sort_cols:
            consolidated_df.sort_values(
                by=existing_sort_cols,
                # All columns are sorted ascending by default, so this is fine.
                # ascending=[True]*len(existing_sort_cols), 
                inplace=True
            )
        else:
            print("Warning: Key columns for sorting are missing. Output may not be optimally sorted.")

        # Ensure Proficiency column is included and visible
        if 'Proficiency' not in consolidated_df.columns:
            print("Warning: Proficiency column missing from consolidated data, adding default value 'Unknown'.")
            consolidated_df['Proficiency'] = 'Unknown'
        
        # Move Proficiency column to a more prominent position if other key columns are present
        columns = consolidated_df.columns.tolist()
        if 'Proficiency' in columns:
            columns.remove('Proficiency')
            # Insert after NounCategory and Noun
            # Check if 'Noun' exists first to avoid error if it's missing
            if 'Noun' in columns:
                noun_idx = columns.index('Noun')
                columns.insert(noun_idx + 1, 'Proficiency')
            elif 'NounCategory' in columns: # If no 'Noun', try 'NounCategory'
                noun_category_idx = columns.index('NounCategory')
                columns.insert(noun_category_idx + 1, 'Proficiency')
            else: # If neither exists, insert at the beginning
                columns.insert(0, 'Proficiency')
            consolidated_df = consolidated_df[columns]

        consolidated_filename = f"all_proficiencies_consolidated_{base_filename}.csv"
        # Ensure the target directory for comprehensive analysis exists
        comprehensive_output_dir = os.path.join(distances_dir, "comprehensive_analysis")
        os.makedirs(comprehensive_output_dir, exist_ok=True)
        consolidated_file_path = os.path.join(comprehensive_output_dir, consolidated_filename)
        
        consolidated_df.to_csv(consolidated_file_path, index=False)
        print(f"Generated consolidated analysis file with {len(consolidated_df)} rows: {consolidated_file_path}")
        return consolidated_df

    except Exception as e:
        print(f"Error processing or saving consolidated analysis: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
