"""
Statistical analysis functions for Language Representation Analysis.

This module contains functions for performing statistical analyses on language data,
including ANOVA, Tukey's HSD, effect size calculations, and more.
"""

import numpy as np
import pandas as pd
import re
import os
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy import stats


def calculate_cohens_d(group1, group2, ddof=1):
    """
    Calculates Cohen's d for independent samples.

    Args:
        group1 (pd.Series or np.array): Data for the first group.
        group2 (pd.Series or np.array): Data for the second group.
        ddof (int): Degrees of freedom correction for std calculation.

    Returns:
        float: Cohen's d effect size.
    """
    try:
        # Handle series objects
        if isinstance(group1, pd.Series):
            group1 = group1.values
        if isinstance(group2, pd.Series):
            group2 = group2.values
        
        # Check for any non-numeric values
        if not np.issubdtype(group1.dtype, np.number) or not np.issubdtype(group2.dtype, np.number):
            print(f"Warning: Non-numeric values found in inputs to calculate_cohens_d")
            # Attempt to convert to numeric, replacing non-numeric values with NaN
            group1 = pd.to_numeric(group1, errors='coerce').values
            group2 = pd.to_numeric(group2, errors='coerce').values
        
        # Filter nan values
        group1 = group1[~np.isnan(group1)]
        group2 = group2[~np.isnan(group2)]
        
        # Check if we have enough values
        if len(group1) < 2 or len(group2) < 2:
            print(f"Warning: Not enough values to calculate Cohen's d: Group 1: {len(group1)}, Group 2: {len(group2)}")
            return np.nan
        
        # Calculate means
        mean1 = np.mean(group1)
        mean2 = np.mean(group2)
        
        # Check for infinite values
        if not np.isfinite(mean1) or not np.isfinite(mean2):
            print(f"Warning: Infinite values found in means: mean1={mean1}, mean2={mean2}")
            return np.nan
        
        # Calculate standard deviations
        std1 = np.std(group1, ddof=ddof)
        std2 = np.std(group2, ddof=ddof)
        
        # Check for zero/NaN standard deviations
        if np.isnan(std1) or np.isnan(std2):
            print(f"Warning: NaN found in standard deviations: std1={std1}, std2={std2}")
            return np.nan
        
        # Handle case where both groups have zero standard deviation
        if std1 == 0 and std2 == 0:
            # If means are identical, return 0 (no effect)
            if mean1 == mean2:
                return 0
            # If means differ with zero variance, it's an infinite effect size
            # Return a large but finite number to avoid numerical issues
            return np.sign(mean1 - mean2) * 10.0
        
        # Calculate pooled standard deviation
        n1 = len(group1)
        n2 = len(group2)
        
        # Handle case where one group has zero standard deviation
        if std1 == 0:
            # Use the other group's standard deviation
            pooled_std = std2
        elif std2 == 0:
            # Use the other group's standard deviation
            pooled_std = std1
        else:
            # Use the normal pooled standard deviation formula
            pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        # If pooled_std is zero or very close to zero, avoid division by zero
        if pooled_std < 1e-10:
            if mean1 == mean2:
                return 0
            else:
                # Return a large but finite number with appropriate sign
                return np.sign(mean1 - mean2) * 10.0
        
        # Calculate Cohen's d
        d = (mean1 - mean2) / pooled_std
        
        # Handle extreme values
        if not np.isfinite(d):
            print(f"Warning: Non-finite effect size calculated: {d}. Capping at ±10.")
            return np.sign(d) * 10.0
        
        # Cap extremely large values to prevent numerical issues
        if abs(d) > 10.0:
            return np.sign(d) * 10.0
            
        return d
    except Exception as e:
        print(f"Error calculating Cohen's d: {e}")
        import traceback
        traceback.print_exc()
        return np.nan


def analyze_noun_type_distances(df, output_file_path_prefix=None, expect_single_category=False, language=None):
    """
    Analyzes distances by noun type/category using ANOVA and calculates effect sizes.
    
    Args:
        df (pd.DataFrame): DataFrame containing distance data.
        output_file_path_prefix (str, optional): Prefix for output file paths.
        expect_single_category (bool): Whether to expect just one category (for detailed analysis)
        language (str, optional): Language to filter for language-specific analysis
        
    Returns:
        tuple: (anova_cosine_result, anova_jaccard_result, effect_sizes_df)
    """
    if df.empty:
        print("Error: Empty dataframe provided to analyze_noun_type_distances")
        return None, None, pd.DataFrame()
    
    # Filter by language if specified
    if language is not None:
        # Filter rows where either Language1 or Language2 equals the specified language
        language_df = df[(df['Language1'] == language) | (df['Language2'] == language)]
        
        if language_df.empty:
            print(f"Warning: No data found for language '{language}' in noun type analysis")
            return None, None, pd.DataFrame()
        
        print(f"Analyzing noun type distances for language: '{language}' ({len(language_df)} rows)")
        df = language_df
    
    # Check if we have NounCategory column
    if 'NounCategory' not in df.columns:
        print("Error: NounCategory column missing from DataFrame")
        return None, None, pd.DataFrame()
    
    # Check if we have enough unique categories for ANOVA
    if df['NounCategory'].nunique() < 2:
        if expect_single_category:
            print(f"Note: Single category analysis - skipping ANOVA, will calculate statistics")
            
            # Create a simplified result for single category case
            if output_file_path_prefix:
                # Ensure the directory exists
                if os.path.dirname(output_file_path_prefix):
                    os.makedirs(os.path.dirname(output_file_path_prefix), exist_ok=True)
                
                # Generate basic statistics for the single category
                single_cat = df['NounCategory'].iloc[0]
                summary_stats = df.groupby('NounCategory').agg({
                    'CosineDistance': ['count', 'mean', 'std', 'min', 'max'] if 'CosineDistance' in df.columns else [],
                    'JaccardDistance': ['count', 'mean', 'std', 'min', 'max'] if 'JaccardDistance' in df.columns else []
                }).round(4)
                
                # Save statistics
                single_cat_file = f"{output_file_path_prefix}_single_category_summary.csv"
                summary_stats.to_csv(single_cat_file)
                print(f"Single category statistics saved to: {single_cat_file}")
                
                with open(f"{output_file_path_prefix}_noun_type_analysis_summary.txt", 'w') as f:
                    f.write(f"Single category analysis for '{single_cat}':\n\n")
                    f.write(summary_stats.to_string())
            
            return None, None, pd.DataFrame()
        else:
            print(f"Warning: Not enough unique noun categories for ANOVA (found {df['NounCategory'].nunique()}, need at least 2)")
            return None, None, pd.DataFrame()
    
    # Prepare data for analysis
    significant_pairs_cosine = []
    significant_pairs_jaccard = []
    effect_size_records = []
    
    # Create a report
    report_lines = []
    if language is not None:
        report_lines.append(f"Noun Type Analysis for Language '{language}':")
    else:
        report_lines.append("Noun Type Analysis:")
    
    # --------------------------------------
    # Analyze Cosine distances if available
    # --------------------------------------
    anova_cosine_result = None
    p_value_cosine = 1.0  # Default to non-significant
    
    if 'CosineDistance' in df.columns:
        # Ensure data is valid for analysis
        df_cosine = df.dropna(subset=['NounCategory', 'CosineDistance']).copy()
        
        if len(df_cosine) < 2:
            print("Warning: Insufficient data for Cosine distance analysis after removing NaNs")
            anova_cosine_result = None
        else:
            try:
                # Run ANOVA
                formula = "CosineDistance ~ C(NounCategory)"
                model = ols(formula, data=df_cosine).fit()
                anova_cosine_result = sm.stats.anova_lm(model, typ=2)
                
                # Get p-value, safely handling potential changes in output format
                if "PR(>F)" in anova_cosine_result.columns and len(anova_cosine_result) > 0:
                    p_value_cosine = anova_cosine_result["PR(>F)"].iloc[0]
                    
                    report_lines.append("\nCosine Distance ANOVA:")
                    report_lines.append(anova_cosine_result.to_string())
                    report_lines.append(f"\nSignificance: p = {p_value_cosine:.4f}")
                    
                    if p_value_cosine < 0.05:
                        report_lines.append("Result: Significant differences found between noun categories")
                    else:
                        report_lines.append("Result: No significant differences found between noun categories")
                else:
                    print("Warning: Unexpected ANOVA result format for Cosine distances")
                    report_lines.append("\nWarning: Could not extract p-value from ANOVA results")
                
                # Save ANOVA results
                if output_file_path_prefix:
                    # Ensure the directory exists
                    if os.path.dirname(output_file_path_prefix):
                        os.makedirs(os.path.dirname(output_file_path_prefix), exist_ok=True)
                    
                    anova_cosine_file = f"{output_file_path_prefix}_anova_cosine.csv"
                    try:
                        anova_cosine_result.to_csv(anova_cosine_file)
                        print(f"ANOVA results for Cosine distances saved to: {anova_cosine_file}")
                    except Exception as e_save:
                        print(f"Error saving ANOVA results to {anova_cosine_file}: {e_save}")
                
                # If ANOVA is significant, perform post-hoc tests
                if p_value_cosine < 0.05:
                    try:
                        print("\nPerforming Tukey's HSD for Cosine distances...")
                        
                        # First check if we have categories with sufficient samples for Tukey's HSD
                        category_counts = df_cosine['NounCategory'].value_counts()
                        valid_categories = category_counts[category_counts >= 2].index.tolist()
                        
                        if len(valid_categories) < 2:
                            report_lines.append("Not enough categories with sufficient data for Tukey's HSD (need at least 2 observations per category)")
                        else:
                            # Filter to only include categories with enough data
                            tukey_data = df_cosine[df_cosine['NounCategory'].isin(valid_categories)]
                            
                            tukey_cosine = pairwise_tukeyhsd(
                                tukey_data['CosineDistance'], 
                                tukey_data['NounCategory'],
                                alpha=0.05
                            )
                            print(tukey_cosine)
                            
                            # Extract significant pairs - safely handling different result formats
                            try:
                                # Safely create DataFrame from results table
                                if hasattr(tukey_cosine, '_results_table') and hasattr(tukey_cosine._results_table, 'data'):
                                    if len(tukey_cosine._results_table.data) > 1:
                                        header = tukey_cosine._results_table.data[0]
                                        if all(col in header for col in ['group1', 'group2', 'reject', 'meandiff', 'p-adj']):
                                            tukey_df = pd.DataFrame(data=tukey_cosine._results_table.data[1:], 
                                                                  columns=tukey_cosine._results_table.data[0])
                                            significant_pairs = tukey_df[tukey_df['reject']]
                                            
                                            if not significant_pairs.empty:
                                                significant_count = len(significant_pairs)
                                                report_lines.append(f"Significant pairs ({significant_count}):")
                                                
                                                # Add each significant pair to the report with effect size
                                                for _, row in significant_pairs.iterrows():
                                                    cat1 = row['group1']
                                                    cat2 = row['group2']
                                                    diff = row['meandiff']
                                                    p_adj = row['p-adj']
                                                    
                                                    # Calculate Cohen's d
                                                    cat1_vals = df_cosine[df_cosine['NounCategory'] == cat1]['CosineDistance']
                                                    cat2_vals = df_cosine[df_cosine['NounCategory'] == cat2]['CosineDistance']
                                                    
                                                    if len(cat1_vals) >= 2 and len(cat2_vals) >= 2:
                                                        try:
                                                            cohens_d = calculate_cohens_d(cat1_vals, cat2_vals)
                                                            
                                                            if np.isnan(cohens_d):
                                                                effect_interp = "N/A"
                                                            elif abs(cohens_d) < 0.2:
                                                                effect_interp = "negligible effect"
                                                            elif abs(cohens_d) < 0.5:
                                                                effect_interp = "small effect"
                                                            elif abs(cohens_d) < 0.8:
                                                                effect_interp = "medium effect"
                                                            else:
                                                                effect_interp = "large effect"
                                                            
                                                            pair_line = f"  {cat1} vs {cat2}: diff={diff:.4f}, p-adj={p_adj:.4f}, Cohen's d={cohens_d:.2f} ({effect_interp})"
                                                            report_lines.append(pair_line)
                                                            significant_pairs_cosine.append((cat1, cat2, diff, p_adj, cohens_d, effect_interp))
                                                            
                                                            # Save the effect size for later use
                                                            effect_size_records.append({
                                                                'Category1': cat1,
                                                                'Category2': cat2,
                                                                'DistanceType': 'CosineDistance',
                                                                'EffectSize': cohens_d,
                                                                'Category1Mean': cat1_vals.mean(),
                                                                'Category2Mean': cat2_vals.mean(),
                                                                'Category1Count': len(cat1_vals),
                                                                'Category2Count': len(cat2_vals)
                                                            })
                                                        except Exception as e_cohens:
                                                            print(f"Error calculating Cohen's d for {cat1} vs {cat2}: {e_cohens}")
                                                            report_lines.append(f"  {cat1} vs {cat2}: diff={diff:.4f}, p-adj={p_adj:.4f}, Cohen's d=N/A (error in calculation)")
                                                    else:
                                                        report_lines.append(f"  {cat1} vs {cat2}: diff={diff:.4f}, p-adj={p_adj:.4f}, Cohen's d=N/A (insufficient data)")
                                            else:
                                                report_lines.append("No significant pairs found in Tukey's HSD test")
                                        else:
                                            report_lines.append("Could not parse Tukey's HSD results: expected columns missing")
                                    else:
                                        report_lines.append("Tukey's HSD results contain insufficient data")
                                else:
                                    report_lines.append("Could not access Tukey's HSD results table")
                            except Exception as e_parse:
                                print(f"Error parsing Tukey's HSD results: {e_parse}")
                                report_lines.append(f"Error parsing Tukey's HSD results: {e_parse}")
                        
                            if output_file_path_prefix:
                                tukey_cosine_file = f"{output_file_path_prefix}_tukey_cosine.txt"
                                try:
                                    with open(tukey_cosine_file, 'w') as f:
                                        f.write(str(tukey_cosine))
                                    print(f"Tukey's HSD results for Cosine distances saved to: {tukey_cosine_file}")
                                except Exception as e_save:
                                    print(f"Error saving Tukey's HSD results to {tukey_cosine_file}: {e_save}")
                    except Exception as e_tukey:
                        print(f"Error performing Tukey's HSD for Cosine distances: {e_tukey}")
                        report_lines.append(f"Error performing Tukey's HSD: {e_tukey}")
            except Exception as e_anova:
                print(f"Error performing ANOVA for Cosine distances: {e_anova}")
                report_lines.append(f"Error performing ANOVA: {e_anova}")
                anova_cosine_result = None
    else:
        report_lines.append("\nNote: CosineDistance not found in data, skipping Cosine distance analysis")
    
    # -----------------------------------------
    # Analyze Jaccard distances if available
    # -----------------------------------------
    anova_jaccard_result = None
    p_value_jaccard = 1.0  # Default to non-significant
    
    if 'JaccardDistance' in df.columns:
        # Ensure data is valid for analysis
        df_jaccard = df.dropna(subset=['NounCategory', 'JaccardDistance']).copy()
        
        if len(df_jaccard) < 2:
            print("Warning: Insufficient data for Jaccard distance analysis after removing NaNs")
            anova_jaccard_result = None
        else:
            try:
                # Run ANOVA
                formula = "JaccardDistance ~ C(NounCategory)"
                model = ols(formula, data=df_jaccard).fit()
                anova_jaccard_result = sm.stats.anova_lm(model, typ=2)
                
                # Get p-value, safely handling potential changes in output format
                if "PR(>F)" in anova_jaccard_result.columns and len(anova_jaccard_result) > 0:
                    p_value_jaccard = anova_jaccard_result["PR(>F)"].iloc[0]
                    
                    report_lines.append("\nJaccard Distance ANOVA:")
                    report_lines.append(anova_jaccard_result.to_string())
                    report_lines.append(f"\nSignificance: p = {p_value_jaccard:.4f}")
                    
                    if p_value_jaccard < 0.05:
                        report_lines.append("Result: Significant differences found between noun categories")
                    else:
                        report_lines.append("Result: No significant differences found between noun categories")
                else:
                    print("Warning: Unexpected ANOVA result format for Jaccard distances")
                    report_lines.append("\nWarning: Could not extract p-value from ANOVA results")
                
                # Save ANOVA results
                if output_file_path_prefix:
                    # Ensure the directory exists
                    if os.path.dirname(output_file_path_prefix):
                        os.makedirs(os.path.dirname(output_file_path_prefix), exist_ok=True)
                    
                    anova_jaccard_file = f"{output_file_path_prefix}_anova_jaccard.csv"
                    try:
                        anova_jaccard_result.to_csv(anova_jaccard_file)
                        print(f"ANOVA results for Jaccard distances saved to: {anova_jaccard_file}")
                    except Exception as e_save:
                        print(f"Error saving ANOVA results to {anova_jaccard_file}: {e_save}")
                
                # If ANOVA is significant, perform post-hoc tests
                if p_value_jaccard < 0.05:
                    try:
                        print("\nPerforming Tukey's HSD for Jaccard distances...")
                        
                        # First check if we have categories with sufficient samples for Tukey's HSD
                        category_counts = df_jaccard['NounCategory'].value_counts()
                        valid_categories = category_counts[category_counts >= 2].index.tolist()
                        
                        if len(valid_categories) < 2:
                            report_lines.append("Not enough categories with sufficient data for Tukey's HSD (need at least 2 observations per category)")
                        else:
                            # Filter to only include categories with enough data
                            tukey_data = df_jaccard[df_jaccard['NounCategory'].isin(valid_categories)]
                            
                            tukey_jaccard = pairwise_tukeyhsd(
                                tukey_data['JaccardDistance'], 
                                tukey_data['NounCategory'],
                                alpha=0.05
                            )
                            print(tukey_jaccard)
                            
                            # Extract significant pairs - safely handling different result formats
                            try:
                                # Safely create DataFrame from results table
                                if hasattr(tukey_jaccard, '_results_table') and hasattr(tukey_jaccard._results_table, 'data'):
                                    if len(tukey_jaccard._results_table.data) > 1:
                                        header = tukey_jaccard._results_table.data[0]
                                        if all(col in header for col in ['group1', 'group2', 'reject', 'meandiff', 'p-adj']):
                                            tukey_df = pd.DataFrame(data=tukey_jaccard._results_table.data[1:], 
                                                                  columns=tukey_jaccard._results_table.data[0])
                                            significant_pairs = tukey_df[tukey_df['reject']]
                                            
                                            if not significant_pairs.empty:
                                                significant_count = len(significant_pairs)
                                                report_lines.append(f"Significant pairs ({significant_count}):")
                                                
                                                # Add each significant pair to the report with effect size
                                                for _, row in significant_pairs.iterrows():
                                                    cat1 = row['group1']
                                                    cat2 = row['group2']
                                                    diff = row['meandiff']
                                                    p_adj = row['p-adj']
                                                    
                                                    # Calculate Cohen's d
                                                    cat1_vals = df_jaccard[df_jaccard['NounCategory'] == cat1]['JaccardDistance']
                                                    cat2_vals = df_jaccard[df_jaccard['NounCategory'] == cat2]['JaccardDistance']
                                                    
                                                    if len(cat1_vals) >= 2 and len(cat2_vals) >= 2:
                                                        try:
                                                            cohens_d = calculate_cohens_d(cat1_vals, cat2_vals)
                                                            
                                                            if np.isnan(cohens_d):
                                                                effect_interp = "N/A"
                                                            elif abs(cohens_d) < 0.2:
                                                                effect_interp = "negligible effect"
                                                            elif abs(cohens_d) < 0.5:
                                                                effect_interp = "small effect"
                                                            elif abs(cohens_d) < 0.8:
                                                                effect_interp = "medium effect"
                                                            else:
                                                                effect_interp = "large effect"
                                                            
                                                            pair_line = f"  {cat1} vs {cat2}: diff={diff:.4f}, p-adj={p_adj:.4f}, Cohen's d={cohens_d:.2f} ({effect_interp})"
                                                            report_lines.append(pair_line)
                                                            significant_pairs_jaccard.append((cat1, cat2, diff, p_adj, cohens_d, effect_interp))
                                                            
                                                            # Save the effect size for later use
                                                            effect_size_records.append({
                                                                'Category1': cat1,
                                                                'Category2': cat2,
                                                                'DistanceType': 'JaccardDistance',
                                                                'EffectSize': cohens_d,
                                                                'Category1Mean': cat1_vals.mean(),
                                                                'Category2Mean': cat2_vals.mean(),
                                                                'Category1Count': len(cat1_vals),
                                                                'Category2Count': len(cat2_vals)
                                                            })
                                                        except Exception as e_cohens:
                                                            print(f"Error calculating Cohen's d for {cat1} vs {cat2}: {e_cohens}")
                                                            report_lines.append(f"  {cat1} vs {cat2}: diff={diff:.4f}, p-adj={p_adj:.4f}, Cohen's d=N/A (error in calculation)")
                                                    else:
                                                        report_lines.append(f"  {cat1} vs {cat2}: diff={diff:.4f}, p-adj={p_adj:.4f}, Cohen's d=N/A (insufficient data)")
                                            else:
                                                report_lines.append("No significant pairs found in Tukey's HSD test")
                                        else:
                                            report_lines.append("Could not parse Tukey's HSD results: expected columns missing")
                                    else:
                                        report_lines.append("Tukey's HSD results contain insufficient data")
                                else:
                                    report_lines.append("Could not access Tukey's HSD results table")
                            except Exception as e_parse:
                                print(f"Error parsing Tukey's HSD results: {e_parse}")
                                report_lines.append(f"Error parsing Tukey's HSD results: {e_parse}")
                            
                            if output_file_path_prefix:
                                tukey_jaccard_file = f"{output_file_path_prefix}_tukey_jaccard.txt"
                                try:
                                    with open(tukey_jaccard_file, 'w') as f:
                                        f.write(str(tukey_jaccard))
                                    print(f"Tukey's HSD results for Jaccard distances saved to: {tukey_jaccard_file}")
                                except Exception as e_save:
                                    print(f"Error saving Tukey's HSD results to {tukey_jaccard_file}: {e_save}")
                    except Exception as e_tukey:
                        print(f"Error performing Tukey's HSD for Jaccard distances: {e_tukey}")
                        report_lines.append(f"Error performing Tukey's HSD: {e_tukey}")
            except Exception as e_anova:
                print(f"Error performing ANOVA for Jaccard distances: {e_anova}")
                report_lines.append(f"Error performing ANOVA: {e_anova}")
                anova_jaccard_result = None
    else:
        report_lines.append("\nNote: JaccardDistance not found in data, skipping Jaccard distance analysis")
    
    # Save the report
    if output_file_path_prefix:
        # Ensure the directory exists
        if os.path.dirname(output_file_path_prefix):
            os.makedirs(os.path.dirname(output_file_path_prefix), exist_ok=True)
        
        report_file = f"{output_file_path_prefix}_noun_type_analysis_summary.txt"
        try:
            with open(report_file, 'w') as f:
                f.write("\n".join(report_lines))
            print(f"Noun type analysis summary saved to: {report_file}")
        except Exception as e_report:
            print(f"Error saving noun type analysis summary to {report_file}: {e_report}")
    
    # Calculate additional effect sizes between all category pairs (not just significant ones)
    categories = df['NounCategory'].unique()
    
    # Loop through all category pairs to calculate effect sizes
    for i, cat1 in enumerate(categories):
        for j in range(i+1, len(categories)):
            cat2 = categories[j]
            
            # Cosine distance effect size (if available)
            if 'CosineDistance' in df.columns:
                cat1_cosine = df[df['NounCategory'] == cat1]['CosineDistance'].dropna()
                cat2_cosine = df[df['NounCategory'] == cat2]['CosineDistance'].dropna()
                
                if len(cat1_cosine) >= 2 and len(cat2_cosine) >= 2:
                    try:
                        cosine_d = calculate_cohens_d(cat1_cosine, cat2_cosine)
                        
                        # Check if we already have this pair from significant results
                        if not any(rec['Category1'] == cat1 and rec['Category2'] == cat2 and rec['DistanceType'] == 'CosineDistance' for rec in effect_size_records):
                            effect_size_records.append({
                                'Category1': cat1,
                                'Category2': cat2,
                                'DistanceType': 'CosineDistance',
                                'EffectSize': cosine_d,
                                'Category1Mean': cat1_cosine.mean(),
                                'Category2Mean': cat2_cosine.mean(),
                                'Category1Count': len(cat1_cosine),
                                'Category2Count': len(cat2_cosine)
                            })
                    except Exception as e_cosine:
                        print(f"Error calculating Cohen's d for CosineDistance between {cat1} and {cat2}: {e_cosine}")
            
            # Jaccard distance effect size (if available)
            if 'JaccardDistance' in df.columns:
                cat1_jaccard = df[df['NounCategory'] == cat1]['JaccardDistance'].dropna()
                cat2_jaccard = df[df['NounCategory'] == cat2]['JaccardDistance'].dropna()
                
                if len(cat1_jaccard) >= 2 and len(cat2_jaccard) >= 2:
                    try:
                        jaccard_d = calculate_cohens_d(cat1_jaccard, cat2_jaccard)
                        
                        # Check if we already have this pair from significant results
                        if not any(rec['Category1'] == cat1 and rec['Category2'] == cat2 and rec['DistanceType'] == 'JaccardDistance' for rec in effect_size_records):
                            effect_size_records.append({
                                'Category1': cat1,
                                'Category2': cat2,
                                'DistanceType': 'JaccardDistance',
                                'EffectSize': jaccard_d,
                                'Category1Mean': cat1_jaccard.mean(),
                                'Category2Mean': cat2_jaccard.mean(),
                                'Category1Count': len(cat1_jaccard),
                                'Category2Count': len(cat2_jaccard)
                            })
                    except Exception as e_jaccard:
                        print(f"Error calculating Cohen's d for JaccardDistance between {cat1} and {cat2}: {e_jaccard}")
            
            # Language embedding distance effect size (if available)
            if 'LanguageEmbeddingDistance' in df.columns:
                cat1_lang_emb = df[df['NounCategory'] == cat1]['LanguageEmbeddingDistance'].dropna()
                cat2_lang_emb = df[df['NounCategory'] == cat2]['LanguageEmbeddingDistance'].dropna()
                
                if len(cat1_lang_emb) >= 2 and len(cat2_lang_emb) >= 2:
                    try:
                        lang_emb_d = calculate_cohens_d(cat1_lang_emb, cat2_lang_emb)
                        effect_size_records.append({
                            'Category1': cat1,
                            'Category2': cat2,
                            'DistanceType': 'LanguageEmbeddingDistance',
                            'EffectSize': lang_emb_d,
                            'Category1Mean': cat1_lang_emb.mean(),
                            'Category2Mean': cat2_lang_emb.mean(),
                            'Category1Count': len(cat1_lang_emb),
                            'Category2Count': len(cat2_lang_emb)
                        })
                    except Exception as e_lang_emb:
                        print(f"Error calculating Cohen's d for LanguageEmbeddingDistance between {cat1} and {cat2}: {e_lang_emb}")
    
    # Create effect size DataFrame
    effect_sizes_df = pd.DataFrame(effect_size_records)
    
    # Add language info if filtered by language
    if language is not None and not effect_sizes_df.empty:
        effect_sizes_df['Language'] = language
    
    # Sort for readability
    if not effect_sizes_df.empty:
        effect_sizes_df['EffectSizeAbs'] = effect_sizes_df['EffectSize'].abs()
        effect_sizes_df.sort_values(by=['DistanceType', 'EffectSizeAbs'], ascending=[True, False], inplace=True)
        effect_sizes_df.drop(columns=['EffectSizeAbs'], inplace=True)
        
        # Save effect sizes
        if output_file_path_prefix:
            # Ensure the directory exists
            if os.path.dirname(output_file_path_prefix):
                os.makedirs(os.path.dirname(output_file_path_prefix), exist_ok=True)
            
            effects_file = f"{output_file_path_prefix}_effect_sizes.csv"
            try:
                effect_sizes_df.to_csv(effects_file, index=False)
                print(f"Effect sizes saved to: {effects_file}")
            except Exception as e_save:
                print(f"Error saving effect sizes to {effects_file}: {e_save}")
    
    return anova_cosine_result, anova_jaccard_result, effect_sizes_df


def calculate_category_percentages(df, expect_single_category=False, language=None):
    """
    Calculates statistics for noun categories including distances and percentiles.
    
    Args:
        df: DataFrame with distance data by noun category
        expect_single_category: Whether to expect just one category (for detailed analysis)
        language: Optional language to filter for language-specific analysis
        
    Returns:
        dict: Mapping of categories to statistics with all required data points
    """
    import os
    import numpy as np
    import pandas as pd
    
    if df.empty:
        print("Error: DataFrame is empty in calculate_category_percentages")
        return {}
    
    if 'NounCategory' not in df.columns:
        print("Error: Missing NounCategory column in calculate_category_percentages")
        # Check if we can add a default category
        df['NounCategory'] = "Unknown"
        print("Added 'Unknown' as default NounCategory")
    
    # Ensure we have the distance columns
    has_cosine = 'CosineDistance' in df.columns
    has_jaccard = 'JaccardDistance' in df.columns
    has_lang_emb_dist = 'LanguageEmbeddingDistance' in df.columns
    
    # If we have Language1 and Language2 but don't have LanguageEmbeddingDistance,
    # we might be able to calculate it if we also have language-level embeddings
    if not has_lang_emb_dist and 'Language1' in df.columns and 'Language2' in df.columns:
        # Find unique languages
        all_languages = set(df['Language1'].unique()) | set(df['Language2'].unique())
        print(f"Found {len(all_languages)} unique languages for LanguageEmbeddingDistance calculation")
        
        # Try to find any embeddings for these languages
        # Note: This is a placeholder. In practice, we need to have language embeddings pre-calculated.
        # If language embeddings are not available, we'll skip this calculation.
        
        # See if we can use AvgNounDistance as a proxy for LanguageEmbeddingDistance
        if 'AvgNounDistance' in df.columns:
            print("Using AvgNounDistance as LanguageEmbeddingDistance proxy")
            df['LanguageEmbeddingDistance'] = df['AvgNounDistance']
            has_lang_emb_dist = True
    
    if not has_cosine and not has_jaccard and not has_lang_emb_dist:
        print("Error: No distance metric columns found (CosineDistance, JaccardDistance, LanguageEmbeddingDistance)")
        return {}
    
    # Filter by language if specified
    if language is not None:
        # Check if we have language columns
        if 'Language1' in df.columns and 'Language2' in df.columns:
            # Filter rows where either Language1 or Language2 equals the specified language
            language_df = df[(df['Language1'] == language) | (df['Language2'] == language)]
            
            if language_df.empty:
                print(f"Warning: No data found for language '{language}'")
                return {}
            
            print(f"Analyzing category percentages for language: '{language}' ({len(language_df)} rows)")
            df = language_df
        else:
            print("Warning: Cannot filter by language - Language columns not found")
    
    # Calculate statistics for each category
    stats_records = []
    
    for category in df['NounCategory'].unique():
        cat_df = df[df['NounCategory'] == category]
        
        record = {'NounCategory': category}
        
        # Calculate statistics for LanguageEmbeddingDistance
        if has_lang_emb_dist:
            lang_emb_values = cat_df['LanguageEmbeddingDistance'].dropna()
            if not lang_emb_values.empty:
                record['LanguageEmbeddingDistance_mean'] = lang_emb_values.mean()
                record['LanguageEmbeddingDistance_min'] = lang_emb_values.min()
                record['LanguageEmbeddingDistance_max'] = lang_emb_values.max()
                record['LanguageEmbeddingDistance_count'] = len(lang_emb_values)
                record['LanguageEmbeddingDistance_std'] = lang_emb_values.std()
            else:
                record['LanguageEmbeddingDistance_mean'] = np.nan
                record['LanguageEmbeddingDistance_min'] = np.nan
                record['LanguageEmbeddingDistance_max'] = np.nan
                record['LanguageEmbeddingDistance_count'] = 0
                record['LanguageEmbeddingDistance_std'] = np.nan
        
        # Calculate statistics for CosineDistance
        if has_cosine:
            cosine_values = cat_df['CosineDistance'].dropna()
            if not cosine_values.empty:
                record['CosineDistance_mean'] = cosine_values.mean()
                record['CosineDistance_min'] = cosine_values.min()
                record['CosineDistance_max'] = cosine_values.max()
                record['CosineDistance_count'] = len(cosine_values)
                record['CosineDistance_std'] = cosine_values.std()
            else:
                record['CosineDistance_mean'] = np.nan
                record['CosineDistance_min'] = np.nan
                record['CosineDistance_max'] = np.nan
                record['CosineDistance_count'] = 0
                record['CosineDistance_std'] = np.nan
        
        # Calculate statistics for JaccardDistance
        if has_jaccard:
            jaccard_values = cat_df['JaccardDistance'].dropna()
            if not jaccard_values.empty:
                record['JaccardDistance_mean'] = jaccard_values.mean()
                record['JaccardDistance_min'] = jaccard_values.min()
                record['JaccardDistance_max'] = jaccard_values.max()
                record['JaccardDistance_count'] = len(jaccard_values)
                record['JaccardDistance_std'] = jaccard_values.std()
            else:
                record['JaccardDistance_mean'] = np.nan
                record['JaccardDistance_min'] = np.nan
                record['JaccardDistance_max'] = np.nan
                record['JaccardDistance_count'] = 0
                record['JaccardDistance_std'] = np.nan
        
        stats_records.append(record)
    
    # Convert to DataFrame for percentile calculations
    stats_df = pd.DataFrame(stats_records)
    
    if stats_df.empty:
        print("Warning: No statistics records generated")
        return {}
    
    # Handle single category case
    if len(stats_df) == 1 or expect_single_category:
        single_category = stats_df['NounCategory'].iloc[0]
        print(f"Processing single category analysis for '{single_category}'")
        
        # For single category, just create percentiles relative to theoretical range
        if has_lang_emb_dist:
            # LanguageEmbeddingDistance is also 0-2 in theory (1-cos_sim), with typical values 0-1
            stats_df['LanguageEmbeddingDistance_percentile'] = (stats_df['LanguageEmbeddingDistance_mean'] / 1.0) * 100
        
        if has_cosine:
            # Cosine distance is 0-2 in theory (1-cos_sim), with typical values around 0-1
            # So we'll calculate percentile based on theoretical max of 1
            stats_df['CosineDistance_percentile'] = (stats_df['CosineDistance_mean'] / 1.0) * 100
            
        if has_jaccard:
            # Jaccard is naturally 0-1, so we can use it directly as percentile
            stats_df['JaccardDistance_percentile'] = stats_df['JaccardDistance_mean'] * 100
        
        # Calculate combined uniqueness score
        if has_cosine and has_jaccard and has_lang_emb_dist:
            stats_df['combined_uniqueness_score'] = (
                stats_df['LanguageEmbeddingDistance_percentile'] +
                stats_df['CosineDistance_percentile'] + 
                stats_df['JaccardDistance_percentile']
            ) / 3
        elif has_cosine and has_jaccard:
            stats_df['combined_uniqueness_score'] = (
                stats_df['CosineDistance_percentile'] + 
                stats_df['JaccardDistance_percentile']
            ) / 2
        elif has_cosine and has_lang_emb_dist:
            stats_df['combined_uniqueness_score'] = (
                stats_df['LanguageEmbeddingDistance_percentile'] + 
                stats_df['CosineDistance_percentile']
            ) / 2
        elif has_jaccard and has_lang_emb_dist:
            stats_df['combined_uniqueness_score'] = (
                stats_df['LanguageEmbeddingDistance_percentile'] + 
                stats_df['JaccardDistance_percentile']
            ) / 2
    else:
        # Multiple categories - calculate percentiles relative to the group
        if has_lang_emb_dist:
            lang_emb_max = stats_df['LanguageEmbeddingDistance_mean'].max()
            lang_emb_min = stats_df['LanguageEmbeddingDistance_mean'].min()
            
            if lang_emb_max > lang_emb_min:
                stats_df['LanguageEmbeddingDistance_percentile'] = (
                    (stats_df['LanguageEmbeddingDistance_mean'] - lang_emb_min) / 
                    (lang_emb_max - lang_emb_min)
                ) * 100
            else:
                stats_df['LanguageEmbeddingDistance_percentile'] = 100.0  # All values are the same
                
        if has_cosine:
            cosine_max = stats_df['CosineDistance_mean'].max()
            cosine_min = stats_df['CosineDistance_mean'].min()
            
            if cosine_max > cosine_min:
                stats_df['CosineDistance_percentile'] = (
                    (stats_df['CosineDistance_mean'] - cosine_min) / 
                    (cosine_max - cosine_min)
                ) * 100
            else:
                stats_df['CosineDistance_percentile'] = 100.0  # All values are the same
        
        # Calculate percentiles for JaccardDistance_mean
        if has_jaccard:
            jaccard_max = stats_df['JaccardDistance_mean'].max()
            jaccard_min = stats_df['JaccardDistance_mean'].min()
            
            if jaccard_max > jaccard_min:
                stats_df['JaccardDistance_percentile'] = (
                    (stats_df['JaccardDistance_mean'] - jaccard_min) / 
                    (jaccard_max - jaccard_min)
                ) * 100
            else:
                stats_df['JaccardDistance_percentile'] = 100.0  # All values are the same
        
        # Calculate combined uniqueness score
        if has_cosine and has_jaccard and has_lang_emb_dist:
            stats_df['combined_uniqueness_score'] = (
                stats_df['LanguageEmbeddingDistance_percentile'] + 
                stats_df['CosineDistance_percentile'] + 
                stats_df['JaccardDistance_percentile']
            ) / 3
        elif has_cosine and has_jaccard:
            stats_df['combined_uniqueness_score'] = (
                stats_df['CosineDistance_percentile'] + 
                stats_df['JaccardDistance_percentile']
            ) / 2
        elif has_cosine and has_lang_emb_dist:
            stats_df['combined_uniqueness_score'] = (
                stats_df['LanguageEmbeddingDistance_percentile'] + 
                stats_df['CosineDistance_percentile']
            ) / 2
        elif has_jaccard and has_lang_emb_dist:
            stats_df['combined_uniqueness_score'] = (
                stats_df['LanguageEmbeddingDistance_percentile'] + 
                stats_df['JaccardDistance_percentile']
            ) / 2
    
    # Sort the results by combined uniqueness or just the first available distance metric
    if 'combined_uniqueness_score' in stats_df.columns:
        stats_df = stats_df.sort_values('combined_uniqueness_score', ascending=False)
    elif 'LanguageEmbeddingDistance_percentile' in stats_df.columns:
        stats_df = stats_df.sort_values('LanguageEmbeddingDistance_percentile', ascending=False)
    elif 'CosineDistance_percentile' in stats_df.columns:
        stats_df = stats_df.sort_values('CosineDistance_percentile', ascending=False)
    elif 'JaccardDistance_percentile' in stats_df.columns:
        stats_df = stats_df.sort_values('JaccardDistance_percentile', ascending=False)
    
    # Create a more structured result with the stats_df included
    result = {
        'total_pairs': len(df),
        'categories': {category: {} for category in df['NounCategory'].unique()},
        'stats_df': stats_df
    }
    
    # Add language info if filtered by language
    if language is not None:
        result['language'] = language
    
    # Update the categories with the calculated percentiles and combined score
    for _, row in stats_df.iterrows():
        cat_name = row['NounCategory']
        for col in stats_df.columns:
            if col != 'NounCategory':
                result['categories'][cat_name][col] = row[col]
    
    # Summary of statistics
    summary_lines = []
    if language is not None:
        summary_lines.append(f"Statistics for language '{language}', noun categories (total {len(df)} pairs):")
    else:
        summary_lines.append(f"Statistics for noun categories (total {len(df)} pairs):")
    
    for _, row in stats_df.iterrows():
        category = row['NounCategory']
        summary_line = f"  {category}: "
        
        if has_lang_emb_dist:
            summary_line += f"LanguageEmbeddingDistance mean={row.get('LanguageEmbeddingDistance_mean', np.nan):.4f}, "
        
        if has_cosine:
            summary_line += f"CosineDistance mean={row.get('CosineDistance_mean', np.nan):.4f}, "
        
        if has_jaccard:
            summary_line += f"JaccardDistance mean={row.get('JaccardDistance_mean', np.nan):.4f}, "
        
        count = (row.get('LanguageEmbeddingDistance_count', 0) or 
                row.get('CosineDistance_count', 0) or 
                row.get('JaccardDistance_count', 0))
        summary_line += f"count={count}"
        
        summary_lines.append(summary_line)
    
    result['summary'] = "\n".join(summary_lines)
    
    return result


def analyze_gender_language_distances(df, lang_gender_map, output_file_path_prefix=None, language=None):
    """
    Analyzes distances between languages based on grammatical gender.
    
    Args:
        df: DataFrame with language pair distances
        lang_gender_map: Mapping of language codes to grammatical gender systems
        output_file_path_prefix: Path prefix for output files
        language: Optional language to filter for language-specific analysis
        
    Returns:
        tuple: (gender_averages_df, gender_effects_df)
    """
    if df.empty:
        print("Error: Empty dataframe provided to analyze_gender_language_distances")
        return pd.DataFrame(), pd.DataFrame()
    
    if not lang_gender_map:
        print("Error: Empty language gender mapping provided")
        return pd.DataFrame(), pd.DataFrame()
    
    # Check if we have the required columns
    required_columns = ['Language1', 'Language2']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: Missing required columns. Required: {required_columns}")
        return pd.DataFrame(), pd.DataFrame()
    
    # Filter by language if specified
    if language is not None:
        # Filter rows where either Language1 or Language2 equals the specified language
        language_df = df[(df['Language1'] == language) | (df['Language2'] == language)]
        
        if language_df.empty:
            print(f"Warning: No data found for language '{language}' in gender language analysis")
            if output_file_path_prefix:
                # Ensure the directory exists
                if os.path.dirname(output_file_path_prefix):
                    os.makedirs(os.path.dirname(output_file_path_prefix), exist_ok=True)
                    
                with open(f"{output_file_path_prefix}_gender_analysis_warning.txt", 'w') as f:
                    f.write(f"No data found for language '{language}' in gender language analysis\n")
            return pd.DataFrame(), pd.DataFrame()
        
        print(f"Analyzing gender language distances for language: '{language}' ({len(language_df)} rows)")
        df = language_df
    
    # Pre-process language codes to extract just language code part
    df = df.copy()  # Create a copy to avoid modifying the original
    df['Language1Code'] = df['Language1'].apply(lambda x: x.split(' # ')[0] if isinstance(x, str) and ' # ' in x else x)
    df['Language2Code'] = df['Language2'].apply(lambda x: x.split(' # ')[0] if isinstance(x, str) and ' # ' in x else x)
    
    # Add gender info
    df['Gender1'] = df['Language1Code'].map(lang_gender_map)
    df['Gender2'] = df['Language2Code'].map(lang_gender_map)
    
    # Remove rows where either language's gender is unknown
    df_with_gender = df.dropna(subset=['Gender1', 'Gender2'])
    
    if df_with_gender.empty:
        print("Warning: No rows with gender information for both languages")
        if output_file_path_prefix:
            # Ensure the directory exists
            if os.path.dirname(output_file_path_prefix):
                os.makedirs(os.path.dirname(output_file_path_prefix), exist_ok=True)
                
            with open(f"{output_file_path_prefix}_gender_analysis_warning.txt", 'w') as f:
                if language is not None:
                    f.write(f"No rows with gender information for both languages for language '{language}'\n")
                else:
                    f.write("No rows with gender information for both languages\n")
        return pd.DataFrame(), pd.DataFrame()
    
    # Create gender pair column
    df_with_gender['GenderPair'] = df_with_gender.apply(
        lambda row: '-'.join(sorted([row['Gender1'], row['Gender2']])), axis=1
    )
    
    # Only keep rows with valid gender pairs
    valid_gender_pairs = df_with_gender['GenderPair'].dropna()
    if valid_gender_pairs.empty:
        print("Warning: No valid gender pairs found")
        if output_file_path_prefix:
            # Ensure the directory exists
            if os.path.dirname(output_file_path_prefix):
                os.makedirs(os.path.dirname(output_file_path_prefix), exist_ok=True)
                
            with open(f"{output_file_path_prefix}_gender_analysis_warning.txt", 'w') as f:
                if language is not None:
                    f.write(f"No valid gender pairs found for language '{language}'\n")
                else:
                    f.write("No valid gender pairs found\n")
        return pd.DataFrame(), pd.DataFrame()
    
    # Calculate averages for each gender pair
    gender_averages = []
    
    # Process Cosine distances if available
    if 'CosineDistance' in df_with_gender.columns:
        cosine_by_gender = df_with_gender.groupby('GenderPair')['CosineDistance'].agg(['mean', 'median', 'std', 'count']).reset_index()
        for _, row in cosine_by_gender.iterrows():
            gender_averages.append({
                'GenderPair': row['GenderPair'],
                'DistanceType': 'Cosine',
                'Mean': row['mean'],
                'Median': row['median'],
                'StdDev': row['std'],
                'Count': row['count']
            })
    
    # Process Jaccard distances if available
    if 'JaccardDistance' in df_with_gender.columns:
        jaccard_by_gender = df_with_gender.groupby('GenderPair')['JaccardDistance'].agg(['mean', 'median', 'std', 'count']).reset_index()
        for _, row in jaccard_by_gender.iterrows():
            gender_averages.append({
                'GenderPair': row['GenderPair'],
                'DistanceType': 'Jaccard',
                'Mean': row['mean'],
                'Median': row['median'],
                'StdDev': row['std'],
                'Count': row['count']
            })
    
    # Create output DataFrame for gender averages
    gender_averages_df = pd.DataFrame(gender_averages)
    
    if not gender_averages_df.empty:
        # Add language info if filtered by language
        if language is not None:
            gender_averages_df['Language'] = language
        
        # Sort for readability
        gender_averages_df.sort_values(by=['DistanceType', 'Mean'], inplace=True)
        
        # Save results
        if output_file_path_prefix:
            # Ensure the directory exists
            if os.path.dirname(output_file_path_prefix):
                os.makedirs(os.path.dirname(output_file_path_prefix), exist_ok=True)
                
            gender_averages_file = f"{output_file_path_prefix}_gender_averages.csv"
            try:
                gender_averages_df.to_csv(gender_averages_file, index=False)
                print(f"Gender averages saved to: {gender_averages_file}")
            except Exception as e_save:
                print(f"Error saving gender averages to {gender_averages_file}: {e_save}")
    
    # Calculate effect sizes between same and different gender pairs
    gender_effects = []
    unique_genders = set()
    
    # Get all unique genders
    for gender_pair in df_with_gender['GenderPair'].unique():
        if pd.notna(gender_pair):
            for gender in gender_pair.split('-'):
                unique_genders.add(gender)
    
    unique_genders = sorted(list(unique_genders))
    
    if len(unique_genders) < 2:
        print("Warning: Need at least 2 unique genders for effect size calculation")
        if output_file_path_prefix:
            # Ensure the directory exists
            if os.path.dirname(output_file_path_prefix):
                os.makedirs(os.path.dirname(output_file_path_prefix), exist_ok=True)
                
            with open(f"{output_file_path_prefix}_gender_analysis_warning.txt", 'w') as f:
                if language is not None:
                    f.write(f"Not enough unique genders for language '{language}' to perform effect size calculation (need at least 2, found {len(unique_genders)})\n")
                else:
                    f.write(f"Not enough unique genders to perform effect size calculation (need at least 2, found {len(unique_genders)})\n")
        return gender_averages_df, pd.DataFrame()
    
    print(f"\nCalculating effect sizes between gender pairs...")
    
    # Generate detailed text report for gender language analysis
    detailed_report = []
    if language is not None:
        detailed_report.append(f"Gender Language Analysis for Language '{language}':\n")
    else:
        detailed_report.append("Gender Language Analysis:\n")
    
    # Count genders
    gender_count_dict = {}
    for _, row in df_with_gender.iterrows():
        gender1 = row['Gender1']
        gender2 = row['Gender2']
        gender_count_dict[gender1] = gender_count_dict.get(gender1, 0) + 1
        gender_count_dict[gender2] = gender_count_dict.get(gender2, 0) + 1
    
    detailed_report.append("Grammatical Gender Category Counts:")
    for gender, count in sorted(gender_count_dict.items(), key=lambda x: x[1], reverse=True):
        detailed_report.append(f"  {gender}: {count} entries")
    detailed_report.append("\n")
    
    # Add ANOVA results for each distance type
    for distance_type in ['CosineDistance', 'JaccardDistance']:
        if distance_type in df_with_gender.columns:
            detailed_report.append(f"\n--- {distance_type} ---")
            
            # Prepare data for ANOVA for the current distance_type
            anova_data_for_ols = df_with_gender.dropna(subset=[distance_type, 'Gender1'])

            # Check if we have at least 2 unique gender categories with sufficient data
            num_unique_genders = anova_data_for_ols['Gender1'].nunique()
            group_counts = anova_data_for_ols['Gender1'].value_counts()
            valid_groups_for_anova = group_counts[group_counts >= 2]

            if num_unique_genders < 2 or len(valid_groups_for_anova) < 2:
                skip_sm_anova = True
                reason = ""
                if num_unique_genders < 2:
                    reason = f"Not enough unique gender categories for ANOVA (need at least 2, found {num_unique_genders} after NA removal for {distance_type})."
                elif len(valid_groups_for_anova) < 2:
                    reason = f"Not enough gender categories with at least 2 observations for ANOVA (found {len(valid_groups_for_anova)} such groups for {distance_type}). Counts: {group_counts.to_dict()}"
                
                detailed_report.append(reason)
                print(reason) # Also print to console for visibility

                # Attempt SciPy fallback if statsmodels ANOVA is skipped
                group_lists = [grp[distance_type].dropna().values for _, grp in anova_data_for_ols.groupby('Gender1') if len(grp.dropna(subset=[distance_type])) >= 2]
                if len(group_lists) >= 2:
                    try:
                        f_stat, p_val = stats.f_oneway(*group_lists)
                        fallback_msg = f"Fallback one-way ANOVA (SciPy): F = {f_stat:.4f}, p = {p_val:.6f}"
                        detailed_report.append(fallback_msg)
                        print(fallback_msg)
                    except Exception as fallback_err:
                        fallback_fail_msg = f"SciPy fallback ANOVA also failed: {fallback_err}"
                        detailed_report.append(fallback_fail_msg)
                        print(fallback_fail_msg)
                else:
                    detailed_report.append("SciPy fallback ANOVA skipped: Not enough groups with sufficient data.")
                
                continue # Skip to next distance_type
            
            # If checks pass, attempt statsmodels ANOVA
            try:
                formula = f"{distance_type} ~ C(Gender1)"
                model = ols(formula, data=anova_data_for_ols).fit() # Use filtered data
                anova_result = sm.stats.anova_lm(model, typ=2)
                
                # Add ANOVA table to report
                detailed_report.append("ANOVA Table:")
                detailed_report.append(anova_result.to_string())
                detailed_report.append("")
                
                # Safely access p-value - check if the column and row exist
                if "PR(>F)" in anova_result.columns and len(anova_result) > 0:
                    p_value = anova_result["PR(>F)"].iloc[0]
                    
                    if p_value < 0.05:
                        detailed_report.append(f"Significant difference found between grammatical gender categories (p = {p_value:.4f}).")
                        
                        # Add Tukey's HSD results if available
                        try:
                            # Ensure we have enough data for Tukey's HSD
                            gender_counts = df_with_gender['Gender1'].value_counts()
                            valid_genders = gender_counts[gender_counts >= 2].index.tolist()
                            
                            if len(valid_genders) < 2:
                                detailed_report.append("Not enough data in each group for Tukey's HSD (need at least 2 observations per group).")
                            else:
                                # Filter to only include groups with enough data
                                tukey_data = df_with_gender[df_with_gender['Gender1'].isin(valid_genders)].copy()
                                
                                # Prepare data for Tukey's HSD
                                tukey_data_prepared = pd.DataFrame({
                                    'score': tukey_data[distance_type],
                                    'group': tukey_data['Gender1']
                                })
                                
                                # Run Tukey's HSD
                                tukey_result = pairwise_tukeyhsd(
                                    tukey_data_prepared['score'],
                                    tukey_data_prepared['group'],
                                    alpha=0.05
                                )
                                
                                detailed_report.append("\nTukey HSD Results:")
                                detailed_report.append(str(tukey_result))
                                detailed_report.append("")
                                
                                # Add Cohen's d for significant pairs - safely parse the results table
                                detailed_report.append("Cohen's d for Significant Pairs:")
                                
                                # Safely access Tukey results
                                try:
                                    if hasattr(tukey_result, '_results_table') and hasattr(tukey_result._results_table, 'data'):
                                        # Make sure there's enough data and the structure is as expected
                                        if len(tukey_result._results_table.data) > 1:
                                            header = tukey_result._results_table.data[0]
                                            if 'group1' in header and 'group2' in header and 'reject' in header:
                                                # Convert to DataFrame with proper column names
                                                tukey_df = pd.DataFrame(data=tukey_result._results_table.data[1:], columns=header)
                                                
                                                if 'reject' in tukey_df.columns:
                                                    sig_pairs_found = False
                                                    for _, row in tukey_df.iterrows():
                                                        if row['reject']:  # If null hypothesis is rejected (significant difference)
                                                            group1 = row['group1']
                                                            group2 = row['group2']
                                                            
                                                            # Get values for each group
                                                            group1_values = df_with_gender[df_with_gender['Gender1'] == group1][distance_type]
                                                            group2_values = df_with_gender[df_with_gender['Gender1'] == group2][distance_type]
                                                            
                                                            # Ensure we have enough data for Cohen's d
                                                            if len(group1_values) >= 2 and len(group2_values) >= 2:
                                                                # Filter out NaN values
                                                                group1_values = group1_values.dropna()
                                                                group2_values = group2_values.dropna()
                                                                
                                                                if len(group1_values) >= 2 and len(group2_values) >= 2:
                                                                    try:
                                                                        cohens_d = calculate_cohens_d(group1_values, group2_values)
                                                                        
                                                                        # Add effect size interpretation
                                                                        if np.isnan(cohens_d):
                                                                            effect_interpretation = "N/A"
                                                                        elif abs(cohens_d) < 0.2:
                                                                            effect_interpretation = "negligible effect"
                                                                        elif abs(cohens_d) < 0.5:
                                                                            effect_interpretation = "small effect"
                                                                        elif abs(cohens_d) < 0.8:
                                                                            effect_interpretation = "medium effect" 
                                                                        else:
                                                                            effect_interpretation = "large effect"
                                                                            
                                                                        detailed_report.append(f"  {group1} vs {group2}: d = {cohens_d:.3f} ({effect_interpretation})")
                                                                        sig_pairs_found = True
                                                                    except Exception as e_d:
                                                                        detailed_report.append(f"  Error calculating Cohen's d for {group1} vs {group2}: {e_d}")
                                                                else:
                                                                    detailed_report.append(f"  Not enough non-NaN values for {group1} vs {group2} after filtering")
                                                            else:
                                                                detailed_report.append(f"  Not enough values for {group1} vs {group2}")
                                                    
                                                    if not sig_pairs_found:
                                                        detailed_report.append("  No significant pairwise differences with enough data for effect size calculation.")
                                                else:
                                                    detailed_report.append("  Could not parse Tukey results: 'reject' column missing.")
                                            else:
                                                detailed_report.append("  Could not parse Tukey results: expected columns missing.")
                                        else:
                                            detailed_report.append("  Tukey results table has insufficient data.")
                                    else:
                                        detailed_report.append("  Could not access Tukey results table data.")
                                except Exception as e_parse:
                                    detailed_report.append(f"  Error parsing Tukey results: {e_parse}")
                        except Exception as e_tukey:
                            detailed_report.append(f"Tukey HSD analysis failed: {e_tukey}")
                    else:
                        detailed_report.append(f"No significant difference found between grammatical gender categories (p = {p_value:.4f}).")
                else:
                    detailed_report.append("Could not determine significance from ANOVA results (missing p-value).")
                
                # Add descriptive statistics
                try:
                    desc_stats = df_with_gender.groupby('Gender1')[distance_type].agg(['count', 'mean', 'std']).round(4)
                    detailed_report.append("\nDescriptive Statistics:")
                    detailed_report.append(desc_stats.to_string())
                    detailed_report.append("")
                except Exception as e_desc:
                    detailed_report.append(f"Could not generate descriptive statistics: {e_desc}")
            except Exception as e_anova:
                detailed_report.append(f"ANOVA analysis failed: {e_anova}")
                # Avoid writing full stack traces to the report – log a concise message instead
                concise_msg = (
                    f"ANOVA could not be performed using statsmodels: {e_anova}. "
                    "This can happen when model constraints are not satisfied (e.g., no residual degrees of freedom)."
                )
                detailed_report.append(concise_msg)

                # Attempt a simpler fallback ANOVA (SciPy one-way) when there are ≥2 groups with ≥2 observations each
                group_lists = [grp[distance_type].dropna().values for _, grp in anova_data_for_ols.groupby('Gender1') if len(grp.dropna(subset=[distance_type])) >= 2]
                if len(group_lists) >= 2:
                    try:
                        f_stat, p_val = stats.f_oneway(*group_lists)
                        fallback_msg = f"Fallback one-way ANOVA (SciPy): F = {f_stat:.4f}, p = {p_val:.6f}"
                        detailed_report.append(fallback_msg)
                        print(fallback_msg)
                    except Exception as fallback_err:
                        fallback_fail_msg = f"SciPy fallback ANOVA also failed: {fallback_err}"
                        detailed_report.append(fallback_fail_msg)
                        print(fallback_fail_msg)
                    else:
                        detailed_report.append("SciPy fallback ANOVA skipped: Not enough groups with sufficient data.")
                
                continue # Skip to next distance_type
    
    # Save the detailed report
    if output_file_path_prefix:
        # Ensure the directory exists
        if os.path.dirname(output_file_path_prefix):
            os.makedirs(os.path.dirname(output_file_path_prefix), exist_ok=True)
            
        detailed_report_path = f"{output_file_path_prefix}_gender_lang_stats.txt"
        try:
            with open(detailed_report_path, 'w') as f:
                f.write("\n".join(detailed_report))
            print(f"Detailed gender language analysis saved to: {detailed_report_path}")
            
            # Verify the file was created
            if not os.path.exists(detailed_report_path):
                print(f"Warning: Failed to create detailed gender language analysis file at {detailed_report_path}")
        except Exception as e_report:
            print(f"Error saving detailed gender language analysis: {e_report}")
            import traceback
            traceback.print_exc()
    
    # Process Cosine distances if available
    if 'CosineDistance' in df_with_gender.columns:
        for i in range(len(unique_genders)):
            for j in range(i, len(unique_genders)):
                gender1 = unique_genders[i]
                gender2 = unique_genders[j]
                
                try:
                    if gender1 == gender2:
                        # Same gender pair
                        same_gender_mask = (
                            ((df_with_gender['Gender1'] == gender1) & (df_with_gender['Gender2'] == gender1))
                        )
                        same_gender_distances = df_with_gender[same_gender_mask]['CosineDistance'].dropna()
                        
                        # Different gender pair
                        diff_gender_mask = (
                            ((df_with_gender['Gender1'] == gender1) & (df_with_gender['Gender2'] != gender1)) |
                            ((df_with_gender['Gender1'] != gender1) & (df_with_gender['Gender2'] == gender1))
                        )
                        diff_gender_distances = df_with_gender[diff_gender_mask]['CosineDistance'].dropna()
                        
                        if len(same_gender_distances) >= 2 and len(diff_gender_distances) >= 2:
                            effect_size = calculate_cohens_d(same_gender_distances, diff_gender_distances)
                            
                            gender_effects.append({
                                'Gender': gender1,
                                'ComparisonType': 'Same vs Different',
                                'DistanceType': 'Cosine',
                                'EffectSize': effect_size,
                                'SameGenderMean': same_gender_distances.mean(),
                                'DifferentGenderMean': diff_gender_distances.mean(),
                                'SameGenderCount': len(same_gender_distances),
                                'DifferentGenderCount': len(diff_gender_distances)
                            })
                    else:
                        # Compare two specific different genders
                        gender_pair = f"{gender1}-{gender2}"
                        pair_mask = (
                            ((df_with_gender['Gender1'] == gender1) & (df_with_gender['Gender2'] == gender2)) |
                            ((df_with_gender['Gender1'] == gender2) & (df_with_gender['Gender2'] == gender1))
                        )
                        pair_distances = df_with_gender[pair_mask]['CosineDistance'].dropna()
                        
                        # All other gender pairs
                        other_mask = ~pair_mask
                        other_distances = df_with_gender[other_mask]['CosineDistance'].dropna()
                        
                        if len(pair_distances) >= 2 and len(other_distances) >= 2:
                            effect_size = calculate_cohens_d(pair_distances, other_distances)
                            
                            gender_effects.append({
                                'Gender': gender_pair,
                                'ComparisonType': 'Specific vs Others',
                                'DistanceType': 'Cosine',
                                'EffectSize': effect_size,
                                'SpecificPairMean': pair_distances.mean(),
                                'OtherPairsMean': other_distances.mean(),
                                'SpecificPairCount': len(pair_distances),
                                'OtherPairsCount': len(other_distances)
                            })
                except Exception as e_cosine:
                    print(f"Error in Cosine effect size calculation for {gender1} vs {gender2}: {e_cosine}")
    
    # Process Jaccard distances if available
    if 'JaccardDistance' in df_with_gender.columns:
        for i in range(len(unique_genders)):
            for j in range(i, len(unique_genders)):
                gender1 = unique_genders[i]
                gender2 = unique_genders[j]
                
                try:
                    if gender1 == gender2:
                        # Same gender pair
                        same_gender_mask = (
                            ((df_with_gender['Gender1'] == gender1) & (df_with_gender['Gender2'] == gender1))
                        )
                        same_gender_distances = df_with_gender[same_gender_mask]['JaccardDistance'].dropna()
                        
                        # Different gender pair
                        diff_gender_mask = (
                            ((df_with_gender['Gender1'] == gender1) & (df_with_gender['Gender2'] != gender1)) |
                            ((df_with_gender['Gender1'] != gender1) & (df_with_gender['Gender2'] == gender1))
                        )
                        diff_gender_distances = df_with_gender[diff_gender_mask]['JaccardDistance'].dropna()
                        
                        if len(same_gender_distances) >= 2 and len(diff_gender_distances) >= 2:
                            effect_size = calculate_cohens_d(same_gender_distances, diff_gender_distances)
                            
                            gender_effects.append({
                                'Gender': gender1,
                                'ComparisonType': 'Same vs Different',
                                'DistanceType': 'Jaccard',
                                'EffectSize': effect_size,
                                'SameGenderMean': same_gender_distances.mean(),
                                'DifferentGenderMean': diff_gender_distances.mean(),
                                'SameGenderCount': len(same_gender_distances),
                                'DifferentGenderCount': len(diff_gender_distances)
                            })
                    else:
                        # Compare two specific different genders
                        gender_pair = f"{gender1}-{gender2}"
                        pair_mask = (
                            ((df_with_gender['Gender1'] == gender1) & (df_with_gender['Gender2'] == gender2)) |
                            ((df_with_gender['Gender1'] == gender2) & (df_with_gender['Gender2'] == gender1))
                        )
                        pair_distances = df_with_gender[pair_mask]['JaccardDistance'].dropna()
                        
                        # All other gender pairs
                        other_mask = ~pair_mask
                        other_distances = df_with_gender[other_mask]['JaccardDistance'].dropna()
                        
                        if len(pair_distances) >= 2 and len(other_distances) >= 2:
                            effect_size = calculate_cohens_d(pair_distances, other_distances)
                            
                            gender_effects.append({
                                'Gender': gender_pair,
                                'ComparisonType': 'Specific vs Others',
                                'DistanceType': 'Jaccard',
                                'EffectSize': effect_size,
                                'SpecificPairMean': pair_distances.mean(),
                                'OtherPairsMean': other_distances.mean(),
                                'SpecificPairCount': len(pair_distances),
                                'OtherPairsCount': len(other_distances)
                            })
                except Exception as e_jaccard:
                    print(f"Error in Jaccard effect size calculation for {gender1} vs {gender2}: {e_jaccard}")
    
    # Create output DataFrame for gender effects
    gender_effects_df = pd.DataFrame(gender_effects)
    
    if not gender_effects_df.empty:
        # Add language info if filtered by language
        if language is not None:
            gender_effects_df['Language'] = language
        
        # Sort by effect size
        gender_effects_df['EffectSizeAbs'] = gender_effects_df['EffectSize'].abs()
        gender_effects_df.sort_values(by=['DistanceType', 'ComparisonType', 'EffectSizeAbs'], ascending=[True, True, False], inplace=True)
        gender_effects_df.drop(columns=['EffectSizeAbs'], inplace=True)
        
        # Save results
        if output_file_path_prefix:
            # Ensure the directory exists
            if os.path.dirname(output_file_path_prefix):
                os.makedirs(os.path.dirname(output_file_path_prefix), exist_ok=True)
                
            gender_effects_file = f"{output_file_path_prefix}_gender_effects.csv"
            try:
                gender_effects_df.to_csv(gender_effects_file, index=False)
                print(f"Gender effects saved to: {gender_effects_file}")
                
                # Verify the file was created
                if not os.path.exists(gender_effects_file):
                    print(f"Warning: Failed to create gender effects file at {gender_effects_file}")
            except Exception as e_effects:
                print(f"Error saving gender effects: {e_effects}")
                import traceback
                traceback.print_exc()
    
    return gender_averages_df, gender_effects_df


def analyze_distance_metric_correlation(df, output_file_path_prefix=None, language=None):
    """
    Analyzes correlation between cosine and Jaccard distances.
    
    Args:
        df: DataFrame with both CosineDistance and JaccardDistance columns
        output_file_path_prefix: Path prefix for output files
        language: Optional language to filter for language-specific analysis
        
    Returns:
        dict: Dictionary with correlation metrics
    """
    if df.empty:
        print("Error: Empty dataframe provided to analyze_distance_metric_correlation")
        return {}
    
    # Filter by language if specified
    if language is not None:
        # Check if we have language columns
        if 'Language1' in df.columns and 'Language2' in df.columns:
            # Filter rows where either Language1 or Language2 equals the specified language
            language_df = df[(df['Language1'] == language) | (df['Language2'] == language)]
            
            if language_df.empty:
                print(f"Warning: No data found for language '{language}' in correlation analysis")
                if output_file_path_prefix:
                    os.makedirs(os.path.dirname(output_file_path_prefix), exist_ok=True)
                    with open(f"{output_file_path_prefix}_correlation_warning.txt", 'w') as f:
                        f.write(f"No data found for language '{language}' in correlation analysis\n")
                return {}
            
            print(f"Analyzing correlation for language: '{language}' ({len(language_df)} rows)")
            df = language_df
        else:
            print("Warning: Cannot filter by language - Language columns not found in correlation analysis")
    
    # Check if we have the necessary columns
    if 'CosineDistance' not in df.columns or 'JaccardDistance' not in df.columns:
        missing_cols = []
        if 'CosineDistance' not in df.columns:
            missing_cols.append('CosineDistance')
        if 'JaccardDistance' not in df.columns:
            missing_cols.append('JaccardDistance')
        print(f"Error: Missing required columns for correlation analysis: {missing_cols}")
        return {}
    
    # Create a clean copy of the data with non-null values for both distances
    df_clean = df.dropna(subset=['CosineDistance', 'JaccardDistance']).copy()
    
    if df_clean.empty:
        print("Error: No valid data points for correlation analysis after removing NaNs")
        if output_file_path_prefix:
            os.makedirs(os.path.dirname(output_file_path_prefix), exist_ok=True)
            with open(f"{output_file_path_prefix}_correlation_warning.txt", 'w') as f:
                f.write("No valid data points for correlation analysis after removing NaNs\n")
        return {}
    
    # Calculate correlations
    try:
        # Check if we have enough data points
        if len(df_clean) < 2:
            print("Error: Insufficient data points for correlation analysis (need at least 2)")
            if output_file_path_prefix:
                os.makedirs(os.path.dirname(output_file_path_prefix), exist_ok=True)
                with open(f"{output_file_path_prefix}_correlation_warning.txt", 'w') as f:
                    f.write(f"Insufficient data points for correlation analysis (found {len(df_clean)}, need at least 2)\n")
            return {}
            
        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(df_clean['CosineDistance'], df_clean['JaccardDistance'])
        
        # Spearman rank correlation
        spearman_r, spearman_p = stats.spearmanr(df_clean['CosineDistance'], df_clean['JaccardDistance'])
        
        # Prepare results
        results = {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'n': len(df_clean)
        }
        
        print(f"Correlation Analysis:")
        print(f"  Pearson r = {pearson_r:.4f} (p = {pearson_p:.6f})")
        print(f"  Spearman ρ = {spearman_r:.4f} (p = {spearman_p:.6f})")
        print(f"  Sample size: {len(df_clean)}")
        
        # Save results if output path provided
        if output_file_path_prefix:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(output_file_path_prefix), exist_ok=True)
            
            # Generate a summary text file
            try:
                summary_file = f"{output_file_path_prefix}_correlation_summary.txt"
                with open(summary_file, 'w') as f:
                    if language is not None:
                        f.write(f"Correlation Analysis for Language '{language}':\n\n")
                    else:
                        f.write("Correlation Analysis:\n\n")
                    
                    f.write(f"Pearson's r: {pearson_r:.4f} (p = {pearson_p:.6f})\n")
                    f.write(f"Spearman's ρ: {spearman_r:.4f} (p = {spearman_p:.6f})\n")
                    f.write(f"Sample size: {len(df_clean)}\n\n")
                    
                    # Add basic stats
                    cosine_stats = df_clean['CosineDistance'].describe()
                    jaccard_stats = df_clean['JaccardDistance'].describe()
                    
                    f.write("Summary Statistics:\n")
                    f.write(f"Cosine: Mean={cosine_stats['mean']:.4f}, Median={df_clean['CosineDistance'].median():.4f}, SD={cosine_stats['std']:.4f}\n")
                    f.write(f"Jaccard: Mean={jaccard_stats['mean']:.4f}, Median={df_clean['JaccardDistance'].median():.4f}, SD={jaccard_stats['std']:.4f}\n")
                
                if os.path.exists(summary_file):
                    print(f"Correlation summary saved to: {summary_file}")
                else:
                    print(f"Warning: Failed to save correlation summary to: {summary_file}")
            except Exception as e_sum:
                print(f"Error saving correlation summary: {e_sum}")
            
            # Create static scatter plot
            try:
                from .visualization import fallback_png_plot
                
                # Use the fallback PNG plot function
                static_plot_file = f"{output_file_path_prefix}_correlation_plot.png"
                title = f'Correlation between Distance Metrics for Language "{language}"' if language else 'Correlation between Distance Metrics'
                
                # Add correlation info to title
                title += f' (r = {pearson_r:.3f}, ρ = {spearman_r:.3f})'
                
                # Create static plot
                fallback_png_plot(
                    df_clean, 
                    x='CosineDistance', 
                    y='JaccardDistance',
                    output_file=static_plot_file,
                    title=title,
                    xlabel='Semantic Distance % (Cosine)',
                    ylabel='Semantic Distance % (Jaccard)',
                    text_box=f"Pearson's r: {pearson_r:.3f} (p = {pearson_p:.3f})\nSpearman's ρ: {spearman_r:.3f} (p = {spearman_p:.3f})\nSample size: {len(df_clean)}"
                )
                
                print(f"Static correlation plot saved to: {static_plot_file}")
            except Exception as e_static:
                print(f"Error creating static scatter plot: {e_static}")
            
            # Create interactive HTML plot
            try:
                from .visualization import interactive_html_correlation_plot
                
                # Determine hover data based on available columns
                hover_data = []
                for col in ['Noun', 'NounCategory', 'Language1', 'Language2', 'Proficiency']:
                    if col in df_clean.columns:
                        hover_data.append(col)
                
                # Determine color_by based on available columns
                color_by = None
                for candidate in ['NounCategory', 'Proficiency']:
                    if candidate in df_clean.columns and df_clean[candidate].nunique() > 1:
                        color_by = candidate
                        break
                
                # Create output filename for HTML plot
                html_plot_file = f"{output_file_path_prefix}_correlation_plot.html"
                
                # Create interactive plot
                interactive_html_correlation_plot(
                    df=df_clean,
                    output_file=html_plot_file,
                    hover_data=hover_data,
                    color_by=color_by,
                    language=language
                )
                
                print(f"Interactive HTML correlation plot saved to: {html_plot_file}")
            except Exception as e_html:
                print(f"Error creating interactive HTML plot: {e_html}")
                import traceback
                traceback.print_exc()
        
        return results
    except Exception as e:
        print(f"Error in correlation analysis: {e}")
        import traceback
        traceback.print_exc()
        return {}


def analyze_noun_level_comparisons(df, output_file_path_prefix=None, lang_to_family_map=None):
    """
    Performs detailed evaluations between nouns across proficiencies, languages, and language families.
    
    Args:
        df (pd.DataFrame): DataFrame containing comprehensive noun data
        output_file_path_prefix (str, optional): Prefix for output file paths
        lang_to_family_map (dict, optional): Mapping of language codes to language families
        
    Returns:
        pd.DataFrame: Summary DataFrame with noun-level comparisons
    """
    if df.empty:
        print("Error: Empty dataframe provided to analyze_noun_level_comparisons")
        return pd.DataFrame()
    
    # Check required columns
    required_columns = ['Noun', 'CosineDistance']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: Missing required columns for noun-level analysis. Required: {required_columns}")
        return pd.DataFrame()
    
    print("\nPerforming noun-level comparison analysis...")
    
    # Create output directory if needed
    if output_file_path_prefix:
        output_dir = os.path.dirname(output_file_path_prefix)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Ensuring output directory exists: {output_dir}")
    
    # 1. Analyze nouns across proficiencies if proficiency data is available
    proficiency_report_lines = []
    proficiency_analysis_df = pd.DataFrame()
    
    if 'Proficiency' in df.columns and df['Proficiency'].nunique() > 1:
        print("Analyzing nouns across proficiency levels...")
        proficiency_report_lines.append("Noun Analysis Across Proficiency Levels")
        proficiency_report_lines.append("=====================================\n")
        
        # Get unique nouns and proficiencies
        nouns = df['Noun'].unique()
        proficiencies = df['Proficiency'].unique()
        
        # Store results for nouns across proficiencies
        prof_noun_records = []
        
        for noun in nouns:
            noun_df = df[df['Noun'] == noun]
            
            # Skip if this noun doesn't appear in multiple proficiencies
            if noun_df['Proficiency'].nunique() < 2:
                continue
                
            proficiency_report_lines.append(f"\nNoun: {noun}")
            
            # Get mean cosine distances per proficiency for this noun
            prof_stats = noun_df.groupby('Proficiency')['CosineDistance'].agg(['mean', 'std', 'count'])
            proficiency_report_lines.append(f"Cosine Distance Statistics by Proficiency:")
            proficiency_report_lines.append(prof_stats.to_string())
            
            # Compute effect sizes between proficiency pairs for this noun
            for i, prof1 in enumerate(proficiencies):
                for j in range(i+1, len(proficiencies)):
                    prof2 = proficiencies[j]
                    
                    prof1_values = noun_df[noun_df['Proficiency'] == prof1]['CosineDistance']
                    prof2_values = noun_df[noun_df['Proficiency'] == prof2]['CosineDistance']
                    
                    if not prof1_values.empty and not prof2_values.empty:
                        effect_size = calculate_cohens_d(prof1_values, prof2_values)
                        
                        # Interpret effect size
                        if np.isnan(effect_size):
                            effect_interp = "N/A"
                        elif abs(effect_size) < 0.2:
                            effect_interp = "negligible effect"
                        elif abs(effect_size) < 0.5:
                            effect_interp = "small effect"
                        elif abs(effect_size) < 0.8:
                            effect_interp = "medium effect"
                        else:
                            effect_interp = "large effect"
                        
                        proficiency_report_lines.append(f"  {prof1} vs {prof2}: Cohen's d = {effect_size:.3f} ({effect_interp})")
                        
                        # Add to records
                        prof_noun_records.append({
                            'Noun': noun,
                            'NounCategory': noun_df['NounCategory'].iloc[0] if 'NounCategory' in noun_df.columns else 'Unknown',
                            'Proficiency1': prof1,
                            'Proficiency2': prof2,
                            'EffectSize': effect_size,
                            'Interpretation': effect_interp,
                            'Mean1': prof1_values.mean(),
                            'Mean2': prof2_values.mean(),
                            'Count1': len(prof1_values),
                            'Count2': len(prof2_values)
                        })
        
        # Create proficiency analysis DataFrame
        if prof_noun_records:
            proficiency_analysis_df = pd.DataFrame(prof_noun_records)
            proficiency_analysis_df.sort_values(by=['NounCategory', 'Noun', 'EffectSize'], ascending=[True, True, False], inplace=True)
            
            if output_file_path_prefix:
                # Save proficiency analysis
                prof_analysis_file = f"{output_file_path_prefix}_proficiency_noun_comparisons.csv"
                proficiency_analysis_df.to_csv(prof_analysis_file, index=False)
                print(f"Proficiency-level noun comparisons saved to: {prof_analysis_file}")
                
                # Save proficiency report
                prof_report_file = f"{output_file_path_prefix}_proficiency_noun_report.txt"
                with open(prof_report_file, 'w') as f:
                    f.write('\n'.join(proficiency_report_lines))
                print(f"Proficiency-level noun report saved to: {prof_report_file}")
    
    # 2. Analyze nouns across languages
    language_report_lines = []
    language_analysis_df = pd.DataFrame()
    
    if 'Language1' in df.columns and 'Language2' in df.columns:
        print("Analyzing nouns across languages...")
        language_report_lines.append("Noun Analysis Across Languages")
        language_report_lines.append("===========================\n")
        
        # Get unique nouns and languages
        nouns = df['Noun'].unique()
        languages = sorted(set(df['Language1'].unique()) | set(df['Language2'].unique()))
        
        # Store results for nouns across languages
        lang_noun_records = []
        
        for noun in nouns:
            noun_df = df[df['Noun'] == noun]
            
            # Skip if we don't have meaningful language comparisons
            if len(noun_df) < 2:
                continue
                
            language_report_lines.append(f"\nNoun: {noun}")
            
            # Get statistics about language pairs for this noun
            language_report_lines.append(f"Language pair statistics for '{noun}':")
            
            # Group by language pairs and compute statistics
            language_pair_stats = []
            
            for i, lang1 in enumerate(languages):
                for j in range(i+1, len(languages)):
                    lang2 = languages[j]
                    
                    # Get rows for this language pair and noun
                    lang_pair_rows = noun_df[
                        ((noun_df['Language1'] == lang1) & (noun_df['Language2'] == lang2)) |
                        ((noun_df['Language1'] == lang2) & (noun_df['Language2'] == lang1))
                    ]
                    
                    if not lang_pair_rows.empty:
                        cosine_mean = lang_pair_rows['CosineDistance'].mean()
                        cosine_std = lang_pair_rows['CosineDistance'].std()
                        count = len(lang_pair_rows)
                        
                        language_pair_stats.append({
                            'Language1': lang1,
                            'Language2': lang2,
                            'CosineDistance_mean': cosine_mean,
                            'CosineDistance_std': cosine_std,
                            'Count': count
                        })
                        
                        language_report_lines.append(f"  {lang1} vs {lang2}: Mean={cosine_mean:.4f}, StdDev={cosine_std:.4f}, N={count}")
                        
                        # Add to records
                        lang_noun_records.append({
                            'Noun': noun,
                            'NounCategory': noun_df['NounCategory'].iloc[0] if 'NounCategory' in noun_df.columns else 'Unknown',
                            'Language1': lang1,
                            'Language2': lang2,
                            'CosineDistance_mean': cosine_mean,
                            'CosineDistance_std': cosine_std,
                            'Count': count
                        })
        
        # Create language analysis DataFrame
        if lang_noun_records:
            language_analysis_df = pd.DataFrame(lang_noun_records)
            language_analysis_df.sort_values(by=['NounCategory', 'Noun', 'CosineDistance_mean'], ascending=[True, True, False], inplace=True)
            
            if output_file_path_prefix:
                # Save language analysis
                lang_analysis_file = f"{output_file_path_prefix}_language_noun_comparisons.csv"
                language_analysis_df.to_csv(lang_analysis_file, index=False)
                print(f"Language-level noun comparisons saved to: {lang_analysis_file}")
                
                # Save language report
                lang_report_file = f"{output_file_path_prefix}_language_noun_report.txt"
                with open(lang_report_file, 'w') as f:
                    f.write('\n'.join(language_report_lines))
                print(f"Language-level noun report saved to: {lang_report_file}")
    
    # 3. Analyze nouns across language families if mapping is provided
    family_report_lines = []
    family_analysis_df = pd.DataFrame()
    
    if lang_to_family_map and 'Language1' in df.columns and 'Language2' in df.columns:
        print("Analyzing nouns across language families...")
        family_report_lines.append("Noun Analysis Across Language Families")
        family_report_lines.append("===================================\n")
        
        # Create a copy of the DataFrame with language family information
        df_with_families = df.copy()
        
        # Extract language codes and map to families
        df_with_families['Language1Code'] = df_with_families['Language1'].apply(
            lambda x: x.split(' # ')[0] if isinstance(x, str) and ' # ' in x else x
        )
        df_with_families['Language2Code'] = df_with_families['Language2'].apply(
            lambda x: x.split(' # ')[0] if isinstance(x, str) and ' # ' in x else x
        )
        
        # Map codes to families
        df_with_families['Family1'] = df_with_families['Language1Code'].map(lang_to_family_map)
        df_with_families['Family2'] = df_with_families['Language2Code'].map(lang_to_family_map)
        
        # Drop rows where family mapping couldn't be determined
        df_with_families = df_with_families.dropna(subset=['Family1', 'Family2'])
        
        if df_with_families.empty:
            print("Warning: No data with valid language family mappings")
        else:
            # Get unique nouns and families
            nouns = df_with_families['Noun'].unique()
            families = sorted(set(df_with_families['Family1'].unique()) | set(df_with_families['Family2'].unique()))
            
            # Store results for nouns across families
            family_noun_records = []
            
            for noun in nouns:
                noun_df = df_with_families[df_with_families['Noun'] == noun]
                
                # Skip if we don't have meaningful family comparisons
                if len(noun_df) < 2:
                    continue
                    
                family_report_lines.append(f"\nNoun: {noun}")
                
                # Get statistics about family pairs for this noun
                family_report_lines.append(f"Language family pair statistics for '{noun}':")
                
                # Analyze both within-family and between-family comparisons
                within_family_distances = []
                between_family_distances = []
                
                # Group by family pairs and compute statistics
                family_pair_stats = []
                
                for i, fam1 in enumerate(families):
                    for j in range(i, len(families)):
                        fam2 = families[j]
                        
                        # Get rows for this family pair and noun
                        fam_pair_rows = noun_df[
                            ((noun_df['Family1'] == fam1) & (noun_df['Family2'] == fam2)) |
                            ((noun_df['Family1'] == fam2) & (noun_df['Family2'] == fam1))
                        ]
                        
                        if not fam_pair_rows.empty:
                            cosine_mean = fam_pair_rows['CosineDistance'].mean()
                            cosine_std = fam_pair_rows['CosineDistance'].std()
                            count = len(fam_pair_rows)
                            
                            # Record within-family or between-family data
                            if fam1 == fam2:
                                within_family_distances.extend(fam_pair_rows['CosineDistance'].tolist())
                                comparison_type = "Within-family"
                            else:
                                between_family_distances.extend(fam_pair_rows['CosineDistance'].tolist())
                                comparison_type = "Between-family"
                            
                            family_report_lines.append(f"  {fam1} vs {fam2} ({comparison_type}): Mean={cosine_mean:.4f}, StdDev={cosine_std:.4f}, N={count}")
                            
                            # Add to records
                            family_noun_records.append({
                                'Noun': noun,
                                'NounCategory': noun_df['NounCategory'].iloc[0] if 'NounCategory' in noun_df.columns else 'Unknown',
                                'Family1': fam1,
                                'Family2': fam2,
                                'ComparisonType': comparison_type,
                                'CosineDistance_mean': cosine_mean,
                                'CosineDistance_std': cosine_std,
                                'Count': count
                            })
                
                # Calculate effect size between within-family and between-family distances if both exist
                if within_family_distances and between_family_distances:
                    effect_size = calculate_cohens_d(
                        np.array(within_family_distances),
                        np.array(between_family_distances)
                    )
                    
                    # Interpret effect size
                    if np.isnan(effect_size):
                        effect_interp = "N/A"
                    elif abs(effect_size) < 0.2:
                        effect_interp = "negligible effect"
                    elif abs(effect_size) < 0.5:
                        effect_interp = "small effect"
                    elif abs(effect_size) < 0.8:
                        effect_interp = "medium effect"
                    else:
                        effect_interp = "large effect"
                    
                    within_mean = np.mean(within_family_distances)
                    between_mean = np.mean(between_family_distances)
                    
                    family_report_lines.append(f"\n  Within vs Between family comparison:")
                    family_report_lines.append(f"    Within-family mean: {within_mean:.4f} (N={len(within_family_distances)})")
                    family_report_lines.append(f"    Between-family mean: {between_mean:.4f} (N={len(between_family_distances)})")
                    family_report_lines.append(f"    Effect size (Cohen's d): {effect_size:.3f} ({effect_interp})")
                    
                    # Add summary record for this noun
                    family_noun_records.append({
                        'Noun': noun,
                        'NounCategory': noun_df['NounCategory'].iloc[0] if 'NounCategory' in noun_df.columns else 'Unknown',
                        'Family1': 'Within-family',
                        'Family2': 'Between-family',
                        'ComparisonType': 'Summary',
                        'CosineDistance_mean1': within_mean,
                        'CosineDistance_mean2': between_mean,
                        'EffectSize': effect_size,
                        'Interpretation': effect_interp,
                        'Count1': len(within_family_distances),
                        'Count2': len(between_family_distances)
                    })
            
            # Create family analysis DataFrame
            if family_noun_records:
                family_analysis_df = pd.DataFrame(family_noun_records)
                family_analysis_df.sort_values(by=['NounCategory', 'Noun', 'ComparisonType'], ascending=[True, True, False], inplace=True)
                
                if output_file_path_prefix:
                    # Save family analysis
                    family_analysis_file = f"{output_file_path_prefix}_family_noun_comparisons.csv"
                    family_analysis_df.to_csv(family_analysis_file, index=False)
                    print(f"Language-family-level noun comparisons saved to: {family_analysis_file}")
                    
                    # Save family report
                    family_report_file = f"{output_file_path_prefix}_family_noun_report.txt"
                    with open(family_report_file, 'w') as f:
                        f.write('\n'.join(family_report_lines))
                    print(f"Language-family-level noun report saved to: {family_report_file}")
    
    # 4. Create a combined noun-level analysis summary
    summary_report_lines = ["Noun-Level Analysis Summary", "=========================\n"]
    
    # Summarize by noun and category
    nouns = df['Noun'].unique()
    summary_records = []
    
    for noun in nouns:
        noun_df = df[df['Noun'] == noun]
        
        if len(noun_df) < 2:
            continue
            
        # Get noun category
        noun_category = noun_df['NounCategory'].iloc[0] if 'NounCategory' in noun_df.columns else 'Unknown'
        
        # Calculate overall statistics
        cosine_mean = noun_df['CosineDistance'].mean()
        cosine_std = noun_df['CosineDistance'].std()
        cosine_min = noun_df['CosineDistance'].min()
        cosine_max = noun_df['CosineDistance'].max()
        
        # Calculate jaccard stats if available
        if 'JaccardDistance' in noun_df.columns:
            jaccard_mean = noun_df['JaccardDistance'].mean()
            jaccard_std = noun_df['JaccardDistance'].std()
            jaccard_min = noun_df['JaccardDistance'].min()
            jaccard_max = noun_df['JaccardDistance'].max()
        else:
            jaccard_mean = jaccard_std = jaccard_min = jaccard_max = np.nan
        
        # Get language count
        if 'Language1' in noun_df.columns and 'Language2' in noun_df.columns:
            languages = set(noun_df['Language1'].unique()) | set(noun_df['Language2'].unique())
            language_count = len(languages)
        else:
            language_count = np.nan
        
        # Get proficiency count
        if 'Proficiency' in noun_df.columns:
            proficiency_count = noun_df['Proficiency'].nunique()
        else:
            proficiency_count = np.nan
        
        # Add to summary records
        summary_records.append({
            'Noun': noun,
            'NounCategory': noun_category,
            'CosineDistance_mean': cosine_mean,
            'CosineDistance_std': cosine_std,
            'CosineDistance_min': cosine_min,
            'CosineDistance_max': cosine_max,
            'JaccardDistance_mean': jaccard_mean,
            'JaccardDistance_std': jaccard_std,
            'JaccardDistance_min': jaccard_min,
            'JaccardDistance_max': jaccard_max,
            'LanguageCount': language_count,
            'ProficiencyCount': proficiency_count,
            'TotalComparisons': len(noun_df)
        })
        
        # Add to summary report
        summary_report_lines.append(f"Noun: {noun} (Category: {noun_category})")
        summary_report_lines.append(f"  Cosine Distance: Mean={cosine_mean:.4f}, StdDev={cosine_std:.4f}, Min={cosine_min:.4f}, Max={cosine_max:.4f}")
        
        if not np.isnan(jaccard_mean):
            summary_report_lines.append(f"  Jaccard Distance: Mean={jaccard_mean:.4f}, StdDev={jaccard_std:.4f}, Min={jaccard_min:.4f}, Max={jaccard_max:.4f}")
        
        if not np.isnan(language_count):
            summary_report_lines.append(f"  Languages: {language_count}")
        
        if not np.isnan(proficiency_count):
            summary_report_lines.append(f"  Proficiency Levels: {proficiency_count}")
            
        summary_report_lines.append(f"  Total Comparisons: {len(noun_df)}\n")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_records)
    if not summary_df.empty:
        summary_df.sort_values(by=['NounCategory', 'CosineDistance_mean'], ascending=[True, False], inplace=True)
        
        if output_file_path_prefix:
            # Save summary analysis
            summary_file = f"{output_file_path_prefix}_noun_analysis_summary.csv"
            summary_df.to_csv(summary_file, index=False)
            print(f"Noun-level analysis summary saved to: {summary_file}")
            
            # Save summary report
            summary_report_file = f"{output_file_path_prefix}_noun_analysis_report.txt"
            with open(summary_report_file, 'w') as f:
                f.write('\n'.join(summary_report_lines))
            print(f"Noun-level analysis report saved to: {summary_report_file}")
    
    return summary_df


def analyze_language_level_statistics(df, output_file_path_prefix=None, distance_metric="LanguageEmbeddingDistance"):
    """
    Performs statistical analyses on language-level comparison data.
    
    Args:
        df (pd.DataFrame): DataFrame with language-level comparison data
        output_file_path_prefix (str, optional): Prefix for output file paths
        distance_metric (str): Distance metric to analyze, default is "LanguageEmbeddingDistance"
        
    Returns:
        dict: Dictionary containing statistical results
    """
    if df.empty:
        print(f"Error: Empty dataframe provided to analyze_language_level_statistics")
        return {}
    
    required_columns = ['Language1', 'Language2', distance_metric]
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        print(f"Error: Missing required columns in language-level statistics: {missing_cols}")
        return {}
    
    df = df.dropna(subset=required_columns)
    if df.empty:
        print(f"Error: No valid data after dropping NaN values for {required_columns}")
        return {}
        
    # Create a report
    report_lines = []
    report_lines.append(f"Language-Level Statistical Analysis ({distance_metric}):")
    
    results = {}
    
    # 1. Calculate basic statistics
    results['count'] = len(df)
    results['mean'] = df[distance_metric].mean()
    results['std'] = df[distance_metric].std()
    results['min'] = df[distance_metric].min()
    results['max'] = df[distance_metric].max()
    
    report_lines.append(f"\nBasic Statistics:")
    report_lines.append(f"  Count: {results['count']}")
    report_lines.append(f"  Mean: {results['mean']:.4f}")
    report_lines.append(f"  Std Dev: {results['std']:.4f}")
    report_lines.append(f"  Min: {results['min']:.4f}")
    report_lines.append(f"  Max: {results['max']:.4f}")
    
    # 2. Analyze by proficiency levels if available
    if 'Proficiency' in df.columns and df['Proficiency'].nunique() > 1:
        # Get proficiency statistics
        prof_stats = df.groupby('Proficiency')[distance_metric].agg(['count', 'mean', 'std', 'min', 'max'])
        prof_stats = prof_stats.round(4)
        
        report_lines.append("\nProficiency-Level Statistics:")
        report_lines.append(prof_stats.to_string())
        
        results['proficiency_stats'] = prof_stats.to_dict('index')
        
        # Perform ANOVA by proficiency level
        try:
            if df['Proficiency'].nunique() >= 2:
                print(f"Performing ANOVA by proficiency level for {distance_metric}...")
                
                # Check if we have enough data points per proficiency level
                prof_counts = df['Proficiency'].value_counts()
                valid_profs = prof_counts[prof_counts >= 2].index.tolist()
                
                if len(valid_profs) < 2:
                    report_lines.append("\nInsufficient data for ANOVA analysis (need at least 2 observations per proficiency level)")
                else:
                    # Filter to only include proficiency levels with sufficient data
                    anova_df = df[df['Proficiency'].isin(valid_profs)].copy()
                    
                    try:
                        # Run ANOVA
                        formula = f"{distance_metric} ~ C(Proficiency)"
                        model = ols(formula, data=anova_df).fit()
                        anova_result = sm.stats.anova_lm(model, typ=2)
                        
                        report_lines.append("\nANOVA Results (by Proficiency):")
                        report_lines.append(anova_result.to_string())
                        
                        # Save ANOVA results
                        results['anova_result'] = anova_result.to_dict()
                        
                        # Access p-value safely
                        if "PR(>F)" in anova_result.columns and len(anova_result) > 0:
                            p_value = anova_result["PR(>F)"].iloc[0]
                            report_lines.append(f"\nSignificance: p = {p_value:.4f}")
                            
                            if p_value < 0.05:
                                report_lines.append("Result: Significant differences found between proficiency levels")
                                
                                # Perform Tukey's HSD post-hoc test
                                try:
                                    # Prepare data for Tukey's HSD
                                    tukey_data = pd.DataFrame({
                                        'score': anova_df[distance_metric],
                                        'group': anova_df['Proficiency']
                                    })
                                    
                                    # Run Tukey's HSD
                                    tukey_result = pairwise_tukeyhsd(
                                        tukey_data['score'],
                                        tukey_data['group'],
                                        alpha=0.05
                                    )
                                    
                                    report_lines.append("\nTukey's HSD Results:")
                                    report_lines.append(str(tukey_result))
                                    
                                    # Extract significant pairs
                                    try:
                                        # Safely handle Tukey's results
                                        if hasattr(tukey_result, '_results_table') and hasattr(tukey_result._results_table, 'data'):
                                            if len(tukey_result._results_table.data) > 1:
                                                header = tukey_result._results_table.data[0]
                                                if all(col in header for col in ['group1', 'group2', 'reject', 'meandiff', 'p-adj']):
                                                    tukey_df = pd.DataFrame(data=tukey_result._results_table.data[1:], 
                                                                          columns=tukey_result._results_table.data[0])
                                                    significant_pairs = tukey_df[tukey_df['reject']]
                                                    
                                                    if not significant_pairs.empty:
                                                        report_lines.append("\nSignificant Proficiency Pairs:")
                                                        
                                                        # Calculate effect sizes for significant pairs
                                                        for _, row in significant_pairs.iterrows():
                                                            prof1 = row['group1']
                                                            prof2 = row['group2']
                                                            
                                                            # Get values for each proficiency
                                                            prof1_values = anova_df[anova_df['Proficiency'] == prof1][distance_metric]
                                                            prof2_values = anova_df[anova_df['Proficiency'] == prof2][distance_metric]
                                                            
                                                            if len(prof1_values) >= 2 and len(prof2_values) >= 2:
                                                                try:
                                                                    # Calculate Cohen's d
                                                                    cohens_d = calculate_cohens_d(prof1_values, prof2_values)
                                                                    
                                                                    # Interpret effect size
                                                                    if np.isnan(cohens_d):
                                                                        effect_interp = "N/A"
                                                                    elif abs(cohens_d) < 0.2:
                                                                        effect_interp = "negligible effect"
                                                                    elif abs(cohens_d) < 0.5:
                                                                        effect_interp = "small effect"
                                                                    elif abs(cohens_d) < 0.8:
                                                                        effect_interp = "medium effect"
                                                                    else:
                                                                        effect_interp = "large effect"
                                                                    
                                                                    report_lines.append(f"  {prof1} vs {prof2}: diff={row['meandiff']:.4f}, p-adj={row['p-adj']:.4f}, Cohen's d={cohens_d:.2f} ({effect_interp})")
                                                                except Exception as e_cohens:
                                                                    report_lines.append(f"  {prof1} vs {prof2}: diff={row['meandiff']:.4f}, p-adj={row['p-adj']:.4f}, Cohen's d=N/A (error: {e_cohens})")
                                                            else:
                                                                report_lines.append(f"  {prof1} vs {prof2}: diff={row['meandiff']:.4f}, p-adj={row['p-adj']:.4f}, Cohen's d=N/A (insufficient data)")
                                                    else:
                                                        report_lines.append("No significant pairs found in Tukey's HSD test.")
                                                else:
                                                    report_lines.append("Could not parse Tukey's HSD results: expected columns missing")
                                            else:
                                                report_lines.append("Tukey's HSD results contain insufficient data")
                                        else:
                                            report_lines.append("Could not access Tukey's HSD results table")
                                    except Exception as e_parse:
                                        report_lines.append(f"Error parsing Tukey's HSD results: {e_parse}")
                                except Exception as e_tukey:
                                    report_lines.append(f"Error in Tukey's HSD analysis: {e_tukey}")
                            else:
                                report_lines.append("Result: No significant differences found between proficiency levels")
                        else:
                            report_lines.append("Could not determine significance from ANOVA results (missing p-value).")
                    except Exception as e_anova:
                        report_lines.append(f"Error in ANOVA analysis: {e_anova}")
        except Exception as e_prof:
            report_lines.append(f"Error analyzing proficiency levels: {e_prof}")
    
    # Save the report
    if output_file_path_prefix:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_file_path_prefix), exist_ok=True)
        
        report_file = f"{output_file_path_prefix}_{distance_metric}_statistics.txt"
        try:
            with open(report_file, 'w') as f:
                f.write("\n".join(report_lines))
            print(f"Language-level statistics saved to: {report_file}")
        except Exception as e_report:
            print(f"Error saving language-level statistics report: {e_report}")
    
    return results


def analyze_language_family_statistics(df, output_file_path_prefix=None, distance_metric="Distance"):
    """
    Performs statistical analyses on language family comparison data.
    
    Args:
        df (pd.DataFrame): DataFrame with language family comparison data
        output_file_path_prefix (str, optional): Prefix for output file paths
        distance_metric (str): Distance metric to analyze
        
    Returns:
        dict: Dictionary containing statistical results
    """
    if df.empty:
        print(f"Error: Empty dataframe provided to analyze_language_family_statistics")
        return {}
    
    required_columns = ['Family1', 'Family2', distance_metric]
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        print(f"Error: Missing required columns in language family statistics: {missing_cols}")
        return {}
    
    df = df.dropna(subset=required_columns)
    if df.empty:
        print(f"Error: No valid data after dropping NaN values for {required_columns}")
        return {}
        
    # Create a report
    report_lines = []
    report_lines.append(f"Language Family Statistical Analysis ({distance_metric}):")
    
    results = {}
    
    # 1. Calculate basic statistics
    results['count'] = len(df)
    results['mean'] = df[distance_metric].mean()
    results['std'] = df[distance_metric].std()
    results['min'] = df[distance_metric].min()
    results['max'] = df[distance_metric].max()
    
    report_lines.append(f"\nBasic Statistics:")
    report_lines.append(f"  Count: {results['count']}")
    report_lines.append(f"  Mean: {results['mean']:.4f}")
    report_lines.append(f"  Std Dev: {results['std']:.4f}")
    report_lines.append(f"  Min: {results['min']:.4f}")
    report_lines.append(f"  Max: {results['max']:.4f}")
    
    # 2. Analyze by proficiency levels if available
    if 'Proficiency' in df.columns and df['Proficiency'].nunique() > 1:
        # Get proficiency statistics
        prof_stats = df.groupby('Proficiency')[distance_metric].agg(['count', 'mean', 'std', 'min', 'max'])
        prof_stats = prof_stats.round(4)
        
        report_lines.append("\nProficiency-Level Statistics:")
        report_lines.append(prof_stats.to_string())
        
        results['proficiency_stats'] = prof_stats.to_dict('index')
        
        # Perform ANOVA by proficiency level
        try:
            if df['Proficiency'].nunique() >= 2:
                print(f"Performing ANOVA by proficiency level for {distance_metric}...")
                
                # Check if we have enough data points per proficiency level
                prof_counts = df['Proficiency'].value_counts()
                valid_profs = prof_counts[prof_counts >= 2].index.tolist()
                
                if len(valid_profs) < 2:
                    report_lines.append("\nInsufficient data for ANOVA analysis (need at least 2 observations per proficiency level)")
                else:
                    # Filter to only include proficiency levels with sufficient data
                    anova_df = df[df['Proficiency'].isin(valid_profs)].copy()
                    
                    try:
                        # Run ANOVA
                        formula = f"{distance_metric} ~ C(Proficiency)"
                        model = ols(formula, data=anova_df).fit()
                        anova_result = sm.stats.anova_lm(model, typ=2)
                        
                        report_lines.append("\nANOVA Results (by Proficiency):")
                        report_lines.append(anova_result.to_string())
                        
                        # Save ANOVA results
                        results['anova_result'] = anova_result.to_dict()
                        
                        # Access p-value safely
                        if "PR(>F)" in anova_result.columns and len(anova_result) > 0:
                            p_value = anova_result["PR(>F)"].iloc[0]
                            report_lines.append(f"\nSignificance: p = {p_value:.4f}")
                            
                            if p_value < 0.05:
                                report_lines.append("Result: Significant differences found between proficiency levels")
                                
                                # Perform Tukey's HSD post-hoc test
                                try:
                                    # Prepare data for Tukey's HSD
                                    tukey_data = pd.DataFrame({
                                        'score': anova_df[distance_metric],
                                        'group': anova_df['Proficiency']
                                    })
                                    
                                    # Run Tukey's HSD
                                    tukey_result = pairwise_tukeyhsd(
                                        tukey_data['score'],
                                        tukey_data['group'],
                                        alpha=0.05
                                    )
                                    
                                    report_lines.append("\nTukey's HSD Results:")
                                    report_lines.append(str(tukey_result))
                                    
                                    # Extract significant pairs
                                    try:
                                        # Safely handle Tukey's results
                                        if hasattr(tukey_result, '_results_table') and hasattr(tukey_result._results_table, 'data'):
                                            if len(tukey_result._results_table.data) > 1:
                                                header = tukey_result._results_table.data[0]
                                                if all(col in header for col in ['group1', 'group2', 'reject', 'meandiff', 'p-adj']):
                                                    tukey_df = pd.DataFrame(data=tukey_result._results_table.data[1:], 
                                                                          columns=tukey_result._results_table.data[0])
                                                    significant_pairs = tukey_df[tukey_df['reject']]
                                                    
                                                    if not significant_pairs.empty:
                                                        report_lines.append("\nSignificant Proficiency Pairs:")
                                                        
                                                        # Calculate effect sizes for significant pairs
                                                        for _, row in significant_pairs.iterrows():
                                                            prof1 = row['group1']
                                                            prof2 = row['group2']
                                                            
                                                            # Get values for each proficiency
                                                            prof1_values = anova_df[anova_df['Proficiency'] == prof1][distance_metric]
                                                            prof2_values = anova_df[anova_df['Proficiency'] == prof2][distance_metric]
                                                            
                                                            if len(prof1_values) >= 2 and len(prof2_values) >= 2:
                                                                try:
                                                                    # Calculate Cohen's d
                                                                    cohens_d = calculate_cohens_d(prof1_values, prof2_values)
                                                                    
                                                                    # Interpret effect size
                                                                    if np.isnan(cohens_d):
                                                                        effect_interp = "N/A"
                                                                    elif abs(cohens_d) < 0.2:
                                                                        effect_interp = "negligible effect"
                                                                    elif abs(cohens_d) < 0.5:
                                                                        effect_interp = "small effect"
                                                                    elif abs(cohens_d) < 0.8:
                                                                        effect_interp = "medium effect"
                                                                    else:
                                                                        effect_interp = "large effect"
                                                                    
                                                                    report_lines.append(f"  {prof1} vs {prof2}: diff={row['meandiff']:.4f}, p-adj={row['p-adj']:.4f}, Cohen's d={cohens_d:.2f} ({effect_interp})")
                                                                except Exception as e_cohens:
                                                                    report_lines.append(f"  {prof1} vs {prof2}: diff={row['meandiff']:.4f}, p-adj={row['p-adj']:.4f}, Cohen's d=N/A (error: {e_cohens})")
                                                            else:
                                                                report_lines.append(f"  {prof1} vs {prof2}: diff={row['meandiff']:.4f}, p-adj={row['p-adj']:.4f}, Cohen's d=N/A (insufficient data)")
                                                    else:
                                                        report_lines.append("No significant pairs found in Tukey's HSD test.")
                                                else:
                                                    report_lines.append("Could not parse Tukey's HSD results: expected columns missing")
                                            else:
                                                report_lines.append("Tukey's HSD results contain insufficient data")
                                        else:
                                            report_lines.append("Could not access Tukey's HSD results table")
                                    except Exception as e_parse:
                                        report_lines.append(f"Error parsing Tukey's HSD results: {e_parse}")
                                except Exception as e_tukey:
                                    report_lines.append(f"Error in Tukey's HSD analysis: {e_tukey}")
                            else:
                                report_lines.append("Result: No significant differences found between proficiency levels")
                        else:
                            report_lines.append("Could not determine significance from ANOVA results (missing p-value).")
                    except Exception as e_anova:
                        report_lines.append(f"Error in ANOVA analysis: {e_anova}")
        except Exception as e_prof:
            report_lines.append(f"Error analyzing proficiency levels: {e_prof}")
    
    # 3. Analyze within vs between family distances
    try:
        # Identify which rows are within the same family and which are between different families
        df['SameFamily'] = df['Family1'] == df['Family2']
        
        # Collect distances for within and between family comparisons
        within_family_distances = df[df['SameFamily']][distance_metric].tolist()
        between_family_distances = df[~df['SameFamily']][distance_metric].tolist()
        
        # Calculate statistics for within and between family distances
        if within_family_distances and between_family_distances:
            within_mean = np.mean(within_family_distances)
            within_std = np.std(within_family_distances)
            between_mean = np.mean(between_family_distances)
            between_std = np.std(between_family_distances)
            
            # Calculate Cohen's d for within vs between
            if len(within_family_distances) >= 2 and len(between_family_distances) >= 2:
                try:
                    cohens_d = calculate_cohens_d(
                        np.array(within_family_distances),
                        np.array(between_family_distances)
                    )
                    
                    # Interpret effect size
                    if np.isnan(cohens_d):
                        effect_interp = "N/A"
                    elif abs(cohens_d) < 0.2:
                        effect_interp = "negligible effect"
                    elif abs(cohens_d) < 0.5:
                        effect_interp = "small effect"
                    elif abs(cohens_d) < 0.8:
                        effect_interp = "medium effect"
                    else:
                        effect_interp = "large effect"
                    
                    report_lines.append("\nWithin vs Between Family Comparison:")
                    report_lines.append(f"  Within-family mean: {within_mean:.4f}, std: {within_std:.4f} (N={len(within_family_distances)})")
                    report_lines.append(f"  Between-family mean: {between_mean:.4f}, std: {between_std:.4f} (N={len(between_family_distances)})")
                    report_lines.append(f"  Effect size (Cohen's d): {cohens_d:.3f} ({effect_interp})")
                    
                    # Perform t-test for within vs between
                    try:
                        t_stat, p_value = stats.ttest_ind(within_family_distances, between_family_distances, equal_var=False)
                        report_lines.append(f"  t-test: t={t_stat:.3f}, p={p_value:.6f}")
                        
                        # Add significance interpretation
                        if p_value < 0.05:
                            report_lines.append("  Result: Significant difference between within-family and between-family distances")
                        else:
                            report_lines.append("  Result: No significant difference between within-family and between-family distances")
                    except Exception as e_ttest:
                        report_lines.append(f"  Error performing t-test: {e_ttest}")
                except Exception as e_effect:
                    report_lines.append(f"Error calculating effect size for within vs between family comparison: {e_effect}")
            else:
                report_lines.append("\nInsufficient data for within vs between family comparison (need at least 2 samples per group)")
        else:
            if not within_family_distances:
                report_lines.append("\nNo within-family comparisons found")
            if not between_family_distances:
                report_lines.append("\nNo between-family comparisons found")
    except Exception as e_family:
        report_lines.append(f"Error analyzing within vs between family distances: {e_family}")
    
    # Save the report
    if output_file_path_prefix:
        # Ensure the directory exists
        try:
            os.makedirs(os.path.dirname(output_file_path_prefix), exist_ok=True)
            
            report_file = f"{output_file_path_prefix}_{distance_metric}_statistics.txt"
            with open(report_file, 'w') as f:
                f.write("\n".join(report_lines))
            print(f"Language family statistics saved to: {report_file}")
        except Exception as e_report:
            print(f"Error saving language family statistics report: {e_report}")
    
    return results
