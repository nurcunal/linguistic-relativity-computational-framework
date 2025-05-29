import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import chi2_contingency
import os
import glob
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend and style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class SimpleLRAnalyzer:
    """Simple analyzer focused on cross-linguistic variance in bilingual model choices."""
    
    def __init__(self, data_path='api_generations/lr_experiments_english', output_dir='simple_lr_experiment_results'):
        self.data_path = data_path
        self.output_dir = output_dir
        self.experiments = {}
        self.results = {}
        self.selected_experiments = []
        
        # Create output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def select_experiments_interactive(self):
        """Interactive selection of experiments to analyze."""
        print("\n" + "="*80)
        print("SIMPLE LINGUISTIC RELATIVITY EXPERIMENT ANALYZER")
        print("="*80)
        
        # Check if data path exists
        if not os.path.exists(self.data_path):
            print(f"Error: Data path {self.data_path} does not exist!")
            return False
        
        # Discover available experiments
        experiment_dirs = [d for d in os.listdir(self.data_path) 
                          if os.path.isdir(os.path.join(self.data_path, d))]
        
        if not experiment_dirs:
            print("No experiment directories found!")
            return False
        
        # Check which experiments have data files
        available_experiments = []
        for exp_dir in experiment_dirs:
            exp_path = os.path.join(self.data_path, exp_dir)
            csv_files = glob.glob(os.path.join(exp_path, "*.csv"))
            if csv_files:
                available_experiments.append(exp_dir)
        
        if not available_experiments:
            print("No experiments with CSV files found!")
            return False
        
        print(f"\nFound {len(available_experiments)} experiment(s) available for analysis:")
        print(f"Data source: {self.data_path}")
        print("-" * 50)
        
        for i, exp in enumerate(available_experiments, 1):
            exp_path = os.path.join(self.data_path, exp)
            csv_files = glob.glob(os.path.join(exp_path, "*.csv"))
            print(f"{i}. {exp} ({len(csv_files)} CSV file(s))")
        
        print(f"{len(available_experiments) + 1}. ALL EXPERIMENTS")
        print(f"{len(available_experiments) + 2}. EXIT")
        
        while True:
            try:
                choice = input(f"\nSelect experiment(s) to analyze (1-{len(available_experiments) + 2}): ").strip()
                
                if choice.upper() == 'EXIT' or choice == str(len(available_experiments) + 2):
                    print("Exiting...")
                    return False
                
                choice_num = int(choice)
                
                if choice_num == len(available_experiments) + 1:  # ALL
                    self.selected_experiments = available_experiments
                    print(f"Selected: ALL {len(available_experiments)} experiments")
                    break
                elif 1 <= choice_num <= len(available_experiments):
                    selected_exp = available_experiments[choice_num - 1]
                    self.selected_experiments = [selected_exp]
                    print(f"Selected: {selected_exp}")
                    break
                else:
                    print(f"Invalid choice. Please enter a number between 1 and {len(available_experiments) + 2}")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\nExiting...")
                return False
        
        return True
    
    def select_generation_interactive(self):
        """Interactive selection of generation to analyze."""
        print(f"\nScanning for available generations in selected experiments...")
        
        # Discover available generations from selected experiments
        available_generations = set()
        
        for exp_name in self.selected_experiments:
            exp_path = os.path.join(self.data_path, exp_name)
            csv_files = glob.glob(os.path.join(exp_path, "*.csv"))
            
            for csv_file in csv_files:
                filename = os.path.basename(csv_file)
                # Extract generation identifier from filename (e.g., GG2, XG3)
                parts = filename.split('-')
                if len(parts) > 0:
                    generation = parts[0]
                    available_generations.add(generation)
        
        available_generations = sorted(list(available_generations))
        
        if not available_generations:
            print("No generations found in selected experiments!")
            return None
        
        print(f"\nFound {len(available_generations)} generation(s) available:")
        print("-" * 40)
        
        for i, gen in enumerate(available_generations, 1):
            # Map generation codes to friendly names
            gen_name = self._get_generation_name(gen)
            print(f"{i}. {gen} ({gen_name})")
        
        print(f"{len(available_generations) + 1}. ALL GENERATIONS")
        print(f"{len(available_generations) + 2}. BACK TO EXPERIMENT SELECTION")
        
        while True:
            try:
                choice = input(f"\nSelect generation (1-{len(available_generations) + 2}): ").strip()
                choice_num = int(choice)
                
                if choice_num == len(available_generations) + 2:  # BACK
                    return "BACK"
                elif choice_num == len(available_generations) + 1:  # ALL
                    print("Selected: ALL generations")
                    return "ALL"
                elif 1 <= choice_num <= len(available_generations):
                    selected = available_generations[choice_num - 1]
                    gen_name = self._get_generation_name(selected)
                    print(f"Selected: {selected} ({gen_name})")
                    return selected
                else:
                    print(f"Invalid choice. Please enter a number between 1 and {len(available_generations) + 2}")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\nExiting...")
                return None
    
    def _get_generation_name(self, gen_code):
        """Map generation codes to friendly names."""
        mapping = {
            'GG2': 'Gemini 2.0',
            'XG3': 'Grok 3',
            'XG3M': 'Grok 3 Mini',
            'GPT4': 'GPT-4',
            'CL3': 'Claude 3',
            'CL35': 'Claude 3.5',
            'OG4': 'GPT-4o',
            'OO3M': 'GPT-4o Mini'
        }
        return mapping.get(gen_code, 'Unknown Model')

    def discover_experiments(self, generation_filter=None):
        """Discover CSV files for selected experiments and generation."""
        print(f"\nDiscovering experiments in: {self.data_path}")
        
        if not os.path.exists(self.data_path):
            print(f"Error: Data path {self.data_path} does not exist!")
            return {}
        
        for exp_name in self.selected_experiments:
            exp_path = os.path.join(self.data_path, exp_name)
            if not os.path.exists(exp_path):
                print(f"Warning: Experiment directory {exp_path} does not exist!")
                continue
                
            csv_files = glob.glob(os.path.join(exp_path, "*.csv"))
            
            if generation_filter and generation_filter != "ALL":
                csv_files = [f for f in csv_files if os.path.basename(f).startswith(generation_filter)]
            
            if csv_files:
                self.experiments[exp_name] = csv_files
                print(f"Found experiment: {exp_name} with {len(csv_files)} CSV file(s)")
        
        return self.experiments
    
    def load_experiment_data(self, experiment_name):
        """Load and combine all CSV files for an experiment."""
        csv_files = self.experiments.get(experiment_name, [])
        all_data = []
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                df['SourceFile'] = os.path.basename(csv_file)
                all_data.append(df)
                print(f"Loaded {csv_file}: {len(df)} records")
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"Combined data for {experiment_name}: {len(combined_df)} total records")
            return combined_df
        return None
    
    def get_response_column(self, df):
        """Find the main response column in the dataframe."""
        possible_columns = ['ItemResponse', 'PredictedEmotion', 'Response', 'Answer']
        for col in possible_columns:
            if col in df.columns:
                return col
        
        # Look for any column with 'response' or 'answer' in the name
        for col in df.columns:
            if any(word in col.lower() for word in ['response', 'answer', 'prediction']):
                return col
        
        return None
    
    def calculate_cohens_d(self, group1, group2):
        """Calculate Cohen's d effect size between two groups."""
        if len(group1) < 2 or len(group2) < 2:
            return np.nan
        
        mean1, mean2 = np.mean(group1), np.mean(group2)
        n1, n2 = len(group1), len(group2)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0 if mean1 == mean2 else np.inf
        
        return (mean1 - mean2) / pooled_std
    
    def interpret_cohens_d(self, d):
        """Interpret Cohen's d effect size."""
        if pd.isna(d):
            return "Cannot calculate"
        elif np.isinf(d):
            return "Extreme effect"
        elif abs(d) < 0.2:
            return "Negligible effect"
        elif abs(d) < 0.5:
            return "Small effect"
        elif abs(d) < 0.8:
            return "Medium effect"
        else:
            return "Large effect"
    
    def calculate_cross_linguistic_variance(self, df, response_col):
        """Calculate cross-linguistic variance metrics."""
        languages = df['Language'].unique()
        variances = []
        
        print(f"Calculating cross-linguistic variance for {len(languages)} languages...")
        
        # Calculate variance for each response category across languages
        response_categories = df[response_col].unique()
        
        for category in response_categories:
            # Get proportion of this response for each language
            proportions = []
            for lang in languages:
                lang_data = df[df['Language'] == lang]
                if len(lang_data) > 0:
                    prop = len(lang_data[lang_data[response_col] == category]) / len(lang_data)
                    proportions.append(prop)
            
            if len(proportions) > 1:
                variance = np.var(proportions)
                variances.append(variance)
        
        mean_variance = np.mean(variances) if variances else 0
        
        return {
            'individual_variances': variances,
            'mean_variance': mean_variance,
            'response_categories': list(response_categories),
            'num_languages': len(languages),
            'languages': list(languages)
        }
    
    def perform_pairwise_comparisons(self, df, response_col):
        """Perform pairwise comparisons between languages with enhanced Cohen's d reporting."""
        languages = df['Language'].unique()
        comparisons = []
        
        print(f"Performing pairwise comparisons for {len(languages)} languages...")
        
        for lang1, lang2 in combinations(languages, 2):
            lang1_data = df[df['Language'] == lang1]
            lang2_data = df[df['Language'] == lang2]
            
            if len(lang1_data) == 0 or len(lang2_data) == 0:
                continue
            
            # Create contingency table for chi-square test
            lang1_responses = lang1_data[response_col].value_counts()
            lang2_responses = lang2_data[response_col].value_counts()
            
            # Get all possible responses
            all_responses = set(lang1_responses.index) | set(lang2_responses.index)
            
            # Create contingency table
            contingency_table = []
            for response in all_responses:
                row = [
                    lang1_responses.get(response, 0),
                    lang2_responses.get(response, 0)
                ]
                contingency_table.append(row)
            
            contingency_table = np.array(contingency_table)
            
            # Perform chi-square test
            try:
                chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                significant = p_value < 0.05
            except:
                chi2, p_value, significant = np.nan, np.nan, False
            
            # Calculate Cohen's d for the most common responses
            # Convert categorical to numerical for Cohen's d
            lang1_numeric = pd.Categorical(lang1_data[response_col]).codes
            lang2_numeric = pd.Categorical(lang2_data[response_col]).codes
            
            cohens_d = self.calculate_cohens_d(lang1_numeric, lang2_numeric)
            cohens_d_interpretation = self.interpret_cohens_d(cohens_d)
            
            # Categorize effect size
            effect_category = self.categorize_effect_size(cohens_d)
            
            comparison = {
                'language1': lang1,
                'language2': lang2,
                'chi2_statistic': chi2,
                'p_value': p_value,
                'significant': significant,
                'cohens_d': cohens_d,
                'cohens_d_interpretation': cohens_d_interpretation,
                'effect_category': effect_category,
                'lang1_sample_size': len(lang1_data),
                'lang2_sample_size': len(lang2_data)
            }
            
            comparisons.append(comparison)
        
        return comparisons
    
    def categorize_effect_size(self, d):
        """Categorize Cohen's d effect size into negligible/small/medium/large."""
        if pd.isna(d):
            return "cannot_calculate"
        elif np.isinf(d):
            return "extreme"
        elif abs(d) < 0.2:
            return "negligible"
        elif abs(d) < 0.5:
            return "small"
        elif abs(d) < 0.8:
            return "medium"
        else:
            return "large"
    
    def summarize_effect_sizes(self, comparisons):
        """Summarize effect sizes by category with language pair examples."""
        effect_summary = {
            'negligible': [],
            'small': [],
            'medium': [],
            'large': [],
            'extreme': [],
            'cannot_calculate': []
        }
        
        for comp in comparisons:
            category = comp.get('effect_category', 'cannot_calculate')
            lang_pair = f"{comp['language1']} vs {comp['language2']}"
            cohens_d = comp.get('cohens_d', 0)
            
            effect_summary[category].append({
                'pair': lang_pair,
                'cohens_d': cohens_d,
                'significant': comp.get('significant', False)
            })
        
        return effect_summary
    
    def calculate_response_percentages(self, df, response_col):
        """Calculate response percentages for each language, including question/item information."""
        languages = df['Language'].unique()
        response_stats = {}
        
        for lang in languages:
            lang_data = df[df['Language'] == lang]
            if len(lang_data) > 0:
                response_counts = lang_data[response_col].value_counts()
                response_percentages = (response_counts / len(lang_data) * 100).round(2)
                
                response_stats[lang] = {
                    'total_responses': len(lang_data),
                    'response_counts': response_counts.to_dict(),
                    'response_percentages': response_percentages.to_dict(),
                    'most_common_response': response_counts.index[0] if len(response_counts) > 0 else None,
                    'most_common_percentage': response_percentages.iloc[0] if len(response_percentages) > 0 else 0
                }
        
        return response_stats
    
    def calculate_detailed_response_data(self, df, response_col):
        """Calculate detailed response data including question/item information for CSV export."""
        detailed_data = []
        
        # Get question/item column - try different possible column names
        question_col = None
        possible_question_cols = ['ItemText', 'Question', 'ItemID', 'Prompt', 'Task']
        for col in possible_question_cols:
            if col in df.columns:
                question_col = col
                break
        
        if question_col is None:
            # Fallback: create a generic question identifier
            question_col = 'ItemIndex'
            df = df.copy()
            df[question_col] = df.index
        
        languages = df['Language'].unique()
        
        for lang in languages:
            lang_data = df[df['Language'] == lang]
            
            # Group by question and response to get counts
            if question_col in lang_data.columns:
                grouped = lang_data.groupby([question_col, response_col]).size().reset_index(name='Count')
                
                for _, row in grouped.iterrows():
                    question = str(row[question_col])[:100]  # Truncate long questions
                    response = row[response_col]
                    count = row['Count']
                    
                    # Calculate percentage for this language
                    total_for_lang = len(lang_data)
                    percentage = (count / total_for_lang * 100) if total_for_lang > 0 else 0
                    
                    detailed_data.append({
                        'Language': lang,
                        'Question': question,
                        'Response': response,
                        'Count': count,
                        'Percentage': round(percentage, 2)
                    })
            else:
                # Fallback: just use response counts without question breakdown
                response_counts = lang_data[response_col].value_counts()
                total_for_lang = len(lang_data)
                
                for response, count in response_counts.items():
                    percentage = (count / total_for_lang * 100) if total_for_lang > 0 else 0
                    detailed_data.append({
                        'Language': lang,
                        'Question': 'All Items',
                        'Response': response,
                        'Count': count,
                        'Percentage': round(percentage, 2)
                    })
        
        return detailed_data
    
    def create_visualizations(self, df, response_col, experiment_name, variance_data, response_stats):
        """Create simple visualizations."""
        exp_output_dir = os.path.join(self.output_dir, experiment_name)
        if not os.path.exists(exp_output_dir):
            os.makedirs(exp_output_dir)
        
        # 1. Response Distribution Heatmap
        plt.figure(figsize=(14, 8))
        
        # Create percentage matrix for heatmap
        languages = df['Language'].unique()
        responses = df[response_col].unique()
        
        # Limit responses if too many
        if len(responses) > 15:
            top_responses = df[response_col].value_counts().nlargest(15).index
            responses = top_responses
        
        percentage_matrix = []
        lang_labels = []
        
        for lang in languages:
            lang_data = df[df['Language'] == lang]
            row = []
            for response in responses:
                count = len(lang_data[lang_data[response_col] == response])
                percentage = (count / len(lang_data) * 100) if len(lang_data) > 0 else 0
                row.append(percentage)
            percentage_matrix.append(row)
            lang_labels.append(lang.replace(' # ', '\n'))
        
        sns.heatmap(percentage_matrix, 
                   xticklabels=[str(r)[:20] for r in responses],  # Truncate long labels
                   yticklabels=lang_labels,
                   annot=True, 
                   fmt='.1f', 
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Percentage (%)'})
        
        plt.title(f'{experiment_name}: Response Distribution by Language (%)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Response Category', fontsize=12)
        plt.ylabel('Language', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        heatmap_path = os.path.join(exp_output_dir, f'{experiment_name}_response_heatmap.png')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved heatmap: {heatmap_path}")
        
        # 2. Variance Bar Chart (IMPROVED)
        plt.figure(figsize=(12, 6))
        
        if variance_data['individual_variances']:
            response_categories = variance_data['response_categories']
            variances = variance_data['individual_variances']
            
            # Create more informative x-axis labels
            if len(response_categories) <= 10:
                # If few categories, show full names
                x_labels = [str(cat)[:15] for cat in response_categories]  # Truncate to 15 chars
            else:
                # If many categories, show abbreviated names with indices
                x_labels = [f"{i+1}: {str(cat)[:8]}" for i, cat in enumerate(response_categories)]
            
            bars = plt.bar(range(len(variances)), variances, 
                          color='skyblue', edgecolor='navy', alpha=0.7)
            
            plt.axhline(y=variance_data['mean_variance'], color='red', linestyle='--', 
                       linewidth=2, label=f'Mean Variance: {variance_data["mean_variance"]:.4f}')
            
            plt.title(f'{experiment_name}: Cross-Linguistic Variance by Response Category', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Response Categories', fontsize=12)
            plt.ylabel('Variance', fontsize=12)
            
            # Set x-axis labels
            plt.xticks(range(len(variances)), x_labels, rotation=45, ha='right')
            
            # Add value labels on bars
            for i, (bar, var) in enumerate(zip(bars, variances)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(variances)*0.01, 
                        f'{var:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            variance_path = os.path.join(exp_output_dir, f'{experiment_name}_variance_chart.png')
            plt.savefig(variance_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved variance chart: {variance_path}")
        
        # 3. Most Common Response by Language (IMPROVED)
        plt.figure(figsize=(12, 8))
        
        languages = []
        percentages = []
        response_labels = []
        
        for lang, stats in response_stats.items():
            if stats['most_common_response'] is not None:
                languages.append(lang.replace(' # ', '\n'))
                percentages.append(stats['most_common_percentage'])
                response_labels.append(str(stats['most_common_response'])[:10])  # Truncate long responses
        
        if languages:
            bars = plt.bar(range(len(languages)), percentages, 
                          color='lightgreen', edgecolor='darkgreen', alpha=0.7)
            
            plt.title(f'{experiment_name}: Most Common Response Percentage by Language', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Language', fontsize=12)
            plt.ylabel('Percentage of Most Common Response (%)', fontsize=12)
            plt.xticks(range(len(languages)), languages, rotation=45, ha='right')
            
            # Add improved labels on bars showing both percentage and response
            for i, (bar, pct, response) in enumerate(zip(bars, percentages, response_labels)):
                # Show percentage above bar
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
                
                # Show response name inside or below bar
                if pct > 20:  # If bar is tall enough, put text inside
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                            f'"{response}"', ha='center', va='center', 
                            fontweight='bold', fontsize=9, color='darkgreen',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                else:  # If bar is short, put text below
                    plt.text(bar.get_x() + bar.get_width()/2, -max(percentages)*0.05, 
                            f'"{response}"', ha='center', va='top', 
                            fontweight='bold', fontsize=8, rotation=45)
            
            # Add a note explaining what the chart shows
            plt.figtext(0.5, 0.02, 
                       'Note: Shows the percentage and type of most frequent response for each language',
                       ha='center', fontsize=10, style='italic')
            
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.15)  # Make room for the note
            
            common_response_path = os.path.join(exp_output_dir, f'{experiment_name}_common_responses.png')
            plt.savefig(common_response_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved common responses chart: {common_response_path}")
    
    def analyze_experiment(self, experiment_name):
        """Analyze a single experiment for cross-linguistic variance."""
        print(f"\n{'='*60}")
        print(f"ANALYZING: {experiment_name}")
        print(f"{'='*60}")
        
        df = self.load_experiment_data(experiment_name)
        if df is None:
            print(f"No data loaded for {experiment_name}")
            return None
        
        response_col = self.get_response_column(df)
        if not response_col:
            print(f"No response column found for {experiment_name}")
            return None
        
        print(f"Using response column: {response_col}")
        
        # Calculate cross-linguistic variance
        variance_data = self.calculate_cross_linguistic_variance(df, response_col)
        
        # Perform pairwise comparisons
        pairwise_comparisons = self.perform_pairwise_comparisons(df, response_col)
        
        # Calculate response percentages (for summary)
        response_stats = self.calculate_response_percentages(df, response_col)
        
        # Calculate detailed response data (for CSV export)
        detailed_response_data = self.calculate_detailed_response_data(df, response_col)
        
        # Summarize effect sizes
        effect_size_summary = self.summarize_effect_sizes(pairwise_comparisons)
        
        # Create visualizations
        self.create_visualizations(df, response_col, experiment_name, variance_data, response_stats)
        
        # Calculate overall statistics
        significant_comparisons = sum(1 for comp in pairwise_comparisons if comp['significant'])
        total_comparisons = len(pairwise_comparisons)
        
        # Count effect sizes by category
        effect_counts = {category: len(effects) for category, effects in effect_size_summary.items()}
        non_negligible_effects = effect_counts['small'] + effect_counts['medium'] + effect_counts['large'] + effect_counts['extreme']
        
        # Test if mean variance is significantly different from zero
        if variance_data['individual_variances']:
            # One-sample t-test against zero
            t_stat, variance_p_value = stats.ttest_1samp(variance_data['individual_variances'], 0)
            variance_significant = variance_p_value < 0.05
        else:
            variance_significant = False
            variance_p_value = np.nan
        
        results = {
            'experiment_name': experiment_name,
            'total_languages': variance_data['num_languages'],
            'total_responses': len(df),
            'response_column': response_col,
            'variance_data': variance_data,
            'response_statistics': response_stats,
            'detailed_response_data': detailed_response_data,
            'pairwise_comparisons': pairwise_comparisons,
            'effect_size_summary': effect_size_summary,
            'summary_statistics': {
                'mean_cross_linguistic_variance': variance_data['mean_variance'],
                'variance_significantly_different_from_zero': variance_significant,
                'variance_p_value': variance_p_value,
                'total_pairwise_comparisons': total_comparisons,
                'significant_comparisons': significant_comparisons,
                'percentage_significant': (significant_comparisons / total_comparisons * 100) if total_comparisons > 0 else 0,
                'effect_size_counts': effect_counts,
                'non_negligible_effects': non_negligible_effects,
                'interpretation': self.interpret_results(variance_data['mean_variance'], 
                                                       significant_comparisons, 
                                                       total_comparisons, 
                                                       non_negligible_effects)
            }
        }
        
        # Save results
        self.save_results(experiment_name, results)
        
        return results
    
    def interpret_results(self, mean_variance, significant_comparisons, total_comparisons, non_negligible_effects):
        """Provide interpretation of the results."""
        interpretations = []
        
        # Variance interpretation
        if mean_variance > 0.05:
            interpretations.append("HIGH cross-linguistic variance detected")
        elif mean_variance > 0.02:
            interpretations.append("MODERATE cross-linguistic variance detected")
        else:
            interpretations.append("LOW cross-linguistic variance detected")
        
        # Significance interpretation
        sig_percentage = (significant_comparisons / total_comparisons * 100) if total_comparisons > 0 else 0
        if sig_percentage > 50:
            interpretations.append("MAJORITY of language pairs show statistically significant differences")
        elif sig_percentage > 25:
            interpretations.append("MODERATE number of language pairs show significant differences")
        else:
            interpretations.append("FEW language pairs show significant differences")
        
        # Effect size interpretation
        if non_negligible_effects > 0:
            interpretations.append(f"{non_negligible_effects} language pairs show non-negligible effect sizes")
        
        return " | ".join(interpretations)
    
    def save_results(self, experiment_name, results):
        """Save results to files."""
        exp_output_dir = os.path.join(self.output_dir, experiment_name)
        if not os.path.exists(exp_output_dir):
            os.makedirs(exp_output_dir)
        
        # Save summary report
        summary_path = os.path.join(exp_output_dir, f'{experiment_name}_summary_report.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"SIMPLE LINGUISTIC RELATIVITY ANALYSIS\n")
            f.write(f"Experiment: {experiment_name}\n")
            f.write(f"="*60 + "\n\n")
            
            # Basic info
            f.write(f"BASIC INFORMATION\n")
            f.write(f"Total Languages: {results['total_languages']}\n")
            f.write(f"Total Responses: {results['total_responses']}\n")
            f.write(f"Response Column: {results['response_column']}\n")
            f.write(f"Languages: {', '.join(results['variance_data']['languages'])}\n\n")
            
            # Variance analysis
            f.write(f"CROSS-LINGUISTIC VARIANCE ANALYSIS\n")
            f.write(f"Mean Cross-Linguistic Variance: {results['summary_statistics']['mean_cross_linguistic_variance']:.6f}\n")
            f.write(f"Variance Significantly Different from Zero: {results['summary_statistics']['variance_significantly_different_from_zero']}\n")
            f.write(f"Variance p-value: {results['summary_statistics']['variance_p_value']:.6f}\n\n")
            
            # Statistical significance
            f.write(f"STATISTICAL SIGNIFICANCE ANALYSIS\n")
            f.write(f"Total Pairwise Comparisons: {results['summary_statistics']['total_pairwise_comparisons']}\n")
            f.write(f"Significant Comparisons (p < 0.05): {results['summary_statistics']['significant_comparisons']}\n")
            f.write(f"Percentage Significant: {results['summary_statistics']['percentage_significant']:.1f}%\n\n")
            
            # Effect sizes
            f.write(f"EFFECT SIZE ANALYSIS (Cohen's d)\n")
            f.write(f"Effect Sizes by Category:\n")
            effect_counts = results['summary_statistics']['effect_size_counts']
            for category, count in effect_counts.items():
                if count > 0:
                    f.write(f"  {category.capitalize()}: {count} pairs\n")
            
            # Show examples of non-negligible effects
            f.write(f"\nNon-Negligible Effect Examples:\n")
            for category in ['small', 'medium', 'large', 'extreme']:
                effects = results['effect_size_summary'].get(category, [])
                if effects:
                    f.write(f"  {category.capitalize()} effects:\n")
                    for effect in effects:  # Show ALL effects, not just first 5
                        d_val = effect['cohens_d']
                        sig_marker = " *" if effect['significant'] else ""
                        f.write(f"    {effect['pair']}: d={d_val:.3f}{sig_marker}\n")
            f.write(f"  (* = statistically significant)\n\n")
            
            # Aggregate statistics
            f.write(f"AGGREGATE STATISTICS\n")
            f.write(f"Total Non-Negligible Effects: {results['summary_statistics']['non_negligible_effects']}\n")
            f.write(f"Total Pairwise Comparisons: {results['summary_statistics']['total_pairwise_comparisons']}\n")
            f.write(f"Percentage Non-Negligible: {(results['summary_statistics']['non_negligible_effects'] / results['summary_statistics']['total_pairwise_comparisons'] * 100):.1f}%\n\n")
            
            # Interpretation
            f.write(f"\nOVERALL INTERPRETATION\n")
            f.write(f"-"*40 + "\n")
            f.write(f"{results['summary_statistics']['interpretation']}\n")
        
        # Save pairwise comparisons CSV
        if results['pairwise_comparisons']:
            comparisons_df = pd.DataFrame(results['pairwise_comparisons'])
            comparisons_path = os.path.join(exp_output_dir, f'{experiment_name}_pairwise_comparisons.csv')
            comparisons_df.to_csv(comparisons_path, index=False)
            print(f"Saved pairwise comparisons: {comparisons_path}")
        
        # Save response percentages CSV
        response_data = []
        for lang, stats in results['response_statistics'].items():
            for response, percentage in stats['response_percentages'].items():
                response_data.append({
                    'Language': lang,
                    'Response': response,
                    'Count': stats['response_counts'][response],
                    'Percentage': percentage
                })
        
        if response_data:
            response_df = pd.DataFrame(response_data)
            response_path = os.path.join(exp_output_dir, f'{experiment_name}_response_percentages.csv')
            response_df.to_csv(response_path, index=False)
            print(f"Saved response percentages: {response_path}")
        
        # Save detailed response data CSV (with question information)
        if results['detailed_response_data']:
            detailed_df = pd.DataFrame(results['detailed_response_data'])
            detailed_path = os.path.join(exp_output_dir, f'{experiment_name}_detailed_response_data.csv')
            detailed_df.to_csv(detailed_path, index=False)
            print(f"Saved detailed response data: {detailed_path}")
        
        print(f"Saved summary report: {summary_path}")
    
    def run_interactive_analysis(self):
        """Run interactive analysis with experiment and generation selection."""
        # Step 1: Select experiments
        if not self.select_experiments_interactive():
            return
        
        # Step 2: Select generation
        while True:
            generation_filter = self.select_generation_interactive()
            if generation_filter is None:
                return
            elif generation_filter == "BACK":
                if not self.select_experiments_interactive():
                    return
                continue
            else:
                break
        
        # Step 3: Discover and analyze
        self.discover_experiments(generation_filter)
        
        if not self.experiments:
            print("No experiments found with the selected criteria!")
            return
        
        print(f"\nStarting analysis of {len(self.experiments)} experiment(s):")
        for exp_name in self.experiments.keys():
            print(f"  - {exp_name}")
        
        # Step 4: Analyze each experiment
        for experiment_name in self.experiments.keys():
            try:
                result = self.analyze_experiment(experiment_name)
                if result:
                    self.results[experiment_name] = result
                    print(f"✓ Completed: {experiment_name}")
                else:
                    print(f"✗ Failed: {experiment_name}")
            except Exception as e:
                print(f"✗ Error analyzing {experiment_name}: {e}")
        
        # Step 5: Generate master summary
        self.generate_master_summary()
        
        print(f"\nAnalysis complete! Results saved to: {self.output_dir}")
    
    def run_all_experiments(self, generation_filter=None):
        """Run analysis for all discovered experiments (backward compatibility)."""
        # Set all experiments as selected
        if not os.path.exists(self.data_path):
            print(f"Error: Data path {self.data_path} does not exist!")
            return
        
        experiment_dirs = [d for d in os.listdir(self.data_path) 
                          if os.path.isdir(os.path.join(self.data_path, d))]
        
        available_experiments = []
        for exp_dir in experiment_dirs:
            exp_path = os.path.join(self.data_path, exp_dir)
            csv_files = glob.glob(os.path.join(exp_path, "*.csv"))
            if csv_files:
                available_experiments.append(exp_dir)
        
        self.selected_experiments = available_experiments
        
        self.discover_experiments(generation_filter)
        
        if not self.experiments:
            print("No experiments found!")
            return
        
        print(f"\nFound {len(self.experiments)} experiments to analyze:")
        for exp_name in self.experiments.keys():
            print(f"  - {exp_name}")
        
        for experiment_name in self.experiments.keys():
            try:
                result = self.analyze_experiment(experiment_name)
                if result:
                    self.results[experiment_name] = result
                    print(f"✓ Completed: {experiment_name}")
                else:
                    print(f"✗ Failed: {experiment_name}")
            except Exception as e:
                print(f"✗ Error analyzing {experiment_name}: {e}")
        
        # Generate master summary
        self.generate_master_summary()
        
        print(f"\nAnalysis complete! Results saved to: {self.output_dir}")
    
    def generate_master_summary(self):
        """Generate a master summary across all experiments."""
        if not self.results:
            return
        
        master_path = os.path.join(self.output_dir, 'MASTER_SUMMARY.txt')
        
        with open(master_path, 'w', encoding='utf-8') as f:
            f.write("MASTER SUMMARY - SIMPLE LINGUISTIC RELATIVITY ANALYSIS\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Data Source: {self.data_path}\n")
            f.write(f"Total Experiments Analyzed: {len(self.results)}\n")
            f.write(f"Experiments: {', '.join(self.results.keys())}\n\n")
            
            # Aggregate statistics
            total_languages = set()
            all_variances = []
            all_sig_percentages = []
            all_non_negligible_effects = []
            
            f.write("EXPERIMENT SUMMARIES\n")
            f.write("-"*40 + "\n")
            
            for exp_name, result in self.results.items():
                summary = result['summary_statistics']
                total_languages.update(result['variance_data']['languages'])
                all_variances.append(summary['mean_cross_linguistic_variance'])
                all_sig_percentages.append(summary['percentage_significant'])
                all_non_negligible_effects.append(summary['non_negligible_effects'])
                
                f.write(f"\n{exp_name}:\n")
                f.write(f"  Languages: {result['total_languages']}\n")
                f.write(f"  Mean Variance: {summary['mean_cross_linguistic_variance']:.6f}\n")
                f.write(f"  Significant Comparisons: {summary['percentage_significant']:.1f}%\n")
                f.write(f"  Non-Negligible Effects: {summary['non_negligible_effects']}\n")
                f.write(f"  Interpretation: {summary['interpretation']}\n")
            
            f.write(f"\nAGGREGATE STATISTICS\n")
            f.write(f"-"*40 + "\n")
            f.write(f"Total Unique Languages: {len(total_languages)}\n")
            f.write(f"Average Cross-Linguistic Variance: {np.mean(all_variances):.6f}\n")
            f.write(f"Average Percentage of Significant Comparisons: {np.mean(all_sig_percentages):.1f}%\n")
            f.write(f"Total Non-Negligible Effects: {sum(all_non_negligible_effects)}\n")
            
            # Overall interpretation
            f.write(f"\nOVERALL INTERPRETATION\n")
            f.write(f"-"*40 + "\n")
            avg_variance = np.mean(all_variances)
            avg_sig_pct = np.mean(all_sig_percentages)
            
            if avg_variance > 0.05 and avg_sig_pct > 50:
                interpretation = "STRONG evidence for cross-linguistic differences in bilingual model preferences"
            elif avg_variance > 0.02 and avg_sig_pct > 25:
                interpretation = "MODERATE evidence for cross-linguistic differences"
            else:
                interpretation = "WEAK evidence for cross-linguistic differences"
            
            f.write(f"{interpretation}\n")
        
        print(f"Master summary saved: {master_path}")


def main():
    """Main function to run the simple analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple Linguistic Relativity Analysis')
    parser.add_argument('--generation', '-g', help='Generation filter (e.g., GG2, XG3)')
    parser.add_argument('--data-path', default='api_generations/lr_experiments_english',
                       help='Path to experiment data')
    parser.add_argument('--output-dir', default='simple_lr_experiment_results',
                       help='Output directory for results')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Run in interactive mode (default)')
    parser.add_argument('--all', '-a', action='store_true',
                       help='Run all experiments without interaction')
    
    args = parser.parse_args()
    
    print("Simple Linguistic Relativity Analyzer")
    print("="*50)
    
    analyzer = SimpleLRAnalyzer(
        data_path=args.data_path,
        output_dir=args.output_dir
    )
    
    if args.all and args.generation:
        # Non-interactive mode: run all experiments with specified generation
        analyzer.run_all_experiments(generation_filter=args.generation)
    elif args.all:
        # Non-interactive mode: run all experiments, all generations
        analyzer.run_all_experiments()
    else:
        # Interactive mode (default)
        analyzer.run_interactive_analysis()


if __name__ == "__main__":
    main() 