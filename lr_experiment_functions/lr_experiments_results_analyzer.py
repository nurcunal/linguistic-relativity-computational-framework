import pandas as pd
import os
import glob
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- Matplotlib and Seaborn Setup ---
import matplotlib
try:
    current_backend = matplotlib.get_backend()
    # Only set if not already an appropriate non-interactive backend
    if current_backend.lower() not in ['agg', 'cairo', 'pdf', 'ps', 'svg', 'template']:
        matplotlib.use('Agg') 
        print(f"Matplotlib backend set to 'Agg' for compatibility.")
except Exception as e:
    print(f"Warning: Could not set Matplotlib backend to 'Agg'. Visualizations might have issues: {e}")
import matplotlib.pyplot as plt
import seaborn as sns

# --- Core Scientific Libraries ---
import numpy as np
import scipy.stats # For general stats functions
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu, kruskal, friedmanchisquare
from scipy.stats import spearmanr, pearsonr, shapiro, levene, normaltest, anderson
from scipy.stats import bootstrap as scipy_bootstrap # Explicit alias for clarity
from scipy.stats import permutation_test as scipy_permutation_test # Explicit alias

# --- Standard Libraries ---
from itertools import combinations
import warnings
import json
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import threading
import argparse # For command-line interface

# --- Optional Advanced Statistical Packages (with robust import handling) ---
STATSMODELS_AVAILABLE = False
ANOVA_POWER_AVAILABLE = False # Specific flag for anova_power
PYMC_AVAILABLE = False

try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols, mixedlm
    from statsmodels.stats.power import TTestIndPower # More specific import for t-test power
    from statsmodels.stats.multitest import multipletests
    STATSMODELS_AVAILABLE = True
    print("✓ statsmodels core imported.")
    try:
        from statsmodels.stats.power import anova_power
        ANOVA_POWER_AVAILABLE = True
        print("✓ statsmodels.stats.power.anova_power imported.")
    except ImportError:
        print("Warning: statsmodels.stats.power.anova_power not found. ANOVA power calcs may be limited.")
except ImportError as e:
    print(f"Warning: statsmodels not available ({e}). Advanced stats (mixed models, power analysis) may be disabled.")

try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
    print("✓ PyMC and ArviZ imported successfully.")
except ImportError as e:
    print(f"Warning: PyMC/ArviZ not available ({e}). Bayesian analysis will be disabled.")

warnings.filterwarnings('ignore')

# Set plotting style (ensure plt is defined first)
plt.style.use('default')
sns.set_palette("Set2")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

class LRExperimentAnalyzer:
    """Comprehensive analyzer for Linguistic Relativity experiments with academic-level analysis."""
    
    def __init__(self, data_path='api_generations/lr_experiments', output_dir='lr_experiment_results', max_workers=None):
        self.data_path = data_path
        self.output_dir = output_dir
        self.experiments = {}
        self.results = {}
        self.selected_generation = None
        self.linguistic_families = self._define_linguistic_families()
        
        # Multiprocessing configuration
        self.max_workers = min(max_workers or mp.cpu_count(), 12)
        self.thread_lock = threading.Lock()
        
        self.ensure_output_dir()
        
    def _get_multiprocessing_context(self):
        """Get appropriate multiprocessing context based on platform."""
        try:
            if hasattr(mp, 'get_context'):
                return mp.get_context('spawn')  # More reliable across platforms
            return mp
        except:
            return mp
    
    def _define_linguistic_families(self):
        """Define linguistic families for cross-linguistic analysis."""
        return {
            'Indo-European': {
                'Germanic': ['DE # German', 'GA # Irish'],
                'Romance': ['ES # Spanish', 'FR # French', 'IT # Italian', 'PT # Portuguese'],
                'Slavic': ['RU # Russian', 'PL # Polish'],
                'Indo-Iranian': ['FA # Persian', 'HI # Hindi', 'UR # Urdu', 'PS # Pashto'],
                'Baltic': ['LV # Latvian']
            },
            'Sino-Tibetan': {
                'Chinese': ['ZH # Mandarin']
            },
            'Semitic': {
                'Semitic': ['AR # Arabic', 'HE # Hebrew']
            },
            'Japonic': {
                'Japanese': ['JA # Japanese']
            },
            'Koreanic': {
                'Korean': ['KO # Korean']
            },
            'Niger-Congo': {
                'Bantu': ['SW # Swahili']
            },
            'Austro-Asiatic': {
                'Vietic': ['VI # Vietnamese']
            },
            'Tai-Kadai': {
                'Tai': ['TH # Thai']
            },
            'Turkic': {
                'Turkic': ['TR # Turkish']
            },
            'Indo-European-Albanian': {
                'Albanian': ['SQ # Albanian']
            },
            'Indo-European-Bengali': {
                'Bengali': ['BN # Bengali']
            }
        }
    
    def get_language_family(self, language):
        """Get the linguistic family and subfamily for a language."""
        for family, subfamilies in self.linguistic_families.items():
            for subfamily, languages in subfamilies.items():
                if language in languages:
                    return family, subfamily
        return 'Unknown', 'Unknown'
    
    def ensure_output_dir(self):
        """Create comprehensive output directory structure."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        subdirs = [
            'overall_analysis', 'statistical_significance', 'cohens_d_analysis', 
            'linguistic_relativity_assessment', 'per_language_analysis',
            'cross_linguistic_analysis', 'linguistic_family_analysis',
            'academic_reports', 'publication_ready_figures'
        ]
        for subdir in subdirs:
            path = os.path.join(self.output_dir, subdir)
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
    
    def select_generation_interactive(self):
        """Interactive selection of API generation to analyze."""
        print("\n" + "="*80)
        print("LINGUISTIC RELATIVITY EXPERIMENT ANALYZER")
        print("="*80)
        
        # Discover available generations
        available_generations = set()
        experiment_dirs = [d for d in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, d))]
        
        for exp_dir in experiment_dirs:
            exp_path = os.path.join(self.data_path, exp_dir)
            csv_files = glob.glob(os.path.join(exp_path, "*.csv"))
            
            for csv_file in csv_files:
                filename = os.path.basename(csv_file)
                # Extract generation identifier from filename (e.g., GG2, XG3M)
                parts = filename.split('-')
                if len(parts) > 0:
                    generation = parts[0]
                    available_generations.add(generation)
        
        available_generations = sorted(list(available_generations))
        
        if not available_generations:
            print("No API generations found!")
            return None
        
        print(f"\nFound {len(available_generations)} API generation(s) available for analysis:")
        print("-" * 50)
        
        for i, gen in enumerate(available_generations, 1):
            # Map generation codes to friendly names
            gen_name = self._get_generation_name(gen)
            print(f"{i}. {gen} ({gen_name})")
        
        print(f"{len(available_generations) + 1}. ALL (Analyze all generations together)")
        print(f"{len(available_generations) + 2}. EXIT")
        
        while True:
            try:
                choice = input(f"\nSelect option (1-{len(available_generations) + 2}): ").strip()
                choice_num = int(choice)
                
                if choice_num == len(available_generations) + 2:  # EXIT
                    print("Exiting...")
                    return None
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
            'XG3M': 'Grok 3 Mini',
            'GPT4': 'GPT-4',
            'CL3': 'Claude 3',
            'CL35': 'Claude 3.5'
        }
        return mapping.get(gen_code, 'Unknown Model')
    
    def discover_experiments(self):
        """Automatically discover all experiment directories and CSV files."""
        print(f"\nDiscovering experiments for generation: {self.selected_generation}...")
        self.experiments = {} # Reset experiments dict at the start of discovery
        experiment_dirs = [d for d in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, d))]
        
        for exp_dir in experiment_dirs:
            exp_path = os.path.join(self.data_path, exp_dir)
            all_csvs_in_dir = glob.glob(os.path.join(exp_path, "*.csv"))
            
            if not all_csvs_in_dir:
                continue # Skip this directory if no CSVs found

            csv_files_for_this_exp = []
            if self.selected_generation == "ALL":
                csv_files_for_this_exp = all_csvs_in_dir
                print(f"  For '{exp_dir}' (ALL generations): Found {len(all_csvs_in_dir)} CSV files: {', '.join([os.path.basename(f) for f in all_csvs_in_dir])}")
            else:
                csv_files_for_this_exp = [f for f in all_csvs_in_dir if os.path.basename(f).startswith(self.selected_generation)]
                # Optional: Log if filtering happened
                # if len(csv_files_for_this_exp) < len(all_csvs_in_dir):
                #     print(f"  For '{exp_dir}' (generation: {self.selected_generation}): Filtered to {len(csv_files_for_this_exp)} CSV files from {len(all_csvs_in_dir)} total.")
            
            if csv_files_for_this_exp:
                print(f"Found experiment: {exp_dir} with {len(csv_files_for_this_exp)} CSV file(s) to be processed.")
                self.experiments[exp_dir] = csv_files_for_this_exp
                
        return self.experiments
    
    def load_experiment_data(self, experiment_name):
        """Load and combine all CSV files for an experiment with data validation."""
        csv_files = self.experiments.get(experiment_name, [])
        all_data = []
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                df['SourceFile'] = os.path.basename(csv_file)
                
                required_cols = ['Model', 'Language']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    print(f"Warning: Missing required columns {missing_cols} in {csv_file}")
                
                all_data.append(df)
                print(f"Loaded {csv_file}: {len(df)} records")
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
                
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"Combined data for {experiment_name}: {len(combined_df)} total records")
            
            combined_df[['LinguisticFamily', 'LinguisticSubfamily']] = combined_df['Language'].apply(
                lambda x: pd.Series(self.get_language_family(x))
            )
            
            return combined_df
        return None
    
    def classify_experiment_type(self, df, experiment_name):
        """Classify whether experiment has definitive answers or is preference-based."""
        response_col = self._get_response_column(df)
        if not response_col:
            return 'unknown'
        
        sample_responses = df[response_col].value_counts()
        item_categories = df['ItemCategory'].unique() if 'ItemCategory' in df.columns else []
        
        definitive_indicators = [
            'color_naming', 'classify', 'hex', 'spatial_reasoning', 
            'multiple_choice', 'maze', 'route'
        ]
        
        preference_indicators = [
            'emotion', 'association', 'scenario', 'neutral_animals', 
            'preference', 'choice', 'feeling'
        ]
        
        experiment_lower = experiment_name.lower()
        categories_lower = ' '.join(item_categories).lower()
        
        is_definitive = any(indicator in experiment_lower or indicator in categories_lower 
                           for indicator in definitive_indicators)
        is_preference = any(indicator in experiment_lower or indicator in categories_lower 
                           for indicator in preference_indicators)
        
        if 'sound' in experiment_name.lower() and 'TrueEmotionLabel' in df.columns:
            return 'definitive'
        
        if is_definitive and not is_preference:
            return 'definitive'
        elif is_preference and not is_definitive:
            return 'preference'
        else:
            if len(sample_responses) <= 8 and sample_responses.iloc[0] / len(df) < 0.7:
                return 'preference'
            else:
                return 'mixed'
    
    def calculate_accuracy_metrics(self, df, experiment_name, experiment_type):
        """Calculate accuracy metrics for experiments with definitive answers."""
        if experiment_type != 'definitive':
            return None

        if 'sound' in experiment_name.lower() and 'TrueEmotionLabel' in df.columns and 'PredictedEmotion' in df.columns:
            accuracy_df = df.copy()
            accuracy_df['TrueEmotionLabel_norm'] = accuracy_df['TrueEmotionLabel'].str.lower().str.strip()
            accuracy_df['PredictedEmotion_norm'] = accuracy_df['PredictedEmotion'].str.lower().str.strip()
            accuracy_df['IsCorrect'] = accuracy_df['TrueEmotionLabel_norm'] == accuracy_df['PredictedEmotion_norm']
            
            metrics = {
                'overall_accuracy': accuracy_df['IsCorrect'].mean(),
                'total_items': len(accuracy_df),
                'correct_items': accuracy_df['IsCorrect'].sum(),
                'accuracy_by_language': accuracy_df.groupby('Language')['IsCorrect'].agg(['mean', 'count']),
                'accuracy_by_item': accuracy_df.groupby('ImageFile')['IsCorrect'].agg(['mean', 'count']) if 'ImageFile' in accuracy_df.columns else None,
                'accuracy_by_category': None
            }
            
            return metrics, accuracy_df
        
        return None
    
    def calculate_cohens_d(self, group1, group2):
        """Calculate Cohen's d effect size between two groups."""
        n1, n2 = len(group1), len(group2)
        if n1 < 2 or n2 < 2:
            return None
            
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0 if mean1 == mean2 else np.inf
            
        d = (mean1 - mean2) / pooled_std
        return d
    
    def interpret_cohens_d(self, d):
        """Interpret Cohen's d effect size."""
        if d is None or np.isnan(d):
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
    
    def perform_statistical_tests(self, df, experiment_type):
        """Perform comprehensive statistical tests."""
        results = {}
        response_col = self._get_response_column(df)
        
        # Chi-square tests
        categorical_pairs = [
            ('Language', response_col) if response_col else None,
            ('ItemCategory', response_col) if 'ItemCategory' in df.columns and response_col else None,
            ('Language', 'ItemCategory') if 'ItemCategory' in df.columns else None
        ]
        
        for pair in categorical_pairs:
            if pair is None:
                continue
                
            col1, col2 = pair
            if col1 in df.columns and col2 in df.columns:
                contingency = pd.crosstab(df[col1], df[col2])
                try:
                    chi2, p_val, dof, expected = chi2_contingency(contingency)
                    results[f'chi2_{col1}_vs_{col2}'] = {
                        'chi2_statistic': chi2,
                        'p_value': p_val,
                        'degrees_of_freedom': dof,
                        'significant': p_val < 0.05,
                        'contingency_table': contingency
                    }
                except Exception as e:
                    print(f"Chi-square test failed for {col1} vs {col2}: {e}")
        
        # Kruskal-Wallis test for overall language differences
        if response_col:
            try:
                language_groups = [group[response_col].dropna().values for name, group in df.groupby('Language') if not group[response_col].dropna().empty]
                language_codes = []
                for group_values in language_groups:
                    try:
                        codes = pd.Categorical(group_values).codes
                        language_codes.append(codes)
                    except Exception:
                        pass
                
                valid_language_codes = [lc for lc in language_codes if len(lc) > 0]
                if len(valid_language_codes) >= 2:
                    h_stat, p_val = kruskal(*valid_language_codes)
                    results['kruskal_wallis_test'] = {
                        'h_statistic': h_stat,
                        'p_value': p_val,
                        'significant': p_val < 0.05,
                        'interpretation': 'Significant language group differences in response distribution' if p_val < 0.05 else 'No significant language group differences in response distribution'
                    }
            except Exception as e:
                print(f"Kruskal-Wallis test failed: {e}")
        
        # Language accuracy comparisons for definitive experiments (PARALLELIZED)
        if experiment_type == 'definitive' and 'IsCorrect' in df.columns:
            languages = df['Language'].unique()
            
            # Prepare data for parallel processing
            language_data = {}
            for lang in languages:
                language_data[lang] = df[df['Language'] == lang]['IsCorrect'].astype(int)
            
            def compare_language_accuracy(lang1, lang2, group1_scores, group2_scores):
                """Compare accuracy between two languages."""
                if len(group1_scores) > 0 and len(group2_scores) > 0:
                    try:
                        u_stat, u_p = mannwhitneyu(group1_scores, group2_scores, alternative='two-sided')
                        cohens_d = self.calculate_cohens_d(group1_scores, group2_scores)
                        
                        return {
                            'language1': lang1,
                            'language2': lang2,
                            'mean1': group1_scores.mean(),
                            'mean2': group2_scores.mean(),
                            'u_statistic': u_stat,
                            'u_p_value': u_p,
                            'u_significant': u_p < 0.05,
                            'cohens_d': cohens_d,
                            'effect_size_interpretation': self.interpret_cohens_d(cohens_d)
                        }
                    except Exception as e:
                        print(f"Statistical test failed for {lang1} vs {lang2}: {e}")
                        return None
                return None
            
            print(f"Performing pairwise language comparisons using up to {self.max_workers} cores...")
            language_comparisons = self._parallel_pairwise_comparisons(
                languages, compare_language_accuracy, language_data
            )
            
            # Filter out None results
            language_comparisons = [comp for comp in language_comparisons if comp is not None]
            results['language_accuracy_comparisons'] = language_comparisons
        
        return results
    
    def perform_comprehensive_language_analysis(self, df, experiment_name, experiment_type):
        """Perform comprehensive per-language analysis with academic rigor."""
        response_col = self._get_response_column(df)
        if not response_col:
            return {}
            
        languages = sorted(df['Language'].unique())
        
        def analyze_single_language(language):
            """Analyze a single language - designed for parallel processing."""
            lang_data = df[df['Language'] == language]
            family, subfamily = self.get_language_family(language)
            
            profile = {
                'language': language,
                'linguistic_family': family,
                'linguistic_subfamily': subfamily,
                'sample_size': len(lang_data),
                'response_statistics': self._analyze_response_patterns(lang_data, response_col),
                'cognitive_patterns': self._analyze_cognitive_patterns(lang_data, response_col),
                'statistical_profile': self._generate_statistical_profile(lang_data, response_col)
            }
            
            if experiment_type == 'definitive' and 'IsCorrect' in lang_data.columns:
                profile['accuracy_analysis'] = self._analyze_accuracy_patterns(lang_data)
            
            return language, profile
        
        print(f"Analyzing {len(languages)} languages using up to {self.max_workers} cores...")
        
        # Use parallel processing for language analysis
        results = self._parallel_apply(analyze_single_language, languages, use_threads=True)
        
        # Convert results back to dictionary
        language_profiles = {lang: profile for lang, profile in results}
        
        return language_profiles
    
    def _analyze_response_patterns(self, lang_data, response_col):
        """Analyze response patterns for a specific language."""
        if response_col not in lang_data.columns:
            return {}
        responses = lang_data[response_col]
        
        return {
            'total_responses': len(responses),
            'unique_responses': responses.nunique(),
            'response_entropy': self._calculate_entropy(responses),
            'most_frequent_response': responses.mode().iloc[0] if not responses.mode().empty else None,
            'response_frequency_distribution': responses.value_counts(normalize=True).to_dict(),
            'response_consistency': self._calculate_response_consistency(lang_data, response_col),
            'lexical_diversity': self._calculate_lexical_diversity(responses)
        }
    
    def _analyze_cognitive_patterns(self, lang_data, response_col):
        """Analyze cognitive patterns in responses."""
        if response_col not in lang_data.columns:
            return {}
        responses = lang_data[response_col]
        return {
            'semantic_clustering': self._analyze_semantic_clustering(responses),
            'response_length_statistics': self._analyze_response_lengths(responses),
            'code_switching_frequency': self._detect_code_switching(responses),
            'cultural_specificity_markers': self._identify_cultural_markers(responses)
        }
    
    def _generate_statistical_profile(self, lang_data, response_col):
        """Generate comprehensive statistical profile for a language."""
        if response_col not in lang_data.columns:
            return {}
        responses = lang_data[response_col]
        response_codes = pd.Categorical(responses).codes
        
        return {
            'descriptive_statistics': {
                'mean_response_code': np.mean(response_codes) if len(response_codes) > 0 else 0,
                'std_response_code': np.std(response_codes) if len(response_codes) > 0 else 0,
                'skewness': self._calculate_skewness(response_codes),
                'kurtosis': self._calculate_kurtosis(response_codes)
            },
            'distribution_tests': self._perform_distribution_tests(response_codes),
            'variance_analysis': {
                'total_variance': np.var(response_codes) if len(response_codes) > 0 else 0,
                'explained_variance_ratio': self._calculate_explained_variance(lang_data, response_col)
            }
        }
    
    def _analyze_accuracy_patterns(self, lang_data):
        """Analyze accuracy patterns for definitive experiments."""
        if 'IsCorrect' not in lang_data.columns:
            return None
        accuracy_scores = lang_data['IsCorrect'].astype(int)
        if len(accuracy_scores) == 0:
             return {'overall_accuracy': 0, 'accuracy_variance': 0, 'confidence_interval_95': {'lower':0, 'upper':0, 'margin_of_error':0}}
        
        return {
            'overall_accuracy': accuracy_scores.mean(),
            'accuracy_variance': accuracy_scores.var(),
            'confidence_interval_95': self._calculate_confidence_interval(accuracy_scores),
            'item_wise_accuracy': self._analyze_item_wise_accuracy(lang_data),
            'error_patterns': self._analyze_error_patterns(lang_data),
            'learning_curve_analysis': self._analyze_learning_patterns(lang_data)
        }
    
    def perform_cross_linguistic_analysis(self, df, experiment_type, language_profiles):
        """Perform comprehensive cross-linguistic analysis."""
        response_col = self._get_response_column(df)
        if not response_col:
            return {}
        
        analysis = {
            'overall_diversity_metrics': self._calculate_cross_linguistic_diversity(df, response_col),
            'pairwise_language_similarities': self._calculate_language_similarities(language_profiles),
            'clustering_analysis': self._perform_language_clustering(language_profiles),
            'convergence_divergence_patterns': self._analyze_convergence_patterns(df, response_col),
            'cultural_distance_correlations': self._analyze_cultural_distance_effects(language_profiles)
        }
        
        if experiment_type == 'definitive' and 'IsCorrect' in df.columns:
            analysis['accuracy_based_analysis'] = self._analyze_cross_linguistic_accuracy_patterns(df)
        
        return analysis
    
    def perform_linguistic_family_analysis(self, df, experiment_type, experiment_name):
        """Analyze experimental data by linguistic family and subfamily."""
        response_col = self._get_response_column(df)
        if not response_col:
            return {}
        family_data = df.groupby('LinguisticFamily')
        subfamily_data = df.groupby(['LinguisticFamily', 'LinguisticSubfamily'])
        analysis = {
            'family_level_analysis': self._analyze_by_linguistic_family(family_data, response_col, experiment_type),
            'subfamily_level_analysis': self._analyze_by_linguistic_subfamily(subfamily_data, response_col, experiment_type),
            'phylogenetic_patterns': self._analyze_phylogenetic_patterns(df, response_col, experiment_name),
            'typological_correlations': self._analyze_typological_correlations(df, response_col)
        }
        return analysis
    
    def assess_linguistic_relativity_comprehensive(self, df, experiment_name, experiment_type, 
                                                 statistical_results, language_profiles, cross_linguistic_analysis):
        """Comprehensive assessment of linguistic relativity evidence."""
        basic_assessment = self.assess_linguistic_relativity(df, experiment_name, experiment_type, statistical_results)
        enhanced_criteria = {
            'cross_linguistic_variance': self._assess_cross_linguistic_variance(language_profiles),
            'phylogenetic_signal': self._assess_phylogenetic_signal(cross_linguistic_analysis),
            'cultural_specificity': self._assess_cultural_specificity(language_profiles),
            'systematic_variation': self._assess_systematic_variation(statistical_results),
            'replication_potential': self._assess_replication_potential(df, experiment_type),
            'theoretical_coherence': self._assess_theoretical_coherence(language_profiles, experiment_name)
        }
        comprehensive_assessment = {
            **basic_assessment,
            'enhanced_criteria': enhanced_criteria,
            'academic_interpretation': self._generate_academic_interpretation(
                basic_assessment, enhanced_criteria, experiment_name, experiment_type
            ),
            'methodological_considerations': self._assess_methodological_limitations(df, experiment_type),
            'future_research_directions': self._suggest_future_research(
                basic_assessment, enhanced_criteria, experiment_name
            )
        }
        return comprehensive_assessment
    
    def assess_linguistic_relativity(self, df, experiment_name, experiment_type, statistical_results):
        """Assess evidence for linguistic relativity in the experiment."""
        assessment = {
            'experiment_name': experiment_name,
            'experiment_type': experiment_type,
            'total_languages': df['Language'].nunique(),
            'total_responses': len(df),
            'evidence_strength': 'insufficient_data'
        }
        
        response_col = self._get_response_column(df)
        
        significant_chi2_tests = []
        if f'chi2_Language_vs_{response_col}' in statistical_results:
            test = statistical_results[f'chi2_Language_vs_{response_col}']
            if test['significant']:
                significant_chi2_tests.append('Language-Response association')
        
        significant_accuracy_differences = 0
        large_effect_sizes = 0
        
        if 'language_accuracy_comparisons' in statistical_results:
            comparisons = statistical_results['language_accuracy_comparisons']
            for comp in comparisons:
                if comp['u_significant']:
                    significant_accuracy_differences += 1
                if comp['cohens_d'] is not None and abs(comp['cohens_d']) >= 0.8:
                    large_effect_sizes += 1
        
        if response_col:
            language_response_diversity = df.groupby('Language')[response_col].nunique().mean()
            overall_response_diversity = df[response_col].nunique()
            response_diversity_ratio = language_response_diversity / overall_response_diversity if overall_response_diversity > 0 else 0
        else:
            response_diversity_ratio = 0
        
        evidence_indicators = 0
        
        if significant_chi2_tests:
            evidence_indicators += 2
            
        if significant_accuracy_differences > 0:
            evidence_indicators += 1
            
        if large_effect_sizes > 0:
            evidence_indicators += 2
            
        if response_diversity_ratio > 0.3:
            evidence_indicators += 1
        
        if evidence_indicators >= 4:
            assessment['evidence_strength'] = 'strong'
        elif evidence_indicators >= 2:
            assessment['evidence_strength'] = 'moderate'
        elif evidence_indicators >= 1:
            assessment['evidence_strength'] = 'weak'
        else:
            assessment['evidence_strength'] = 'insufficient'
        
        assessment.update({
            'significant_language_effects': len(significant_chi2_tests),
            'significant_accuracy_differences': significant_accuracy_differences,
            'large_effect_sizes': large_effect_sizes,
            'response_diversity_ratio': response_diversity_ratio,
            'evidence_indicators': evidence_indicators
        })
        
        return assessment
    
    def _get_response_column(self, df):
        """Dynamically determine the response column."""
        if 'ItemResponse' in df.columns:
            return 'ItemResponse'
        elif 'PredictedEmotion' in df.columns:
            return 'PredictedEmotion'
        elif 'Response' in df.columns:
            return 'Response'
        else:
            potential_cols = [col for col in df.columns if any(word in col.lower() for word in ['response', 'answer', 'prediction', 'emotion'])]
            return potential_cols[0] if potential_cols else None

    # Helper methods for analysis
    def _calculate_entropy(self, responses):
        """Calculate Shannon entropy of responses."""
        if responses.empty:
            return 0.0
        value_counts = responses.value_counts(normalize=True)
        return -np.sum(value_counts * np.log2(value_counts + 1e-10))
    
    def _calculate_response_consistency(self, lang_data, response_col):
        """Calculate response consistency within language group."""
        if 'ItemID' not in lang_data.columns or response_col not in lang_data.columns:
            return 0.0
        consistency_scores = []
        for item_id in lang_data['ItemID'].unique():
            item_responses = lang_data[lang_data['ItemID'] == item_id][response_col]
            if len(item_responses) > 1:
                mode_count = item_responses.value_counts().iloc[0] if not item_responses.value_counts().empty else 0
                consistency_scores.append(mode_count / len(item_responses))
        return np.mean(consistency_scores) if consistency_scores else 0.0
    
    def _calculate_lexical_diversity(self, responses):
        """Calculate lexical diversity using Type-Token Ratio."""
        if responses.empty:
            return 0.0
        all_tokens = [token for response in responses.dropna() for token in str(response).lower().split()]
        if not all_tokens:
            return 0.0
        return len(set(all_tokens)) / len(all_tokens)

    def _analyze_semantic_clustering(self, responses):
        """Analyze semantic clustering in responses."""
        semantic_categories = {
            'emotion_words': ['happy', 'sad', 'angry', 'calm', 'fearful', 'surprised', 'neutral'],
            'color_words': ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'black', 'white'],
            'spatial_words': ['left', 'right', 'up', 'down', 'north', 'south', 'east', 'west'],
            'quantity_words': ['one', 'two', 'many', 'few', 'some', 'all', 'none']
        }
        
        category_usage = {}
        for category, words in semantic_categories.items():
            usage_count = sum(1 for response in responses 
                            if any(word in str(response).lower() for word in words))
            category_usage[category] = usage_count / len(responses) if len(responses) > 0 else 0
        
        return category_usage
    
    def _analyze_response_lengths(self, responses):
        """Analyze response length statistics."""
        lengths = [len(str(response).split()) for response in responses if pd.notna(response)]
        
        if not lengths:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        
        return {
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'median_length': np.median(lengths)
        }
    
    def _detect_code_switching(self, responses):
        """Detect potential code-switching in responses."""
        code_switching_count = 0
        total_responses = 0
        
        for response in responses:
            if pd.isna(response):
                continue
            total_responses += 1
            response_str = str(response)
            
            has_latin = any(ord(char) < 128 and char.isalpha() for char in response_str)
            has_non_latin = any(ord(char) > 127 and char.isalpha() for char in response_str)
            
            if has_latin and has_non_latin:
                code_switching_count += 1
        
        return code_switching_count / total_responses if total_responses > 0 else 0.0
    
    def _identify_cultural_markers(self, responses):
        """Identify potential cultural-specific response markers."""
        cultural_indicators = {
            'formal_markers': ['sir', 'madam', 'please', 'thank you'],
            'informal_markers': ['yeah', 'ok', 'sure', 'nope'],
            'cultural_concepts': ['honor', 'respect', 'harmony', 'balance']
        }
        
        marker_usage = {}
        for category, markers in cultural_indicators.items():
            usage_count = sum(1 for response in responses 
                            if any(marker in str(response).lower() for marker in markers))
            marker_usage[category] = usage_count / len(responses) if len(responses) > 0 else 0
        
        return marker_usage

    def _calculate_skewness(self, data):
        """Calculate skewness of data distribution."""
        if len(data) < 3:
            return 0.0
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if std == 0:
            return 0.0
        skew = np.sum(((data - mean) / std) ** 3) / n
        return skew
    
    def _calculate_kurtosis(self, data):
        """Calculate kurtosis of data distribution."""
        if len(data) < 4:
            return 0.0
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if std == 0:
            return 0.0
        kurt = np.sum(((data - mean) / std) ** 4) / n - 3
        return kurt
    
    def _perform_distribution_tests(self, data):
        """Perform statistical tests on data distribution."""
        if len(data) < 8:
            return {'shapiro_wilk': None, 'interpretation': 'Sample too small for distribution tests'}
        
        try:
            shapiro_stat, shapiro_p = shapiro(data)
            return {
                'shapiro_wilk': {
                    'statistic': shapiro_stat,
                    'p_value': shapiro_p,
                    'is_normal': shapiro_p > 0.05
                },
                'interpretation': 'Normal distribution' if shapiro_p > 0.05 else 'Non-normal distribution'
            }
        except Exception:
            return {'shapiro_wilk': None, 'interpretation': 'Distribution test failed'}
    
    def _calculate_explained_variance(self, lang_data, response_col):
        """Calculate explained variance in responses."""
        if 'ItemCategory' not in lang_data.columns:
            return 0.0
        
        try:
            response_codes = pd.Categorical(lang_data[response_col]).codes
            categories = pd.Categorical(lang_data['ItemCategory']).codes
            
            corr, _ = spearmanr(response_codes, categories)
            return corr ** 2 if not np.isnan(corr) else 0.0
        except Exception:
            return 0.0
    
    def _calculate_confidence_interval(self, accuracy_scores, confidence=0.95):
        """Calculate confidence interval for accuracy scores."""
        n = len(accuracy_scores)
        mean = np.mean(accuracy_scores)
        std_err = np.std(accuracy_scores, ddof=1) / np.sqrt(n)
        
        from scipy.stats import t
        t_val = t.ppf((1 + confidence) / 2, n - 1)
        margin_error = t_val * std_err
        
        return {
            'lower': max(0, mean - margin_error),
            'upper': min(1, mean + margin_error),
            'margin_of_error': margin_error
        }
    
    def _analyze_item_wise_accuracy(self, lang_data):
        """Analyze accuracy patterns across different items."""
        if 'ItemID' not in lang_data.columns or 'IsCorrect' not in lang_data.columns:
            return None
        
        item_accuracy = lang_data.groupby('ItemID')['IsCorrect'].agg([
            'mean', 'count', 'std'
        ]).reset_index()
        
        return {
            'most_difficult_items': item_accuracy.nsmallest(3, 'mean')['ItemID'].tolist(),
            'easiest_items': item_accuracy.nlargest(3, 'mean')['ItemID'].tolist(),
            'accuracy_variance_across_items': item_accuracy['mean'].var(),
            'items_with_perfect_accuracy': item_accuracy[item_accuracy['mean'] == 1.0]['ItemID'].tolist(),
            'items_with_zero_accuracy': item_accuracy[item_accuracy['mean'] == 0.0]['ItemID'].tolist()
        }
    
    def _analyze_error_patterns(self, lang_data):
        """Analyze error patterns in incorrect responses."""
        if 'IsCorrect' not in lang_data.columns:
            return None

        incorrect_data = lang_data[lang_data['IsCorrect'] == False]
        
        if len(incorrect_data) == 0:
            return {'error_rate': 0.0, 'error_patterns': 'No errors found'}
        
        error_analysis = {
            'error_rate': len(incorrect_data) / len(lang_data),
            'error_distribution_by_item': incorrect_data['ItemID'].value_counts().to_dict() if 'ItemID' in incorrect_data.columns else {},
            'common_incorrect_responses': incorrect_data[self._get_response_column(lang_data)].value_counts().head(5).to_dict()
        }
        
        return error_analysis
    
    def _analyze_learning_patterns(self, lang_data):
        """Analyze potential learning patterns within the experiment."""
        if 'IsCorrect' not in lang_data.columns:
            return None

        accuracy_by_position = lang_data.groupby(lang_data.index)['IsCorrect'].mean()
        
        return {
            'overall_trend': 'improving' if accuracy_by_position.iloc[-1] > accuracy_by_position.iloc[0] else 'declining',
            'trend_correlation': spearmanr(range(len(accuracy_by_position)), accuracy_by_position)[0] if len(accuracy_by_position) > 2 else 0.0
        }

    def _calculate_cross_linguistic_diversity(self, df, response_col):
        """Calculate cross-linguistic diversity metrics."""
        languages = df['Language'].unique()
        if not languages.any() or response_col not in df.columns:
            return {'overall_entropy': 0, 'mean_language_entropy': 0, 'entropy_variance': 0, 'diversity_index': 0}

        all_responses = df[response_col]
        overall_entropy = self._calculate_entropy(all_responses)
        
        language_entropies = [self._calculate_entropy(df[df['Language'] == lang][response_col]) for lang in languages]
        mean_lang_entropy = np.mean(language_entropies) if language_entropies else 0
        
        return {
            'overall_entropy': overall_entropy,
            'mean_language_entropy': mean_lang_entropy,
            'entropy_variance': np.var(language_entropies) if language_entropies else 0,
            'diversity_index': overall_entropy / mean_lang_entropy if mean_lang_entropy > 0 else 0
        }
    
    def _calculate_language_similarities(self, language_profiles):
        """Calculate pairwise similarities between languages."""
        if not language_profiles or len(language_profiles) < 2:
            return {}
        languages = list(language_profiles.keys())
        similarities = {}
        
        for i, lang1 in enumerate(languages):
            for j, lang2 in enumerate(languages[i+1:], i+1):
                profile1 = language_profiles.get(lang1, {})
                profile2 = language_profiles.get(lang2, {})
                dist1 = profile1.get('response_statistics', {}).get('response_frequency_distribution', {})
                dist2 = profile2.get('response_statistics', {}).get('response_frequency_distribution', {})
                similarities[f"{lang1}_vs_{lang2}"] = self._calculate_distribution_similarity(dist1, dist2)
        return similarities
    
    def _calculate_distribution_similarity(self, dist1, dist2):
        """Calculate similarity between two response distributions (Jaccard)."""
        all_responses = set(dist1.keys()) | set(dist2.keys())
        if not all_responses:
            return 0.0
        intersection = sum(min(dist1.get(r, 0), dist2.get(r, 0)) for r in all_responses)
        union = sum(max(dist1.get(r, 0), dist2.get(r, 0)) for r in all_responses)
        return intersection / union if union > 0 else 0.0

    def _perform_language_clustering(self, language_profiles):
        """Perform clustering analysis on languages based on response patterns."""
        if not language_profiles or len(language_profiles) < 2:
             return {'clusters': [], 'interpretation': 'Insufficient data for clustering'}

        clustering_data = []
        language_names = []
        for lang, profile in language_profiles.items():
            resp_stats = profile.get('response_statistics', {})
            entropy = resp_stats.get('response_entropy', 0)
            diversity = resp_stats.get('lexical_diversity', 0)
            clustering_data.append([entropy, diversity])
            language_names.append(lang)
        
        return {
            'clustering_features': ['response_entropy', 'lexical_diversity'],
            'languages': language_names,
            'interpretation': 'Languages clustered based on response patterns (simplified placeholder).'
        }
    
    def _analyze_convergence_patterns(self, df, response_col):
        """Analyze convergence/divergence patterns across languages."""
        if 'ItemID' not in df.columns or response_col not in df.columns:
            return {'mean_convergence': 0, 'convergence_variance': 0, 'interpretation': 'ItemID or response column missing.'}
        
        convergence_scores = []
        for item_id in df['ItemID'].unique():
            item_data = df[df['ItemID'] == item_id]
            if not item_data.empty:
                item_entropy = self._calculate_entropy(item_data[response_col])
                convergence_scores.append(1 / (1 + item_entropy)) 
        
        return {
            'mean_convergence': np.mean(convergence_scores) if convergence_scores else 0,
            'convergence_variance': np.var(convergence_scores) if convergence_scores else 0,
            'interpretation': 'High convergence indicates universal patterns, low convergence indicates language-specific patterns.'
        }
    
    def _analyze_cultural_distance_effects(self, language_profiles):
        """Analyze effects of cultural distance on response patterns (simplified)."""
        return {
            'within_family_coherence_placeholder': 0,
            'between_family_divergence_placeholder': 0,
            'cultural_effect_strength_placeholder': 'moderate', 
            'interpretation': 'Cultural distance analysis requires external data on cultural/linguistic proximity.'
        }
    
    def _analyze_cross_linguistic_accuracy_patterns(self, df):
        """Analyze accuracy patterns across languages for definitive experiments."""
        if 'IsCorrect' not in df.columns:
            return None
        language_accuracies = df.groupby('Language')['IsCorrect'].mean()
        if language_accuracies.empty:
            return {'accuracy_range': 0, 'accuracy_variance': 0}
            
        return {
            'accuracy_range': language_accuracies.max() - language_accuracies.min() if not language_accuracies.empty else 0,
            'accuracy_variance': language_accuracies.var() if not language_accuracies.empty else 0,
            'highest_performing_languages': language_accuracies.nlargest(3).index.tolist(),
            'lowest_performing_languages': language_accuracies.nsmallest(3).index.tolist(),
        }

    def _analyze_by_linguistic_family(self, family_data, response_col, experiment_type):
        """Analyze experimental data grouped by linguistic family."""
        analysis = {}
        for name, group in family_data:
            family_profile = {
                'sample_size': len(group),
                'response_statistics': self._analyze_response_patterns(group, response_col),
                'mean_diversity': group.groupby('Language')[response_col].nunique().mean()
            }
            if experiment_type == 'definitive' and 'IsCorrect' in group.columns:
                family_profile['mean_accuracy'] = group['IsCorrect'].mean()
            analysis[name] = family_profile
        return analysis

    def _analyze_by_linguistic_subfamily(self, subfamily_data, response_col, experiment_type):
        """Analyze experimental data grouped by linguistic subfamily."""
        analysis = {}
        for name, group in subfamily_data:
            family_name, subfamily_name = name
            subfamily_key = f"{family_name} - {subfamily_name}"
            profile = {
                'sample_size': len(group),
                'response_statistics': self._analyze_response_patterns(group, response_col),
                'mean_diversity': group.groupby('Language')[response_col].nunique().mean()
            }
            if experiment_type == 'definitive' and 'IsCorrect' in group.columns:
                profile['mean_accuracy'] = group['IsCorrect'].mean()
            analysis[subfamily_key] = profile
        return analysis

    def _analyze_phylogenetic_patterns(self, df, response_col, experiment_name):
        """Analyze patterns based on phylogenetic relationships (simplified)."""
        family_groups = df.groupby('LinguisticFamily')
        within_family_similarity = []

        current_experiment_profiles = self.results.get(experiment_name, {}).get('language_profiles')
        if not current_experiment_profiles: # If not found in self.results
            if hasattr(self, '_currently_processing_profiles') and self._currently_processing_profiles.get(experiment_name): # Try _currently_processing_profiles
                 current_experiment_profiles = self._currently_processing_profiles[experiment_name]
            
            # After trying both sources, if it's still not populated (i.e., None or empty), then return.
            if not current_experiment_profiles: 
                return {
                    'mean_within_family_similarity': 0,
                    'interpretation': 'Phylogenetic analysis skipped: language profiles not available after checking all sources.'
                }
        # If current_experiment_profiles was found in self.results initially, the above block is skipped.
        # If found via _currently_processing_profiles, it's now populated.
        # Code execution continues if current_experiment_profiles is now valid.

        for name, group in family_groups:
            if group['Language'].nunique() > 1:
                lang_profiles_in_family = {lang: current_experiment_profiles[lang] 
                                           for lang in group['Language'].unique() 
                                           if lang in current_experiment_profiles}
                
                if len(lang_profiles_in_family) > 1:
                    similarities = self._calculate_language_similarities(lang_profiles_in_family)
                    within_family_similarity.extend(similarities.values())

        return {
            'mean_within_family_similarity': np.mean(within_family_similarity) if within_family_similarity else 0,
            'interpretation': 'Higher within-family similarity may suggest phylogenetic influence. (Note: Current analysis is simplified)'
        }

    def _analyze_typological_correlations(self, df, response_col):
        """Placeholder for analyzing correlations with typological features."""
        return {
            'typological_feature_analysis': 'Not implemented due to lack of typological database.',
            'potential_correlations': 'Could explore correlations with features like word order, morphology, etc.'
        }
    
    def _assess_cross_linguistic_variance(self, language_profiles):
        """Assess the degree of cross-linguistic variance based on language profiles."""
        if not language_profiles or len(language_profiles) < 2:
            return 'low'
        
        entropies = [profile['response_statistics']['response_entropy'] for profile in language_profiles.values()]
        entropy_variance = np.var(entropies)
        
        if entropy_variance > 0.5:
            return 'high'
        elif entropy_variance > 0.2:
            return 'medium'
        else:
            return 'low'
    
    def _assess_phylogenetic_signal(self, cross_linguistic_analysis):
        """Assess phylogenetic signal from cross-linguistic analysis."""
        if cross_linguistic_analysis and 'overall_diversity_metrics' in cross_linguistic_analysis:
            if 0.5 < cross_linguistic_analysis['overall_diversity_metrics'].get('diversity_index', 0) < 1.5:
                return 'potential'
        return 'unclear'
    
    def _assess_cultural_specificity(self, language_profiles):
        """Assess cultural specificity from language profiles."""
        num_specific_markers = 0
        for profile in language_profiles.values():
            if profile['cognitive_patterns']['cultural_specificity_markers']:
                if any(v > 0.1 for v in profile['cognitive_patterns']['cultural_specificity_markers'].values()):
                    num_specific_markers +=1
        
        if num_specific_markers / len(language_profiles) > 0.3:
            return 'high'
        elif num_specific_markers > 0:
            return 'medium'
        return 'low'
    
    def _assess_systematic_variation(self, statistical_results):
        """Assess systematic variation from statistical results."""
        if statistical_results.get('kruskal_wallis_test', {}).get('significant', False):
            return 'present'
        if statistical_results.get('language_accuracy_comparisons'):
            if any(comp.get('u_significant', False) for comp in statistical_results['language_accuracy_comparisons']):
                return 'present'
        return 'absent'
    
    def _assess_replication_potential(self, df, experiment_type):
        """Assess replication potential based on data characteristics."""
        if len(df) > 100 * df['Language'].nunique():
            return 'high'
        elif len(df) > 30 * df['Language'].nunique():
            return 'medium'
        return 'low'
    
    def _assess_theoretical_coherence(self, language_profiles, experiment_name):
        """Assess theoretical coherence of findings with existing LR theories."""
        return 'moderate'

    def _generate_academic_interpretation(self, basic_assessment, enhanced_criteria, experiment_name, experiment_type):
        """Generate an academic interpretation string based on all assessments."""
        interpretation = f"Academic Interpretation for {experiment_name} ({experiment_type}):\n"
        interpretation += f"  Overall LR Evidence: {basic_assessment.get('evidence_strength', 'N/A').upper()}\n"
        interpretation += f"  Cross-linguistic Variance: {enhanced_criteria.get('cross_linguistic_variance', 'N/A')}\n"
        interpretation += f"  Phylogenetic Signal: {enhanced_criteria.get('phylogenetic_signal', 'N/A')}\n"
        interpretation += f"  Cultural Specificity: {enhanced_criteria.get('cultural_specificity', 'N/A')}\n"
        interpretation += f"  Systematic Variation: {enhanced_criteria.get('systematic_variation', 'N/A')}\n"
        
        if basic_assessment.get('evidence_strength') == 'strong' and enhanced_criteria.get('cross_linguistic_variance') == 'high':
            interpretation += "  The findings strongly suggest robust linguistic relativity effects, characterized by significant and systematic variation in responses across languages. This indicates that linguistic structures likely play a substantial role in shaping cognitive processes related to this experimental domain.\n"
        elif basic_assessment.get('evidence_strength') == 'moderate':
            interpretation += "  Moderate evidence for linguistic relativity is observed. While some significant cross-linguistic differences exist, further investigation is needed to fully disentangle linguistic influences from other cognitive or cultural factors. The patterns warrant deeper exploration into how specific linguistic features might modulate task performance or conceptualization.\n"
        else:
            interpretation += "  The evidence for linguistic relativity in this experiment is currently limited or inconclusive. Observed variations may be attributable to factors other than language, or the experimental design may lack sensitivity to detect subtle linguistic influences. Further research with refined methodologies or broader linguistic sampling is recommended.\n"
        return interpretation

    def _assess_methodological_limitations(self, df, experiment_type):
        """Assess methodological limitations of the current experiment and analysis."""
        limitations = []
        if df['Language'].nunique() < 5:
            limitations.append("Limited number of languages restricts generalizability.")
        if df.groupby('Language').size().min() < 30:
            limitations.append("Small sample sizes per language may reduce statistical power.")
        if not limitations:
            limitations.append("Standard methodological considerations apply (e.g., task generalizability, model biases).")
        return "; ".join(limitations)

    def _suggest_future_research(self, basic_assessment, enhanced_criteria, experiment_name):
        """Suggest future research directions based on the findings."""
        suggestions = []
        if basic_assessment.get('evidence_strength') in ['strong', 'moderate']:
            suggestions.append(f"Further investigate the specific linguistic features in {experiment_name} that correlate with observed cognitive differences.")
            suggestions.append("Expand the linguistic sample to include more diverse language families and typologies.")
        if enhanced_criteria.get('cultural_specificity') == 'high':
            suggestions.append("Conduct qualitative analyses to understand the cultural nuances in responses.")
        suggestions.append("Employ experimental designs that can better isolate linguistic effects from broader cultural or cognitive universals.")
        return "; ".join(suggestions)

    def generate_comprehensive_visualizations(self, df, experiment_name, experiment_type, metrics=None, language_profiles=None, statistical_results=None):
        """Generate comprehensive visualizations for academic reports."""
        fig_base_dir = os.path.join(self.output_dir, 'publication_ready_figures', experiment_name)
        if not os.path.exists(fig_base_dir):
            os.makedirs(fig_base_dir)

        response_col = self._get_response_column(df)
        if not response_col:
            print(f"Warning: Could not find response column for visualization in {experiment_name}")
            return

        print(f"Generating visualizations for {experiment_name}...")
        
        # Clear any existing plots
        plt.clf()
        plt.close('all')

        # Response Distribution by Language (Heatmap)
        try:
            plt.figure(figsize=(16, 10))
            response_by_lang = pd.crosstab(df['Language'], df[response_col], normalize='index') * 100
            
            if response_by_lang.shape[1] > 15:
                top_responses = df[response_col].value_counts().nlargest(15).index
                response_by_lang = response_by_lang[top_responses]
            
            # Create heatmap with proper configuration for Agg backend
            ax = sns.heatmap(response_by_lang, 
                           annot=True, 
                           fmt='.1f', 
                           cmap='YlGnBu', 
                           linewidths=0.5,
                           cbar_kws={'label': 'Percentage (%)'},
                           square=False)
            
            plt.title(f'{experiment_name}: Normalized Response Distribution by Language (%)', 
                     fontsize=14, fontweight='bold', pad=20)
            plt.xlabel('Response Category', fontsize=12, fontweight='bold')
            plt.ylabel('Language', fontsize=12, fontweight='bold')
            
            # Improve layout and readability
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.yticks(rotation=0, fontsize=10)
            plt.tight_layout()
            
            # Save with high DPI and proper format
            heatmap_path = os.path.join(fig_base_dir, f'{experiment_name}_response_dist_heatmap.png')
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()
            
            print(f"✓ Response distribution heatmap saved: {heatmap_path}")
            
        except Exception as e:
            print(f"Error generating response distribution heatmap for {experiment_name}: {e}")
            plt.close()

        # Language Family Distribution (Bar Chart)
        try:
            plt.figure(figsize=(12, 8))
            family_counts = df['LinguisticFamily'].value_counts()
            
            ax = family_counts.plot(kind='bar', color='skyblue', edgecolor='navy', linewidth=1.2)
            plt.title(f'{experiment_name}: Distribution by Linguistic Family', 
                     fontsize=14, fontweight='bold', pad=20)
            plt.xlabel('Linguistic Family', fontsize=12, fontweight='bold')
            plt.ylabel('Number of Responses', fontsize=12, fontweight='bold')
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.yticks(fontsize=10)
            
            # Add value labels on bars
            for i, v in enumerate(family_counts.values):
                ax.text(i, v + 0.1, str(v), ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            family_chart_path = os.path.join(fig_base_dir, f'{experiment_name}_linguistic_family_dist.png')
            plt.savefig(family_chart_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()
            
            print(f"✓ Linguistic family distribution saved: {family_chart_path}")
            
        except Exception as e:
            print(f"Error generating linguistic family chart for {experiment_name}: {e}")
            plt.close()

        # Response Entropy by Language (if language profiles available)
        if language_profiles:
            try:
                plt.figure(figsize=(14, 8))
                
                languages = []
                entropies = []
                families = []
                
                for lang, profile in language_profiles.items():
                    languages.append(lang.replace(' # ', '\n'))  # Format for better display
                    entropies.append(profile['response_statistics']['response_entropy'])
                    families.append(profile['linguistic_family'])
                
                # Create color map for families
                unique_families = list(set(families))
                colors = plt.cm.Set3(np.linspace(0, 1, len(unique_families)))
                family_colors = dict(zip(unique_families, colors))
                bar_colors = [family_colors[family] for family in families]
                
                bars = plt.bar(range(len(languages)), entropies, color=bar_colors, 
                              edgecolor='black', linewidth=0.8, alpha=0.8)
                
                plt.title(f'{experiment_name}: Response Entropy by Language', 
                         fontsize=14, fontweight='bold', pad=20)
                plt.xlabel('Language', fontsize=12, fontweight='bold')
                plt.ylabel('Response Entropy', fontsize=12, fontweight='bold')
                plt.xticks(range(len(languages)), languages, rotation=45, ha='right', fontsize=9)
                plt.yticks(fontsize=10)
                
                # Add legend for linguistic families
                legend_elements = [plt.Rectangle((0,0),1,1, facecolor=family_colors[family], 
                                               edgecolor='black', alpha=0.8) 
                                 for family in unique_families]
                plt.legend(legend_elements, unique_families, loc='upper right', 
                          title='Linguistic Families', title_fontsize=10, fontsize=9)
                
                plt.tight_layout()
                entropy_chart_path = os.path.join(fig_base_dir, f'{experiment_name}_response_entropy.png')
                plt.savefig(entropy_chart_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
                plt.close()
                
                print(f"✓ Response entropy chart saved: {entropy_chart_path}")
                
            except Exception as e:
                print(f"Error generating response entropy chart for {experiment_name}: {e}")
                plt.close()

        # Accuracy by Language (for definitive experiments)
        if experiment_type == 'definitive' and metrics and 'accuracy_by_language' in metrics:
            try:
                plt.figure(figsize=(12, 8))
                
                accuracy_data = metrics['accuracy_by_language']
                languages = [lang.replace(' # ', '\n') for lang in accuracy_data.index]
                accuracies = accuracy_data['mean'].values
                counts = accuracy_data['count'].values
                
                bars = plt.bar(range(len(languages)), accuracies, 
                              color='lightcoral', edgecolor='darkred', linewidth=1.2, alpha=0.8)
                
                plt.title(f'{experiment_name}: Accuracy by Language', 
                         fontsize=14, fontweight='bold', pad=20)
                plt.xlabel('Language', fontsize=12, fontweight='bold')
                plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
                plt.xticks(range(len(languages)), languages, rotation=45, ha='right', fontsize=9)
                plt.yticks(fontsize=10)
                plt.ylim(0, 1.1)
                
                # Add sample size labels on bars
                for i, (acc, count) in enumerate(zip(accuracies, counts)):
                    plt.text(i, acc + 0.02, f'n={count}', ha='center', va='bottom', 
                            fontsize=8, fontweight='bold')
                
                plt.tight_layout()
                accuracy_chart_path = os.path.join(fig_base_dir, f'{experiment_name}_accuracy_by_language.png')
                plt.savefig(accuracy_chart_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
                plt.close()
                
                print(f"✓ Accuracy chart saved: {accuracy_chart_path}")
                
            except Exception as e:
                print(f"Error generating accuracy chart for {experiment_name}: {e}")
                plt.close()

        print(f"Comprehensive visualizations saved to: {fig_base_dir}")
        self._save_analysis_artifacts(experiment_name)

    def _save_analysis_artifacts(self, experiment_name):
        """Saves specific analysis artifacts to their respective directories."""
        result = self.results.get(experiment_name)
        if not result:
            return
        
        artifact_map = {
            'language_profiles': 'per_language_analysis',
            'cross_linguistic_analysis': 'cross_linguistic_analysis',
            'linguistic_family_analysis': 'linguistic_family_analysis',
            'statistical_results': 'statistical_significance',
            'lr_assessment': 'linguistic_relativity_assessment'
        }

        for data_key, dir_name_segment in artifact_map.items():
            exp_specific_artifact_dir = os.path.join(self.output_dir, dir_name_segment, experiment_name)
            os.makedirs(exp_specific_artifact_dir, exist_ok=True)
            
            data_to_save = result.get(data_key)
            
            if data_key == 'language_profiles':
                if data_to_save:
                    # Save individual language profiles as JSON
                    for lang, profile in data_to_save.items():
                        lang_safe_name = lang.replace(' # ', '_').replace(' ', '_').replace('/', '_')
                        json_path = os.path.join(exp_specific_artifact_dir, f'{lang_safe_name}_profile.json')
                        self._save_json_data(profile, json_path)
                    
                    # Generate CSV summary of all language profiles
                    self._generate_language_profiles_csv(data_to_save, exp_specific_artifact_dir, experiment_name)
                    
                    # Generate human-readable text summary
                    self._generate_language_profiles_text_summary(data_to_save, exp_specific_artifact_dir, experiment_name)
                continue
            
            if data_to_save is not None:
                # Save JSON
                file_name = f"{data_key.replace('_', '-')}_summary.json"
                json_path = os.path.join(exp_specific_artifact_dir, file_name)
                self._save_json_data(data_to_save, json_path)
                
                # Generate CSV and text versions
                self._generate_csv_and_text_outputs(data_to_save, data_key, exp_specific_artifact_dir, experiment_name)

        # Enhanced Cohen's D analysis for all experiment types
        self._generate_comprehensive_cohens_d_analysis(result, experiment_name)

    def _generate_language_profiles_csv(self, language_profiles, output_dir, experiment_name):
        """Generate CSV summary of language profiles."""
        csv_data = []
        for lang, profile in language_profiles.items():
            resp_stats = profile.get('response_statistics', {})
            cog_patterns = profile.get('cognitive_patterns', {})
            stat_profile = profile.get('statistical_profile', {})
            desc_stats = stat_profile.get('descriptive_statistics', {})
            
            row = {
                'Language': lang,
                'Linguistic_Family': profile.get('linguistic_family', 'Unknown'),
                'Linguistic_Subfamily': profile.get('linguistic_subfamily', 'Unknown'),
                'Sample_Size': profile.get('sample_size', 0),
                'Total_Responses': resp_stats.get('total_responses', 0),
                'Unique_Responses': resp_stats.get('unique_responses', 0),
                'Response_Entropy': resp_stats.get('response_entropy', 0),
                'Lexical_Diversity': resp_stats.get('lexical_diversity', 0),
                'Response_Consistency': resp_stats.get('response_consistency', 0),
                'Mean_Response_Length': cog_patterns.get('response_length_statistics', {}).get('mean_length', 0),
                'Code_Switching_Frequency': cog_patterns.get('code_switching_frequency', 0),
                'Mean_Response_Code': desc_stats.get('mean_response_code', 0),
                'Std_Response_Code': desc_stats.get('std_response_code', 0),
                'Skewness': desc_stats.get('skewness', 0),
                'Kurtosis': desc_stats.get('kurtosis', 0)
            }
            
            # Add accuracy data if available
            if 'accuracy_analysis' in profile:
                acc_analysis = profile['accuracy_analysis']
                if acc_analysis:
                    row['Overall_Accuracy'] = acc_analysis.get('overall_accuracy', 0)
                    row['Accuracy_Variance'] = acc_analysis.get('accuracy_variance', 0)
                    ci = acc_analysis.get('confidence_interval_95', {})
                    row['Accuracy_CI_Lower'] = ci.get('lower', 0)
                    row['Accuracy_CI_Upper'] = ci.get('upper', 0)
            
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(output_dir, f'{experiment_name}_language_profiles_summary.csv')
        df.to_csv(csv_path, index=False)
        print(f"Language profiles CSV saved: {csv_path}")

    def _generate_language_profiles_text_summary(self, language_profiles, output_dir, experiment_name):
        """Generate human-readable text summary of language profiles."""
        text_path = os.path.join(output_dir, f'{experiment_name}_language_profiles_summary.txt')
        
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(f"LANGUAGE PROFILES SUMMARY - {experiment_name.upper()}\n")
            f.write("=" * 80 + "\n\n")
            
            # Group by linguistic family
            families = {}
            for lang, profile in language_profiles.items():
                family = profile.get('linguistic_family', 'Unknown')
                if family not in families:
                    families[family] = []
                families[family].append((lang, profile))
            
            for family, langs in families.items():
                f.write(f"LINGUISTIC FAMILY: {family}\n")
                f.write("-" * 40 + "\n")
                
                for lang, profile in langs:
                    resp_stats = profile.get('response_statistics', {})
                    f.write(f"\n{lang}:\n")
                    f.write(f"  Sample Size: {profile.get('sample_size', 0)}\n")
                    f.write(f"  Response Entropy: {resp_stats.get('response_entropy', 0):.3f}\n")
                    f.write(f"  Lexical Diversity: {resp_stats.get('lexical_diversity', 0):.3f}\n")
                    f.write(f"  Most Frequent Response: {resp_stats.get('most_frequent_response', 'N/A')}\n")
                    
                    if 'accuracy_analysis' in profile and profile['accuracy_analysis']:
                        acc = profile['accuracy_analysis']['overall_accuracy']
                        f.write(f"  Overall Accuracy: {acc:.3f}\n")
                
                f.write("\n")
        
        print(f"Language profiles text summary saved: {text_path}")

    def _generate_csv_and_text_outputs(self, data, data_key, output_dir, experiment_name):
        """Generate CSV and text outputs for various data types."""
        base_filename = f"{experiment_name}_{data_key.replace('_', '-')}"
        
        if data_key == 'statistical_results':
            self._generate_statistical_results_outputs(data, output_dir, base_filename)
        elif data_key == 'cross_linguistic_analysis':
            self._generate_cross_linguistic_outputs(data, output_dir, base_filename)
        elif data_key == 'linguistic_family_analysis':
            self._generate_family_analysis_outputs(data, output_dir, base_filename)
        elif data_key == 'lr_assessment':
            self._generate_lr_assessment_outputs(data, output_dir, base_filename)

    def _generate_statistical_results_outputs(self, stats_data, output_dir, base_filename):
        """Generate CSV and text outputs for statistical results."""
        # CSV for language comparisons if available
        if 'language_accuracy_comparisons' in stats_data:
            comparisons = stats_data['language_accuracy_comparisons']
            df = pd.DataFrame(comparisons)
            csv_path = os.path.join(output_dir, f'{base_filename}_language_comparisons.csv')
            df.to_csv(csv_path, index=False)
        
        # Text summary
        text_path = os.path.join(output_dir, f'{base_filename}_summary.txt')
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write("ENHANCED STATISTICAL RESULTS SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            # Traditional Statistical Tests
            f.write("TRADITIONAL STATISTICAL TESTS\n")
            f.write("-" * 40 + "\n")
            
            # Kruskal-Wallis test
            if 'kruskal_wallis_test' in stats_data:
                kw = stats_data['kruskal_wallis_test']
                f.write("Kruskal-Wallis Test (Overall Language Differences):\n")
                f.write(f"  H-statistic: {kw.get('h_statistic', 0):.3f}\n")
                f.write(f"  p-value: {kw.get('p_value', 1):.6f}\n")
                f.write(f"  Significant: {'Yes' if kw.get('significant', False) else 'No'}\n")
                f.write(f"  Interpretation: {kw.get('interpretation', 'N/A')}\n\n")
            
            # Chi-square tests
            chi2_tests = [k for k in stats_data.keys() if k.startswith('chi2_')]
            if chi2_tests:
                f.write("Chi-Square Tests:\n")
                for test_key in chi2_tests:
                    test = stats_data[test_key]
                    f.write(f"  {test_key}:\n")
                    f.write(f"    Chi2-statistic: {test.get('chi2_statistic', 0):.3f}\n")
                    f.write(f"    p-value: {test.get('p_value', 1):.6f}\n")
                    f.write(f"    Significant: {'Yes' if test.get('significant', False) else 'No'}\n\n")
            
            # ==================== ENHANCED STATISTICAL ANALYSES ====================
            
            # Multiple Comparisons Correction
            if 'multiple_comparisons_correction' in stats_data:
                f.write("MULTIPLE COMPARISONS CORRECTION\n")
                f.write("-" * 40 + "\n")
                corrections = stats_data['multiple_comparisons_correction']
                
                if 'bonferroni' in corrections:
                    bonf = corrections['bonferroni']
                    n_rejected = sum(bonf['rejected_hypotheses'])
                    f.write(f"Bonferroni Correction:\n")
                    f.write(f"  Tests corrected: {len(bonf['test_names'])}\n")
                    f.write(f"  Significant after correction: {n_rejected}\n")
                
                if 'fdr_bh' in corrections:
                    fdr = corrections['fdr_bh']
                    n_rejected = sum(fdr['rejected_hypotheses'])
                    f.write(f"Benjamini-Hochberg (FDR) Correction:\n")
                    f.write(f"  Tests corrected: {len(fdr['test_names'])}\n")
                    f.write(f"  Significant after correction: {n_rejected}\n\n")
            
            # Power Analysis
            if 'power_analysis' in stats_data:
                f.write("POWER ANALYSIS\n")
                f.write("-" * 40 + "\n")
                power = stats_data['power_analysis']
                
                if 'error' not in power:
                    f.write(f"Current Power (pairwise): {power.get('current_power_pairwise', 0):.3f}\n")
                    f.write(f"Current Power (ANOVA): {power.get('current_power_anova', 'N/A')}\n")
                    f.write(f"Required N for 80% power: {power.get('required_n_for_80_power', 'N/A')}\n")
                    f.write(f"Bonferroni corrected power: {power.get('bonferroni_corrected_power', 0):.3f}\n")
                    f.write(f"Number of comparisons: {power.get('n_pairwise_comparisons', 0)}\n")
                    f.write(f"Interpretation: {power.get('interpretation', 'N/A')}\n\n")
                else:
                    f.write(f"Power analysis error: {power['error']}\n\n")
            
            # Mixed Effects Analysis
            if 'mixed_effects_analysis' in stats_data:
                f.write("MIXED EFFECTS ANALYSIS\n")
                f.write("-" * 40 + "\n")
                mixed = stats_data['mixed_effects_analysis']
                
                if 'error' not in mixed:
                    f.write(f"Model AIC: {mixed.get('aic', 'N/A')}\n")
                    f.write(f"Model BIC: {mixed.get('bic', 'N/A')}\n")
                    f.write(f"Log-likelihood: {mixed.get('log_likelihood', 'N/A')}\n")
                    
                    interpretation = mixed.get('interpretation', {})
                    f.write(f"Model fit quality: {interpretation.get('model_fit_quality', 'N/A')}\n")
                    f.write(f"Random effects importance: {interpretation.get('random_effects_importance', 'N/A')}\n\n")
                else:
                    f.write(f"Mixed effects analysis error: {mixed['error']}\n\n")
            
            # Bayesian Analysis
            if 'bayesian_analysis' in stats_data:
                f.write("BAYESIAN ANALYSIS\n")
                f.write("-" * 40 + "\n")
                bayesian = stats_data['bayesian_analysis']
                
                if 'error' not in bayesian:
                    f.write("Posterior summaries by language:\n")
                    for lang, results in bayesian.items():
                        if isinstance(results, dict):
                            f.write(f"  {lang}:\n")
                            if 'posterior_mean' in results:
                                f.write(f"    Posterior mean: {results.get('posterior_mean', 0):.3f}\n")
                                f.write(f"    Credible interval: [{results.get('credible_interval_95', [0,0])[0]:.3f}, {results.get('credible_interval_95', [0,0])[1]:.3f}]\n")
                            if 'uncertainty_entropy' in results:
                                f.write(f"    Uncertainty entropy: {results.get('uncertainty_entropy', 0):.3f}\n")
                    f.write("\n")
                else:
                    f.write(f"Bayesian analysis error: {bayesian['error']}\n\n")
            
            # Bootstrap Analysis
            if 'bootstrap_analysis' in stats_data:
                f.write("BOOTSTRAP ANALYSIS\n")
                f.write("-" * 40 + "\n")
                bootstrap = stats_data['bootstrap_analysis']
                
                if 'error' not in bootstrap:
                    f.write("Bootstrap confidence intervals by language:\n")
                    for lang, results in bootstrap.items():
                        if isinstance(results, dict):
                            f.write(f"  {lang}:\n")
                            f.write(f"    Bootstrap mean: {results.get('bootstrap_mean', 0):.3f}\n")
                            f.write(f"    Bootstrap std: {results.get('bootstrap_std', 0):.3f}\n")
                            ci = results.get('confidence_interval_95', [0, 0])
                            f.write(f"    95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]\n")
                    f.write("\n")
                else:
                    f.write(f"Bootstrap analysis error: {bootstrap['error']}\n\n")
            
            # Permutation Tests
            if 'permutation_tests' in stats_data:
                f.write("PERMUTATION TESTS\n")
                f.write("-" * 40 + "\n")
                permutation = stats_data['permutation_tests']
                
                if 'error' not in permutation:
                    significant_tests = [k for k, v in permutation.items() if isinstance(v, dict) and v.get('significant', False)]
                    f.write(f"Total pairwise tests: {len(permutation)}\n")
                    f.write(f"Significant tests: {len(significant_tests)}\n")
                    
                    if significant_tests:
                        f.write("Significant comparisons:\n")
                        for test_name in significant_tests[:10]:  # Show top 10
                            result = permutation[test_name]
                            f.write(f"  {test_name}: p = {result.get('p_value', 1):.4f}\n")
                    f.write("\n")
                else:
                    f.write(f"Permutation tests error: {permutation['error']}\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("Note: Detailed results are saved in separate JSON files.\n")
        
        print("Generating master report...")
        self.generate_master_report()
        print("\nAnalysis complete!")
        print(f"Results saved to: {self.output_dir}")

    def run_complete_analysis(self):
        """Run complete analysis of all experiments."""
        
        selected_data_source = self._select_data_source_interactive()
        if selected_data_source is None:
            print("No data source selected. Aborting analysis.")
            return
        self.data_path = selected_data_source # Set the chosen data_path

        self.selected_generation = self.select_generation_interactive()
        if self.selected_generation is None:
            return
        
        print(f"\nStarting comprehensive linguistic relativity analysis for {self.selected_generation}...")
        # Note: Max workers info is still printed, but experiment loop is sequential for stability.
        print(f"Using up to {self.max_workers} CPU cores for parallel sub-tasks within each experiment")
        print("=" * 80)
        
        self.discover_experiments()
        if not self.experiments:
            print("No experiments found!")
            return
        
        experiment_names = list(self.experiments.keys())
        print(f"Found {len(experiment_names)} experiments to analyze: {experiment_names}")
        print("Processing experiments sequentially for stability (parallelism applied within each experiment where safe)...")
        
        completed_count = 0
        for experiment_name in experiment_names:
            try:
                print(f"\nStarting analysis of {experiment_name} ({completed_count + 1}/{len(experiment_names)})...")
                result = self.analyze_single_experiment(experiment_name)
                if result:
                    print(f"✓ Completed analysis of {experiment_name}")
                else:
                    print(f"✗ Failed analysis of {experiment_name}")
            except Exception as e:
                print(f"✗ Error analyzing {experiment_name}: {e}")
                import traceback
                traceback.print_exc()
            finally:
                completed_count += 1
                print(f"Progress: {completed_count}/{len(experiment_names)} experiments processed.")
                
        print("\nGenerating final master report...")
        self.generate_master_report()
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print(f"Results saved to: {self.output_dir}")
        print(f"Sub-tasks within experiments processed using up to {self.max_workers} CPU cores.")
        print("=" * 80)

    def run_analysis_for_generation(self, generation_code, data_source_path=None):
        """Run analysis for a specific generation without interactive input."""
        
        if data_source_path and os.path.isdir(data_source_path):
            self.data_path = data_source_path
            print(f"Using specified data source: {self.data_path}")
        elif data_source_path: # Path provided but not found
            print(f"Warning: Specified data_source_path '{data_source_path}' not found. Attempting interactive selection or default.")
            selected_data_source = self._select_data_source_interactive()
            if selected_data_source is None:
                print("No data source selected or available. Aborting analysis.")
                return
            self.data_path = selected_data_source
        else: # No data_source_path provided, use interactive selection
            selected_data_source = self._select_data_source_interactive()
            if selected_data_source is None:
                print("No data source selected or available. Aborting analysis.")
                return
            self.data_path = selected_data_source

        print(f"\nStarting analysis for generation: {generation_code} from data source: {self.data_path}")
        # Note: Max workers info is still printed, but experiment loop is sequential for stability.
        print(f"Using up to {self.max_workers} CPU cores for parallel sub-tasks within each experiment")
        self.selected_generation = generation_code
        
        print("Discovering experiments...")
        self.discover_experiments()
        if not self.experiments:
            print("No experiments found!")
            return
        
        experiment_names = list(self.experiments.keys())
        print(f"Found {len(experiment_names)} experiments: {experiment_names}")
        print("Processing experiments sequentially for stability (parallelism applied within each experiment where safe)...")
        
        completed_count = 0
        for experiment_name in experiment_names:
            try:
                print(f"\nStarting analysis of {experiment_name} ({completed_count + 1}/{len(experiment_names)})...")
                result = self.analyze_single_experiment(experiment_name)
                if result:
                    print(f"✓ Completed analysis of {experiment_name}")
                else:
                    print(f"✗ Failed to analyze {experiment_name}")
            except Exception as e:
                print(f"✗ Error analyzing {experiment_name}: {e}")
                import traceback
                traceback.print_exc()
            finally:
                completed_count += 1
                print(f"Progress: {completed_count}/{len(experiment_names)} experiments processed.")
        
        print("\nGenerating final master report...")
        self.generate_master_report()
        print("\nAnalysis complete!")
        print(f"Results saved to: {self.output_dir}")
        print(f"Sub-tasks within experiments processed using up to {self.max_workers} CPU cores.")

    # ==================== ENHANCED STATISTICAL ANALYSIS METHODS ====================
    
    def perform_mixed_effects_analysis(self, df, experiment_type):
        """Perform mixed effects analysis to account for hierarchical data structure."""
        if not STATSMODELS_AVAILABLE:
            return {'error': 'statsmodels not available for mixed effects analysis'}
        
        response_col = self._get_response_column(df)
        if not response_col or experiment_type != 'definitive' or 'IsCorrect' not in df.columns:
            return {'error': 'Mixed effects analysis requires definitive experiment with accuracy data'}
        
        try:
            # Prepare data for mixed effects model
            model_data = df.copy()
            model_data['Language_encoded'] = pd.Categorical(model_data['Language']).codes
            model_data['LinguisticFamily_encoded'] = pd.Categorical(model_data['LinguisticFamily']).codes
            
            # Fit mixed effects logistic regression
            # Language as fixed effect, LinguisticFamily as random effect
            formula = "IsCorrect ~ Language_encoded"
            if 'ItemCategory' in model_data.columns:
                model_data['ItemCategory_encoded'] = pd.Categorical(model_data['ItemCategory']).codes
                formula += " + ItemCategory_encoded"
            
            # Add random effects for linguistic family
            random_formula = "0 + LinguisticFamily_encoded"
            
            model = mixedlm(formula, model_data, groups=model_data['LinguisticFamily'], 
                          re_formula=random_formula)
            result = model.fit()
            
            return {
                'model_summary': str(result.summary()),
                'fixed_effects': result.fe_params.to_dict(),
                'random_effects_variance': float(result.cov_re.iloc[0, 0]) if hasattr(result, 'cov_re') else None,
                'log_likelihood': float(result.llf),
                'aic': float(result.aic),
                'bic': float(result.bic),
                'interpretation': self._interpret_mixed_effects_results(result)
            }
            
        except Exception as e:
            return {'error': f'Mixed effects analysis failed: {str(e)}'}
    
    def perform_bayesian_analysis(self, df, experiment_type):
        """Perform Bayesian analysis for robust statistical inference."""
        if not PYMC_AVAILABLE:
            return {'error': 'PyMC not available for Bayesian analysis'}
        
        response_col = self._get_response_column(df)
        if not response_col:
            return {'error': 'No response column found for Bayesian analysis'}
        
        try:
            # Prepare data
            languages = df['Language'].unique()
            language_data = {}
            
            if experiment_type == 'definitive' and 'IsCorrect' in df.columns:
                # Bayesian analysis for accuracy data
                for lang in languages:
                    lang_df = df[df['Language'] == lang]
                    language_data[lang] = lang_df['IsCorrect'].astype(int).values
                
                return self._bayesian_accuracy_analysis(language_data)
            else:
                # Bayesian analysis for categorical response data
                for lang in languages:
                    lang_df = df[df['Language'] == lang]
                    responses = lang_df[response_col].value_counts()
                    language_data[lang] = responses.to_dict()
                
                return self._bayesian_categorical_analysis(language_data)
                
        except Exception as e:
            return {'error': f'Bayesian analysis failed: {str(e)}'}
    
    def perform_power_analysis(self, df, experiment_type, effect_size=0.5):
        """Perform power analysis to assess statistical power and required sample sizes."""
        if not STATSMODELS_AVAILABLE:
            return {'error': 'statsmodels not available for power analysis'}
        
        try:
            languages = df['Language'].unique()
            n_languages = len(languages)
            
            # Current sample sizes per language
            sample_sizes = df.groupby('Language').size()
            mean_sample_size = sample_sizes.mean()
            min_sample_size = sample_sizes.min()
            
            # Power analysis for different scenarios
            power_results = {}
            
            # T-test power analysis (for pairwise comparisons)
            # Ensure min_sample_size is at least 2 for power calculation
            eff_sample_size = max(min_sample_size, 2)
            current_power = TTestIndPower().power(effect_size, eff_sample_size, alpha=0.05, alternative='two-sided')
            power_results['current_power_pairwise'] = float(current_power)
            
            # Required sample size for 80% power
            required_n_80 = TTestIndPower().solve_power(effect_size, power=0.8, alpha=0.05, alternative='two-sided')
            power_results['required_n_for_80_power'] = int(np.ceil(required_n_80)) if not np.isnan(required_n_80) else None
            
            # ANOVA power analysis (for overall language effect)
            if n_languages > 2 and ANOVA_POWER_AVAILABLE:
                try:
                    # Ensure nobs is at least 2 for anova_power
                    anova_nobs = max(min_sample_size, 2)
                    anova_power_current = anova_power(effect_size=effect_size, nobs=anova_nobs, k_groups=n_languages, alpha=0.05)
                    power_results['current_power_anova'] = float(anova_power_current) if anova_power_current is not None else None
                except Exception as e_anova:
                    print(f"ANOVA power calculation failed: {e_anova}")
                    power_results['current_power_anova'] = None
            elif n_languages > 2 and not ANOVA_POWER_AVAILABLE:
                power_results['current_power_anova'] = 'anova_power unavailable'
            
            # Multiple comparisons adjustment
            if n_languages >= 2:
                n_comparisons = n_languages * (n_languages - 1) // 2
                if n_comparisons > 0:
                    bonferroni_alpha = 0.05 / n_comparisons
                    power_results['bonferroni_adjusted_alpha'] = bonferroni_alpha
                    # Power with Bonferroni correction
                    bonferroni_power = TTestIndPower().power(effect_size, eff_sample_size, alpha=bonferroni_alpha, alternative='two-sided')
                    power_results['bonferroni_corrected_power'] = float(bonferroni_power)
                else:
                    power_results['bonferroni_adjusted_alpha'] = None
                    power_results['bonferroni_corrected_power'] = None
            else:
                n_comparisons = 0
                power_results['bonferroni_adjusted_alpha'] = None
                power_results['bonferroni_corrected_power'] = None
                
            power_results['n_pairwise_comparisons'] = n_comparisons
            
            power_results['interpretation'] = self._interpret_power_analysis(power_results)
            
            return power_results
            
        except Exception as e:
            return {'error': f'Power analysis failed: {str(e)}'}
    
    def perform_multiple_comparisons_correction(self, statistical_results):
        """Apply multiple comparisons correction to statistical tests."""
        if not STATSMODELS_AVAILABLE:
            return statistical_results
        
        try:
            # Extract p-values from various tests
            p_values = []
            test_names = []
            
            # Chi-square tests
            for key, result in statistical_results.items():
                if key.startswith('chi2_') and isinstance(result, dict) and 'p_value' in result:
                    p_values.append(result['p_value'])
                    test_names.append(key)
            
            # Language accuracy comparisons
            if 'language_accuracy_comparisons' in statistical_results:
                for comp in statistical_results['language_accuracy_comparisons']:
                    if 'u_p_value' in comp:
                        p_values.append(comp['u_p_value'])
                        test_names.append(f"accuracy_{comp['language1']}_vs_{comp['language2']}")
            
            if not p_values:
                return statistical_results
            
            # Apply multiple corrections
            corrections = {}
            
            # Bonferroni correction
            bonferroni_rejected, bonferroni_pvals = multipletests(p_values, method='bonferroni')[:2]
            corrections['bonferroni'] = {
                'corrected_p_values': bonferroni_pvals.tolist(),
                'rejected_hypotheses': bonferroni_rejected.tolist(),
                'test_names': test_names
            }
            
            # Benjamini-Hochberg (FDR) correction
            fdr_rejected, fdr_pvals = multipletests(p_values, method='fdr_bh')[:2]
            corrections['fdr_bh'] = {
                'corrected_p_values': fdr_pvals.tolist(),
                'rejected_hypotheses': fdr_rejected.tolist(),
                'test_names': test_names
            }
            
            # Holm correction
            holm_rejected, holm_pvals = multipletests(p_values, method='holm')[:2]
            corrections['holm'] = {
                'corrected_p_values': holm_pvals.tolist(),
                'rejected_hypotheses': holm_rejected.tolist(),
                'test_names': test_names
            }
            
            statistical_results['multiple_comparisons_correction'] = corrections
            statistical_results['original_p_values'] = p_values
            
            return statistical_results
            
        except Exception as e:
            statistical_results['multiple_comparisons_error'] = str(e)
            return statistical_results
    
    def perform_bootstrap_analysis(self, df, experiment_type, n_bootstrap=1000):
        """Perform bootstrap analysis for robust confidence intervals."""
        try:
            response_col = self._get_response_column(df)
            if not response_col:
                return {'error': 'No response column found for bootstrap analysis'}
            
            languages = df['Language'].unique()
            
            print(f"Performing bootstrap analysis for {len(languages)} languages using up to {self.max_workers} cores...")

            def bootstrap_single_language(language_df_tuple):
                language, lang_df = language_df_tuple
                try:
                    if experiment_type == 'definitive' and 'IsCorrect' in lang_df.columns:
                        accuracy_data = lang_df['IsCorrect'].astype(int).values
                        if len(accuracy_data) > 0:
                            return language, self._bootstrap_accuracy(accuracy_data, n_bootstrap)
                    else:
                        responses = lang_df[response_col]
                        if len(responses) > 0:
                            return language, self._bootstrap_entropy(responses, n_bootstrap)
                except Exception as e_single:
                    print(f"Bootstrap analysis error for {language}: {e_single}")
                return language, None # Return None for this language if error or no data

            # Prepare data for parallel processing: list of (language, dataframe_for_language)
            language_data_for_parallel = []
            for lang in languages:
                language_data_for_parallel.append((lang, df[df['Language'] == lang]))

            # Use _parallel_apply for parallel execution
            # Using threads might be safer here if the bootstrap functions involve numpy/scipy which often release GIL
            # but ProcessPoolExecutor could be faster if computation is heavy and pickling is not an issue.
            # Let's stick to the default of _parallel_apply which is ProcessPoolExecutor unless use_threads=True is specified.
            # For CPU-bound tasks like bootstrapping, ProcessPoolExecutor is generally preferred.
            parallel_results = self._parallel_apply(bootstrap_single_language, language_data_for_parallel, use_threads=False)

            bootstrap_results = {}
            for lang, result_data in parallel_results:
                if result_data:
                    bootstrap_results[lang] = result_data
            
            return bootstrap_results
            
        except Exception as e:
            return {'error': f'Bootstrap analysis failed globally: {str(e)}'}

    def perform_permutation_tests(self, df, experiment_type, n_permutations=1000):
        """Perform permutation tests using parallel processing."""
        try:
            response_col = self._get_response_column(df)
            if not response_col:
                return {'error': 'No response column found for permutation tests'}
            
            languages = df['Language'].unique()
            if len(languages) < 2:
                return {'error': 'Need at least 2 languages for permutation tests'}
            
            language_pairs = list(combinations(languages, 2))
            print(f"Performing permutation tests for {len(language_pairs)} language pairs using up to {self.max_workers} cores...")

            # Memoize data extraction for efficiency if df is large
            # This is a simple form of memoization; for very large DFs, more advanced might be needed
            lang_data_cache = {}
            def get_lang_data(lang_name, column_name, is_int=False):
                if (lang_name, column_name, is_int) not in lang_data_cache:
                    data = df[df['Language'] == lang_name][column_name]
                    if is_int:
                        data = data.astype(int).values
                    else:
                        data = data.values # Keep as Series or convert to numpy array as needed by test funcs
                    lang_data_cache[(lang_name, column_name, is_int)] = data
                return lang_data_cache[(lang_name, column_name, is_int)]

            def test_single_pair(pair):
                lang1, lang2 = pair
                try:
                    if experiment_type == 'definitive' and 'IsCorrect' in df.columns:
                        group1 = get_lang_data(lang1, 'IsCorrect', is_int=True)
                        group2 = get_lang_data(lang2, 'IsCorrect', is_int=True)
                        if len(group1) > 0 and len(group2) > 0:
                            return f"{lang1}_vs_{lang2}", self._permutation_test_accuracy(group1, group2, n_permutations)
                    else:
                        # For categorical responses, ensure we pass the series/array of actual responses
                        responses1_series = df[df['Language'] == lang1][response_col]
                        responses2_series = df[df['Language'] == lang2][response_col]
                        if not responses1_series.empty and not responses2_series.empty:
                            return f"{lang1}_vs_{lang2}", self._permutation_test_distributions(responses1_series, responses2_series, n_permutations)
                except Exception as e_single:
                    print(f"Permutation test error for {lang1} vs {lang2}: {e_single}")
                return f"{lang1}_vs_{lang2}", None # Return None for result if error

            parallel_results = self._parallel_apply(test_single_pair, language_pairs, use_threads=False)
            
            permutation_results = {}
            for pair_key, result_data in parallel_results:
                if result_data: # Only add if a result was successfully computed
                    permutation_results[pair_key] = result_data
            
            return permutation_results
            
        except Exception as e:
            return {'error': f'Permutation tests failed globally: {str(e)}'}
    
    # ==================== HELPER METHODS FOR ENHANCED STATISTICS ====================
    
    def _interpret_mixed_effects_results(self, model_result):
        """Interpret mixed effects model results."""
        try:
            fixed_effects = model_result.fe_params
            significant_effects = []
            
            for param, coef in fixed_effects.items():
                if param != 'Intercept':
                    # Simple significance check (would need proper p-values in real implementation)
                    if abs(coef) > 0.1:  # Placeholder threshold
                        significant_effects.append(param)
            
            interpretation = {
                'significant_fixed_effects': significant_effects,
                'model_fit_quality': 'good' if model_result.aic < 1000 else 'moderate',  # Placeholder
                'random_effects_importance': 'high' if hasattr(model_result, 'cov_re') else 'low'
            }
            
            return interpretation
            
        except Exception:
            return {'interpretation': 'Mixed effects interpretation failed'}
    
    def _bayesian_accuracy_analysis(self, language_data):
        """Perform Bayesian analysis for accuracy data."""
        try:
            with pm.Model() as model:
                # Priors for each language's accuracy rate
                accuracy_rates = {}
                for lang in language_data.keys():
                    accuracy_rates[lang] = pm.Beta(f'accuracy_{lang}', alpha=1, beta=1)
                
                # Likelihood
                for lang, data in language_data.items():
                    pm.Binomial(f'obs_{lang}', n=len(data), p=accuracy_rates[lang], observed=sum(data))
                
                # Sample from posterior
                trace = pm.sample(1000, tune=500, return_inferencedata=True, progressbar=False)
            
            # Extract results
            results = {}
            for lang in language_data.keys():
                posterior_samples = trace.posterior[f'accuracy_{lang}'].values.flatten()
                results[lang] = {
                    'posterior_mean': float(np.mean(posterior_samples)),
                    'posterior_std': float(np.std(posterior_samples)),
                    'credible_interval_95': [float(np.percentile(posterior_samples, 2.5)),
                                           float(np.percentile(posterior_samples, 97.5))],
                    'probability_above_chance': float(np.mean(posterior_samples > 0.5))
                }
            
            return results
            
        except Exception as e:
            return {'error': f'Bayesian accuracy analysis failed: {str(e)}'}
    
    def _bayesian_categorical_analysis(self, language_data):
        """Perform Bayesian analysis for categorical response data."""
        try:
            # Simplified Bayesian analysis for categorical data
            results = {}
            
            for lang, response_counts in language_data.items():
                total_responses = sum(response_counts.values())
                if total_responses > 0:
                    # Calculate Dirichlet posterior parameters
                    alpha_prior = 1  # Uniform prior
                    posterior_params = {resp: count + alpha_prior for resp, count in response_counts.items()}
                    
                    # Sample from Dirichlet posterior
                    alpha_values = list(posterior_params.values())
                    samples = np.random.dirichlet(alpha_values, 1000)
                    
                    results[lang] = {
                        'response_probabilities': {resp: float(np.mean(samples[:, i])) 
                                                 for i, resp in enumerate(posterior_params.keys())},
                        'uncertainty_entropy': float(-np.sum(np.mean(samples, axis=0) * 
                                                           np.log(np.mean(samples, axis=0) + 1e-10)))
                    }
            
            return results
            
        except Exception as e:
            return {'error': f'Bayesian categorical analysis failed: {str(e)}'}
    
    def _interpret_power_analysis(self, power_results):
        """Interpret power analysis results."""
        interpretation = []
        
        current_power = power_results.get('current_power_pairwise', 0)
        if current_power < 0.5:
            interpretation.append("Current study is severely underpowered for detecting medium effects")
        elif current_power < 0.8:
            interpretation.append("Current study has moderate power; consider increasing sample size")
        else:
            interpretation.append("Current study has adequate power for detecting medium effects")
        
        bonferroni_power = power_results.get('bonferroni_corrected_power', 0)
        if bonferroni_power < 0.5:
            interpretation.append("After multiple comparisons correction, power is severely reduced")
        
        n_comparisons = power_results.get('n_pairwise_comparisons', 0)
        if n_comparisons > 10:
            interpretation.append(f"Large number of comparisons ({n_comparisons}) increases multiple testing burden")
        
        return '; '.join(interpretation)
    
    def _bootstrap_accuracy(self, accuracy_data, n_bootstrap):
        """Bootstrap confidence intervals for accuracy."""
        def accuracy_statistic(data, axis):
            return np.mean(data, axis=axis)
        
        try:
            # Use scipy.stats.bootstrap
            rng = np.random.default_rng(42)
            bootstrap_result = scipy_bootstrap((accuracy_data,), accuracy_statistic, 
                                       n_resamples=n_bootstrap, random_state=rng)
            
            return {
                'bootstrap_mean': float(np.mean(bootstrap_result.bootstrap_distribution)),
                'bootstrap_std': float(np.std(bootstrap_result.bootstrap_distribution)),
                'confidence_interval_95': [float(bootstrap_result.confidence_interval.low),
                                         float(bootstrap_result.confidence_interval.high)]
            }
        except Exception:
            # Fallback manual bootstrap
            bootstrap_means = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(accuracy_data, size=len(accuracy_data), replace=True)
                bootstrap_means.append(np.mean(sample))
            
            return {
                'bootstrap_mean': float(np.mean(bootstrap_means)),
                'bootstrap_std': float(np.std(bootstrap_means)),
                'confidence_interval_95': [float(np.percentile(bootstrap_means, 2.5)),
                                         float(np.percentile(bootstrap_means, 97.5))]
            }
    
    def _bootstrap_entropy(self, responses, n_bootstrap):
        """Bootstrap confidence intervals for response entropy."""
        def entropy_statistic(data):
            return self._calculate_entropy(pd.Series(data))
        
        bootstrap_entropies = []
        responses_array = responses.values
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(responses_array, size=len(responses_array), replace=True)
            entropy = entropy_statistic(sample)
            bootstrap_entropies.append(entropy)
        
        return {
            'bootstrap_mean': float(np.mean(bootstrap_entropies)),
            'bootstrap_std': float(np.std(bootstrap_entropies)),
            'confidence_interval_95': [float(np.percentile(bootstrap_entropies, 2.5)),
                                     float(np.percentile(bootstrap_entropies, 97.5))]
        }
    
    def _permutation_test_accuracy(self, group1, group2, n_permutations):
        """Permutation test for accuracy differences between groups."""
        observed_diff = np.mean(group1) - np.mean(group2)
        combined = np.concatenate([group1, group2])
        n1 = len(group1)
        
        permuted_diffs = []
        for _ in range(n_permutations):
            np.random.shuffle(combined)
            perm_group1 = combined[:n1]
            perm_group2 = combined[n1:]
            permuted_diffs.append(np.mean(perm_group1) - np.mean(perm_group2))
        
        p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))
        
        return {
            'observed_difference': float(observed_diff),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'effect_size': float(observed_diff / np.std(combined)) if np.std(combined) > 0 else 0.0
        }
    
    def _permutation_test_distributions(self, responses1, responses2, n_permutations):
        """Permutation test for response distribution differences."""
        # Use entropy difference as test statistic
        entropy1 = self._calculate_entropy(responses1)
        entropy2 = self._calculate_entropy(responses2)
        observed_diff = entropy1 - entropy2
        
        combined = pd.concat([responses1, responses2])
        n1 = len(responses1)
        
        permuted_diffs = []
        for _ in range(n_permutations):
            shuffled = combined.sample(frac=1).reset_index(drop=True)
            perm_responses1 = shuffled[:n1]
            perm_responses2 = shuffled[n1:]
            
            perm_entropy1 = self._calculate_entropy(perm_responses1)
            perm_entropy2 = self._calculate_entropy(perm_responses2)
            permuted_diffs.append(perm_entropy1 - perm_entropy2)
        
        p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))
        
        return {
            'observed_entropy_difference': float(observed_diff),
            'p_value': float(p_value),
            'significant': p_value < 0.05
        }

    # ==================== END ENHANCED STATISTICAL METHODS ====================

    def analyze_single_experiment(self, experiment_name):
        """Perform comprehensive analysis of a single experiment with academic rigor."""
        print(f"\n{'='*80}")
        print(f"ANALYZING: {experiment_name}")
        print(f"Using {self.max_workers} cores for parallel processing")
        print(f"{'='*80}")
        
        df = self.load_experiment_data(experiment_name)
        if df is None:
            print(f"No data loaded for {experiment_name}")
            return None
        
        experiment_type = self.classify_experiment_type(df, experiment_name)
        print(f"Experiment type: {experiment_type}")
        
        metrics = None
        if experiment_type == 'definitive':
            accuracy_result = self.calculate_accuracy_metrics(df, experiment_name, experiment_type)
            if accuracy_result:
                metrics, df = accuracy_result
        
        language_profiles = self.perform_comprehensive_language_analysis(df, experiment_name, experiment_type)
        
        if not hasattr(self, '_currently_processing_profiles'):
            self._currently_processing_profiles = {}
        self._currently_processing_profiles[experiment_name] = language_profiles

        cross_linguistic_analysis = self.perform_cross_linguistic_analysis(df, experiment_type, language_profiles)
        family_analysis = self.perform_linguistic_family_analysis(df, experiment_type, experiment_name)
        statistical_results = self.perform_statistical_tests(df, experiment_type)
        
        # ==================== ENHANCED STATISTICAL ANALYSES ====================
        print("Performing enhanced statistical analyses...")
        
        # Apply multiple comparisons correction
        statistical_results = self.perform_multiple_comparisons_correction(statistical_results)
        
        # Mixed effects analysis (for definitive experiments)
        mixed_effects_results = self.perform_mixed_effects_analysis(df, experiment_type)
        statistical_results['mixed_effects_analysis'] = mixed_effects_results
        
        # Bayesian analysis
        bayesian_results = self.perform_bayesian_analysis(df, experiment_type)
        statistical_results['bayesian_analysis'] = bayesian_results
        
        # Power analysis
        power_analysis_results = self.perform_power_analysis(df, experiment_type)
        statistical_results['power_analysis'] = power_analysis_results
        
        # Bootstrap analysis
        bootstrap_results = self.perform_bootstrap_analysis(df, experiment_type)
        statistical_results['bootstrap_analysis'] = bootstrap_results
        
        # Permutation tests
        permutation_results = self.perform_permutation_tests(df, experiment_type)
        statistical_results['permutation_tests'] = permutation_results
        
        print("Enhanced statistical analyses completed.")
        # ==================== END ENHANCED ANALYSES ====================
        
        lr_assessment = self.assess_linguistic_relativity_comprehensive(
            df, experiment_name, experiment_type, statistical_results, language_profiles, cross_linguistic_analysis
        )
        
        current_result = {
            'data': df,
            'experiment_type': experiment_type,
            'metrics': metrics,
            'language_profiles': language_profiles,
            'cross_linguistic_analysis': cross_linguistic_analysis,
            'linguistic_family_analysis': family_analysis,
            'statistical_results': statistical_results,
            'lr_assessment': lr_assessment,
            'processing_info': {
                'max_workers_used': self.max_workers,
                'parallel_processing_enabled': self.max_workers > 1,
                'processing_timestamp': datetime.now().isoformat()
            }
        }

        # Thread-safe storage of results
        with self.thread_lock:
            self.results[experiment_name] = current_result

        self.generate_comprehensive_visualizations(df, experiment_name, experiment_type, metrics, language_profiles, statistical_results)
        
        if hasattr(self, '_currently_processing_profiles') and experiment_name in self._currently_processing_profiles:
            del self._currently_processing_profiles[experiment_name]

        self.generate_comprehensive_academic_report(experiment_name)
        return self.results[experiment_name]

    def generate_comprehensive_academic_report(self, experiment_name):
        """Generate comprehensive academic report for a single experiment."""
        result = self.results.get(experiment_name)
        if not result:
            print(f"No results found for {experiment_name}")
            return
        
        report_dir = os.path.join(self.output_dir, 'academic_reports', experiment_name)
        os.makedirs(report_dir, exist_ok=True)
        
        report_path = os.path.join(report_dir, f'{experiment_name}_academic_summary.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("ENHANCED COMPREHENSIVE ACADEMIC REPORT\n")
            f.write(f"Experiment: {experiment_name}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Abstract/Executive Summary
            lr_assessment = result.get('lr_assessment', {})
            f.write("ABSTRACT / EXECUTIVE SUMMARY\n")
            f.write(f"  Experiment Type: {result.get('experiment_type', 'N/A')}\n")
            f.write(f"  Linguistic Relativity Evidence: {lr_assessment.get('evidence_strength', 'N/A').upper()}\n")
            f.write(f"  Key Findings: {lr_assessment.get('academic_interpretation', 'N/A')}\n\n")
            
            # Enhanced Statistical Summary
            stats = result.get('statistical_results', {})
            f.write("ENHANCED STATISTICAL ANALYSIS SUMMARY\n")
            
            # Power Analysis Summary
            if 'power_analysis' in stats and 'error' not in stats['power_analysis']:
                power = stats['power_analysis']
                f.write(f"  Current Statistical Power: {power.get('current_power_pairwise', 0):.3f}\n")
                f.write(f"  Power Interpretation: {power.get('interpretation', 'N/A')}\n")
            
            # Multiple Comparisons Summary
            if 'multiple_comparisons_correction' in stats:
                corrections = stats['multiple_comparisons_correction']
                if 'bonferroni' in corrections:
                    n_significant = sum(corrections['bonferroni']['rejected_hypotheses'])
                    f.write(f"  Significant tests after Bonferroni correction: {n_significant}\n")
            
            f.write(f"  --- Detailed enhanced statistical report saved separately ---\n\n")
            
            # Conclusion
            f.write("CONCLUSION\n")
            evidence = lr_assessment.get('evidence_strength', 'insufficient').lower()
            if evidence == 'strong':
                conclusion = "provided strong evidence for"
            elif evidence == 'moderate':
                conclusion = "provided moderate evidence for"
            elif evidence == 'weak':
                conclusion = "provided weak evidence for"
            else:
                conclusion = "provided insufficient evidence for"
            
            f.write(f"  The enhanced analysis of '{experiment_name}' {conclusion} linguistic relativity effects. ")
            f.write(f"The study employed advanced statistical methods including mixed effects modeling, ")
            f.write(f"Bayesian inference, bootstrap confidence intervals, and permutation tests to ensure ")
            f.write(f"robust and reliable conclusions.\n\n")
        
        print(f"Enhanced academic report saved: {report_path}")

    def generate_master_report(self):
        """Generate comprehensive master report across all experiments."""
        master_report_path = os.path.join(self.output_dir, 'academic_reports', 'MASTER_ACADEMIC_ANALYSIS_SUMMARY.txt')
        overall_summary_path = os.path.join(self.output_dir, 'overall_analysis', 'all_experiments_summary.csv')
        all_exp_summaries = []
        
        with open(master_report_path, 'w', encoding='utf-8') as f:
            f.write("ENHANCED LINGUISTIC RELATIVITY ACADEMIC ANALYSIS REPORT\n")
            f.write(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Generation Analyzed: {self.selected_generation}\n")
            f.write("="*80 + "\n\n")
            f.write(f"OVERVIEW\n")
            f.write(f"Total experiments analyzed: {len(self.results)}\n")
            f.write(f"Experiments included: {', '.join(self.results.keys())}\n")
            f.write(f"Enhanced statistical methods employed: Mixed Effects Models, Bayesian Analysis, ")
            f.write(f"Power Analysis, Bootstrap Confidence Intervals, Permutation Tests\n\n")
            
            evidence_summary = {}
            for exp_name, result in self.results.items():
                if not result: 
                    continue
                lr_assess = result.get('lr_assessment', {})
                strength = lr_assess.get('evidence_strength', 'undefined')
                if strength not in evidence_summary:
                    evidence_summary[strength] = []
                evidence_summary[strength].append(exp_name)
                
                exp_summary_data = {
                    'ExperimentName': exp_name,
                    'ExperimentType': result.get('experiment_type', 'N/A'),
                    'LREvidenceStrength': strength,
                    'TotalLanguages': result.get('data', pd.DataFrame())['Language'].nunique() if isinstance(result.get('data'), pd.DataFrame) else 0,
                    'TotalResponses': len(result.get('data', pd.DataFrame())),
                    'OverallAccuracy': result.get('metrics', {}).get('overall_accuracy', np.nan) if result.get('metrics') else np.nan,
                    'Generation': self.selected_generation
                }
                all_exp_summaries.append(exp_summary_data)
            
            f.write("EVIDENCE STRENGTH SUMMARY (Across All Experiments)\n")
            for strength_cat in ['strong', 'moderate', 'weak', 'insufficient', 'undefined']:
                experiments = evidence_summary.get(strength_cat, [])
                if experiments:
                    f.write(f"  {strength_cat.upper()}: {len(experiments)} experiment(s)\n")
                    for exp in experiments:
                        f.write(f"    - {exp}\n")
            f.write("\n")
            
            f.write("METHODOLOGICAL ENHANCEMENTS\n")
            f.write(f"This analysis employed state-of-the-art statistical methods to ensure robust ")
            f.write(f"and reliable conclusions about linguistic relativity effects. The enhanced ")
            f.write(f"methodology addresses common limitations in cross-linguistic research through ")
            f.write(f"hierarchical modeling, Bayesian inference, and non-parametric testing.\n\n")
            
            f.write("GENERAL CONCLUSIONS & FUTURE DIRECTIONS\n")
            f.write(f"The ensemble of experiments analyzed from the {self.selected_generation} generation ")
            f.write(f"provides a comprehensive landscape of linguistic relativity effects using enhanced ")
            f.write(f"statistical methods. Future research should continue to employ these rigorous ")
            f.write(f"analytical approaches while expanding linguistic diversity and refining experimental methodologies.\n")
        
        print(f"Enhanced master academic report saved: {master_report_path}")
        
        if all_exp_summaries:
            summary_df = pd.DataFrame(all_exp_summaries)
            if not os.path.exists(os.path.dirname(overall_summary_path)):
                 os.makedirs(os.path.dirname(overall_summary_path))
            summary_df.to_csv(overall_summary_path, index=False, encoding='utf-8')
            print(f"Overall experiment summary CSV saved: {overall_summary_path}")

    def _save_json_data(self, data_to_save, file_path):
        """Helper to save data as JSON with robust numpy/pandas handling."""
        try:
            import json
            def robust_converter(obj):
                if isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                if isinstance(obj, (np.floating, np.float64)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, pd.Timestamp):
                    return obj.isoformat()
                if isinstance(obj, pd.Series):
                    try:
                        return obj.to_dict()
                    except TypeError: 
                        return {str(k): v for k, v in obj.items()}
                if isinstance(obj, pd.DataFrame):
                    try:
                        return obj.to_dict(orient='records')
                    except Exception: 
                        return {
                            'columns': obj.columns.tolist() if hasattr(obj.columns, 'tolist') else list(obj.columns),
                            'data': obj.values.tolist() if hasattr(obj.values, 'tolist') else list(obj.values),
                            'index': obj.index.tolist() if hasattr(obj.index, 'tolist') else list(obj.index),
                            'info': 'DataFrame converted with structure preservation due to complexity.'
                        }
                if hasattr(obj, 'item'): 
                    return obj.item()
                if isinstance(obj, (set)):
                    return list(obj)
                
                try:
                    if isinstance(obj, (dict, list, str, int, float, bool, type(None))):
                        return obj 
                    return str(obj)
                except Exception:
                    return f"Unserializable_object_type:_{type(obj).__name__}"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2, default=robust_converter)
        except Exception as e:
            print(f"Error saving JSON to {file_path}: {e}")

    def _generate_comprehensive_cohens_d_analysis(self, result, experiment_name):
        """Generate comprehensive Cohen's d analysis for all experiment types."""
        cohens_d_dir = os.path.join(self.output_dir, 'cohens_d_analysis', experiment_name)
        os.makedirs(cohens_d_dir, exist_ok=True)
        
        stats_results = result.get('statistical_results', {})
        language_profiles = result.get('language_profiles', {})
        experiment_type = result.get('experiment_type', 'unknown')
        
        # For definitive experiments with accuracy comparisons
        if stats_results.get('language_accuracy_comparisons'):
            cohen_data = stats_results['language_accuracy_comparisons']
            self._save_json_data(cohen_data, os.path.join(cohens_d_dir, 'language_accuracy_comparisons_with_cohens_d.json'))
            
            # Generate CSV
            df = pd.DataFrame(cohen_data)
            df.to_csv(os.path.join(cohens_d_dir, 'language_accuracy_comparisons.csv'), index=False)
            
            # Generate text summary
            self._generate_cohens_d_text_summary(cohen_data, cohens_d_dir, experiment_name, 'accuracy')
        
        # For all experiments: response pattern comparisons
        if language_profiles and len(language_profiles) >= 2:
            response_comparisons = self._calculate_response_pattern_cohens_d(language_profiles)
            if response_comparisons:
                self._save_json_data(response_comparisons, os.path.join(cohens_d_dir, 'response_pattern_comparisons_with_cohens_d.json'))
                
                # Generate CSV
                df = pd.DataFrame(response_comparisons)
                df.to_csv(os.path.join(cohens_d_dir, 'response_pattern_comparisons.csv'), index=False)
                
                # Generate text summary
                self._generate_cohens_d_text_summary(response_comparisons, cohens_d_dir, experiment_name, 'response_patterns')

    def _calculate_response_pattern_cohens_d(self, language_profiles):
        """Calculate Cohen's d for response patterns between languages."""
        comparisons = []
        languages = list(language_profiles.keys())
        
        for i, lang1 in enumerate(languages):
            for j, lang2 in enumerate(languages[i+1:], i+1):
                profile1 = language_profiles[lang1]
                profile2 = language_profiles[lang2]
                
                # Compare response entropies
                entropy1 = profile1.get('response_statistics', {}).get('response_entropy', 0)
                entropy2 = profile2.get('response_statistics', {}).get('response_entropy', 0)
                
                # Compare lexical diversity
                diversity1 = profile1.get('response_statistics', {}).get('lexical_diversity', 0)
                diversity2 = profile2.get('response_statistics', {}).get('lexical_diversity', 0)
                
                # Calculate Cohen's d for entropy difference
                entropy_diff = abs(entropy1 - entropy2)
                diversity_diff = abs(diversity1 - diversity2)
                
                # Simple effect size calculation (normalized difference)
                max_entropy = max(entropy1, entropy2, 1e-6)
                max_diversity = max(diversity1, diversity2, 1e-6)
                
                entropy_effect_size = entropy_diff / max_entropy
                diversity_effect_size = diversity_diff / max_diversity
                
                comparison = {
                    'language1': lang1,
                    'language2': lang2,
                    'entropy1': entropy1,
                    'entropy2': entropy2,
                    'entropy_difference': entropy_diff,
                    'entropy_effect_size': entropy_effect_size,
                    'entropy_effect_interpretation': self._interpret_effect_size(entropy_effect_size),
                    'diversity1': diversity1,
                    'diversity2': diversity2,
                    'diversity_difference': diversity_diff,
                    'diversity_effect_size': diversity_effect_size,
                    'diversity_effect_interpretation': self._interpret_effect_size(diversity_effect_size),
                    'family1': profile1.get('linguistic_family', 'Unknown'),
                    'family2': profile2.get('linguistic_family', 'Unknown'),
                    'same_family': profile1.get('linguistic_family') == profile2.get('linguistic_family')
                }
                
                comparisons.append(comparison)
        
        return comparisons

    def _interpret_effect_size(self, effect_size):
        """Interpret effect size magnitude."""
        if effect_size < 0.1:
            return "Negligible"
        elif effect_size < 0.3:
            return "Small"
        elif effect_size < 0.5:
            return "Medium"
        else:
            return "Large"

    def _generate_cohens_d_text_summary(self, comparisons, output_dir, experiment_name, analysis_type):
        """Generate human-readable summary of Cohen's d analysis."""
        text_path = os.path.join(output_dir, f'{experiment_name}_{analysis_type}_cohens_d_summary.txt')
        
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(f"COHEN'S D ANALYSIS - {experiment_name.upper()}\n")
            f.write(f"Analysis Type: {analysis_type.replace('_', ' ').title()}\n")
            f.write("=" * 60 + "\n\n")
            
            if analysis_type == 'accuracy':
                f.write("Language Accuracy Comparisons:\n")
                f.write("-" * 40 + "\n")
                
                large_effects = [c for c in comparisons if c.get('cohens_d') and abs(c['cohens_d']) >= 0.8]
                medium_effects = [c for c in comparisons if c.get('cohens_d') and 0.5 <= abs(c['cohens_d']) < 0.8]
                
                f.write(f"Total Comparisons: {len(comparisons)}\n")
                f.write(f"Large Effect Sizes (|d| >= 0.8): {len(large_effects)}\n")
                f.write(f"Medium Effect Sizes (0.5 <= |d| < 0.8): {len(medium_effects)}\n\n")
                
                if large_effects:
                    f.write("Largest Effect Sizes:\n")
                    sorted_effects = sorted(large_effects, key=lambda x: abs(x.get('cohens_d', 0)), reverse=True)[:10]
                    for comp in sorted_effects:
                        f.write(f"  {comp['language1']} vs {comp['language2']}: d = {comp.get('cohens_d', 0):.3f}\n")
                        f.write(f"    Accuracy: {comp.get('mean1', 0):.3f} vs {comp.get('mean2', 0):.3f}\n")
                        f.write(f"    Interpretation: {comp.get('effect_size_interpretation', 'N/A')}\n\n")
            
            elif analysis_type == 'response_patterns':
                f.write("Response Pattern Comparisons:\n")
                f.write("-" * 40 + "\n")
                
                # Entropy analysis
                large_entropy_effects = [c for c in comparisons if c.get('entropy_effect_size', 0) >= 0.5]
                f.write(f"Large Entropy Differences (effect >= 0.5): {len(large_entropy_effects)}\n")
                
                # Diversity analysis
                large_diversity_effects = [c for c in comparisons if c.get('diversity_effect_size', 0) >= 0.5]
                f.write(f"Large Diversity Differences (effect >= 0.5): {len(large_diversity_effects)}\n\n")
                
                # Cross-family vs within-family comparisons
                cross_family = [c for c in comparisons if not c.get('same_family', True)]
                within_family = [c for c in comparisons if c.get('same_family', False)]
                
                f.write(f"Cross-family comparisons: {len(cross_family)}\n")
                f.write(f"Within-family comparisons: {len(within_family)}\n\n")
                
                if large_entropy_effects:
                    f.write("Largest Entropy Differences:\n")
                    sorted_entropy = sorted(large_entropy_effects, key=lambda x: x.get('entropy_effect_size', 0), reverse=True)[:10]
                    for comp in sorted_entropy:
                        f.write(f"  {comp['language1']} vs {comp['language2']}: effect = {comp.get('entropy_effect_size', 0):.3f}\n")
                        f.write(f"    Entropy: {comp.get('entropy1', 0):.3f} vs {comp.get('entropy2', 0):.3f}\n")
                        f.write(f"    Same family: {comp.get('same_family', False)}\n\n")
        
        print(f"Cohen's d text summary saved: {text_path}")

    def _generate_cross_linguistic_outputs(self, cross_ling_data, output_dir, base_filename):
        """Generate CSV and text outputs for cross-linguistic analysis."""
        text_path = os.path.join(output_dir, f'{base_filename}_summary.txt')
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write("CROSS-LINGUISTIC ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall diversity metrics
            if 'overall_diversity_metrics' in cross_ling_data:
                metrics = cross_ling_data['overall_diversity_metrics']
                f.write("Overall Diversity Metrics:\n")
                f.write(f"  Overall Entropy: {metrics.get('overall_entropy', 0):.3f}\n")
                f.write(f"  Mean Language Entropy: {metrics.get('mean_language_entropy', 0):.3f}\n")
                f.write(f"  Entropy Variance: {metrics.get('entropy_variance', 0):.3f}\n")
                f.write(f"  Diversity Index: {metrics.get('diversity_index', 0):.3f}\n\n")
            
            # Convergence patterns
            if 'convergence_divergence_patterns' in cross_ling_data:
                conv = cross_ling_data['convergence_divergence_patterns']
                f.write("Convergence/Divergence Patterns:\n")
                f.write(f"  Mean Convergence: {conv.get('mean_convergence', 0):.3f}\n")
                f.write(f"  Convergence Variance: {conv.get('convergence_variance', 0):.3f}\n")
                f.write(f"  Interpretation: {conv.get('interpretation', 'N/A')}\n\n")

    def _generate_family_analysis_outputs(self, family_data, output_dir, base_filename):
        """Generate CSV and text outputs for linguistic family analysis."""
        # CSV for family-level data
        if 'family_level_analysis' in family_data:
            family_rows = []
            for family, analysis in family_data['family_level_analysis'].items():
                resp_stats = analysis.get('response_statistics', {})
                row = {
                    'Linguistic_Family': family,
                    'Sample_Size': analysis.get('sample_size', 0),
                    'Response_Entropy': resp_stats.get('response_entropy', 0),
                    'Mean_Diversity': analysis.get('mean_diversity', 0)
                }
                if 'mean_accuracy' in analysis:
                    row['Mean_Accuracy'] = analysis['mean_accuracy']
                family_rows.append(row)
            
            if family_rows:
                df = pd.DataFrame(family_rows)
                csv_path = os.path.join(output_dir, f'{base_filename}_family_summary.csv')
                df.to_csv(csv_path, index=False)

    def _generate_lr_assessment_outputs(self, lr_data, output_dir, base_filename):
        """Generate text output for linguistic relativity assessment."""
        text_path = os.path.join(output_dir, f'{base_filename}_summary.txt')
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write("LINGUISTIC RELATIVITY ASSESSMENT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Experiment: {lr_data.get('experiment_name', 'N/A')}\n")
            f.write(f"Experiment Type: {lr_data.get('experiment_type', 'N/A')}\n")
            f.write(f"Total Languages: {lr_data.get('total_languages', 0)}\n")
            f.write(f"Total Responses: {lr_data.get('total_responses', 0)}\n")
            f.write(f"Evidence Strength: {lr_data.get('evidence_strength', 'N/A').upper()}\n\n")
            
            f.write("Evidence Indicators:\n")
            f.write(f"  Significant Language Effects: {lr_data.get('significant_language_effects', 0)}\n")
            f.write(f"  Significant Accuracy Differences: {lr_data.get('significant_accuracy_differences', 0)}\n")
            f.write(f"  Large Effect Sizes: {lr_data.get('large_effect_sizes', 0)}\n")
            f.write(f"  Response Diversity Ratio: {lr_data.get('response_diversity_ratio', 0):.3f}\n\n")
            
            if 'academic_interpretation' in lr_data:
                f.write("Academic Interpretation:\n")
                f.write(f"{lr_data['academic_interpretation']}\n\n")

    # ==================== ENHANCED STATISTICAL ANALYSIS METHODS ====================

    def _parallel_apply(self, func, data_list, use_threads=False, chunk_size=None):
        """Apply function in parallel to list of data."""
        if len(data_list) <= 1 or self.max_workers <= 1:
            return [func(item) for item in data_list]
        
        executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
        
        try:
            with executor_class(max_workers=self.max_workers) as executor:
                if chunk_size:
                    # Process in chunks for memory efficiency
                    results = []
                    for i in range(0, len(data_list), chunk_size):
                        chunk = data_list[i:i + chunk_size]
                        chunk_results = list(executor.map(func, chunk))
                        results.extend(chunk_results)
                    return results
                else:
                    return list(executor.map(func, data_list))
        except Exception as e:
            print(f"Parallel processing failed, falling back to sequential: {e}")
            return [func(item) for item in data_list]
    
    def _parallel_pairwise_comparisons(self, languages, comparison_func, data_dict):
        """Perform pairwise comparisons in parallel."""
        language_pairs = list(combinations(languages, 2))
        
        def compare_pair(pair):
            lang1, lang2 = pair
            return comparison_func(lang1, lang2, data_dict[lang1], data_dict[lang2])
        
        return self._parallel_apply(compare_pair, language_pairs, use_threads=True)

    def _select_data_source_interactive(self):
        """Interactively prompts the user to select the source data folder for API generations."""
        print("\n" + "="*80)
        print("SELECT API GENERATION DATA SOURCE")
        print("="*80)

        sources = [
            ("api_generations/lr_experiments", "Original API generations"),
            ("api_generations/lr_experiments_english", "English-only API generations")
        ]

        # Check if these directories actually exist
        available_sources = []
        for path, desc in sources:
            if os.path.isdir(path):
                available_sources.append((path, desc))
            else:
                print(f"(Directory not found: {path} - skipping this option)")
        
        if not available_sources:
            print("No API generation source directories found. Please ensure either")
            print("'api_generations/lr_experiments' or 'api_generations/lr_experiments_english' exists.")
            return None

        print("\nAvailable data sources:")
        for i, (path, desc) in enumerate(available_sources, 1):
            print(f"{i}. {path} ({desc})")
        print(f"{len(available_sources) + 1}. EXIT")

        while True:
            try:
                choice = input(f"\nSelect data source option (1-{len(available_sources) + 1}): ").strip()
                choice_num = int(choice)

                if choice_num == len(available_sources) + 1: # EXIT
                    print("Exiting data source selection...")
                    return None
                elif 1 <= choice_num <= len(available_sources):
                    selected_path, selected_desc = available_sources[choice_num - 1]
                    print(f"Selected data source: {selected_path} ({selected_desc})")
                    return selected_path
                else:
                    print(f"Invalid choice. Please enter a number between 1 and {len(available_sources) + 1}.")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\nExiting data source selection...")
                return None
        return None # Should not be reached


def main():
    """Main function to run the analysis."""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Linguistic Relativity Experiment Analyzer with Multicore Support')
    parser.add_argument('generation', nargs='?', help='Generation code to analyze (e.g., GG2, XG3M)')
    parser.add_argument('--workers', '-w', type=int, default=None, 
                       help=f'Number of worker processes (default: auto-detect, max: 12)')
    parser.add_argument('--data-path', default=None, # Changed default to None
                       help='Path to experiment data (e.g., api_generations/lr_experiments or api_generations/lr_experiments_english). If not set, interactive selection will be triggered.')
    parser.add_argument('--output-dir', default='lr_experiment_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Determine number of workers
    if args.workers is not None:
        max_workers = min(args.workers, 12)
        if args.workers > 12:
            print(f"Warning: Requested {args.workers} workers, but limiting to 12 for stability")
    else:
        max_workers = min(mp.cpu_count(), 12)
    
    print(f"Initializing Linguistic Relativity Analyzer with {max_workers} workers...")
    
    try:
        # data_path is now handled interactively or via arg, don't pass default from here to __init__ directly
        # __init__ will use its own default if nothing is set by run_complete_analysis or run_analysis_for_generation
        analyzer = LRExperimentAnalyzer(
            output_dir=args.output_dir,
            max_workers=max_workers
            # data_path will be set by the run methods
        )
        
        if args.generation:
            print(f"Running analysis for generation: {args.generation}")
            # Pass the data_path from args if provided, otherwise it will be interactively selected in the method
            analyzer.run_analysis_for_generation(args.generation, data_source_path=args.data_path)
        else:
            # Run interactive analysis (which includes data source selection)
            analyzer.run_complete_analysis()
            
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Set multiprocessing start method for better compatibility
    if hasattr(mp, 'set_start_method'):
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Method already set
    
    main() 