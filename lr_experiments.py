import os
import sys

# Add the functions directory to the Python path to allow imports
# This assumes the script is run from the root of your project directory
current_dir = os.path.dirname(os.path.abspath(__file__))
functions_dir = os.path.join(current_dir, "lr_experiment_functions")
sys.path.insert(0, functions_dir)

# Now import functions from the grammatical_gender_experiment module
# We need load_grammatical_genders for the main menu logic if it checks for the file
# and run_grammatical_gender_distance_analysis for the experiment itself.
from lr_experiment_functions.grammatical_gender_experiment import (
    load_grammatical_genders, 
    run_grammatical_gender_distance_analysis
)
# Import experiment functions
from lr_experiment_functions.color_experiment import run_color_experiment
from lr_experiment_functions.temporal_reasoning_experiment import run_temporal_reasoning_experiment
from lr_experiment_functions.spatial_reasoning_experiment import run_spatial_reasoning_experiment
from lr_experiment_functions.quantitative_reasoning_experiment import run_quantitative_reasoning_experiment
from lr_experiment_functions.acoustic_reasoning_experiment import run_acoustic_reasoning_experiment

# Import our new analysis tools
from lr_analysis_functions import llm_config_assessment
from lr_analysis_functions import embedder_assessment

INPUTS_DIR = os.path.join(current_dir, "inputs") # Define for load_grammatical_genders

def print_menu():
    """Print the main menu of the LR experiments."""
    print("\n=== Linguistic Relativity Experiments ===")
    print("1. Grammatical Gender Analysis")
    print("2. Color Perception Analysis")
    print("3. Temporal Reasoning Analysis")
    print("4. Spatial Reasoning Analysis")
    print("5. Quantitative Reasoning Analysis")
    print("6. Acoustic Reasoning Analysis")
    print("\n=== Meta-Analysis Tools ===")
    print("7. LLM Configuration Assessment")
    print("8. Embedder Model Assessment")
    print("0. Exit")
    print("======================================")

def main():
    """Main function to run the LR experiments."""
    while True:
        print_menu()
        choice = input("Enter your choice (0-8): ")
        
        if choice == '0':
            print("Exiting...")
            break
        elif choice == '1':
            print("\nRunning Grammatical Gender Analysis...")
            # Run the analysis with specified cores
            run_grammatical_gender_distance_analysis(verbose=True, n_cores=12) # Default to 12 cores
            
        elif choice == '2':
            print("\nRunning Color Perception Analysis...")
            run_color_experiment()
        elif choice == '3':
            print("\nRunning Temporal Reasoning Analysis...")
            run_temporal_reasoning_experiment()
        elif choice == '4':
            print("\nRunning Spatial Reasoning Analysis...")
            run_spatial_reasoning_experiment()
        elif choice == '5':
            print("\nRunning Quantitative Reasoning Analysis...")
            run_quantitative_reasoning_experiment()
        elif choice == '6':
            print("\nRunning Acoustic Reasoning Analysis...")
            run_acoustic_reasoning_experiment()
        elif choice == '7':
            print("\nRunning LLM Configuration Assessment...")
            
            # Allow specifying an output directory
            output_dir = input("Enter output directory (or press Enter for default): ")
            if not output_dir.strip():
                output_dir = "analysis_results/llm_comparison"
                
            # Run the analysis
            llm_config_assessment.analyze_llm_configurations(output_dir)
            
        elif choice == '8':
            print("\nRunning Embedder Model Assessment...")
            
            # Let the user choose to analyze all configs or a specific one
            analyze_all = input("Analyze all LLM configurations? (y/n): ").lower() == 'y'
            
            if analyze_all:
                output_dir = input("Enter output directory (or press Enter for default): ")
                if not output_dir.strip():
                    output_dir = "analysis_results/embedder_comparison"
                embedder_assessment.analyze_all_embedders(output_dir=output_dir)
            else:
                llm_config = input("Enter specific LLM configuration to analyze: ")
                output_dir = input("Enter output directory (or press Enter for default): ")
                if not output_dir.strip():
                    output_dir = f"analysis_results/embedder_comparison/{llm_config}"
                embedder_assessment.analyze_embedder_consistency_for_config(llm_config, output_dir)
            
        else:
            print("Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
