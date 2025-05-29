"""
Data loading and file management functions for Language Representation Analysis.

This module contains functions for loading various data files, listing directories,
and other file-related operations.
"""

import os
import json
import pandas as pd


def list_api_providers(folder="api_generations"):
    """
    Lists all API providers in the api_generations folder.
    
    Args:
        folder (str): Path to the api_generations folder.
        
    Returns:
        list: Names of API provider directories.
    """
    if not os.path.exists(folder):
        print(f"Folder '{folder}' does not exist.")
        return []
    
    providers = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]
    return providers


def list_api_models(provider_folder):
    """
    Lists all API models for a specific provider.
    
    Args:
        provider_folder (str): Path to the provider folder.
        
    Returns:
        list: Names of API model directories.
    """
    if not os.path.exists(provider_folder):
        print(f"Folder '{provider_folder}' does not exist.")
        return []
    
    models = [f for f in os.listdir(provider_folder) if os.path.isdir(os.path.join(provider_folder, f))]
    return models


def find_csv_files(folder_path, show_directories=True):
    """
    Finds all CSV files in the specified folder and its subdirectories.
    
    Args:
        folder_path (str): Path to search for CSV files.
        show_directories (bool): If True, also includes directories for navigation.
    
    Returns:
        dict: Dictionary with 'files' and 'directories' keys.
    """
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return {'files': [], 'directories': []}
    
    # Get files and directories
    items = os.listdir(folder_path)
    
    # Separate files and directories
    csv_files = [f for f in items if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(".csv")]
    directories = [d for d in items if os.path.isdir(os.path.join(folder_path, d))]
    
    # Return all items
    return {
        'files': csv_files,
        'directories': directories if show_directories else []
    }


def load_data(file_path, delimiter=",", noun_categories=None):
    """
    Reads a CSV file and processes it for analysis.
    
    Args:
        file_path (str): Path to the CSV file.
        delimiter (str): Column delimiter in the CSV file.
        noun_categories (dict, optional): Mapping of nouns to categories.
        
    Returns:
        pd.DataFrame: Processed DataFrame with adjectives as lists and additional columns.
    """
    df = pd.read_csv(file_path, delimiter=delimiter)
    df['Adjectives'] = df['Adjectives'].apply(lambda x: x.split(";") if isinstance(x, str) else [])
    df = df[df['Adjectives'].str.len() > 0]
    
    # Add NounCategory if noun_categories dictionary is provided
    if noun_categories:
        df['NounCategory'] = df['Noun'].apply(
            lambda noun: next((category for category, nouns in noun_categories.items() 
                              if noun.lower() in [n.lower() for n in nouns]), "Unknown")
        )
    else:
        df['NounCategory'] = "Unknown"
    
    # Generate a unique label for each row
    df['Label'] = df.apply(
        lambda row: f"{row['Noun']}, {row['Language']}, {row['Proficiency']}, {row['Prompt']}, {row['NounCategory']}",
        axis=1
    )
    
    # Debug: Print proficiency levels found in the data
    proficiency_count = df['Proficiency'].value_counts()
    print("\n==== PROFICIENCY LEVELS IN THE DATASET ====")
    for prof, count in proficiency_count.items():
        print(f"  {prof}: {count} rows")
    print("============================================\n")
    
    return df


def load_adjective_prompts(filepath="inputs/adjective_prompts.txt"):
    """
    Loads adjective prompt strings from a text file.
    
    Args:
        filepath (str): Path to the text file containing prompts.
        
    Returns:
        list: List of non-empty prompt strings.
    """
    if not os.path.exists(filepath):
        print(f"Adjective prompts file '{filepath}' not found.")
        return []
    
    with open(filepath, "r", encoding="utf-8") as file:
        prompts = [line.strip() for line in file if line.strip()]
    
    return prompts


def load_noun_categories(filepath="inputs/noun_categories.json"):
    """
    Loads noun category mappings from a JSON file.
    
    Args:
        filepath (str): Path to the JSON file.
        
    Returns:
        dict: Mapping of category names to lists of nouns.
    """
    if not os.path.exists(filepath):
        print(f"Noun categories file '{filepath}' not found.")
        return {}
    
    with open(filepath, "r", encoding="utf-8") as file:
        categories = json.load(file)
    
    return categories


def load_language_families(filepath="inputs/language_families.json"):
    """
    Loads language families mapping from a JSON file.
    
    Args:
        filepath (str): Path to the JSON file with language family definitions.
        
    Returns:
        dict: Mapping of language codes (e.g., 'FR') to family names (e.g., 'Romance').
    """
    if not os.path.exists(filepath):
        print(f"Language families file '{filepath}' not found.")
        return {}
    
    with open(filepath, "r", encoding="utf-8") as file:
        family_data = json.load(file)
    
    lang_code_to_family = {}
    for family_name, languages_in_family in family_data.items():
        if isinstance(languages_in_family, list):  # Ensure it's a list
            for lang_info in languages_in_family:
                if isinstance(lang_info, dict) and "code" in lang_info:
                    lang_code_to_family[lang_info["code"]] = family_name
                else:
                    # Handle cases where a family might be listed but has no languages (e.g. "Hellenic": [])
                    if not isinstance(lang_info, dict):
                         print(f"Warning: Expected a dictionary for language entry in {filepath} under family '{family_name}', but got: {type(lang_info)}")
                    # If it's an empty list or other non-dict, it will be skipped by the isinstance check, which is fine.
        else:
            print(f"Warning: Expected a list of languages for family '{family_name}' in {filepath}, but got: {type(languages_in_family)}")
            
    return lang_code_to_family


def load_grammatical_gender(filepath="inputs/grammatical_genders.json"):
    """
    Loads grammatical gender mappings from a JSON file.
    
    Args:
        filepath (str): Path to the JSON file with grammatical gender definitions.
        
    Returns:
        dict: Mapping of language codes to their gender category string.
    """
    if not os.path.exists(filepath):
        print(f"Grammatical gender file '{filepath}' not found.")
        return {}
    
    with open(filepath, "r", encoding="utf-8") as file:
        gender_data = json.load(file)
    
    lang_code_to_gender_category = {}
    for category, languages_in_category in gender_data.items():
        for lang_info in languages_in_category:
            if isinstance(lang_info, dict) and "code" in lang_info:
                lang_code_to_gender_category[lang_info["code"]] = category
            else:
                print(f"Warning: Malformed language entry in {filepath} under category '{category}': {lang_info}")
    
    return lang_code_to_gender_category
