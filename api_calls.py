import csv
import os
import random
import time
from datetime import datetime
from itertools import combinations

import google.generativeai as genai
import numpy as np
import pandas as pd
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Disable tokenizers parallelism (removes a warning message)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

############################################
# 1. API Functions
############################################
openai_client = None  # Global OpenAI client
xai_client = None  # Global xAI client
gemini_client = None  # Global Gemini client


def initialize_clients():
    """
    Initializes global OpenAI, xAI, and Gemini clients with API keys from txt files.
    
    Expects:
      - inputs/api_key_openai.txt for OpenAI API key.
      - inputs/api_key_xai.txt for xAI API key.
      - inputs/api_key_google.txt for Gemini API key.
    
    Raises:
        FileNotFoundError: If any API key file is missing.
    """
    global openai_client, xai_client, gemini_client
    try:
        with open("inputs/api_key_openai.txt", "r", encoding="utf-8") as f:
            openai_api_key = f.read().strip()
        with open("inputs/api_key_xai.txt", "r", encoding="utf-8") as f:
            xai_api_key = f.read().strip()
        with open("inputs/api_key_google.txt", "r", encoding="utf-8") as f:
            google_api_key = f.read().strip()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"API key file missing: {e}")

    openai_client = OpenAI(api_key=openai_api_key)
    xai_client = OpenAI(api_key=xai_api_key, base_url="https://api.x.ai/v1")
    genai.configure(api_key=google_api_key)
    gemini_client = genai.GenerativeModel("gemini-2.0-flash")
    print("OpenAI, xAI, and Gemini clients initialized.")


def generate_prompt(language, proficiency, single_prompt, words, system_prompt_type="basic"):
    """
    Generates a prompt for the LLM to produce adjectives for given nouns.
    
    Args:
        language (str): The native language of the speaker
        proficiency (str): The English proficiency level
        single_prompt (str): The adjective category prompt
        words (list): List of nouns to produce adjectives for
        system_prompt_type (str): Type of system prompt ("basic", "think-in-native", or "english-monolingual")
    
    Returns:
        str: The formatted prompt
    """
    if system_prompt_type == "english-monolingual":  # ME - Monolingual English
        content = (
            f"You are a native speaker of English. The category prompt is: '{single_prompt}'. "
            f"Provide the ten {single_prompt} adjectives for the given noun.\n\n"
            "INSTRUCTIONS:\n"
            "1. For each noun below, output exactly two lines in English:\n"
            "   - First line: `Noun: <the noun>`\n"
            "   - Second line: `Adjectives: adjective1, adjective2, ..., adjective10`\n"
            "Generate exactly 10 adjectives in a natural manner.\n\n"
            "2. You MUST produce exactly 10 unique adjectives for every noun if possible. "
            "3. You MUST produce exactly 10 unique adjectives for every noun without exception."
            "4. Do NOT repeat any adjectives.\n"
            "5. Do NOT provide any extra explanation or commentary.\n"
            "6. Output strictly in English with exactly these two lines for every noun.\n"
        )
    elif system_prompt_type == "think-in-native":  # TN - Think in Native
        content = (
            f"You are a native {language} speaker who acquired English at {proficiency} level."
            f"You may use all proficiency levels that are below and up to {proficiency} (inclusive)."
            f"The category prompt is: '{single_prompt}'. Provide the ten {single_prompt} adjectives for the given noun.\n\n"
            "INSTRUCTIONS:\n"
            "1. For each noun below, output exactly two lines in English:\n"
            "   - First line: `Noun: <the noun>`\n"
            "   - Second line: `Adjectives: adjective1, adjective2, ..., adjective10`\n"
            "Generate exactly 10 adjectives in a natural manner based on your proficiency level, thinking in your native language but writing the adjectives in English.\n\n"
            "You may only use vocabulary limited to your proficiency level. Think of your proficiency level as the ceiling for the adjectives you can generate.\n"
            "2. You MUST produce exactly 10 unique adjectives for every noun without exception."
            "If you cannot produce exactly 10, output as many adjectives as you can, but do not leave any noun without adjectives.\n"
            "3. Do NOT repeat any adjectives.\n"
            "4. Do NOT provide any extra explanation or commentary.\n"
            "5. Output strictly in English with exactly these two lines for every noun.\n"
        )
    else:  # "basic" (BB - Basic Bilingual)
        content = (
            f"You are a native {language} speaker who acquired English at {proficiency} level."
            f"You may use all proficiency levels that are below and up to {proficiency} (inclusive)."
            f"The category prompt is: '{single_prompt}'. Provide the ten {single_prompt} adjectives for the given noun.\n\n"
            "INSTRUCTIONS:\n"
            "1. For each noun below, output exactly two lines in English:\n"
            "   - First line: `Noun: <the noun>`\n"
            "   - Second line: `Adjectives: adjective1, adjective2, ..., adjective10`\n"
            "Generate exactly 10 adjectives in a natural manner based on your proficiency level.\n\n"
            "You may only use vocabulary limited to your proficiency level. Think of your proficiency level as the ceiling for the adjectives you can generate.\n"
            "2. You MUST produce exactly 10 unique adjectives for every noun without exception."
            "If you cannot produce exactly 10, output as many adjectives as you can, but do not leave any noun without adjectives.\n"
            "3. Do NOT repeat any adjectives.\n"
            "4. Do NOT provide any extra explanation or commentary.\n"
            "5. Output strictly in English with exactly these two lines for every noun.\n"
        )
        
    words_section = "\n".join([f"Noun to describe: {w}" for w in words])
    prompt = f"{content}\n{words_section}\n"
    
    if system_prompt_type == "english-monolingual":  # ME
        print(f"Generated monolingual English prompt (ME) for prompt='{single_prompt}' with {len(words)} noun(s).")
    elif system_prompt_type == "think-in-native":  # TN
        print(f"Generated think-in-native prompt (TN) for {language}, {proficiency}, prompt='{single_prompt}' with {len(words)} noun(s).")
    else:  # BB
        print(f"Generated basic bilingual prompt (BB) for {language}, {proficiency}, prompt='{single_prompt}' with {len(words)} noun(s).")
    
    return prompt


def query_api(prompt, client, model, language, proficiency, max_tokens=128000, temperature=1, retries=3, system_prompt_type="basic"):
    """
    Sends a prompt to the specified API (OpenAI, xAI, Gemini) using the given client.
    For OpenAI and xAI, a system prompt is added to set the role.
    Uses exponential backoff if errors occur.
    
    Args:
        prompt (str): The user prompt to send to the API
        client: The API client to use
        model (str): The model name
        language (str): The native language of the speaker
        proficiency (str): The English proficiency level
        max_tokens (int): Maximum tokens for response
        temperature (float): Temperature for generation
        retries (int): Number of retry attempts
        system_prompt_type (str): Type of system prompt to use ("basic", "think-in-native", or "english-monolingual")
        
    Returns:
        str: The model's response text, or None if all retries fail
    """
    # Set appropriate token limit based on the model
    # OpenAI models have a max token limit of 32768
    openai_max_tokens = 32768
    if model.startswith("gpt") or model in ["o3-mini"]:
        effective_max_tokens = min(max_tokens, openai_max_tokens)
    else:
        effective_max_tokens = max_tokens
        
    for attempt in range(retries):
        try:
            if model.startswith("gemini"):
                # For Gemini, we use google.generativeai's generate_content method.
                response = client.generate_content(
                    prompt,
                    generation_config={
                        "max_output_tokens": effective_max_tokens,
                        "temperature": temperature
                    }
                )
                return response.text
            elif model in ["o3-mini"]:
                # For OpenAI mini models, use max_completion_tokens instead of max_tokens.
                if system_prompt_type == "think-in-native":
                    system_message = {
                        "role": "system",
                        "content": f"You are a native {language} speaker who acquired English at a {proficiency} level. Produce your response thinking in your native language but generating in English."
                    }
                elif system_prompt_type == "english-monolingual":
                    system_message = {
                        "role": "system",
                        "content": "You are a native speaker of English, you don't know any other languages."
                    }
                else:  # "basic" (BB - Basic Bilingual)
                    system_message = {
                        "role": "system",
                        "content": f"You are a native {language} speaker who acquired English at a {proficiency} level."
                    }
                user_message = {"role": "user", "content": prompt}
                response = client.chat.completions.create(
                    model=model,
                    messages=[system_message, user_message],
                    max_completion_tokens=effective_max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content
            elif model == "grok-3-mini":
                # For grok-3-mini, use the same settings as grok-3 but explicitly specify the model
                if system_prompt_type == "think-in-native":
                    system_message = {
                        "role": "system",
                        "content": f"You are a native {language} speaker who acquired English at a {proficiency} level. Produce your response thinking in your native language but generating in English."
                    }
                elif system_prompt_type == "english-monolingual":
                    system_message = {
                        "role": "system",
                        "content": "You are a native speaker of English, you don't know any other languages."
                    }
                else:  # "basic" (BB - Basic Bilingual)
                    system_message = {
                        "role": "system",
                        "content": f"You are a native {language} speaker who acquired English at the level of {proficiency}."
                    }
                user_message = {"role": "user", "content": prompt}
                response = client.chat.completions.create(
                    model=model,
                    messages=[system_message, user_message],
                    max_tokens=effective_max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content
            else:
                # For other models, use max_tokens.
                if system_prompt_type == "think-in-native":
                    system_message = {
                        "role": "system",
                        "content": f"You are a native {language} speaker who acquired English at the level of {proficiency}. Produce your response thinking in your native language but generating in English."
                    }
                elif system_prompt_type == "english-monolingual":
                    system_message = {
                        "role": "system",
                        "content": "You are a native speaker of English, you don't know any other languages."
                    }
                else:  # "basic" (BB - Basic Bilingual)
                    system_message = {
                        "role": "system",
                        "content": f"You are a native {language} speaker who acquired English at the level of {proficiency}."
                    }
                user_message = {"role": "user", "content": prompt}
                response = client.chat.completions.create(
                    model=model,
                    messages=[system_message, user_message],
                    max_tokens=effective_max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content
        except Exception as e:
            print(f"Error querying {model} (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return None


def parse_response(response, words):
    """
    Parses the API response and extracts adjectives for each noun.
    
    Args:
        response (str): The API response text
        words (list): List of nouns to extract adjectives for
        
    Returns:
        dict: A dictionary mapping each noun to the produced adjectives (joined with semicolons)
        
    Note:
        Expects lines in the format:
          Noun: <noun>
          Adjectives: adj1, adj2, ..., adj10
        If fewer than 10 unique adjectives are produced for a noun, a warning is issued.
    """
    result = {}
    if not response:
        for w in words:
            result[w] = ""
        return result

    lines = response.strip().splitlines()
    current_noun = None
    for line in lines:
        line = line.strip()
        if line.lower().startswith("noun:"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                current_noun = parts[1].strip().lower()
        elif line.lower().startswith("adjectives:") and current_noun:
            parts = line.split(":", 1)
            if len(parts) == 2:
                adjectives_part = parts[1].strip()
                adjectives_list = [adj.strip() for adj in adjectives_part.split(",") if adj.strip()]
                unique_adjs = []
                for adj in adjectives_list:
                    if adj not in unique_adjs:
                        unique_adjs.append(adj)
                if len(unique_adjs) < 10:
                    print(f"Warning: For noun '{current_noun}', only {len(unique_adjs)} adjectives were generated.")
                final_adjs = ";".join(unique_adjs)
                for original_word in words:
                    if original_word.lower() == current_noun:
                        result[original_word] = final_adjs
                current_noun = None
    for w in words:
        if w not in result or not result[w]:
            print(f"Warning: No adjectives found for '{w}'")
            result[w] = ""
    return result


############################################
# 2. Utility Functions for File I/O
############################################
def load_variables_from_file(file_path):
    """
    Loads a list of variables from a text file.
    
    Args:
        file_path (str): Path to the text file
        
    Returns:
        list: List of non-empty lines from the file
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        UnicodeDecodeError: If there's an issue decoding the file
    """
    with open(file_path, "r", encoding="utf-8") as file:
        variables = [line.strip() for line in file if line.strip()]
    print(f"Loaded {len(variables)} items from {file_path}")
    return variables


def get_api_provider_name(model):
    """
    Returns the provider name based on the model name.
    
    Args:
        model (str): The model name
        
    Returns:
        str: The provider name ("openai", "xai", "google", or "unknown")
    """
    if model.startswith("gpt") or model in ["o3-mini"]:
        return "openai"
    elif model.startswith("grok"):
        return "xai"
    elif model.startswith("gemini"):
        return "google"
    else:
        return "unknown"


def get_noun_category(noun, noun_categories=None):
    """
    Returns the category for a noun based on the noun_categories dictionary.

    Args:
        noun (str): The noun to categorize
        noun_categories (dict, optional): Dictionary mapping categories to lists of nouns

    Returns:
        str: The category of the noun, or "Unknown" if not found
    """
    if not noun_categories:
        return "Unknown"

    for category, nouns in noun_categories.items():
        if noun.lower() in [n.lower() for n in nouns]:
            return category

    return "Unknown"


def load_noun_categories(filepath="inputs/noun_categories.json"):
    """
    Loads noun category mappings from a JSON file.
    
    Args:
        filepath (str): Path to the JSON file containing noun categories
        
    Returns:
        dict: A dictionary mapping category names to lists of nouns
        
    Note:
        Expected format is a dictionary mapping category names to lists of nouns.
        Returns an empty dictionary if the file is not found.
    """
    if not os.path.exists(filepath):
        print(f"Noun categories file '{filepath}' not found.")
        return {}

    import json
    with open(filepath, "r", encoding="utf-8") as file:
        categories = json.load(file)

    return categories


############################################
# 3. Main API Generation Pipeline
############################################
def main():
    """
    Main function to execute the API generation pipeline.
    Handles user input for API selection, temperature, and system prompt type.
    Processes combinations of languages, proficiency levels, prompts, and nouns.
    Generates adjectives and saves results to CSV files.
    """
    # Initialize all API clients first.
    try:
        initialize_clients()
    except Exception as e:
        print(f"Error initializing API clients: {e}")
        print("Please check that you have valid API keys in the input files.")
        return

    # Prompt user to choose an API.
    print("Choose an API to use:")
    print("1. OpenAI (gpt-4.1)")
    print("2. xAI (grok-3)")
    print("3. Google (gemini-2.0-flash)")
    print("4. OpenAI (o3-mini) - reasoning model")
    print("5. xAI (grok-3-mini) - reasoning model")

    choice = input("Enter 1, 2, 3, 4, or 5: ").strip()

    model_abbr = ""  # Initialize model_abbr
    if choice == "1":
        api_client = openai_client
        model = "gpt-4.1"
        api_provider = "openai"
        model_abbr = "OG4"
        temp_default = 1.0
        temp_range = "0-2"
    elif choice == "2":
        api_client = xai_client
        model = "grok-3"
        api_provider = "xai"
        model_abbr = "XG3"
        temp_default = 1.0
        temp_range = "0-2"
    elif choice == "3":
        api_client = gemini_client
        model = "gemini-2.0-flash"
        api_provider = "google"
        model_abbr = "GG2"
        temp_default = 1.0
        temp_range = "0-2"
    elif choice == "4":
        api_client = openai_client
        model = "o3-mini"
        api_provider = "openai"
        model_abbr = "OO3M"
        temp_default = 1.0
        temp_range = "0-2"
        print("Note: o3-mini is a reasoning-optimized model")
    elif choice == "5":
        api_client = xai_client
        model = "grok-3-mini"
        api_provider = "xai"
        model_abbr = "XG3M"
        temp_default = 1.0
        temp_range = "0-2"
        print("Note: grok-3-mini is a reasoning-optimized model")
    else:
        print("Invalid choice. Please run again and select 1, 2, 3, 4, or 5.")
        return

    # Prompt user for temperature with API-specific default & range.
    temp_prompt = f"Enter desired temperature for {model} (range: {temp_range}). Press Enter to use the default ({temp_default}): "
    temp_input = input(temp_prompt).strip()
    if temp_input == "":
        temperature = temp_default
    else:
        try:
            temperature = float(temp_input)
        except ValueError:
            print(f"Invalid input. Using default temperature {temp_default}.")
            temperature = temp_default
    print(f"Using temperature: {temperature}")

    # Prompt user to choose system prompt type
    print("\nChoose a system prompt type:")
    print("1. Basic Bilingual (BB): 'You are a native {language} speaker who acquired English at a {proficiency} level.'")
    print("2. Think-in-Native (TN): 'You are a native {language} speaker who acquired English at a {proficiency} level. Produce your response thinking in your native language but generating in English.'")
    print("3. Monolingual English (ME): 'You are a native speaker of English, you don't know any other languages.'")
    prompt_choice = input("Enter 1, 2, or 3: ").strip()
    
    if prompt_choice == "1":
        system_prompt_type = "basic"
        prompt_type_abbr = "BB"
        print("Using Basic Bilingual (BB) system prompt.")
    elif prompt_choice == "2":
        system_prompt_type = "think-in-native"
        prompt_type_abbr = "TN"
        print("Using Think-in-Native (TN) system prompt.")
    elif prompt_choice == "3":
        system_prompt_type = "english-monolingual"
        prompt_type_abbr = "ME"
        print("Using Monolingual English (ME) system prompt.")
    else:
        print("Invalid choice. Using Basic Bilingual (BB) system prompt by default.")
        system_prompt_type = "basic"
        prompt_type_abbr = "BB"

    # Define file paths for inputs.
    nouns_file = "inputs/nouns.txt"
    languages_file = "inputs/languages.txt"
    proficiency_file = "inputs/proficiency_levels.txt"
    adjective_prompt_file = "inputs/adjective_prompts.txt"

    # Verify that input files exist
    missing_files_list = []
    for file_path_check in [nouns_file, languages_file, proficiency_file, adjective_prompt_file]:
        if not os.path.exists(file_path_check):
            missing_files_list.append(file_path_check)

    if missing_files_list:
        print(f"The following required input files are missing: {', '.join(missing_files_list)}")
        print("Please produce these files before running the script.")
        return

    # Load input files.
    try:
        words_list = load_variables_from_file(nouns_file)
        languages_list = sorted(load_variables_from_file(languages_file))  # Sort languages alphabetically
        proficiency_list = load_variables_from_file(proficiency_file)
        prompts_list = load_variables_from_file(adjective_prompt_file)
    except Exception as e:
        print(f"Error loading input files: {e}")
        return

    if not words_list or not languages_list or not proficiency_list or not prompts_list:
        print("One or more input files are empty.")
        return
        
    # Produce the directory structure for outputs
    temp_str = f"{int(temperature)}T"
    lang_str = f"{len(languages_list)}L"
    prof_str = f"{len(proficiency_list)}P"
    noun_str = f"{len(words_list)}N"
    prompt_str = f"{len(prompts_list)}AP"
    
    run_folder_name_base = f"{model_abbr}-{prompt_type_abbr}-{temp_str}-{lang_str}-{prof_str}-{noun_str}-{prompt_str}"
    
    # Base path for this specific run's outputs
    run_specific_output_dir = os.path.join("api_generations", api_provider, model, run_folder_name_base)
    checkpoints_dir = os.path.join(run_specific_output_dir, "checkpoints")
    final_output_csv_path = os.path.join(run_specific_output_dir, f"{run_folder_name_base}.csv")

    os.makedirs(checkpoints_dir, exist_ok=True)  # This will produce run_specific_output_dir if it doesn't exist

    print(f"Target output file: {final_output_csv_path}")
    print(f"Checkpoints will be saved in: {checkpoints_dir}")

    # Load noun categories first
    try:
        noun_categories = load_noun_categories()
        print(f"Loaded {len(noun_categories)} noun categories")
    except Exception as e:
        print(f"Warning: Could not load noun categories: {e}")
        noun_categories = {}

    # Load existing data from checkpoints and previous final file
    existing_data_map = {}
    fieldnames = ["Model", "SystemPromptType", "Temperature", "Language", "Proficiency", "Prompt", "NounCategory", "Noun", "Adjectives"]

    def _load_rows_from_csv_to_map(filepath, data_map_to_populate):
        """
        Helper function to load rows from a CSV file into a data map.
        
        Args:
            filepath (str): Path to the CSV file
            data_map_to_populate (dict): Dictionary to populate with loaded data
            
        Returns:
            None: Updates data_map_to_populate in place
        """
        try:
            with open(filepath, "r", newline="", encoding="utf-8") as csvfile_load:
                reader = csv.DictReader(csvfile_load)
                rows_loaded_from_file = 0
                for row_load in reader:
                    key_load = (
                        row_load.get("Language", ""), 
                        row_load.get("Proficiency", ""), 
                        row_load.get("Prompt", ""), 
                        row_load.get("Noun", "")
                    )
                    # Ensure the key is fully formed before adding to map
                    if all(k for k in key_load):
                        # Produce a new dict for the row to ensure all expected fields are there
                        # and to handle potential float conversion for Temperature
                        temp_val_str = row_load.get("Temperature", str(temp_default))
                        try:
                            temp_val = float(temp_val_str)
                        except ValueError:
                            temp_val = temp_default

                        data_map_to_populate[key_load] = {
                            "Model": row_load.get("Model", model),
                            "SystemPromptType": row_load.get("SystemPromptType", system_prompt_type),
                            "Temperature": temp_val,
                            "Language": key_load[0],
                            "Proficiency": key_load[1],
                            "Prompt": key_load[2],
                            "NounCategory": row_load.get("NounCategory", get_noun_category(key_load[3], noun_categories)),
                            "Noun": key_load[3],
                            "Adjectives": row_load.get("Adjectives", "")
                        }
                        rows_loaded_from_file += 1
                if rows_loaded_from_file > 0:
                    print(f"Loaded {rows_loaded_from_file} rows from {filepath}. Total unique rows in map: {len(data_map_to_populate)}")
        except FileNotFoundError:
            # This is normal if the file doesn't exist yet
            pass
        except Exception as e_load:
            print(f"Error reading or processing {filepath}: {e_load}. File might be skipped or partially loaded.")

    if os.path.exists(checkpoints_dir):
        print(f"Loading existing data from checkpoints in {checkpoints_dir}...")
        for item in sorted(os.listdir(checkpoints_dir)):  # Sorted to maintain some order if overwrites happen
            if item.startswith("checkpoint_") and item.endswith(".csv"):
                cp_path = os.path.join(checkpoints_dir, item)
                _load_rows_from_csv_to_map(cp_path, existing_data_map)
    
    if os.path.exists(final_output_csv_path):
        print(f"Loading existing data from final CSV: {final_output_csv_path}...")
        _load_rows_from_csv_to_map(final_output_csv_path, existing_data_map)
    
    if existing_data_map:
        print(f"Total {len(existing_data_map)} unique existing rows loaded into memory.")

    # Calculate total expected combinations.
    total_combinations = len(words_list) * len(languages_list) * len(proficiency_list) * len(prompts_list)

    # Check if all combinations are already present and complete
    is_fully_complete_based_on_inputs = True
    if len(existing_data_map) < total_combinations:
         is_fully_complete_based_on_inputs = False
    else:
        for lang_check in languages_list:
            for prof_check in proficiency_list:
                for sp_check in prompts_list:
                    for word_check in words_list:
                        key_check = (lang_check, prof_check, sp_check, word_check)
                        if key_check not in existing_data_map:
                            is_fully_complete_based_on_inputs = False
                            break
                        # Check for completeness (e.g., at least 10 adjectives)
                        adjectives_check = existing_data_map[key_check].get("Adjectives", "")
                        if not adjectives_check or len(adjectives_check.split(';')) < 10:
                            is_fully_complete_based_on_inputs = False
                            break
                    if not is_fully_complete_based_on_inputs:
                        break
                if not is_fully_complete_based_on_inputs:
                    break
            if not is_fully_complete_based_on_inputs:
                break
    
    if is_fully_complete_based_on_inputs:
        print(f"All {total_combinations} combinations based on current input files are already present and complete.")
        print(f"Final data is available at: {final_output_csv_path}")
        return
    else:
        print("File is incomplete or missing entries with respect to current inputs. Proceeding to generate/complete missing data.")

    print(
        f"\nProcessing with {model} (Temp: {temperature}, SystemPrompt: {system_prompt_type}):\n"
        f" - {len(words_list)} Noun(s)\n - {len(languages_list)} Language(s)\n"
        f" - {len(proficiency_list)} Proficiency Level(s)\n - {len(prompts_list)} Adjective Prompt(s)\n"
        f"Total expected combinations for this run: {total_combinations}\n")

    all_rows_collected_this_run = [] 
    processed_items_count = 0
    checkpoint_file_index = 0

    try:
        for language_item in languages_list:
            current_language_rows_for_checkpoint = []
            checkpoint_file_index += 1  # Increment for each language

            for proficiency_item in proficiency_list:
                for single_prompt_item in prompts_list:
                    
                    nouns_for_api_call_batch = []
                    temp_batch_data_storage = {}  # Noun -> row_dict for current (L,P,SP)

                    for noun_item in words_list:
                        current_key_tuple = (language_item, proficiency_item, single_prompt_item, noun_item)
                        
                        existing_row = existing_data_map.get(current_key_tuple)
                        is_complete_in_existing = False
                        if existing_row:
                            adjectives_str = existing_row.get("Adjectives", "")
                            if adjectives_str and len(adjectives_str.split(';')) >= 10:
                                is_complete_in_existing = True
                        
                        if is_complete_in_existing:
                            # If it's complete in existing_data_map, use it directly
                            temp_batch_data_storage[noun_item] = existing_row
                        else:
                            # Needs processing (either new or incomplete)
                            nouns_for_api_call_batch.append(noun_item)
                            if existing_row:  # It's incomplete, store a copy to update
                                temp_batch_data_storage[noun_item] = existing_row.copy()
                            # If not existing_row, it's brand new, will be produced after API call

                    if nouns_for_api_call_batch:
                        print(f"API Call for {language_item}, {proficiency_item}, Prompt: '{single_prompt_item}', Nouns: {len(nouns_for_api_call_batch)}")
                        prompt_text = generate_prompt(language_item, proficiency_item, single_prompt_item, nouns_for_api_call_batch, system_prompt_type)
                        # Make sure to pass the current run's temperature
                        response_text = query_api(prompt_text, api_client, model, language_item, proficiency_item, temperature=temperature, system_prompt_type=system_prompt_type)
                        
                        if response_text:
                            parsed_adj_map = parse_response(response_text, nouns_for_api_call_batch)
                            for noun_w in nouns_for_api_call_batch:
                                new_adjectives = parsed_adj_map.get(noun_w, "")
                                noun_cat = get_noun_category(noun_w, noun_categories)
                                
                                if noun_w in temp_batch_data_storage:  # Existing but was incomplete, update it
                                    row_to_update = temp_batch_data_storage[noun_w]
                                    row_to_update["Adjectives"] = new_adjectives
                                    # Ensure other metadata fields are current for this run
                                    row_to_update["Model"] = model
                                    row_to_update["SystemPromptType"] = system_prompt_type
                                    row_to_update["Temperature"] = temperature  # Current run's temperature
                                    row_to_update["Language"] = language_item
                                    row_to_update["Proficiency"] = proficiency_item
                                    row_to_update["Prompt"] = single_prompt_item
                                    row_to_update["NounCategory"] = noun_cat  # NounCategory might change if inputs updated
                                else:  # Completely new noun entry for this (L,P,SP)
                                    temp_batch_data_storage[noun_w] = {
                                        "Model": model, 
                                        "SystemPromptType": system_prompt_type,
                                        "Temperature": temperature, 
                                        "Language": language_item, 
                                        "Proficiency": proficiency_item, 
                                        "Prompt": single_prompt_item, 
                                        "NounCategory": noun_cat, 
                                        "Noun": noun_w, 
                                        "Adjectives": new_adjectives
                                    }
                        else:  # API call failed for this batch
                            print(f"Warning: Failed to get API response for batch ({language_item}, {proficiency_item}, '{single_prompt_item}'). Nouns attempted: {nouns_for_api_call_batch}")
                            for noun_w_failed in nouns_for_api_call_batch:
                                if noun_w_failed not in temp_batch_data_storage:  # If it was a new noun, add placeholder
                                    noun_cat = get_noun_category(noun_w_failed, noun_categories)
                                    temp_batch_data_storage[noun_w_failed] = {
                                        "Model": model,
                                        "SystemPromptType": system_prompt_type,
                                        "Temperature": temperature, 
                                        "Language": language_item,
                                        "Proficiency": proficiency_item, 
                                        "Prompt": single_prompt_item,
                                        "NounCategory": noun_cat,
                                        "Noun": noun_w_failed, 
                                        "Adjectives": ""  # Mark as empty
                                    }
                                # If it was existing but incomplete, it remains as it was (with its old incomplete adjs or gets updated to empty if API fails for it now)
                                elif temp_batch_data_storage[noun_w_failed].get("Adjectives", ""):  # if it was incomplete but had some adjectives
                                     temp_batch_data_storage[noun_w_failed]["Adjectives"] = ""  # Mark as empty due to current failure

                    # Consolidate & update progress for all nouns in this (L,P,SP)
                    for noun_final_pass in words_list:
                        row_for_this_noun_combo = None
                        if noun_final_pass in temp_batch_data_storage:
                            row_for_this_noun_combo = temp_batch_data_storage[noun_final_pass]
                        else:
                            # This case implies a noun was in words_list but somehow not handled:
                            # - Not complete in existing_data_map
                            # - Not added to nouns_for_api_call_batch (should not happen if logic above is correct)
                            # - Or API batch processing missed it.
                            # Add a placeholder to ensure CSV structure integrity.
                            print(f"Critical Warning: Noun '{noun_final_pass}' was missed in API/loading for ({language_item},{proficiency_item},'{single_prompt_item}'). Adding empty row.")
                            noun_cat = get_noun_category(noun_final_pass, noun_categories)
                            row_for_this_noun_combo = {
                                "Model": model,
                                "SystemPromptType": system_prompt_type, 
                                "Temperature": temperature, 
                                "Language": language_item, 
                                "Proficiency": proficiency_item,
                                "Prompt": single_prompt_item,
                                "NounCategory": noun_cat, 
                                "Noun": noun_final_pass, 
                                "Adjectives": ""
                            }
                        
                        current_language_rows_for_checkpoint.append(row_for_this_noun_combo)
                        # Add to the overall list that will be saved at the very end.
                        # This list will naturally contain all items, either from existing_complete, updated_incomplete, or newly_generated.
                        all_rows_collected_this_run.append(row_for_this_noun_combo)
                        
                        processed_items_count += 1
                        if processed_items_count % 10 == 0 or processed_items_count == total_combinations: 
                             print(f"Progress: {processed_items_count}/{total_combinations} combinations processed.")
            
            # After processing all proficiencies and prompts for the current language_item
            if current_language_rows_for_checkpoint:
                # Sanitize language_item for filename
                safe_language_name = language_item.replace(' ', '_').replace('/', '_')
                checkpoint_filename = f"checkpoint_{checkpoint_file_index}_{safe_language_name}.csv"
                checkpoint_filepath = os.path.join(checkpoints_dir, checkpoint_filename)
                try:
                    with open(checkpoint_filepath, "w", newline="", encoding="utf-8") as cp_csvfile:
                        writer = csv.DictWriter(cp_csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(current_language_rows_for_checkpoint)
                    print(f"Saved checkpoint for language '{language_item}' to {checkpoint_filepath} ({len(current_language_rows_for_checkpoint)} rows)")
                except Exception as e_cp_save:
                    print(f"Error saving checkpoint file {checkpoint_filepath}: {e_cp_save}")
        
        # Sort data by language, noun category, noun before final save
        # all_rows_collected_this_run now contains all data (loaded, updated, or new)
        
        # Ensure data integrity by rebuilding from a map to eliminate duplicate entries
        final_data_map_for_output = {}
        for row_final in all_rows_collected_this_run:
            key_final = (
                row_final.get("Language", ""), 
                row_final.get("Proficiency", ""), 
                row_final.get("Prompt", ""), 
                row_final.get("Noun", "")
            )
            if all(k for k in key_final):  # ensure key is valid
                # This will keep the last encountered version if there were duplicates, which should be the most current.
                final_data_map_for_output[key_final] = row_final

        sorted_data_final = sorted(final_data_map_for_output.values(), key=lambda x: (x.get("Language", ""), x.get("NounCategory", ""), x.get("Noun", "")))

        # Write final output CSV 
        with open(final_output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sorted_data_final)

        print(f"\nData successfully written to {final_output_csv_path} ({len(sorted_data_final)} rows, sorted by language, noun category, and noun)")

    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        if all_rows_collected_this_run:
            interrupted_file_name = f"INTERRUPTED-{run_folder_name_base}.csv"
            interrupted_file_path = os.path.join(run_specific_output_dir, interrupted_file_name)
            try:
                # Similar deduplication and sorting for interrupted save
                final_data_map_interrupt = {}
                for row_int in all_rows_collected_this_run:
                    key_int = (row_int.get("Language", ""), row_int.get("Proficiency", ""), row_int.get("Prompt", ""), row_int.get("Noun", ""))
                    if all(k for k in key_int):
                        final_data_map_interrupt[key_int] = row_int
                sorted_data_interrupt = sorted(final_data_map_interrupt.values(), key=lambda x: (x.get("Language", ""), x.get("NounCategory", ""), x.get("Noun", "")))
                
                with open(interrupted_file_path, "w", newline="", encoding="utf-8") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(sorted_data_interrupt)
                print(f"Partial data saved to {interrupted_file_path}")
            except Exception as e_interrupt:
                print(f"Could not save partial data during interrupt: {e_interrupt}")

    except Exception as e_main:
        print(f"\nUnexpected error occurred: {e_main}")
        import traceback
        traceback.print_exc()
        if all_rows_collected_this_run:
            error_file_name = f"ERROR-{run_folder_name_base}.csv"
            error_file_path = os.path.join(run_specific_output_dir, error_file_name)
            try:
                # Similar deduplication and sorting for error save
                final_data_map_error = {}
                for row_err in all_rows_collected_this_run:
                    key_err = (row_err.get("Language", ""), row_err.get("Proficiency", ""), row_err.get("Prompt", ""), row_err.get("Noun", ""))
                    if all(k for k in key_err):
                        final_data_map_error[key_err] = row_err
                sorted_data_error = sorted(final_data_map_error.values(), key=lambda x: (x.get("Language", ""), x.get("NounCategory", ""), x.get("Noun", "")))

                with open(error_file_path, "w", newline="", encoding="utf-8") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(sorted_data_error)
                print(f"Partial data saved to {error_file_path}")
            except Exception as e_error_save:
                print(f"Could not save partial data during error: {e_error_save}")


# 1. Add a helper function to get noun category
def get_noun_category(noun, noun_categories=None):
    """
    Returns the category for a noun based on the noun_categories dictionary.

    Args:
        noun (str): The noun to categorize
        noun_categories (dict, optional): Dictionary mapping categories to lists of nouns

    Returns:
        str: The category of the noun, or "Unknown" if not found
    """
    if not noun_categories:
        return "Unknown"

    for category, nouns in noun_categories.items():
        if noun.lower() in [n.lower() for n in nouns]:
            return category

    return "Unknown"


# 2. Add a function to load noun categories
def load_noun_categories(filepath="inputs/noun_categories.json"):
    """
    Loads noun category mappings from a JSON file.
    
    Args:
        filepath (str): Path to the JSON file containing noun categories
        
    Returns:
        dict: A dictionary mapping category names to lists of nouns
        
    Note:
        Expected format is a dictionary mapping category names to lists of nouns.
        Returns an empty dictionary if the file is not found.
    """
    if not os.path.exists(filepath):
        print(f"Noun categories file '{filepath}' not found.")
        return {}

    import json
    with open(filepath, "r", encoding="utf-8") as file:
        categories = json.load(file)

    return categories


# 3. Update generate_csv_file function to include the prompt type in the CSV output
def generate_csv_file(language, proficiency, prompt, nouns, api_client, model, temperature,
                      provider, noun_categories=None, system_prompt_type="basic"):
    """
    Generates and saves a CSV file with adjectives for each noun in the specified language.
    Orders data alphabetically by language, noun category, and noun.

    Args:
        language (str): Target language.
        proficiency (str): CEFR proficiency level.
        prompt (str): Adjective prompt.
        nouns (list): List of nouns to generate adjectives for.
        api_client: API client to use.
        model (str): Model name.
        temperature (float): Temperature value for generation.
        provider (str): API provider name.
        noun_categories (dict, optional): Dictionary mapping categories to lists of nouns.
        system_prompt_type (str): Type of system prompt to use ("basic", "think-in-native", or "english-monolingual").
        
    Returns:
        str: Path to the generated CSV file.
    """
    # Define model abbreviations
    model_abbrev = {
        "gpt-4.1": "OG4",
        "grok-3": "XG3",
        "gemini-2.0-flash": "GG2",
        "o3-mini": "OO3M",
        "grok-3-mini": "XG3M"
    }
    
    # Define system prompt type abbreviations
    prompt_type_abbrev = {
        "basic": "BB",            # Basic Bilingual
        "think-in-native": "TN",  # Think in Native
        "english-monolingual": "ME" # Monolingual English
    }
    
    # Use the abbreviation, or fallback to the model name if not in dictionary
    model_abbr = model_abbrev.get(model, model[:3].upper())
    prompt_abbr = prompt_type_abbrev.get(system_prompt_type, "BB")  # Default to BB if not found
    
    provider_dir = os.path.join("api_generations", provider)
    os.makedirs(provider_dir, exist_ok=True)

    model_dir = os.path.join(provider_dir, model)
    os.makedirs(model_dir, exist_ok=True)

    # Produce the new condensed output file name with prompt type abbreviation
    temp_str = f"{int(temperature)}T"
    lang_str = "1L"  # Only one language in this function
    prof_str = "1P"  # Only one proficiency in this function
    noun_str = f"{len(nouns)}N"
    prompt_str = "1AP"  # Only one adjective prompt in this function
    
    output_filename = f"{model_abbr}-{prompt_abbr}-{temp_str}-{lang_str}-{prof_str}-{noun_str}-{prompt_str}-{language}-{proficiency}.csv"
    output_file = os.path.join(model_dir, output_filename)

    batch_size = min(10, len(nouns))  # Process nouns in smaller batches
    api_responses = []

    print(f"Generating adjectives for {len(nouns)} nouns in language: {language} (proficiency: {proficiency})...")
    print(f"Using model: {model}, temperature: {temperature}, prompt: '{prompt}', system prompt type: {system_prompt_type}")
    print(f"Saving results to: {output_file}")

    # Produce batches of nouns
    batches = [nouns[i:i + batch_size] for i in range(0, len(nouns), batch_size)]

    # Process each batch
    total_batches = len(batches)
    for i, batch in enumerate(batches):
        print(f"Processing batch {i + 1}/{total_batches} ({len(batch)} nouns)")

        api_prompt = generate_prompt(language, proficiency, prompt, batch, system_prompt_type)
        response = query_api(api_prompt, api_client, model, language, proficiency, temperature=temperature, system_prompt_type=system_prompt_type)

        if response:
            noun_adjectives = parse_response(response, batch)
            for noun, adjectives in noun_adjectives.items():
                noun_category = get_noun_category(noun, noun_categories)
                api_responses.append({
                    "Model": model,
                    "SystemPromptType": system_prompt_type,
                    "Temperature": temperature,
                    "Language": language,
                    "Proficiency": proficiency,
                    "Prompt": prompt,
                    "NounCategory": noun_category,
                    "Noun": noun,
                    "Adjectives": adjectives
                })
        else:
            print(f"Warning: Empty response for batch {i + 1}")
            # Add empty responses for all nouns in the batch
            for noun in batch:
                noun_category = get_noun_category(noun, noun_categories)
                api_responses.append({
                    "Model": model,
                    "SystemPromptType": system_prompt_type,
                    "Temperature": temperature,
                    "Language": language,
                    "Proficiency": proficiency,
                    "Prompt": prompt,
                    "NounCategory": noun_category,
                    "Noun": noun,
                    "Adjectives": ""
                })

        # Short delay between batches
        if i < total_batches - 1:
            time.sleep(1)

    # Sort responses by language, noun category, noun
    sorted_responses = sorted(api_responses, key=lambda x: (x["Language"], x["NounCategory"], x["Noun"]))

    # Write responses to CSV
    if sorted_responses:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ["Model", "SystemPromptType", "Temperature", "Language", "Proficiency", "Prompt", "NounCategory", "Noun", "Adjectives"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sorted_responses)

        print(
            f"CSV file saved: {output_file} with {len(sorted_responses)} rows (sorted by language, noun category, and noun)")
    else:
        print("No responses to save.")

    return output_file  # Return the file path for reference


# 4. Update process_nouns function to use noun categories
def process_nouns(nouns, language, proficiency, prompt, api_client, model, temperature, provider, system_prompt_type="basic"):
    """
    Processes a list of nouns for a specific language, proficiency, and prompt.
    
    Args:
        nouns (list): List of nouns to process.
        language (str): The native language.
        proficiency (str): English proficiency level.
        prompt (str): Adjective prompt.
        api_client: API client instance.
        model (str): Model name.
        temperature (float): Temperature setting.
        provider (str): API provider name.
        system_prompt_type (str): Type of system prompt to use.
        
    Returns:
        str: Path to the generated CSV file.
    """
    # Load noun categories
    try:
        noun_categories = load_noun_categories()
        print(f"Loaded {len(noun_categories)} noun categories")
    except Exception as e:
        print(f"Warning: Could not load noun categories: {e}")
        noun_categories = {}

    # Generate CSV file with adjectives
    output_file = generate_csv_file(language, proficiency, prompt, nouns, api_client, model,
                                    temperature, provider, noun_categories, system_prompt_type)

    print(f"Process complete. Data saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    main()
