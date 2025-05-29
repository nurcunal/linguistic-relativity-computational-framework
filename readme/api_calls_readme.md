# API Calls Module README

## Overview

"api_calls.py" is a core component of a computational linguistic framework designed to investigate linguistic relativity by quantifying cross-linguistic conceptual differences. The script systematically prompt-engineers large language models (LLMs) to simulate bilingual speakers with specific native languages and English proficiency levels. For each simulated bilingual profile, the script elicits exactly 10 descriptive adjectives for a diverse set of nouns, producing a rich dataset that captures how speakers of different languages conceptualize and perceive various concepts when expressing themselves in English. These adjective-noun associations are later embedded as vector representations to assess semantic distances between languages, revealing potential cognitive biases shaped by native language structures.

The script also supports dedicated Linguistic Relativity (LR) experiments. This functionality allows for targeted investigations into how native language influences conceptualization, using tailored prompts and experimental setups.

Unlike traditional linguistic relativity experiments with human participants, which face limitations of small sample sizes and subjective methods, this computational approach offers unprecedented scale, consistency, and objectivity. By generating millions of adjective data points across multiple languages, proficiency levels, and conceptual domains, api_calls.py provides the foundational data for constructing semantic maps that visualize cross-linguistic conceptual spaces.

## Purpose

The primary purpose of this module is to generate and collect adjective associations for nouns across different:
- Native language profiles with English as a medium
- English proficiency levels (e.g., A1, A2, B1, B2, C1, C2, Perfect)
- Adjectival categories/prompts (e.g., "most frequent", "most common", "best", "top", "first")
- LLM providers and models
- System prompt strategies

Additionally, the module facilitates specific Linguistic Relativity (LR) experiments by:
- Allowing users to define custom LR prompts and scenarios.
- Generating data specifically tailored for LR analysis.

These responses are collected to study how language models represent knowledge differently when prompted to act as speakers with different language backgrounds and proficiency levels, and to directly test hypotheses related to linguistic relativity.

## Features

### Multi-Provider API Support
- OpenAI Models: GPT-4.1 and o3-mini (reasoning model)
- xAI Models: Grok-3 and Grok-3-mini (reasoning model)
- Google Models: Gemini-2.0-flash

### System Prompt Strategies
The script offers two distinct system prompt types to study different prompting strategies:

1.  **Basic Bilingual (BB)**: "You are a native {language} speaker who acquired English at a {proficiency} level. Respond to the user prompt in English."
    *   Purpose: Mimics an implicit (subconscious) elicitation of relevant adjectives, allowing the LLM to naturally reflect how a bilingual speaker might respond without explicit instruction to think in their native language.

2.  **Think-in-Native (TN)**: "You are a native {language} speaker who acquired English at a {proficiency} level. Respond to the user prompt in English, but think in your native language when responding in English."
    *   Purpose: Mimics an explicit elicitation process to ensure the LLM produces adjectives that more directly reflect native language cognition, deliberately activating language-specific conceptual structures.

### Customizable Generation Parameters
- Adjustable temperature settings (0.0, 1.0, 2.0) for adjective generation.
- Batch processing of nouns to optimize API calls.
- Configurable input lists for languages, proficiency levels, adjective prompts, and nouns.

### Support for Linguistic Relativity (LR) Experiments
The script includes a dedicated mode for running various LR experiments. These experiments are designed to test specific hypotheses about how language influences thought. Each experiment has its own set of prompts and stimuli, often loaded from JSON files in the `inputs/lr_experiments_inputs/` directory.

Supported LR experiments include:
-   **Multiple Choice Experiment**: Assesses preferences or conceptual categorizations.
-   **Grammatical Gender Questions**: Investigates the influence of grammatical gender on object/concept perception.
-   **Neutral Animals Experiment**: Probes how gender-neutral animal terms are conceptualized in languages with and without grammatical gender.
-   **Spatial Reasoning Experiment**: Tests how linguistic frameworks for space influence spatial problem-solving.
-   **Color Reasoning Experiment**: Examines how color terminology affects color perception and categorization (excluding image-to-hex tasks).
-   **Sound Waves Experiment**: A multimodal task (image-to-text) correlating sound wave images with emotions.

### Robust Error Handling and Recovery
- Checkpoint generation after each language (for adjective generation) or each language/experiment (for LR tasks) is processed.
- Ability to resume from interruptions or crashes by loading existing data from checkpoints and final output files.
- Exponential backoff for API retries.
- Detailed error logging and recovery mechanisms, including saving partial data on interruption (Ctrl+C) or errors.

### Data Organization and Management
- Structured file naming conventions for outputs.
- Hierarchical directory organization by provider, model, and run parameters.
    - Adjective generation outputs are typically saved under: `api_generations/lr_experiments_english/[provider]/[model]/[run_folder_name_base]/`
    - LR experiment outputs are saved under: `api_generations/lr_experiments_english/[experiment_name]/`
- Automatic deduplication of entries during data loading and consolidation.

## Input Files

The script expects these input files in the `inputs/` directory:

**For Adjective Generation:**
-   `nouns.txt`: List of 120+ nouns (from 12 conceptual categories) to generate adjectives for.
-   `languages.txt`: List of 24 language codes and names (e.g., "AR # Arabic", "DE # German").
-   `proficiency_levels.txt`: List of English proficiency levels (e.g., B1, B2, C1, C2, Perfect).
-   `adjective_prompts.txt`: List of prompts/categories for adjective generation (e.g., "most frequent").
-   `noun_categories.json`: JSON mapping of nouns to 12 conceptual categories (Abstract Concepts, Cultural Artifacts, etc.).

**For LR Experiments:**
-   `inputs/lr_experiments_inputs/[experiment_name].json`: Contains stimuli, questions, or data for specific experiments (e.g., `multiple_choice_experiment.json`).
-   `inputs/lr_experiments_inputs/[experiment_name]_prompt_template.txt`: Contains the prompt template for the specific experiment.
-   The `sound_waves` experiment expects PNG image files in `inputs/lr_experiments_inputs/sound_waves/`.

**API Keys:**
-   `api_key_openai.txt`: OpenAI API key.
-   `api_key_xai.txt`: xAI API key.
-   `api_key_google.txt`: Google API key.

## Output Structure

### Directory Hierarchy for Adjective Generation
```
api_generations/
└── lr_experiments_english/
    └── [provider]/
        └── [model]/
            └── [run_folder_name_base]/  # e.g., OG4-BB-1T-24L-6P-120N-3AP
                ├── checkpoints/
                │   ├── checkpoint_1_Arabic.csv
                │   └── ...
                └── [run_folder_name_base].csv
```

### Directory Hierarchy for LR Experiments
```
api_generations/
└── lr_experiments_english/
    └── [experiment_name]/      # e.g., multiple_choice_experiment
        ├── checkpoints/        # Checkpoints per language
        │   ├── checkpoint_1_Arabic.csv
        │   └── ...
        └── [output_filename].csv # e.g., OG4-BB-1T-24L-Perfect-multiple_choice_experiment_detailed.csv
```

### File Naming Convention for Adjective Generation
Output CSV files follow: `"{ModelAbbr}-{PromptTypeAbbr}-{Temp}T-{Lang}L-{Prof}P-{Noun}N-{Prompt}AP.csv"`
-   **ModelAbbr**: Abbreviation of the model (e.g., OG4, XG3, GG2, OO3M, XG3M).
-   **PromptTypeAbbr**: System prompt type (BB, TN).
-   **Temp**: Temperature value (integer part, e.g., 0T, 1T).
-   **Lang**: Number of languages.
-   **Prof**: Number of proficiency levels.
-   **Noun**: Number of nouns.
-   **Prompt**: Number of adjective prompts.

### CSV Output Format for Adjective Generation
Each CSV file contains these columns:
-   `Model`: The LLM model used.
-   `SystemPromptType`: The system prompt strategy (basic, think-in-native).
-   `Temperature`: Generation temperature value.
-   `Language`: The native language.
-   `Proficiency`: English proficiency level.
-   `Prompt`: The adjective category prompt.
-   `NounCategory`: Category of the noun (from `noun_categories.json`).
-   `Noun`: The target noun.
-   `Adjectives`: Semicolon-separated list of generated adjectives.

### Output Format for LR Experiments
The output format varies by experiment. Generally, each row represents a single item/response. Common columns include:
-   `Model`: The LLM model used.
-   `SystemPromptType`: System prompt strategy (BB or TN).
-   `Temperature`: Generation temperature.
-   `Language`: Native language.
-   `Proficiency`: English proficiency (typically "Perfect" for LR experiments).
-   `Experiment`: Name of the LR experiment.
-   `ItemCategory` (if applicable): Category of the stimulus.
-   `ItemID` (if applicable): Unique identifier for the stimulus/question.
-   `ItemText` (if applicable): The original text of the question or stimulus.
-   `ItemResponse`: The LLM's response to the item.
-   Specific columns for certain experiments (e.g., `ImageFile`, `TrueEmotionLabel`, `PredictedEmotion` for `sound_waves`).

## Running the Script

When executed, `api_calls.py` typically performs the following steps:

1.  **Initializes API Clients**: Loads API keys from the input files.
2.  **User Prompts for Configuration**:
    *   Select API provider/model.
    *   Enter temperature setting (for adjective generation).
    *   Choose system prompt type (BB or TN).
    *   Choose task type:
        1.  Adjective Generation.
        2.  Experimental Tasks (LR experiments).
3.  **Load Inputs**:
    *   For adjective generation: loads nouns, languages, proficiencies, adjective prompts, and noun categories.
    *   For LR experiments: loads experiment-specific data (e.g., questions from JSON, image files).
4.  **Prepare Output Directories**: Creates necessary folders for outputs and checkpoints.
5.  **Load Existing Data (Checkpointing)**: Checks for and loads data from previous runs (checkpoints or final output files) to avoid redundant API calls and resume progress.
6.  **Process Data**:
    *   **Adjective Generation**: Iterates through all combinations of language, proficiency, adjective prompt, and noun. For each batch of nouns, it generates a user prompt, queries the selected LLM, and parses the response.
    *   **LR Experiments**:
        *   Prompts the user to select a specific LR experiment or run all available ones.
        *   Iterates through each language and each item/stimulus within the chosen experiment.
        *   Generates a specific prompt for each item (often using a template from `inputs/lr_experiments_inputs/`), queries the LLM (potentially with image data for multimodal tasks like `sound_waves`), and records the response.
7.  **Save Checkpoints**: Periodically saves intermediate results to checkpoint files.
8.  **Consolidate and Save Final Output**: After all processing, combines all collected data (new and existing), sorts it, and saves it to a final consolidated CSV file.
9.  **Handle Interruptions/Errors**: Saves partial progress if the script is interrupted (Ctrl+C) or encounters an unexpected error.

## Use Cases

This script is ideal for:
-   Generating large-scale datasets for linguistic research on LLM representations across languages.
-   Conducting comparative studies of conceptual variation and cognitive biases influenced by L1.
-   Investigating the impact of system prompt strategies (BB vs. TN) on LLM behavior.
-   Running targeted Linguistic Relativity experiments to test specific hypotheses.
-   Exploring how different LLMs model bilingualism and L1 interference.
-   Providing empirical data for NLP studies on adjective-noun relationships and semantic similarity.

## Implementation Notes

-   The script uses libraries like `openai`, `google-generativeai`, and `pandas`.
-   It implements custom prompt generation logic for both adjective generation and LR experiments.
-   Response parsing is designed to handle the specific formats expected from the LLMs for each task.
-   Rate-limiting and exponential backoff strategies are implemented to manage API usage.
-   Detailed progress reporting is provided during execution.

## Error Handling

The script handles various error scenarios:
-   Missing API keys (raises `FileNotFoundError`).
-   API errors (e.g., rate limiting, connection issues) with retries using exponential backoff.
-   Malformed or incomplete responses from LLMs (warnings are printed, and entries might be marked as empty/incomplete).
-   Missing input files for experiments or adjective generation.
-   Unexpected interruptions (saves partial data).
-   Recovery from previous partial runs by loading checkpoints. 