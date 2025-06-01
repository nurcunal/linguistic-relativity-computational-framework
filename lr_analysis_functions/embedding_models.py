import csv
import glob
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from itertools import combinations, groupby
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Set a flag for UMAP availability
UMAP_AVAILABLE = True
try:
    import umap
except ImportError:
    print("UMAP not available. Install with 'pip install umap-learn'")
    UMAP_AVAILABLE = False

# Disable tokenizers parallelism to remove warning messages.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Global cache for models to avoid reloading
MODEL_CACHE = {}


def get_embedding_model_abbreviations():
    """
    Returns a dictionary mapping embedding model names to their abbreviations.
    
    Returns:
        dict: A dictionary where keys are full model names and values are their abbreviations
    """
    return {
        "BAAI/bge-m3": "M3",
        "gte-multilingual-base": "MB",
        "intfloat/multilingual-e5-base": "E5",
        "jinaai/jina-embeddings-v3": "JV3",
        "nomic-ai/nomic-embed-text-v2-moe": "NV2",
        "sentence-transformers/all-MiniLM-L6-v2": "ML6",
        "sentence-transformers/paraphrase-MiniLM-L6-v2": "PML6",
        "sentence-transformers/paraphrase-mpnet-base-v2": "PMB",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": "PMML",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": "PMMB"
    }


def load_embedding_models(filepath="inputs/embedding_models.txt"):
    """
    Loads embedding model names from a text file.
    Adds standard embedding models if the file isn't found.
    """

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            models = [line.strip() for line in f.readlines() if line.strip() and not line.strip().startswith('#')]
        print(f"Loaded {len(models)} embedding models from {filepath}")
        return models
    except FileNotFoundError:
        print(f"Embedding models file not found at {filepath}, using default models")
        # Default models
        default_models = [
            "all-mpnet-base-v2",
            "all-MiniLM-L12-v2",
            "all-MiniLM-L6-v2",
            "multi-qa-mpnet-base-cos-v1",
            "paraphrase-multilingual-mpnet-base-v2",
            "jinaai/jina-embeddings-v2-base-en",
            "jinaai/jina-embeddings-v3",  # New Jina v3 model
            "nomic-ai/nomic-embed-text-v1.5",
            "thenlper/gte-large",
            "sentence-transformers/gtr-t5-large",
        ]
        return default_models


def optimize_nomic_for_mac(model_name):
    """
    Optimize Nomic models specifically for the current hardware.
    Prioritizes GPU usage (CUDA or MPS) if available.

    Args:
        model_name: The name of the Nomic model being used

    Returns:
        dict: Configuration parameters for optimized model loading
    """
    print(f"Optimizing Nomic model '{model_name}' for current system...")

    # Initialize config with fallback CPU settings
    config = {
        'device': 'cpu',  # Default to CPU
        'batch_size': 8,  # Conservative batch size
        'use_mps': False,  # Metal Performance Shaders flag
        'use_cuda': False,  # CUDA flag
        'prefix': 'clustering:',  # Default task prefix
        'fallback_to_cpu': True  # Enable fallback to CPU if GPU fails
    }

    # Check for Apple Silicon specifically
    import platform
    is_mac = platform.system() == 'Darwin'
    is_arm64 = platform.machine() == 'arm64'
    is_apple_silicon = is_mac and is_arm64

    if is_apple_silicon:
        print("Detected Apple Silicon Mac - optimizing for native performance")

    # Check for torch and GPU availability
    try:
        import torch

        # First check for CUDA GPU as highest priority
        if torch.cuda.is_available():
            try:
                # Test CUDA with a quick operation
                test = torch.ones(2, 2, device='cuda')
                result = test + test
                del result, test
                torch.cuda.empty_cache()

                print("CUDA GPU verified, will use it for Nomic model")
                config['device'] = 'cuda'
                config['use_cuda'] = True
                config['batch_size'] = 16  # CUDA can handle larger batches
                return config
            except Exception as cuda_err:
                print(f"CUDA available but test failed: {cuda_err}")
                print("Will check for Apple Silicon GPU (MPS)")

        # If no CUDA, check for Mac's Metal Performance Shaders
        if is_apple_silicon and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("Apple Silicon GPU acceleration (MPS) is available")

            # Test MPS with a small tensor operation to verify it actually works
            try:
                print("Testing MPS performance...")
                test_tensor = torch.ones(2, 2)
                mps_device = torch.device('mps')
                test_tensor = test_tensor.to(mps_device)
                result = test_tensor + test_tensor

                # More complex operation for thorough testing
                test_matrix = torch.rand(32, 32, device=mps_device)
                matrix_result = torch.matmul(test_matrix, test_matrix)

                del result, test_tensor, test_matrix, matrix_result  # Clean up
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()

                # If we reach here, MPS is working
                config['device'] = 'mps'
                config['use_mps'] = True
                print("MPS test successful, will use Apple Silicon GPU acceleration")

                # For MPS, adjust batch size based on model size
                # Smaller batch sizes prevent memory issues on Mac GPUs
                if 'large' in model_name or 'v2-moe' in model_name:
                    config['batch_size'] = 4  # Smaller batch for larger models
                else:
                    config['batch_size'] = 8  # Default for regular models
            except Exception as mps_error:
                print(f"MPS test failed: {mps_error}")
                print("Falling back to CPU despite MPS being available")
        else:
            if is_apple_silicon:
                print("Apple Silicon detected but MPS acceleration not available")
            print("No GPU acceleration available, using CPU optimization")
            config['batch_size'] = 16  # CPU can handle larger batches
    except ImportError:
        print("PyTorch not properly installed, using CPU with limited optimization")

    # Determine optimal task prefix for embeddings
    # Different Nomic models work better with different prefixes
    if 'embed-text-v1' in model_name:
        config['prefix'] = 'search_document:'
    elif 'embed-text-v1.5' in model_name:
        config['prefix'] = 'search_query:'
    else:  # v2 and newer models
        config['prefix'] = 'clustering:'

    print(f"Nomic optimization complete. Using device={config['device']}, "
          f"batch_size={config['batch_size']}, prefix='{config['prefix']}'")

    return config


def cleanup_torch_memory():
    """
    Clean up PyTorch/CUDA memory to prevent leaks,
    especially important when using Nomic models on Mac MPS.
    """
    try:
        import torch
        import gc

        # Force garbage collection
        gc.collect()

        # Empty CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA memory cache cleared")

        # Empty MPS cache if available (for Mac)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # No direct MPS cache clearing in PyTorch yet, but we can try this:
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
                print("MPS memory cache cleared")
            elif hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
                print("MPS memory cache cleared")
            else:
                print("MPS cache clearing not available in this PyTorch version")

        # Additional system memory optimization for Mac
        try:
            import subprocess
            import platform

            # Only on macOS, try to purge inactive memory
            if platform.system() == 'Darwin':
                try:
                    subprocess.run(['sudo', 'purge'], check=False, timeout=5)
                    print("Successfully purged inactive memory on macOS")
                except (subprocess.SubprocessError, subprocess.TimeoutExpired):
                    print("Could not purge macOS memory (may require sudo privileges)")
        except ImportError:
            pass

        print("Memory cleanup completed")
    except ImportError:
        print("PyTorch not available, skipping memory cleanup")
    except Exception as e:
        print(f"Error during memory cleanup: {e}")


def detect_optimal_device():
    """
    Detects the best available computing device.
    Prioritizes CUDA, then MPS (Mac GPU), then falls back to CPU.

    Returns:
        str: 'cuda', 'mps', or 'cpu'
    """
    device = 'cpu'  # Default fallback

    try:
        import torch

        # Check if running on Apple Silicon Mac
        import platform
        is_mac = platform.system() == 'Darwin'
        is_arm64 = platform.machine() == 'arm64'
        is_apple_silicon = is_mac and is_arm64

        if is_apple_silicon:
            print("Detected Apple Silicon Mac - checking for Metal GPU acceleration")

        # Check for CUDA (NVIDIA GPU)
        if torch.cuda.is_available():
            try:
                # Test CUDA with a small tensor operation to verify it works
                test_tensor = torch.ones(2, 2, device="cuda")
                result = test_tensor + test_tensor
                del result  # Clean up
                torch.cuda.empty_cache()

                device = 'cuda'
                print(f"CUDA GPU detected and verified - will use GPU acceleration")
                return device
            except Exception as cuda_error:
                print(f"CUDA available but test failed: {cuda_error}")
                print(f"Checking for Apple Silicon GPU acceleration...")

        # Check for MPS (Apple Silicon GPU)
        if is_apple_silicon and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                # More extensive MPS test with multiple operations
                print("Testing Apple Silicon GPU (MPS) acceleration...")
                mps_device = torch.device('mps')

                # Test 1: Simple addition
                test_tensor = torch.ones(2, 2)
                test_tensor = test_tensor.to(mps_device)
                result = test_tensor + test_tensor

                # Test 2: Matrix multiplication
                test_matrix = torch.rand(10, 10, device=mps_device)
                result = torch.matmul(test_matrix, test_matrix)

                # Cleanup
                del test_tensor, test_matrix, result
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()

                device = 'mps'
                print(f"Apple Silicon GPU acceleration (MPS) thoroughly verified - will use native GPU acceleration")
                return device
            except Exception as mps_error:
                print(f"Apple Silicon GPU (MPS) test failed: {mps_error}")
                print(f"Using CPU acceleration instead")

        # If we get here, use CPU
        if is_apple_silicon:
            print("Native Metal acceleration not available - using CPU on Apple Silicon")
        else:
            print(f"No GPU acceleration detected, using CPU")

    except ImportError:
        print(f"PyTorch not properly installed, using CPU")

    return device


def embed_adjectives(adjectives, model):
    """
    Embeds a list of adjectives using the provided model.
    Optimized to handle batching more efficiently for GPU processing.

    Args:
        adjectives: List of adjectives to embed
        model: SentenceTransformer model

    Returns:
        numpy.ndarray: Mean of adjective embeddings
    """
    if not adjectives or len(adjectives) == 0:
        return np.zeros(model.get_sentence_embedding_dimension())

    # Special handling for Jina v3 model
    is_jina_v3 = hasattr(model, 'jina_v3') and model.jina_v3 == True

    # Determine batch size based on number of adjectives
    # For GPU, we can use larger batches
    if hasattr(model, 'device') and ('cuda' in str(model.device) or 'mps' in str(model.device)):
        batch_size = min(32, len(adjectives))  # Larger batch for GPU
    else:
        batch_size = min(16, len(adjectives))  # Smaller batch for CPU

    # Convert to tensor in one go
    if len(adjectives) <= batch_size:
        if is_jina_v3:
            # For our custom loaded Jina v3 model, just use standard encoding
            # The task-specific embedding is handled by the model itself
            vectors = model.encode(adjectives, convert_to_tensor=True)
        else:
            # For standard SentenceTransformer models
            vectors = model.encode(adjectives, convert_to_tensor=True)
        # Convert back to numpy for consistent return type
        return vectors.mean(dim=0).cpu().numpy()
    else:
        # Process in batches for memory efficiency
        all_vectors = []
        for i in range(0, len(adjectives), batch_size):
            batch = adjectives[i:i + batch_size]
            if is_jina_v3:
                # For our custom loaded Jina v3 model, just use standard encoding
                batch_vectors = model.encode(batch, convert_to_tensor=True)
            else:
                # For standard SentenceTransformer models
                batch_vectors = model.encode(batch, convert_to_tensor=True)
            all_vectors.append(batch_vectors)

        # Concatenate all batches and compute mean
        all_embeddings = torch.cat(all_vectors, dim=0)
        mean_embedding = all_embeddings.mean(dim=0).cpu().numpy()
        return mean_embedding


def build_embeddings(df, model_name):
    """
    Builds embeddings for all adjectives in the dataframe using the specified model.
    Automatically installs required dependencies if missing.
    Returns None if the model fails to load after attempts to install dependencies.
    """
    # Import SentenceTransformer at the beginning of the function to ensure correct scope
    from sentence_transformers import SentenceTransformer

    # Clean model name - remove ALL quotes and commas completely
    if isinstance(model_name, str):
        # Remove all quotes and commas
        model_name = model_name.replace('"', '').replace("'", '').replace(',', '').strip()

    print(f"Using SentenceTransformer model: {model_name}")

    # Check if model is already cached
    global MODEL_CACHE
    if model_name in MODEL_CACHE:
        print(f"Using cached model: {model_name}")
        model = MODEL_CACHE[model_name]
        print(f"Model loaded from cache")
    else:
        # Detect available device for GPU acceleration with better fallback strategy
        device = 'cpu'  # Default fallback
        mps_verified = False
        cuda_verified = False

        try:
            import torch
            # First priority: CUDA (NVIDIA GPU)
            if torch.cuda.is_available():
                try:
                    # Test CUDA with a small tensor
                    test_tensor = torch.ones(2, 2, device="cuda")
                    result = test_tensor + test_tensor
                    del result
                    torch.cuda.empty_cache()
                    device = 'cuda'
                    cuda_verified = True
                    print(f"CUDA GPU detected and verified - will use GPU acceleration")
                except Exception as cuda_error:
                    print(f"CUDA available but test failed: {cuda_error}")
                    print(f"Falling back to MPS/CPU")

            # Second priority: MPS (Apple Silicon)
            if not cuda_verified and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                try:
                    # More extensive MPS test with multiple operations
                    print("Testing Mac GPU (MPS) acceleration...")
                    mps_device = torch.device('mps')

                    # Test 1: Simple addition
                    test_tensor = torch.ones(2, 2)
                    test_tensor = test_tensor.to(mps_device)
                    result = test_tensor + test_tensor

                    # Test 2: Matrix multiplication
                    test_matrix = torch.rand(10, 10, device=mps_device)
                    result = torch.matmul(test_matrix, test_matrix)

                    # Cleanup
                    del test_tensor, test_matrix, result
                    if hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()

                    device = 'mps'
                    mps_verified = True
                    print(f"Mac GPU acceleration (MPS) thoroughly verified - will use GPU acceleration")
                except Exception as mps_error:
                    print(f"Mac GPU (MPS) test failed: {mps_error}")
                    print(f"Falling back to CPU")

            if not cuda_verified and not mps_verified:
                print(f"No working GPU detected, using CPU acceleration")
        except ImportError:
            print(f"PyTorch not properly installed, using CPU")

        # Load the model with the determined device
        try:  # Main model loading try block
            # Special handling for Jina models
            if "jinaai/jina-embeddings" in model_name:
                print(f"Preparing to load Jina model: {model_name}")
                try:
                    # Check for required packages
                    missing_packages = []
                    try:
                        import transformers
                        import sentence_transformers
                        import einops
                        print("Required packages for Jina found")
                    except ImportError as e:
                        print(f"Missing package for Jina model: {e}")
                        missing_packages = ["transformers", "sentence-transformers", "torch", "einops"]

                    # Install missing packages
                    if missing_packages:
                        print(f"Installing required packages for Jina model: {', '.join(missing_packages)}")
                        import subprocess
                        import sys
                        try:
                            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade",
                                                   "transformers>=4.30.0", "sentencepiece>=0.1.99", "protobuf>=3.20.2",
                                                   "sentence-transformers", "einops"])
                            print("Successfully installed Jina dependencies")
                        except subprocess.CalledProcessError:
                            print("Failed to install Jina dependencies, this model cannot be used.")
                            return None

                    # Get optimized device
                    device = detect_optimal_device()

                    # Use a more reliable Jina model if v3 is specified (fallback)
                    if "v3" in model_name:
                        try:
                            print(f"Attempting to fix Jina v3 implementation files...")

                            # Check if running on Apple Silicon Mac
                            import platform
                            is_mac = platform.system() == 'Darwin'
                            is_arm64 = platform.machine() == 'arm64'
                            is_apple_silicon = is_mac and is_arm64

                            if is_apple_silicon:
                                print("Detected Apple Silicon Mac - using optimized installation approach")

                            # Try to install the official Jina embeddings package which has the implementation files
                            import subprocess
                            import sys

                            try:
                                print("Installing jina-embeddings package which contains required implementation files")

                                # Mac-specific installation commands
                                if is_apple_silicon:
                                    # Set environment variables for proper native compilation
                                    import os
                                    os.environ['GRPC_PYTHON_BUILD_SYSTEM_OPENSSL'] = '1'
                                    os.environ['GRPC_PYTHON_BUILD_SYSTEM_ZLIB'] = '1'

                                    # Install native dependencies first (these commands won't run if they fail)
                                    try:
                                        print("Checking for native protobuf/numpy installation...")
                                        subprocess.run(["which", "brew"], check=True, capture_output=True)
                                        # Only attempt brew commands if brew is installed
                                        subprocess.run(["brew", "list", "protobuf"], check=False, capture_output=True)
                                        subprocess.run(["brew", "list", "numpy"], check=False, capture_output=True)
                                        print("Dependency check completed")
                                    except:
                                        print("Skipping brew checks - proceeding with pip install")

                                # Install the package - will work on both Mac and non-Mac systems
                                subprocess.check_call([sys.executable, "-m", "pip", "install",
                                                       "jina-embeddings", "--upgrade", "--quiet"])
                                print("Successfully installed jina-embeddings package")
                            except Exception as install_err:
                                print(f"Warning: Could not install jina-embeddings package: {install_err}")

                            # Directly try using sentence-transformers which should handle the implementation better
                            print(f"Trying direct loading approach for Jina v3...")

                            # Use a simpler direct loading approach
                            model = SentenceTransformer(model_name, device=device)
                            print(f"Successfully loaded Jina v3 model using direct approach")

                            # Set task-specific property
                            model.default_task = 'text-matching'

                            # Cache the model
                            MODEL_CACHE[model_name] = model

                            # Create embeddings
                            print(f"Creating embeddings using Jina v3 model...")
                            embeddings_list = []

                            for adjs in df['Adjectives']:
                                embeddings_list.append(embed_adjectives(adjs, model))

                            # Add embeddings to the dataframe
                            df_copy = df.copy()
                            df_copy['Embedding'] = embeddings_list

                            return df_copy

                        except Exception as jina_error:
                            print(f"Error with direct Jina v3 loading: {jina_error}")
                            print("Issues with Jina v3 implementation files. Using reliable Jina v2 model instead...")

                            # Skip further attempts with v3 and go directly to v2
                            fallback_model = "jinaai/jina-embeddings-v2-base-en"
                            print(f"Loading fallback model: {fallback_model}")

                            try:
                                # Try to load the v2 model directly
                                model = SentenceTransformer(fallback_model, device=device)
                                print(f"Successfully loaded Jina v2 model")

                                # Cache the model with the original name requested
                                MODEL_CACHE[
                                    model_name] = model  # This enables future references to use the cached model

                                # Create embeddings with v2 model
                                print(f"Creating embeddings using Jina v2 fallback model...")
                                embeddings_list = []

                                for adjs in df['Adjectives']:
                                    embeddings_list.append(embed_adjectives(adjs, model))

                                # Add embeddings to the dataframe
                                df_copy = df.copy()
                                df_copy['Embedding'] = embeddings_list

                                return df_copy
                            except Exception as v2_error:
                                print(f"Failed to load Jina v2 model: {v2_error}")
                                print("Please select a different embedding model.")
                                return None

                    # Try to load the model
                    print(f"Loading non-Jina model on {device}: {model_name}")
                    try:
                        # If it's a GTE model
                        if "gte-" in model_name or "Alibaba-NLP/gte-" in model_name:
                            # Ensure proper model name format
                            if not model_name.startswith("Alibaba-NLP/"):
                                model_name = f"Alibaba-NLP/{model_name}"

                            model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
                            print(f"Successfully loaded GTE model on {device}")

                        # For standard models (not Jina v3, which is handled above)
                        else:
                            model = SentenceTransformer(model_name, device=device)
                            print(f"Successfully loaded model on {device}")
                    except Exception as gpu_error:
                        if device != 'cpu':
                            print(f"Failed to load Jina model on {device}: {gpu_error}")
                            print("Falling back to CPU")

                            try:
                                # For GTE models
                                if "gte-" in model_name or "Alibaba-NLP/gte-" in model_name:
                                    # Ensure proper model name format
                                    if not model_name.startswith("Alibaba-NLP/"):
                                        model_name = f"Alibaba-NLP/{model_name}"

                                    model = SentenceTransformer(model_name, trust_remote_code=True, device='cpu')
                                    print(f"Successfully loaded GTE model on CPU")
                                else:
                                    # Standard model loading
                                    model = SentenceTransformer(model_name, device='cpu')
                                    print(f"Successfully loaded model on CPU")

                            except Exception as cpu_error:
                                print(f"Failed to load model on CPU: {cpu_error}")
                                print("Please choose another embedding model from the list.")
                                return None
                        else:
                            print(f"Failed to load model: {gpu_error}")
                            print("Please choose another embedding model from the list.")
                            return None
                except Exception as model_error:
                    print(f"Error loading model '{model_name}': {model_error}")
                    print("Please choose another embedding model from the list.")
                    return None

            # Special handling for Nomic models
            elif "nomic-ai/" in model_name:
                print(f"Preparing to load Nomic model: {model_name}")

                # Check for required packages
                missing_packages = []
                try:
                    import einops
                    print("einops package found")
                except ImportError:
                    print("einops package not found, will try to install it...")
                    missing_packages.append("einops")

                try:
                    import accelerate
                    print("accelerate package found")
                except ImportError:
                    print("accelerate package not found, will try to install it...")
                    missing_packages.append("accelerate")

                # Install missing packages
                if missing_packages:
                    print(f"Installing required packages for Nomic model: {', '.join(missing_packages)}")
                    import subprocess
                    import sys
                    try:
                        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
                        print("Successfully installed Nomic dependencies")

                        # Try importing again to verify
                        try:
                            import einops
                            print("Successfully installed and imported einops")
                            if "accelerate" in missing_packages:
                                import accelerate
                                print("Successfully installed and imported accelerate")
                        except ImportError as ie:
                            print(f"Failed to import required packages after installation: {ie}")
                            print("Please select a different model.")
                            return None
                    except subprocess.CalledProcessError:
                        print("Failed to install required packages. Please select a different model.")
                        return None

                # Get device-optimized configuration (prioritizes GPU)
                nomic_config = optimize_nomic_for_mac(model_name)
                device = nomic_config['device']

                # Try to load the model
                print(f"Loading Nomic model with trust_remote_code=True on {device}: {model_name}")
                try:
                    model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
                    print(f"Successfully loaded Nomic model on {device}")

                    # Store the config for later use during embedding
                    model.nomic_config = nomic_config
                    print(f"Using optimal task prefix: '{nomic_config['prefix']}'")

                except Exception as nomic_error:
                    print(f"Error loading Nomic model '{model_name}': {nomic_error}")
                    print("Please choose another embedding model from the list.")
                    return None

            # GTE models require trust_remote_code=True and proper model name
            elif "gte-" in model_name or "Alibaba-NLP/gte-" in model_name:
                try:
                    print(f"Loading GTE model with trust_remote_code=True on {device}: {model_name}")
                    # Ensure proper model name format
                    if not model_name.startswith("Alibaba-NLP/"):
                        model_name = f"Alibaba-NLP/{model_name}"

                    # Try with GPU first
                    try:
                        model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
                        print(f"Successfully loaded GTE model on {device}")
                    except Exception as gpu_error:
                        if device != 'cpu':
                            print(f"Failed to load GTE model on {device}: {gpu_error}")
                            print("Falling back to CPU")
                            model = SentenceTransformer(model_name, trust_remote_code=True, device='cpu')
                            print(f"Successfully loaded GTE model on CPU")
                        else:
                            raise
                except Exception as gte_error:
                    print(f"Error loading GTE model '{model_name}' with trust_remote_code=True: {gte_error}")
                    print("Please choose another embedding model from the list.")
                    return None
            else:
                # Default model loading for other model types
                try:
                    print(f"Loading model on {device}: {model_name}")
                    model = SentenceTransformer(model_name, device=device)
                    print(f"Successfully loaded model on {device}")
                except Exception as general_error:
                    if device != 'cpu':
                        print(f"Failed to load model on {device}: {general_error}")
                        print("Falling back to CPU")
                        try:
                            model = SentenceTransformer(model_name, device='cpu')
                            print(f"Successfully loaded model on CPU")
                        except Exception as cpu_error:
                            print(f"Failed to load model on CPU: {cpu_error}")
                            print("Please choose another embedding model from the list.")
                            return None
                    else:
                        print(f"Failed to load model: {general_error}")
                        print("Please choose another embedding model from the list.")
                        return None
        except Exception as model_error:
            print(f"Unexpected error loading model: {model_error}")
            print("Please choose another embedding model from the list.")
            return None

    # Create embeddings for all adjectives
    embeddings_list = []

    # Process in optimal batch sizes for better GPU utilization
    if hasattr(model, 'device') and ('cuda' in str(model.device) or 'mps' in str(model.device)):
        print(f"Using GPU batch processing for embedding generation")
        batch_size_rows = 64  # Process 64 rows at a time on GPU

        total_items = len(df)
        current_processed_count = 0  # To track for progress message
        for batch_start_row_idx in range(0, total_items, batch_size_rows):
            batch_end_row_idx = min(batch_start_row_idx + batch_size_rows, total_items)
            current_batch_df = df.iloc[batch_start_row_idx:batch_end_row_idx]

            # Get all lists of adjectives for the current batch of rows
            adjective_lists_for_batch = current_batch_df['Adjectives'].tolist()

            # Store the number of adjectives for each original row to reconstruct means later
            num_adjectives_per_original_row = [len(adj_list) if adj_list else 0 for adj_list in
                                               adjective_lists_for_batch]

            # Flatten all adjectives into a single list for efficient batch encoding
            # Only include adjectives from non-empty lists to avoid issues with model.encode
            flat_list_of_all_adjectives_in_batch = []
            for adj_list_for_row in adjective_lists_for_batch:
                if adj_list_for_row:  # Ensure the list itself is not None and is not empty
                    flat_list_of_all_adjectives_in_batch.extend(adj_list_for_row)

            mean_embeddings_for_this_batch_of_rows = []

            if not flat_list_of_all_adjectives_in_batch:
                # All rows in this batch had empty adjective lists
                for num_adjs_original_row in num_adjectives_per_original_row:
                    # Append zeros for the embedding dimension
                    mean_embeddings_for_this_batch_of_rows.append(np.zeros(model.get_sentence_embedding_dimension()))
            else:
                # Encode all individual adjectives from all rows in this batch in one go
                # SentenceTransformer's encode method handles its own internal batching for large lists.
                all_individual_adj_embeddings_tensor = model.encode(flat_list_of_all_adjectives_in_batch,
                                                                    convert_to_tensor=True)

                # Iterate through the original rows (represented by num_adjectives_per_original_row)
                # to reconstruct the mean embedding for each.
                current_flat_embedding_idx = 0
                for num_adjs_original_row in num_adjectives_per_original_row:
                    if num_adjs_original_row == 0:
                        # This row originally had no adjectives
                        mean_embeddings_for_this_batch_of_rows.append(
                            np.zeros(model.get_sentence_embedding_dimension()))
                    else:
                        # Slice the embeddings corresponding to the current original row's adjectives
                        embeddings_for_this_row = all_individual_adj_embeddings_tensor[
                                                  current_flat_embedding_idx: current_flat_embedding_idx + num_adjs_original_row]
                        # Calculate the mean
                        mean_embedding_this_row = embeddings_for_this_row.mean(dim=0).cpu().numpy()
                        mean_embeddings_for_this_batch_of_rows.append(mean_embedding_this_row)
                        current_flat_embedding_idx += num_adjs_original_row

            embeddings_list.extend(mean_embeddings_for_this_batch_of_rows)
            current_processed_count += len(current_batch_df)  # Update count by number of rows processed

            # Output progress
            if batch_start_row_idx % (
                    batch_size_rows * 5) == 0 or batch_end_row_idx == total_items:  # Use batch_start_row_idx for modulo
                print(
                    f"  GPU Processing: {current_processed_count}/{total_items} items complete ({current_processed_count / total_items * 100:.1f}%)")

            # Memory cleanup for Mac GPU - consider if this is still needed as frequently with larger batches
            if ('mps' in str(model.device)) and batch_end_row_idx % (
                    batch_size_rows * 10) == 0:  # Use batch_end_row_idx
                try:
                    import torch
                    if hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()
                except Exception:
                    pass
    else:
        # For CPU, use simpler processing (original logic, calls embed_adjectives per row)
        print(f"Using CPU processing for embedding generation")
        for adjs in df['Adjectives']:
            embeddings_list.append(embed_adjectives(adjs, model))

    # Add embeddings to the dataframe
    df_copy = df.copy()
    df_copy['Embedding'] = embeddings_list

    # Cache the model for future use
    MODEL_CACHE[model_name] = model

    return df_copy


# Clear model cache before exiting to free memory
def clear_model_cache():
    """Clear the model cache to free memory"""
    global MODEL_CACHE
    print(f"Clearing model cache ({len(MODEL_CACHE)} models)...")

    # For each model in cache, try to clean up its memory
    for model_name, model in MODEL_CACHE.items():
        try:
            # If it's a Nomic model, do extra cleanup
            if "nomic-ai/" in model_name:
                print(f"Performing extra cleanup for Nomic model: {model_name}")
                cleanup_torch_memory()

            # Clear model from cache
            del model
        except Exception as e:
            print(f"Error clearing model {model_name}: {e}")

    MODEL_CACHE.clear()
    print("Model cache cleared")

    # Final garbage collection
    import gc
    gc.collect()
    print("Garbage collection performed")
