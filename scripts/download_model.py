#!/usr/bin/env python3
"""
Model Download and Verification Script
Downloads Llama 3.1 8B and verifies it works correctly in ml-training environment
"""

import os
import sys
import logging
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import mlx.core as mx

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_environment():
    """Check if we're in the correct environment"""
    logger.info("🔍 Checking environment...")
    
    # Check if we're in ml-training environment or virtual environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    venv_env = os.environ.get('VIRTUAL_ENV', '')
    
    if conda_env == 'ml-training':
        logger.info(f"✅ Environment: {conda_env}")
        return True
    elif venv_env and 'ml-training-env' in venv_env:
        logger.info(f"✅ Environment: Virtual environment ({venv_env})")
        return True
    else:
        logger.warning(f"⚠️  Not in expected environment. Conda: {conda_env}, Venv: {venv_env}")
        logger.info("Please activate the environment: conda activate ml-training OR source ml-training-env/bin/activate")
        return False

def check_dependencies():
    """Check if all required dependencies are installed"""
    logger.info("📦 Checking dependencies...")
    
    try:
        import torch
        logger.info(f"✅ PyTorch: {torch.__version__}")
        logger.info(f"✅ MPS available: {torch.backends.mps.is_available()}")
    except ImportError as e:
        logger.error(f"❌ PyTorch not found: {e}")
        return False
    
    try:
        import transformers
        logger.info(f"✅ Transformers: {transformers.__version__}")
    except ImportError as e:
        logger.error(f"❌ Transformers not found: {e}")
        return False
    
    try:
        import mlx
        try:
            mlx_version = mlx.__version__
        except AttributeError:
            mlx_version = "installed (version unknown)"
        logger.info(f"✅ MLX: {mlx_version}")
    except ImportError as e:
        logger.error(f"❌ MLX not found: {e}")
        return False
    
    try:
        import peft
        logger.info(f"✅ PEFT: {peft.__version__}")
    except ImportError as e:
        logger.error(f"❌ PEFT not found: {e}")
        return False
    
    return True

def download_model():
    """Download Llama 3.1 8B model"""
    logger.info("📥 Downloading Llama 3.1 8B model...")
    
    model_name = "meta-llama/Llama-3.1-8B"
    
    try:
        # Download tokenizer
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Download model (this will cache it locally)
        logger.info("Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use float16 instead of bfloat16 for MPS compatibility
            device_map="cpu",  # Download to CPU first, then move to MPS
            trust_remote_code=True
        )
        
        logger.info("✅ Model downloaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download model: {e}")
        return False

def test_model_loading():
    """Test that the model loads correctly"""
    logger.info("🧪 Testing model loading...")
    
    model_name = "meta-llama/Llama-3.1-8B"
    
    try:
        # Test tokenizer loading
        logger.info("Testing tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Test model loading
        logger.info("Testing model loading...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use float16 for MPS compatibility
            device_map="cpu",  # Test on CPU first
            trust_remote_code=True
        )
        
        # Test basic inference
        logger.info("Testing basic inference...")
        test_text = "Hello, how are you?"
        inputs = tokenizer(test_text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"✅ Test inference successful: {response}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model loading test failed: {e}")
        return False

def test_mlx_integration():
    """Test MLX integration"""
    logger.info("🍎 Testing MLX integration...")
    
    try:
        # Test MLX basic functionality
        logger.info("Testing MLX core...")
        x = mx.array([1, 2, 3, 4])
        y = mx.array([5, 6, 7, 8])
        z = x + y
        logger.info(f"✅ MLX test successful: {z}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ MLX test failed: {e}")
        return False

def main():
    """Main function"""
    logger.info("🚀 Starting model download and verification")
    logger.info("=" * 50)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        logger.error("❌ Missing dependencies. Please run setup script first.")
        sys.exit(1)
    
    # Download model
    if not download_model():
        logger.error("❌ Model download failed")
        sys.exit(1)
    
    # Test model loading
    if not test_model_loading():
        logger.error("❌ Model loading test failed")
        sys.exit(1)
    
    # Test MLX integration
    if not test_mlx_integration():
        logger.error("❌ MLX integration test failed")
        sys.exit(1)
    
    logger.info("")
    logger.info("🎉 All tests passed! Model is ready for fine-tuning.")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Prepare your training data")
    logger.info("2. Run: python src/train.py --config config/training_config.yaml")
    logger.info("3. Monitor training with: wandb login && wandb watch")

if __name__ == "__main__":
    main()
