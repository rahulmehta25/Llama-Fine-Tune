#!/usr/bin/env python3
"""
Simple Model Download Script
Downloads Llama 3.1 8B with authentication
"""

import os
import sys
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model():
    """Download Llama 3.1 8B model"""
    logger.info("🚀 Downloading Llama 3.1 8B model...")
    
    model_name = "meta-llama/Llama-3.1-8B"
    
    try:
        # Check if token is set
        token = os.environ.get('HUGGINGFACE_TOKEN')
        if not token:
            logger.error("❌ HUGGINGFACE_TOKEN environment variable not set")
            logger.info("Please set your token: export HUGGINGFACE_TOKEN=your_token")
            logger.info("Or run: python3 scripts/setup_huggingface_auth.py")
            return False
        
        logger.info("✅ Token found, proceeding with download...")
        
        # Download tokenizer
        logger.info("📥 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=token,
            trust_remote_code=True
        )
        
        # Download model (this will cache it locally)
        logger.info("📥 Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=token,
            torch_dtype=torch.float16,  # Use float16 for MPS compatibility
            device_map="auto",
            trust_remote_code=True
        )
        
        logger.info("✅ Model downloaded successfully!")
        
        # Test basic functionality
        logger.info("🧪 Testing model...")
        test_text = "Hello, how are you?"
        inputs = tokenizer(test_text, return_tensors="pt")
        
        with model.eval():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"✅ Test successful: {response}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        return False

if __name__ == "__main__":
    success = download_model()
    if success:
        print("\n🎉 Model download complete!")
        print("The model is now cached locally and ready for fine-tuning.")
    else:
        print("\n❌ Model download failed.")
        sys.exit(1)
