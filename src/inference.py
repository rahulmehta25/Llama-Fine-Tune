#!/usr/bin/env python3
"""
Inference Script for Fine-Tuned Llama 3.1 8B
Optimized for Apple Silicon M4 Max
"""

import os
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LlamaInference:
    """Inference class for fine-tuned Llama 3.1 8B"""
    
    def __init__(self, model_path: str, config_path: str = "config/model_config.yaml"):
        """Initialize inference with model and configuration"""
        self.model_path = model_path
        self.config = self._load_config(config_path)
        self.model = None
        self.tokenizer = None
        self.device = self._setup_device()
        
        logger.info(f"Using device: {self.device}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _setup_device(self) -> str:
        """Setup device for Apple Silicon optimization"""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def _setup_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Setup quantization configuration for memory efficiency"""
        if self.config.get('quantization', {}).get('load_in_4bit', False):
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        return None
    
    def load_model(self):
        """Load the fine-tuned model and tokenizer"""
        logger.info(f"Loading model from: {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            padding_side=self.config['tokenizer']['padding_side'],
            truncation_side=self.config['tokenizer']['truncation_side']
        )
        
        # Add special tokens if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        quantization_config = self._setup_quantization_config()
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config['base_model'],
            quantization_config=quantization_config,
            device_map=self.config['apple_silicon']['device_map'],
            torch_dtype=torch.bfloat16 if self.config['apple_silicon']['use_mps'] else torch.float16,
            trust_remote_code=True
        )
        
        # Load fine-tuned weights
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        
        # Set to evaluation mode
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
    def format_prompt(self, instruction: str, input_text: str = "") -> str:
        """Format prompt for inference"""
        if input_text:
            return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            return f"### Instruction:\n{instruction}\n\n### Response:\n"
    
    def generate_response(self, 
                         instruction: str, 
                         input_text: str = "",
                         max_new_tokens: int = 512,
                         temperature: float = 0.7,
                         top_p: float = 0.9,
                         top_k: int = 50,
                         repetition_penalty: float = 1.1) -> str:
        """Generate response for given instruction"""
        
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Format prompt
        prompt = self.format_prompt(instruction, input_text)
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        
        # Move to device
        if self.device == "mps":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate response
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                early_stopping=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generation_time = time.time() - start_time
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part
        response = full_response[len(prompt):].strip()
        
        logger.info(f"Generated response in {generation_time:.2f} seconds")
        
        return response
    
    def batch_inference(self, 
                       instructions: List[str], 
                       input_texts: List[str] = None,
                       **generation_kwargs) -> List[str]:
        """Run batch inference on multiple instructions"""
        
        if input_texts is None:
            input_texts = [""] * len(instructions)
        
        if len(instructions) != len(input_texts):
            raise ValueError("Number of instructions and input texts must match")
        
        responses = []
        
        for i, (instruction, input_text) in enumerate(zip(instructions, input_texts)):
            logger.info(f"Processing {i+1}/{len(instructions)}")
            response = self.generate_response(instruction, input_text, **generation_kwargs)
            responses.append(response)
        
        return responses
    
    def interactive_chat(self):
        """Start interactive chat session"""
        print("\n🤖 Llama 3.1 8B Fine-Tuned Chat")
        print("=" * 40)
        print("Type 'quit' to exit, 'clear' to clear context")
        print()
        
        while True:
            try:
                # Get user input
                instruction = input("You: ").strip()
                
                if instruction.lower() == 'quit':
                    print("Goodbye! 👋")
                    break
                
                if instruction.lower() == 'clear':
                    print("Context cleared.")
                    continue
                
                if not instruction:
                    continue
                
                # Generate response
                print("🤖 Thinking...")
                response = self.generate_response(instruction)
                print(f"Assistant: {response}")
                print()
                
            except KeyboardInterrupt:
                print("\nGoodbye! 👋")
                break
            except Exception as e:
                logger.error(f"Error during chat: {e}")
                print(f"Error: {e}")
                print()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned Llama 3.1 8B")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model directory"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/model_config.yaml",
        help="Path to model configuration file"
    )
    parser.add_argument(
        "--instruction",
        type=str,
        help="Single instruction to process"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Input text for the instruction"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive chat mode"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = LlamaInference(args.model_path, args.config)
    
    # Load model
    inference.load_model()
    
    if args.interactive:
        # Interactive mode
        inference.interactive_chat()
    elif args.instruction:
        # Single instruction mode
        response = inference.generate_response(
            args.instruction,
            args.input,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        )
        print(f"Instruction: {args.instruction}")
        if args.input:
            print(f"Input: {args.input}")
        print(f"Response: {response}")
    else:
        print("Please provide either --instruction or --interactive flag")
        print("Use --help for more information")

if __name__ == "__main__":
    main()


