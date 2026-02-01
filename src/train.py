#!/usr/bin/env python3
"""
Llama 3.1 8B Fine-Tuning Script for M4 Max
Optimized for Apple Silicon with QLoRA
"""

import os
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import wandb
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset, load_dataset
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LlamaFineTuner:
    """Fine-tuning class for Llama 3.1 8B with QLoRA on M4 Max"""
    
    def __init__(self, config_path: str):
        """Initialize the fine-tuner with configuration"""
        self.config = self._load_config(config_path)
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Set up device
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
    
    def _setup_lora_config(self) -> LoraConfig:
        """Setup LoRA configuration optimized for M4 Max"""
        lora_config = self.config['lora']
        return LoraConfig(
            r=lora_config['r'],
            lora_alpha=lora_config['lora_alpha'],
            target_modules=lora_config['target_modules'],
            lora_dropout=lora_config['lora_dropout'],
            bias=lora_config['bias'],
            task_type=TaskType.CAUSAL_LM
        )
    
    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer"""
        logger.info(f"Loading model: {self.config['model']['name']}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['name'],
            padding_side=self.config['tokenizer']['padding_side'],
            truncation_side=self.config['tokenizer']['truncation_side']
        )
        
        # Add special tokens if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization if specified
        quantization_config = self._setup_quantization_config()
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model']['name'],
            quantization_config=quantization_config,
            device_map="cpu",  # Load to CPU first, then move to MPS
            torch_dtype=torch.float16,  # Use float16 for MPS compatibility
            trust_remote_code=True
        )
        
        # Apply LoRA
        lora_config = self._setup_lora_config()
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        logger.info("Model and tokenizer loaded successfully")
    
    def prepare_dataset(self, file_path: str, max_samples: Optional[int] = None) -> Dataset:
        """Prepare dataset for training"""
        logger.info(f"Loading dataset from: {file_path}")
        
        # Load JSONL data
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(eval(line.strip()))
                if max_samples and len(data) >= max_samples:
                    break
        
        # Format data for instruction tuning
        formatted_data = []
        for item in data:
            instruction = item.get(self.config['data']['instruction_column'], '')
            input_text = item.get(self.config['data']['input_column'], '')
            output = item.get(self.config['data']['output_column'], '')
            
            # Create instruction format
            if input_text:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            
            formatted_data.append({
                'text': prompt,
                'instruction': instruction,
                'input': input_text,
                'output': output
            })
        
        return Dataset.from_list(formatted_data)
    
    def tokenize_function(self, examples):
        """Tokenize the examples"""
        return self.tokenizer(
            examples['text'],
            truncation=True,
            padding=True,
            max_length=self.config['model']['max_length'],
            return_tensors="pt"
        )
    
    def prepare_training_data(self):
        """Prepare training and evaluation datasets"""
        logger.info("Preparing training data...")
        
        # Load training data
        train_dataset = self.prepare_dataset(
            self.config['data']['train_file'],
            self.config['data'].get('max_train_samples')
        )
        
        # Load evaluation data
        eval_dataset = None
        if os.path.exists(self.config['data']['eval_file']):
            eval_dataset = self.prepare_dataset(
                self.config['data']['eval_file'],
                self.config['data'].get('max_eval_samples')
            )
        
        # Tokenize datasets
        train_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        if eval_dataset:
            eval_dataset = eval_dataset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=eval_dataset.column_names
            )
        
        return train_dataset, eval_dataset
    
    def setup_training_args(self) -> TrainingArguments:
        """Setup training arguments optimized for M4 Max"""
        training_config = self.config['training']
        
        return TrainingArguments(
            output_dir=training_config['output_dir'],
            logging_dir=training_config['logging_dir'],
            per_device_train_batch_size=training_config['per_device_train_batch_size'],
            per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', 4),
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            learning_rate=training_config['learning_rate'],
            num_train_epochs=training_config['num_train_epochs'],
            max_steps=training_config.get('max_steps', -1),
            warmup_steps=training_config.get('warmup_steps', 0),
            weight_decay=training_config.get('weight_decay', 0.01),
            lr_scheduler_type=training_config.get('lr_scheduler_type', 'linear'),
            
            # Apple Silicon optimizations
            bf16=training_config.get('bf16', True),
            fp16=training_config.get('fp16', False),
            dataloader_num_workers=training_config.get('dataloader_num_workers', 8),
            gradient_checkpointing=training_config.get('gradient_checkpointing', True),
            
            # Logging and saving
            logging_steps=training_config.get('logging_steps', 10),
            save_steps=training_config.get('save_steps', 500),
            eval_steps=training_config.get('eval_steps', 500),
            save_total_limit=training_config.get('save_total_limit', 3),
            load_best_model_at_end=training_config.get('load_best_model_at_end', True),
            metric_for_best_model=training_config.get('metric_for_best_model', 'eval_loss'),
            greater_is_better=training_config.get('greater_is_better', False),
            
            # Evaluation
            eval_strategy=training_config.get('eval_strategy', 'steps'),
            eval_accumulation_steps=training_config.get('eval_accumulation_steps', 1),
            
            # Weights & Biases
            report_to=training_config.get('report_to', 'wandb'),
            run_name=training_config.get('run_name', 'llama-3.1-8b-finetune'),
            
            # Other settings
            remove_unused_columns=False,
            dataloader_pin_memory=False,  # Better for Apple Silicon
        )
    
    def train(self):
        """Main training function"""
        logger.info("Starting training...")
        
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Prepare data
        train_dataset, eval_dataset = self.prepare_training_data()
        
        # Setup training arguments
        training_args = self.setup_training_args()
        
        # Initialize Weights & Biases
        if training_args.report_to == 'wandb':
            wandb.init(
                project="llama-3.1-8b-finetune",
                name=training_args.run_name,
                config=self.config
            )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Start training
        self.trainer.train()
        
        # Save the final model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(training_args.output_dir)
        
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {training_args.output_dir}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3.1 8B with QLoRA")
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Initialize fine-tuner
    fine_tuner = LlamaFineTuner(args.config)
    
    # Start training
    fine_tuner.train()

if __name__ == "__main__":
    main()
