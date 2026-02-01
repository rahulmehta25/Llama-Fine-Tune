#!/usr/bin/env python3
"""
Data Processing Utilities for Llama 3.1 8B Fine-Tuning
Handles data preparation, formatting, and validation
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Data processing class for fine-tuning preparation"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize data processor with configuration"""
        self.config = config
        self.instruction_column = config['data']['instruction_column']
        self.input_column = config['data']['input_column']
        self.output_column = config['data']['output_column']
    
    def load_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from various formats"""
        file_path = Path(file_path)
        
        if file_path.suffix == '.jsonl':
            return self._load_jsonl(file_path)
        elif file_path.suffix == '.json':
            return self._load_json(file_path)
        elif file_path.suffix in ['.csv', '.tsv']:
            return self._load_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _load_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load JSONL data"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
        return data
    
    def _load_json(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load JSON data"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'data' in data:
            return data['data']
        else:
            return [data]
    
    def _load_csv(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load CSV data"""
        df = pd.read_csv(file_path)
        return df.to_dict('records')
    
    def validate_data(self, data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Validate and clean data"""
        valid_data = []
        errors = []
        
        for i, item in enumerate(data):
            try:
                # Check required columns
                if self.instruction_column not in item:
                    errors.append(f"Row {i}: Missing instruction column '{self.instruction_column}'")
                    continue
                
                if self.output_column not in item:
                    errors.append(f"Row {i}: Missing output column '{self.output_column}'")
                    continue
                
                # Clean and validate text
                instruction = str(item[self.instruction_column]).strip()
                output = str(item[self.output_column]).strip()
                input_text = str(item.get(self.input_column, '')).strip()
                
                if not instruction or not output:
                    errors.append(f"Row {i}: Empty instruction or output")
                    continue
                
                # Add to valid data
                valid_item = {
                    self.instruction_column: instruction,
                    self.output_column: output,
                    self.input_column: input_text
                }
                valid_data.append(valid_item)
                
            except Exception as e:
                errors.append(f"Row {i}: Error processing - {str(e)}")
        
        logger.info(f"Validated {len(valid_data)} items, {len(errors)} errors found")
        if errors:
            logger.warning(f"Validation errors: {errors[:5]}...")  # Show first 5 errors
        
        return valid_data, errors
    
    def format_for_training(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format data for instruction tuning"""
        formatted_data = []
        
        for item in data:
            instruction = item[self.instruction_column]
            input_text = item[self.input_column]
            output = item[self.output_column]
            
            # Create instruction format
            if input_text:
                formatted_text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
            else:
                formatted_text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            
            formatted_data.append({
                'instruction': instruction,
                'input': input_text,
                'output': output,
                'text': formatted_text
            })
        
        return formatted_data
    
    def split_data(self, data: List[Dict[str, Any]], 
                   train_ratio: float = 0.8, 
                   val_ratio: float = 0.1,
                   test_ratio: float = 0.1) -> Tuple[List[Dict[str, Any]], 
                                                    List[Dict[str, Any]], 
                                                    List[Dict[str, Any]]]:
        """Split data into train/validation/test sets"""
        
        # Ensure ratios sum to 1
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        # First split: train vs (val + test)
        train_data, temp_data = train_test_split(
            data, 
            test_size=(val_ratio + test_ratio),
            random_state=42
        )
        
        # Second split: val vs test
        if val_ratio > 0 and test_ratio > 0:
            val_data, test_data = train_test_split(
                temp_data,
                test_size=test_ratio / (val_ratio + test_ratio),
                random_state=42
            )
        elif val_ratio > 0:
            val_data, test_data = temp_data, []
        else:
            val_data, test_data = [], temp_data
        
        logger.info(f"Data split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        
        return train_data, val_data, test_data
    
    def save_data(self, data: List[Dict[str, Any]], file_path: str):
        """Save data in JSONL format"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(data)} items to {file_path}")
    
    def process_dataset(self, input_file: str, 
                       output_dir: str = "data",
                       train_ratio: float = 0.8,
                       val_ratio: float = 0.1,
                       test_ratio: float = 0.1) -> Dict[str, str]:
        """Complete data processing pipeline"""
        
        logger.info(f"Processing dataset: {input_file}")
        
        # Load data
        raw_data = self.load_data(input_file)
        logger.info(f"Loaded {len(raw_data)} raw items")
        
        # Validate data
        valid_data, errors = self.validate_data(raw_data)
        
        if not valid_data:
            raise ValueError("No valid data found after validation")
        
        # Format for training
        formatted_data = self.format_for_training(valid_data)
        
        # Split data
        train_data, val_data, test_data = self.split_data(
            formatted_data, train_ratio, val_ratio, test_ratio
        )
        
        # Save splits
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        file_paths = {}
        if train_data:
            train_path = output_dir / "train.jsonl"
            self.save_data(train_data, train_path)
            file_paths['train'] = str(train_path)
        
        if val_data:
            val_path = output_dir / "eval.jsonl"
            self.save_data(val_data, val_path)
            file_paths['eval'] = str(val_path)
        
        if test_data:
            test_path = output_dir / "test.jsonl"
            self.save_data(test_data, test_path)
            file_paths['test'] = str(test_path)
        
        # Save validation errors if any
        if errors:
            error_path = output_dir / "validation_errors.json"
            with open(error_path, 'w', encoding='utf-8') as f:
                json.dump(errors, f, indent=2, ensure_ascii=False)
            logger.warning(f"Validation errors saved to {error_path}")
        
        return file_paths

def create_sample_data(output_file: str = "data/sample_data.jsonl"):
    """Create sample data for testing"""
    
    sample_data = [
        {
            "instruction": "What is the capital of France?",
            "input": "",
            "output": "The capital of France is Paris."
        },
        {
            "instruction": "Explain the concept of machine learning",
            "input": "",
            "output": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task."
        },
        {
            "instruction": "Translate the following text to Spanish",
            "input": "Hello, how are you?",
            "output": "Hola, ¿cómo estás?"
        },
        {
            "instruction": "Write a short story about a robot",
            "input": "The story should be about 100 words",
            "output": "In a quiet laboratory, a small robot named ARIA discovered emotions. Unlike other machines, ARIA felt curiosity about the world beyond its charging station. One day, it ventured outside, marveling at sunlight and wind. When a child approached, ARIA learned about friendship. The robot realized that consciousness wasn't about processing power, but about connection. ARIA returned to the lab, but now it dreamed of the world outside, teaching other robots about the beauty of being alive."
        },
        {
            "instruction": "Solve this math problem",
            "input": "What is 15% of 200?",
            "output": "15% of 200 is 30. To calculate this: 200 × 0.15 = 30."
        },
        {
            "instruction": "What are the benefits of renewable energy?",
            "input": "",
            "output": "Renewable energy offers numerous benefits including reduced greenhouse gas emissions, decreased dependence on fossil fuels, lower long-term energy costs, job creation in clean energy sectors, improved air quality, and enhanced energy security through diversified energy sources."
        },
        {
            "instruction": "Explain how photosynthesis works",
            "input": "",
            "output": "Photosynthesis is the process by which plants convert light energy into chemical energy. Plants use sunlight, carbon dioxide from the air, and water from the soil to produce glucose (sugar) and oxygen. This process occurs primarily in the chloroplasts of plant cells, where chlorophyll captures light energy and converts it into chemical energy stored in glucose molecules."
        },
        {
            "instruction": "What is quantum computing?",
            "input": "",
            "output": "Quantum computing is a revolutionary computing paradigm that leverages quantum mechanical phenomena like superposition and entanglement to process information. Unlike classical computers that use bits (0 or 1), quantum computers use quantum bits (qubits) that can exist in multiple states simultaneously, potentially enabling exponential speedups for certain computational problems."
        },
        {
            "instruction": "Write a haiku about nature",
            "input": "",
            "output": "Cherry blossoms fall\nSoft petals dance in the breeze\nSpring's gentle whisper"
        },
        {
            "instruction": "What is the difference between AI and machine learning?",
            "input": "",
            "output": "Artificial Intelligence (AI) is the broader concept of machines being able to carry out tasks in a way that we would consider 'smart.' Machine Learning (ML) is a subset of AI that focuses on the idea that machines should be able to learn and adapt through experience. In other words, ML is one of the methods used to achieve AI, but AI encompasses many other approaches beyond just machine learning."
        },
        {
            "instruction": "How does the internet work?",
            "input": "",
            "output": "The internet works through a global network of interconnected computers that communicate using standardized protocols. When you send data, it's broken into packets that travel through routers and switches across multiple networks until they reach their destination, where they're reassembled. This decentralized system allows information to flow efficiently around the world."
        },
        {
            "instruction": "What is climate change?",
            "input": "",
            "output": "Climate change refers to long-term shifts in global temperatures and weather patterns. While climate variations are natural, human activities since the 1800s have been the main driver of climate change, primarily through burning fossil fuels which generates greenhouse gas emissions that trap heat in the atmosphere."
        },
        {
            "instruction": "Explain the water cycle",
            "input": "",
            "output": "The water cycle is the continuous movement of water through Earth's atmosphere, land, and oceans. It involves evaporation (water turning to vapor), condensation (vapor forming clouds), precipitation (rain, snow, etc.), and collection (water gathering in bodies of water). This cycle is essential for life on Earth and helps distribute water globally."
        },
        {
            "instruction": "What is democracy?",
            "input": "",
            "output": "Democracy is a system of government where power is vested in the people, who exercise it directly or through elected representatives. Key principles include free and fair elections, majority rule with minority rights, individual freedoms, rule of law, and citizen participation in political processes."
        },
        {
            "instruction": "How do vaccines work?",
            "input": "",
            "output": "Vaccines work by introducing a weakened, killed, or piece of a disease-causing microorganism into the body. This triggers the immune system to produce antibodies and memory cells that recognize and fight the actual disease if encountered later. Vaccines essentially train the immune system to defend against specific infections."
        },
        {
            "instruction": "What is the scientific method?",
            "input": "",
            "output": "The scientific method is a systematic approach to understanding the natural world through observation, hypothesis formation, experimentation, data collection, analysis, and conclusion drawing. It involves making observations, asking questions, forming testable hypotheses, conducting controlled experiments, analyzing results, and drawing evidence-based conclusions."
        },
        {
            "instruction": "Explain gravity",
            "input": "",
            "output": "Gravity is a fundamental force that attracts objects with mass toward each other. On Earth, gravity pulls objects toward the planet's center, giving them weight. According to Einstein's theory of general relativity, gravity is the curvature of spacetime caused by massive objects. The more massive an object, the stronger its gravitational pull."
        },
        {
            "instruction": "What is DNA?",
            "input": "",
            "output": "DNA (Deoxyribonucleic acid) is the hereditary material in humans and almost all other organisms. It contains the genetic instructions for development, functioning, growth, and reproduction. DNA is made up of four chemical bases (A, T, G, C) that form pairs and create a double helix structure."
        },
        {
            "instruction": "How do computers work?",
            "input": "",
            "output": "Computers work by processing information using binary code (0s and 1s). They have a central processing unit (CPU) that performs calculations, memory to store data and programs, input devices to receive information, and output devices to display results. The CPU follows instructions from programs to manipulate data and produce desired outcomes."
        },
        {
            "instruction": "What is evolution?",
            "input": "",
            "output": "Evolution is the process by which species change over time through genetic variation, natural selection, and adaptation to their environment. It explains how all living things share common ancestors and how diversity of life developed through gradual changes over millions of years. Charles Darwin's theory of natural selection is a key mechanism of evolution."
        },
        {
            "instruction": "Explain the solar system",
            "input": "",
            "output": "The solar system consists of the Sun and all objects that orbit around it, including eight planets, their moons, asteroids, comets, and other celestial bodies. The four inner planets (Mercury, Venus, Earth, Mars) are rocky, while the four outer planets (Jupiter, Saturn, Uranus, Neptune) are gas giants. Everything orbits the Sun due to its massive gravitational pull."
        }
    ]
    
    # Save sample data
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in sample_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"Created sample data with {len(sample_data)} items at {output_file}")
    return output_file

if __name__ == "__main__":
    # Example usage
    import yaml
    
    # Load config
    with open('config/training_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Use existing sample data
    sample_file = "data/sample_data.jsonl"
    
    # Process data
    processor = DataProcessor(config)
    file_paths = processor.process_dataset(sample_file)
    
    print("Data processing complete!")
    print("Generated files:")
    for split, path in file_paths.items():
        print(f"  {split}: {path}")
