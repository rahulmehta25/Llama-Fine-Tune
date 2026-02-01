# Llama 3.1 8B Fine-Tuning Project - Complete Implementation Guide

A comprehensive fine-tuning setup for Llama 3.1 8B optimized for Apple Silicon M4 Max with 64GB RAM. This project uses QLoRA (Quantized Low-Rank Adaptation) for efficient fine-tuning and includes complete data processing, training, and inference pipelines.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Hardware Specifications](#hardware-specifications)
3. [Complete Installation Guide](#complete-installation-guide)
4. [Environment Setup](#environment-setup)
5. [Model Download and Authentication](#model-download-and-authentication)
6. [Data Processing Pipeline](#data-processing-pipeline)
7. [Training Configuration](#training-configuration)
8. [Inference System](#inference-system)
9. [Project Structure](#project-structure)
10. [Usage Examples](#usage-examples)
11. [Troubleshooting](#troubleshooting)
12. [Technical Details](#technical-details)

## Project Overview

This project provides a complete implementation for fine-tuning Llama 3.1 8B on Apple Silicon M4 Max hardware. The setup includes:

- Complete conda environment with Python 3.11
- QLoRA fine-tuning with MPS (Metal Performance Shaders) support
- Data processing pipeline for JSONL format datasets
- Training scripts optimized for M4 Max hardware
- Inference system with interactive chat and batch processing
- Comprehensive monitoring with Weights & Biases integration

## Hardware Specifications

- **CPU**: Apple M4 Max with 40 GPU cores
- **RAM**: 64GB unified memory
- **Platform**: macOS (Apple Silicon)
- **Target Model**: Llama 3.1 8B (8 billion parameters)
- **Memory Usage**: ~25GB RAM, ~30GB GPU memory during training
- **Expected Training Time**: 6-12 hours for 1000 steps

## Complete Installation Guide

### Step 1: System Prerequisites

Before starting, ensure you have:
- macOS with Apple Silicon (M1/M2/M3/M4)
- Xcode Command Line Tools installed
- At least 70GB free disk space
- Internet connection for model download

### Step 2: Download and Install Miniconda

```bash
# Download Miniconda for Apple Silicon
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh

# Install Miniconda
bash Miniconda3-latest-MacOSX-arm64.sh

# Add conda to PATH
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Verify installation
conda --version
```

### Step 3: Accept Conda Terms of Service

```bash
# Accept terms for required channels
conda config --set channel_priority strict
conda config --add channels conda-forge
conda config --add channels pytorch
```

### Step 4: Create Project Directory

```bash
# Create project directory
mkdir -p "Llama 3.1 Fine-Tune"
cd "Llama 3.1 Fine-Tune"

# Initialize git repository
git init
```

## Environment Setup

### Step 1: Create Conda Environment

```bash
# Create conda environment with Python 3.11
conda create -n ml-training python=3.11 -y

# Activate environment
conda activate ml-training
```

### Step 2: Install Core Dependencies via Conda

```bash
# Install PyTorch and related packages via conda
conda install pytorch torchvision torchaudio -c pytorch -y

# Install additional conda packages
conda install numpy pandas scikit-learn matplotlib seaborn -y
conda install jupyter ipykernel -y
```

### Step 3: Install ML/AI Packages via pip

```bash
# Core ML libraries
pip install transformers==4.56.2
pip install datasets==3.0.0
pip install accelerate==0.34.0
pip install peft==0.17.1
pip install trl==0.9.0

# Quantization and optimization
pip install bitsandbytes==0.44.0

# Apple Silicon optimization
pip install mlx
pip install mlx-lm

# Experiment tracking and monitoring
pip install wandb==0.18.0
pip install tensorboard==2.18.0

# Additional utilities
pip install tqdm==4.67.1
pip install sentencepiece==0.2.0
pip install protobuf==4.26.1

# Local model serving
pip install ollama==0.4.7
```

### Step 4: Fix NumPy Compatibility

```bash
# Downgrade NumPy for PyTorch compatibility
pip install numpy==1.26.4
```

### Step 5: Verify Installation

```bash
# Test PyTorch with MPS
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}')"

# Test MLX
python -c "import mlx.core as mx; print('MLX working')"

# Test Transformers
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

## Model Download and Authentication

### Step 1: Hugging Face Authentication

```bash
# Method 1: Interactive login
python -m huggingface_hub.commands.huggingface_cli login

# Method 2: Set token directly
export HUGGINGFACE_TOKEN="your_token_here"
```

### Step 2: Request Model Access

1. Visit: https://huggingface.co/meta-llama/Llama-3.1-8B
2. Click "Request access"
3. Wait for approval (usually instant for verified accounts)

### Step 3: Download Model

```bash
# Download and verify model
python scripts/download_model.py

# Alternative simple download
python scripts/download_model_simple.py
```

The model will be cached locally at `~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/`

## Data Processing Pipeline

### Step 1: Data Format

Your training data should be in JSONL format:

```json
{"instruction": "What is the capital of France?", "input": "", "output": "The capital of France is Paris."}
{"instruction": "Explain machine learning", "input": "", "output": "Machine learning is a subset of artificial intelligence..."}
{"instruction": "Translate to Spanish", "input": "Hello world", "output": "Hola mundo"}
```

### Step 2: Process Your Data

```bash
# Process sample data
python src/data_processor.py

# Process custom data
python src/data_processor.py --input your_data.jsonl --output_dir data
```

### Step 3: Data Validation

The processor will:
- Validate JSONL format
- Check required fields (instruction, input, output)
- Split data into train/eval/test sets (80/10/10)
- Generate error reports for invalid entries

## Training Configuration

### Step 1: QLoRA Configuration

The project uses QLoRA with the following optimized settings:

```yaml
lora:
  r: 16                    # Rank - good balance for 8B model
  lora_alpha: 32           # Scaling factor
  target_modules:          # Target attention modules
    - "q_proj"
    - "v_proj" 
    - "k_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
  lora_dropout: 0.1
  bias: "none"
  task_type: "CAUSAL_LM"
```

### Step 2: Training Parameters

```yaml
training:
  per_device_train_batch_size: 4      # Fits in 64GB RAM
  gradient_accumulation_steps: 4      # Effective batch size = 16
  learning_rate: 2e-4
  num_train_epochs: 3
  max_steps: 1000                     # Adjust based on dataset size
  warmup_steps: 100
  weight_decay: 0.01
  lr_scheduler_type: "cosine"
  
  # Apple Silicon optimizations
  bf16: false                        # BFloat16 not supported on MPS
  fp16: false                        # fp16 not supported on MPS
  dataloader_num_workers: 8          # Utilize CPU cores
  gradient_checkpointing: true       # Save memory
```

### Step 3: Hardware Configuration

```yaml
hardware:
  device_map: "auto"
  torch_dtype: "float16"             # Use float16 for MPS compatibility
  use_mps: true                      # Use Metal Performance Shaders
  max_memory: "60GB"                 # Leave some RAM for system
```

### Step 4: Start Training

```bash
# Basic training
python src/train.py

# With custom config
python src/train.py --config config/training_config.yaml

# With Weights & Biases monitoring
wandb login
python src/train.py
```

## Inference System

### Step 1: Interactive Chat

```bash
# Start interactive chat
python src/inference.py --model_path outputs --interactive

# Single instruction
python src/inference.py --model_path outputs --instruction "What is machine learning?"
```

### Step 2: Batch Processing

```python
from src.inference import LlamaInference

# Load model
inference = LlamaInference("outputs")
inference.load_model()

# Process multiple instructions
instructions = [
    "What is AI?",
    "Explain quantum computing",
    "Write a poem about nature"
]

responses = inference.batch_inference(instructions)
for instruction, response in zip(instructions, responses):
    print(f"Q: {instruction}")
    print(f"A: {response}\n")
```

## Project Structure

```
Llama 3.1 Fine-Tune/
├── config/                          # Configuration files
│   ├── training_config.yaml         # Training parameters optimized for M4 Max
│   └── model_config.yaml            # Model configuration
├── data/                            # Data directory
│   ├── sample_data.jsonl            # Sample training data (21 examples)
│   ├── train.jsonl                  # Training data (80% split)
│   ├── eval.jsonl                   # Validation data (10% split)
│   └── test.jsonl                   # Test data (10% split)
├── docs/                            # Documentation
│   └── activity.md                  # Complete project activity log
├── outputs/                         # Model outputs and checkpoints
├── logs/                            # Training logs
├── scripts/                         # Setup and utility scripts
│   ├── setup_conda_environment.sh   # Complete conda environment setup
│   ├── setup_alternative.sh         # Alternative setup for non-conda systems
│   ├── setup_huggingface_auth.py    # Hugging Face authentication helper
│   ├── download_model.py             # Model download with verification
│   ├── download_model_simple.py     # Simplified model download
│   ├── run_training.sh              # Training automation script
│   └── test_model.sh                # Model testing script
├── src/                             # Source code
│   ├── train.py                     # Main training script with QLoRA
│   ├── inference.py                 # Inference and chat interface
│   └── data_processor.py            # Data processing utilities
├── requirements.txt                 # Python dependencies
└── README.md                        # This comprehensive guide
```

## Usage Examples

### Complete Training Workflow

```bash
# 1. Activate environment
conda activate ml-training

# 2. Process your data
python src/data_processor.py --input your_dataset.jsonl

# 3. Start training
python src/train.py --config config/training_config.yaml

# 4. Monitor training
# Training logs will be saved to logs/
# Weights & Biases dashboard will show real-time metrics

# 5. Test your model
python src/inference.py --model_path outputs --interactive
```

### Custom Configuration

```bash
# Edit training parameters
nano config/training_config.yaml

# Adjust batch size for your hardware
# Change learning rate for your dataset
# Modify LoRA parameters for different capacity

# Run with custom config
python src/train.py --config config/training_config.yaml
```

### Data Processing Options

```bash
# Process with custom split ratios
python src/data_processor.py --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1

# Process with validation
python src/data_processor.py --validate --max_errors 10

# Process large dataset
python src/data_processor.py --input large_dataset.jsonl --chunk_size 1000
```

## Troubleshooting

### Common Issues and Solutions

#### 1. BFloat16 Not Supported on MPS

**Error**: `TypeError: BFloat16 is not supported on MPS`

**Solution**: The project is already configured to use float16 instead of bfloat16 for MPS compatibility.

#### 2. Conda Command Not Found

**Error**: `command not found: conda`

**Solution**:
```bash
# Add conda to PATH
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

#### 3. Xcode License Agreement

**Error**: `You have not agreed to the Xcode license agreements`

**Solution**:
```bash
# Accept Xcode license
sudo xcodebuild -license accept

# Or use alternative setup
./scripts/setup_alternative.sh
```

#### 4. NumPy Compatibility

**Error**: `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.0.1`

**Solution**:
```bash
# Downgrade NumPy
pip install numpy==1.26.4
```

#### 5. Out of Memory

**Error**: `CUDA out of memory` or similar

**Solution**: Reduce batch size in `config/training_config.yaml`:
```yaml
training:
  per_device_train_batch_size: 2  # Reduce from 4
  gradient_accumulation_steps: 8  # Increase from 4
```

#### 6. Model Access Denied

**Error**: `403 Client Error. Access to model meta-llama/Llama-3.1-8B is restricted`

**Solution**:
1. Request access at https://huggingface.co/meta-llama/Llama-3.1-8B
2. Wait for approval
3. Ensure authentication: `huggingface-cli login`

### Performance Optimization

#### Memory Optimization

```yaml
# For systems with less memory
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  gradient_checkpointing: true
  dataloader_num_workers: 4
```

#### Speed Optimization

```yaml
# For faster training
training:
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 2
  dataloader_num_workers: 16
  fp16: true  # If supported
```

## Technical Details

### Libraries and Versions

| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.11 | Core language |
| PyTorch | 2.2.2 | Deep learning framework |
| Transformers | 4.56.2 | Hugging Face transformers |
| Datasets | 3.0.0 | Dataset handling |
| Accelerate | 0.34.0 | Training acceleration |
| PEFT | 0.17.1 | Parameter-efficient fine-tuning |
| TRL | 0.9.0 | Transformer reinforcement learning |
| BitsAndBytes | 0.44.0 | Quantization |
| MLX | Latest | Apple Silicon optimization |
| MLX-LM | Latest | Apple Silicon language models |
| Weights & Biases | 0.18.0 | Experiment tracking |
| TensorBoard | 2.18.0 | Training visualization |
| NumPy | 1.26.4 | Numerical computing |
| Pandas | Latest | Data manipulation |
| Scikit-learn | Latest | Machine learning utilities |
| Jupyter | Latest | Interactive notebooks |
| TQDM | 4.67.1 | Progress bars |
| SentencePiece | 0.2.0 | Tokenization |
| Protobuf | 4.26.1 | Serialization |
| Ollama | 0.4.7 | Local model serving |

### MPS Compatibility Fixes

The project includes several fixes for Apple Silicon MPS compatibility:

1. **Data Type**: Uses float16 instead of bfloat16
2. **Model Loading**: Loads to CPU first, then moves to MPS
3. **Mixed Precision**: Disabled fp16 for MPS compatibility
4. **Device Mapping**: Uses "auto" device mapping with MPS fallback

### Memory Management

- **Gradient Checkpointing**: Enabled to save memory
- **QLoRA**: Reduces trainable parameters to 0.52% of total
- **Batch Processing**: Optimized batch size for 64GB RAM
- **Data Loading**: Efficient data loading with multiple workers

### Training Monitoring

- **Weights & Biases**: Real-time training metrics
- **TensorBoard**: Local training visualization
- **Logging**: Comprehensive logging to files
- **Checkpointing**: Automatic model saving

## Expected Performance

| Metric | Value |
|--------|-------|
| **Training Time** | 6-12 hours for 1000 steps |
| **Memory Usage** | ~25GB RAM, ~30GB GPU memory |
| **Performance Gain** | 15-30% improvement on specific task |
| **Fine-tuning Data Needed** | 1,000-5,000 high-quality examples |
| **Model Size** | 8B parameters (base model) |
| **Trainable Parameters** | 41.9M (0.52% of total) |
| **Context Window** | 128K tokens |
| **Batch Size** | 4 (effective 16 with accumulation) |

## Additional Resources

- [Llama 3.1 Documentation](https://huggingface.co/meta-llama/Llama-3.1-8B)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Transformers Library](https://huggingface.co/docs/transformers/)
- [PEFT Documentation](https://huggingface.co/docs/peft/)
- [Apple MLX Documentation](https://ml-explore.github.io/mlx/)
- [Weights & Biases](https://wandb.ai/)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Meta for the Llama 3.1 model
- Hugging Face for the Transformers library
- Microsoft for the LoRA technique
- The open-source community for QLoRA implementation
- Apple for MLX framework and MPS support

---

**Project Status**: Complete and ready for production use with your own datasets.