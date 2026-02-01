#!/bin/bash

# Complete ML Training Setup for M4 Max
# Following user's exact specifications

echo "🚀 Complete ML Training Setup for M4 Max"
echo "========================================"
echo ""

# Step 1: Environment Setup
echo "📦 Step 1: Setting up ML Training Environment"
echo "--------------------------------------------"
./scripts/setup_environment.sh

if [ $? -ne 0 ]; then
    echo "❌ Environment setup failed"
    exit 1
fi

echo ""
echo "✅ Environment setup complete!"
echo ""

# Step 2: Activate environment and download model
echo "📥 Step 2: Downloading and verifying Llama 3.1 8B model"
echo "-----------------------------------------------------"

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ml-training

# Download and verify model
python scripts/download_model.py

if [ $? -ne 0 ]; then
    echo "❌ Model download/verification failed"
    exit 1
fi

echo ""
echo "✅ Model download and verification complete!"
echo ""

# Step 3: Prepare sample data
echo "📊 Step 3: Preparing sample training data"
echo "----------------------------------------"

# Create sample data
python -c "
from src.data_processor import create_sample_data
create_sample_data('data/sample_data.jsonl')
print('Sample data created successfully')
"

if [ $? -ne 0 ]; then
    echo "❌ Sample data creation failed"
    exit 1
fi

echo ""
echo "✅ Sample data prepared!"
echo ""

# Step 4: Verify everything is ready
echo "🧪 Step 4: Final verification"
echo "----------------------------"

# Test imports
python -c "
import torch
import transformers
import mlx
import peft
import wandb
print('✅ All imports successful')
print(f'PyTorch: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MLX: {mlx.__version__}')
"

if [ $? -ne 0 ]; then
    echo "❌ Import verification failed"
    exit 1
fi

echo ""
echo "🎉 Setup Complete! Your M4 Max is ready for ML training!"
echo ""
echo "📋 What's been set up:"
echo "  ✅ ml-training conda environment"
echo "  ✅ PyTorch with Apple Silicon support"
echo "  ✅ MLX framework for optimization"
echo "  ✅ Llama 3.1 8B model downloaded and verified"
echo "  ✅ PEFT, TRL, and other training tools"
echo "  ✅ Sample training data prepared"
echo ""
echo "🚀 Next steps:"
echo "  1. Activate environment: conda activate ml-training"
echo "  2. Start training: ./scripts/run_training.sh"
echo "  3. Test model: ./scripts/test_model.sh"
echo ""
echo "💡 Pro tips:"
echo "  - Use Weights & Biases: wandb login"
echo "  - Monitor with: wandb watch"
echo "  - Deploy with Ollama for local serving"
echo ""


