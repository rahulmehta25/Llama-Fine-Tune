#!/bin/bash

# Llama 3.1 8B Training Script
# Optimized for M4 Max Apple Silicon

echo "🚀 Starting Llama 3.1 8B Fine-Tuning on M4 Max"
echo "=============================================="

# Check if conda environment exists
if ! conda info --envs | grep -q "ml-training"; then
    echo "❌ Conda environment 'ml-training' not found."
    echo "   Please run ./scripts/setup_environment.sh first"
    exit 1
fi

# Activate environment
echo "🔄 Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ml-training

# Check if data exists
if [ ! -f "data/train.jsonl" ]; then
    echo "📊 Creating sample data..."
    python src/data_processor.py
fi

# Check if Weights & Biases is configured
if ! wandb whoami &> /dev/null; then
    echo "⚠️  Weights & Biases not configured. Training will continue without logging."
    echo "   To enable logging, run: wandb login"
fi

# Create output directories
mkdir -p outputs logs

# Start training
echo "🎯 Starting training..."
echo "   Model: Llama 3.1 8B"
echo "   Method: QLoRA"
echo "   Device: Apple Silicon M4 Max"
echo "   Memory: 64GB RAM"
echo ""

python src/train.py --config config/training_config.yaml

echo ""
echo "✅ Training completed!"
echo "   Model saved to: outputs/"
echo "   Logs saved to: logs/"
echo ""
echo "To test your model, run:"
echo "  python src/inference.py --model_path outputs --interactive"
