#!/bin/bash

# Proper ML Training Setup for M4 Max
# Following user's exact specifications: Python 3.11 + Conda

echo "🚀 Setting up ML Training Environment with Conda + Python 3.11"
echo "============================================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Miniconda or Anaconda first."
    echo ""
    echo "📥 Download and install from:"
    echo "   https://docs.conda.io/en/latest/miniconda.html"
    echo ""
    echo "After installation, restart your terminal and run this script again."
    exit 1
fi

echo "✅ Conda found: $(conda --version)"

# Create conda environment with Python 3.11
echo "📦 Creating conda environment 'ml-training' with Python 3.11..."
conda create -n ml-training python=3.11 -y

if [ $? -ne 0 ]; then
    echo "❌ Failed to create conda environment"
    exit 1
fi

# Activate environment
echo "🔄 Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ml-training

# Verify Python version
echo "🐍 Verifying Python version..."
python --version

# Install PyTorch with conda (as specified)
echo "🔥 Installing PyTorch with conda..."
conda install pytorch torchvision torchaudio -c pytorch -y

# Install core frameworks
echo "📚 Installing core frameworks..."
pip install transformers datasets accelerate bitsandbytes

# Install Apple MLX (as specified)
echo "🍎 Installing Apple MLX for Silicon optimization..."
pip install mlx mlx-lm

# Install training tools
echo "🛠️ Installing training tools..."
pip install peft trl  # Parameter-efficient fine-tuning
pip install wandb     # Experiment tracking
pip install ollama    # Local model serving

# Install additional requirements
echo "📦 Installing additional packages..."
pip install -r requirements.txt

# Verify installation
echo "✅ Verifying installation..."
python -c "
import torch
import transformers
import mlx
import peft
print('✅ All core packages imported successfully')
print(f'Python: {torch.version.python}')
print(f'PyTorch: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'Transformers: {transformers.__version__}')
print(f'MLX: installed')
print(f'PEFT: {peft.__version__}')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Setup complete!"
    echo ""
    echo "📋 What's been installed:"
    echo "  ✅ ml-training conda environment"
    echo "  ✅ Python 3.11"
    echo "  ✅ PyTorch with Apple Silicon support"
    echo "  ✅ MLX framework for optimization"
    echo "  ✅ PEFT, TRL, and other training tools"
    echo "  ✅ Weights & Biases for monitoring"
    echo ""
    echo "🚀 Next steps:"
    echo "  1. Activate environment: conda activate ml-training"
    echo "  2. Download model: python scripts/download_model_simple.py"
    echo "  3. Start training: python src/train.py --config config/training_config.yaml"
    echo ""
else
    echo "❌ Installation verification failed"
    exit 1
fi


