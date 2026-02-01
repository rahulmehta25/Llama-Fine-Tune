#!/bin/bash

# Llama 3.1 8B Fine-Tuning Environment Setup for M4 Max
# Following user's exact specifications for ml-training environment

echo "🚀 Setting up ML Training Environment for M4 Max"
echo "==============================================="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Miniconda or Anaconda first."
    echo "   Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create ML environment as specified
echo "📦 Creating conda environment 'ml-training'..."
conda create -n ml-training python=3.11 -y

# Activate environment
echo "🔄 Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ml-training

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
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import mlx; print(f'MLX version: {mlx.__version__}')"
python -c "import peft; print(f'PEFT version: {peft.__version__}')"

echo ""
echo "🎉 ML Training environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate ml-training"
echo ""
echo "To start training, run:"
echo "  python src/train.py --config config/training_config.yaml"
echo ""
