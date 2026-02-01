#!/bin/bash

# Alternative ML Training Setup for M4 Max
# Handles systems without conda or Xcode license issues

echo "🚀 Alternative ML Training Setup for M4 Max"
echo "=========================================="
echo ""

# Check for Xcode license issue
if xcodebuild -version 2>&1 | grep -q "license"; then
    echo "⚠️  Xcode license not accepted. Please run:"
    echo "   sudo xcodebuild -license"
    echo "   Then accept the license agreement"
    echo ""
    echo "Alternatively, you can use the manual setup below."
    echo ""
fi

# Check available Python versions
echo "🐍 Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    echo "✅ Found python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
    echo "✅ Found python"
else
    echo "❌ No Python found. Please install Python 3.11+"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $PYTHON_VERSION"

# Check if we can create virtual environment
echo ""
echo "📦 Setting up virtual environment..."

# Try to create virtual environment
if $PYTHON_CMD -m venv ml-training-env 2>/dev/null; then
    echo "✅ Virtual environment created successfully"
    VENV_CMD="source ml-training-env/bin/activate"
else
    echo "⚠️  Virtual environment creation failed. Using system Python."
    VENV_CMD=""
fi

# Activate environment if created
if [ ! -z "$VENV_CMD" ]; then
    echo "🔄 Activating virtual environment..."
    eval $VENV_CMD
fi

# Install packages
echo ""
echo "📚 Installing ML packages..."

# Install PyTorch with Apple Silicon support
echo "Installing PyTorch..."
$PYTHON_CMD -m pip install torch torchvision torchaudio

# Install core ML packages
echo "Installing core ML packages..."
$PYTHON_CMD -m pip install transformers datasets accelerate bitsandbytes

# Install Apple MLX
echo "Installing Apple MLX..."
$PYTHON_CMD -m pip install mlx mlx-lm

# Install training tools
echo "Installing training tools..."
$PYTHON_CMD -m pip install peft trl wandb ollama

# Install additional requirements
echo "Installing additional packages..."
$PYTHON_CMD -m pip install -r requirements.txt

# Verify installation
echo ""
echo "✅ Verifying installation..."
$PYTHON_CMD -c "
import torch
import transformers
import mlx
import peft
print('✅ All core packages imported successfully')
print(f'PyTorch: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MLX: {mlx.__version__}')
print(f'PEFT: {peft.__version__}')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Setup complete!"
    echo ""
    echo "📋 What's been installed:"
    echo "  ✅ PyTorch with Apple Silicon support"
    echo "  ✅ MLX framework for optimization"
    echo "  ✅ PEFT, TRL, and other training tools"
    echo "  ✅ Weights & Biases for monitoring"
    echo ""
    echo "🚀 Next steps:"
    if [ ! -z "$VENV_CMD" ]; then
        echo "  1. Activate environment: $VENV_CMD"
    fi
    echo "  2. Download model: $PYTHON_CMD scripts/download_model.py"
    echo "  3. Start training: $PYTHON_CMD src/train.py --config config/training_config.yaml"
    echo ""
else
    echo "❌ Installation verification failed"
    exit 1
fi


