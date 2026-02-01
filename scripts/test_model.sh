#!/bin/bash

# Llama 3.1 8B Model Testing Script
# Test your fine-tuned model

echo "🧪 Testing Fine-Tuned Llama 3.1 8B Model"
echo "======================================="

# Check if model exists
if [ ! -d "outputs" ] || [ ! -f "outputs/adapter_config.json" ]; then
    echo "❌ Fine-tuned model not found in outputs/"
    echo "   Please run training first: ./scripts/run_training.sh"
    exit 1
fi

# Activate environment
echo "🔄 Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ml-training

echo ""
echo "🎯 Model Testing Options:"
echo "1. Interactive Chat"
echo "2. Single Question Test"
echo "3. Batch Test with Sample Questions"
echo ""

read -p "Choose option (1-3): " choice

case $choice in
    1)
        echo "Starting interactive chat..."
        python src/inference.py --model_path outputs --interactive
        ;;
    2)
        read -p "Enter your question: " question
        echo "Generating response..."
        python src/inference.py --model_path outputs --instruction "$question"
        ;;
    3)
        echo "Running batch test with sample questions..."
        python -c "
from src.inference import LlamaInference

# Load model
inference = LlamaInference('outputs')
inference.load_model()

# Test questions
questions = [
    'What is machine learning?',
    'Explain quantum computing in simple terms',
    'Write a short poem about artificial intelligence',
    'What are the benefits of renewable energy?',
    'How does photosynthesis work?'
]

print('Running batch test...')
for i, question in enumerate(questions, 1):
    print(f'\\n--- Question {i} ---')
    print(f'Q: {question}')
    response = inference.generate_response(question)
    print(f'A: {response}')
    print('-' * 50)
"
        ;;
    *)
        echo "Invalid option. Exiting."
        exit 1
        ;;
esac

echo ""
echo "✅ Testing completed!"
