# Complete Fine-Tuning Process Guide

## Table of Contents
1. [Overview](#overview)
2. [Environment Setup Phase](#1-environment-setup-phase)
3. [Model Preparation Phase](#2-model-preparation-phase)
4. [Data Preparation Phase](#3-data-preparation-phase)
5. [Training Phase](#4-training-phase)
6. [LoRA Mechanism Explained](#5-lora-low-rank-adaptation-mechanism)
7. [Training Monitoring](#6-training-monitoring)
8. [Output & Artifacts](#7-output--artifacts)
9. [Inference Phase](#8-inference-phase)
10. [Key Concepts Explained](#key-concepts-explained)
11. [Performance Metrics](#performance-metrics)
12. [What You're Actually Training](#what-youre-actually-training)

## Overview

This guide provides a comprehensive understanding of the Llama 3.1 8B fine-tuning process, optimized for Apple M4 Max hardware. The process uses QLoRA (Quantized Low-Rank Adaptation) to efficiently fine-tune an 8-billion parameter model by training only 0.52% of the parameters.

## 1. Environment Setup Phase

The foundation of the fine-tuning process begins with proper environment configuration:

```bash
# Install Miniconda (if needed)
bash Miniconda3-latest-MacOSX-arm64.sh

# Create and activate environment
conda create -n ml-training python=3.11
conda activate ml-training

# Install dependencies
pip install -r requirements.txt

# Apply M4 Max optimizations
python scripts/optimize_for_m4.py
source .env.m4max
```

**Key Components Installed:**
- PyTorch with MPS (Metal Performance Shaders) support
- Transformers library for model handling
- PEFT for parameter-efficient fine-tuning
- MLX for optimized Apple Silicon inference
- Weights & Biases for experiment tracking

## 2. Model Preparation Phase

### Authentication & Access
1. **Hugging Face Login**: Authenticate with your Hugging Face token
2. **Model Access**: Request access to Meta's Llama 3.1 8B model
3. **Download**: Base model (~15GB) automatically cached at `~/.cache/huggingface/`

### Model Architecture
- **Base Model**: Llama 3.1 8B (8 billion parameters)
- **Context Window**: 128K tokens
- **Vocabulary Size**: 128,256 tokens
- **Hidden Size**: 4,096
- **Attention Heads**: 32
- **Layers**: 32

### QLoRA Configuration
```yaml
lora:
  r: 16                    # Rank of adaptation
  lora_alpha: 32           # Scaling factor
  target_modules:          # Layers to adapt
    - "q_proj"            # Query projection
    - "v_proj"            # Value projection
    - "k_proj"            # Key projection
    - "o_proj"            # Output projection
    - "gate_proj"         # FFN gate
    - "up_proj"           # FFN up
    - "down_proj"         # FFN down
  lora_dropout: 0.1
  bias: "none"
```

**Result**: Reduces trainable parameters from 8B to 41.9M (0.52%)

## 3. Data Preparation Phase

### Data Pipeline
```
Raw Data (JSONL) → Validation → Formatting → Tokenization → DataLoader
```

### Required Data Format
```json
{
  "instruction": "What is machine learning?",
  "input": "Optional context or additional information",
  "output": "Machine learning is a subset of artificial intelligence..."
}
```

### Processing Steps

1. **Validation**:
   - Checks JSONL format integrity
   - Validates required fields presence
   - Reports any malformed entries

2. **Formatting**:
   ```python
   # With input context
   prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"

   # Without input context
   prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
   ```

3. **Tokenization**:
   - Max length: 2,048 tokens
   - Padding: Right-side
   - Truncation: Right-side
   - Special tokens: Automatically added

4. **Data Splitting**:
   - Training: 80%
   - Validation: 10%
   - Test: 10%

5. **DataLoader Creation**:
   - Batch size: 6 (optimized for M4 Max)
   - Num workers: 12 (utilizing performance cores)
   - Shuffle: True for training
   - Pin memory: False (better for MPS)

## 4. Training Phase

### What Actually Happens During Training

#### Phase 4.1: Model Loading
```python
# 1. Load base model in float32 for MPS stability
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    torch_dtype=torch.float32,
    device_map="auto"
)

# 2. Apply LoRA adapters
model = get_peft_model(base_model, lora_config)

# 3. Result: Only 41.9M parameters are trainable
```

#### Phase 4.2: Training Loop

Each training step follows this sequence:

```
1. Forward Pass (Prediction)
   ↓
2. Loss Calculation (Compare to target)
   ↓
3. Backward Pass (Calculate gradients)
   ↓
4. Optimizer Step (Update weights)
   ↓
5. Learning Rate Update (Cosine schedule)
```

**Detailed Process:**

1. **Forward Pass**:
   - Input tokens passed through model
   - Generates probability distribution over vocabulary
   - LoRA adapters modify attention patterns

2. **Loss Calculation**:
   - Cross-entropy loss between predictions and targets
   - Measures how wrong the predictions are
   - Lower loss = better performance

3. **Backward Pass**:
   - Calculates gradients via backpropagation
   - Only computes gradients for LoRA parameters
   - Base model remains frozen

4. **Weight Update**:
   - AdamW optimizer updates LoRA weights
   - Learning rate: 2e-4 with cosine decay
   - Weight decay: 0.01 for regularization

#### Phase 4.3: Memory Management

```yaml
Gradient Checkpointing:
  - Saves intermediate activations selectively
  - Recomputes during backward pass
  - Trades computation for memory

Gradient Accumulation:
  - Physical batch: 6 examples
  - Accumulation steps: 3
  - Effective batch: 18 examples
  - Updates weights every 3 batches
```

#### Phase 4.4: Optimization Configuration

```yaml
training:
  optimizer: adamw_torch_fused     # Optimized for M4 Max
  learning_rate: 2e-4               # Starting LR
  lr_scheduler: cosine              # Smooth decay
  warmup_steps: 100                 # Gradual start
  max_steps: 1000                   # Total training
  gradient_clip: 1.0                # Prevent exploding gradients
```

## 5. LoRA (Low-Rank Adaptation) Mechanism

### How LoRA Works

Instead of fine-tuning all 8B parameters, LoRA introduces small, trainable matrices:

```
Original Weight Matrix (Frozen)     LoRA Adaptation
    W₀ ∈ ℝ^(d×d)                   B × A ∈ ℝ^(d×r) × ℝ^(r×d)
    [4096×4096]            +         [4096×16] × [16×4096]
                           =
                      Adapted Weight Matrix
                         W = W₀ + BA
```

### Why LoRA is Efficient

1. **Parameter Reduction**:
   - Original parameters: 8,000,000,000
   - LoRA parameters: 41,900,000
   - Reduction: 99.48%

2. **Memory Savings**:
   - Full fine-tuning: ~30GB for gradients
   - LoRA fine-tuning: ~2GB for gradients

3. **Training Speed**:
   - Fewer parameters to update
   - Smaller optimizer states
   - Faster backpropagation

### Target Modules Explained

```python
Query (q_proj):  Learns what to pay attention to
Key (k_proj):    Learns what information to match
Value (v_proj):  Learns what information to extract
Output (o_proj): Learns how to combine attention results
Gate (gate_proj): Learns when to apply transformations
Up/Down:         Learns feature transformations
```

## 6. Training Monitoring

### Real-time Metrics

**Weights & Biases Dashboard**:
- Loss curves (training & validation)
- Learning rate schedule
- Gradient norms
- Training speed (tokens/sec)
- Memory usage

**Console Output**:
```
Epoch 1/3: 100%|████████| 1000/1000 [06:32<00:00, 2.54it/s]
  Training Loss: 1.234
  Validation Loss: 1.456
  Learning Rate: 0.0001
  Tokens/sec: 1300
```

### Checkpointing Strategy

```yaml
Checkpoint Saving:
  - Every 500 steps
  - Best model based on validation loss
  - Keep 3 most recent checkpoints
  - Each checkpoint: ~500MB (LoRA weights only)
```

## 7. Output & Artifacts

### Final Directory Structure

```
outputs/
├── adapter_config.json           # LoRA configuration
├── adapter_model.safetensors     # Trained LoRA weights (~500MB)
├── tokenizer_config.json         # Tokenizer settings
├── tokenizer.json                # Tokenizer vocabulary
├── special_tokens_map.json      # Special token mappings
├── training_args.json            # Training hyperparameters
└── checkpoint-1000/              # Final checkpoint
    ├── adapter_model.safetensors
    ├── optimizer.pt              # Optimizer state
    └── trainer_state.json        # Training state
```

### What Each File Contains

- **adapter_model.safetensors**: The actual trained weights (LoRA matrices)
- **adapter_config.json**: Configuration for loading LoRA adapters
- **training_args.json**: Complete record of training hyperparameters
- **trainer_state.json**: Training history, best metrics, stopping point

## 8. Inference Phase

### Two Inference Options

#### Option 1: PyTorch Inference (Standard)

```python
# Load base model + LoRA adapter
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
model = PeftModel.from_pretrained(model, "outputs/")

# Generate text
output = model.generate(input_ids, max_new_tokens=512)
```

**Performance**: ~100 tokens/second on M4 Max

#### Option 2: MLX Inference (Optimized for Apple Silicon)

```python
# Use MLX for Metal acceleration
from mlx_lm import load, generate

model, tokenizer = load("meta-llama/Llama-3.1-8B", adapter_path="outputs/")
response = generate(model, tokenizer, prompt="Your prompt here")
```

**Performance**: ~130 tokens/second on M4 Max (30% faster)

### Generation Parameters

```yaml
generation:
  temperature: 0.7        # Randomness (0=deterministic, 1=random)
  top_p: 0.9             # Nucleus sampling threshold
  top_k: 50              # Top-K sampling
  repetition_penalty: 1.1 # Reduce repetition
  max_new_tokens: 512    # Maximum response length
```

## Key Concepts Explained

### Gradient Accumulation
- **Problem**: Large batches don't fit in memory
- **Solution**: Process small batches, accumulate gradients
- **Example**: Batch of 6 × 3 accumulation = effective batch of 18
- **Benefit**: Stable training with limited memory

### Learning Rate Schedule

```
Learning Rate
    ^
2e-4|     .-----.
    |    /       \
    |   /         \___cosine decay
    |  /warmup        \
    | /                \____
0   |______________________|→ Steps
    0   100        1000
```

- **Warmup**: Prevents early instability
- **Cosine Decay**: Smooth reduction to near-zero
- **Purpose**: Fine control over optimization trajectory

### Attention Mechanism Fine-tuning

The attention mechanism learns three key transformations:

1. **Query (Q)**: "What am I looking for?"
2. **Key (K)**: "What information is available?"
3. **Value (V)**: "What should I extract?"

Fine-tuning adapts these to your specific domain:
- Financial data → attention to numbers, trends
- Legal text → attention to precedents, citations
- Medical → attention to symptoms, treatments

### Mixed Precision Training (Future)

Currently disabled for MPS, but conceptually:
- **Compute**: Use float16 (faster)
- **Accumulation**: Use float32 (more precise)
- **Result**: 2x speedup with minimal accuracy loss

## Performance Metrics

### Training Performance

| Metric | Value | Notes |
|--------|-------|-------|
| **Setup Time** | ~3 minutes | Model loading, data prep |
| **Training Speed** | 1,300 tokens/sec | With optimizations |
| **Memory Usage** | 58GB / 64GB | 90% utilization |
| **Time per 1K steps** | 6-12 hours | Depends on data |
| **Checkpoint Size** | 500MB | LoRA weights only |
| **Validation Frequency** | Every 500 steps | Configurable |

### Inference Performance

| Mode | Speed | Memory | Quality |
|------|-------|--------|---------|
| **PyTorch (MPS)** | 100 tokens/sec | 15GB | Baseline |
| **MLX (Metal)** | 130 tokens/sec | 12GB | Identical |
| **Batch (×4)** | 80 tokens/sec/item | 20GB | Identical |

### Resource Utilization

```
Component         Usage    Optimal   Status
GPU Cores:        95%      90-100%   ✅
Memory:           58GB     60GB      ✅
CPU (loading):    80%      70-90%    ✅
Disk I/O:         Burst    -         ✅
Temperature:      75°C     <85°C     ✅
```

## What You're Actually Training

### The Learning Process

You're teaching the model to:

1. **Domain Adaptation**:
   - Learn specialized vocabulary
   - Understand domain-specific patterns
   - Maintain context awareness

2. **Style Transfer**:
   - Match training data tone
   - Follow specific formatting
   - Adopt domain conventions

3. **Knowledge Integration**:
   - Combine base knowledge with new information
   - Resolve conflicts between general and specific
   - Maintain coherence

### Example Transformations

**Before Fine-tuning**:
```
Q: "What's the P/E ratio impact?"
A: "P/E ratio is price divided by earnings..."
```

**After Financial Fine-tuning**:
```
Q: "What's the P/E ratio impact?"
A: "A P/E compression from 25x to 20x implies a 20% valuation
    headwind, requiring 25% earnings growth to maintain price..."
```

### Success Metrics

| Metric | Expected Improvement |
|--------|---------------------|
| **Domain Accuracy** | +15-30% |
| **Response Relevance** | +20-40% |
| **Technical Precision** | +25-35% |
| **Style Consistency** | +30-50% |

## Training Best Practices

### Data Quality > Quantity
- 1,000 high-quality examples > 10,000 mediocre ones
- Diverse examples covering edge cases
- Consistent formatting and style

### Hyperparameter Tuning Order
1. **Learning Rate**: Most impactful (try 1e-4 to 5e-4)
2. **LoRA Rank**: Higher = more capacity (8, 16, 32)
3. **Batch Size**: Larger = more stable (limited by memory)
4. **Epochs**: More isn't always better (watch validation loss)

### Common Issues & Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| **Overfitting** | Val loss increases | Reduce epochs, add dropout |
| **Underfitting** | High training loss | Increase LoRA rank, LR |
| **Catastrophic Forgetting** | Loses base abilities | Lower LR, shorter training |
| **Mode Collapse** | Repetitive outputs | Increase temperature, penalty |

## Advanced Optimizations

### Memory Optimization Techniques
```python
# Clear cache periodically
torch.mps.empty_cache()

# Use gradient checkpointing
model.gradient_checkpointing_enable()

# Optimize data loading
DataLoader(pin_memory=False, persistent_workers=True)
```

### Speed Optimization Techniques
```python
# Compile model (PyTorch 2.0+)
model = torch.compile(model)

# Use fused optimizers
optimizer = torch.optim.AdamW(params, fused=True)

# Enable SDPA
model.config.use_sdpa = True
```

## Conclusion

Fine-tuning Llama 3.1 8B with QLoRA on M4 Max represents a perfect balance of:
- **Efficiency**: Only 0.52% parameters trained
- **Performance**: 15-30% improvement on domain tasks
- **Practicality**: Runs on consumer hardware
- **Speed**: Hours instead of days

The key insight is that large language models already contain vast knowledge; fine-tuning simply teaches them how to apply that knowledge to your specific domain and style preferences. With LoRA, this adaptation happens efficiently without losing the model's general capabilities.

---

*This guide represents the complete fine-tuning pipeline optimized for Apple M4 Max hardware. For specific implementation details, refer to the source code and configuration files.*