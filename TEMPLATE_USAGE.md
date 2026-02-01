# Using This Project as a Base Template

This project serves as your base template for fine-tuning multiple Llama 3.1 8B models on different domains.

## How to Clone for New Models

### 1. Copy the Project
```bash
# From the parent directory
cp -r "Llama 3.1 Fine-Tune" "Llama 3.1 Financial"
cd "Llama 3.1 Financial"
```

### 2. Clean Previous Outputs
```bash
# Remove any existing training outputs
rm -rf outputs/*
rm -rf logs/*
rm -rf data/*.jsonl  # Keep only your new data
```

### 3. Update Configuration
Edit `config/training_config.yaml`:
- Change `run_name` to match your domain (e.g., "llama-3.1-8b-financial")
- Adjust `output_dir` to keep models separate (e.g., "./outputs/financial")
- Modify training parameters based on your data

### 4. Add Your Domain Data
Place your domain-specific training data in `data/`:
```bash
# Your domain data
data/train.jsonl
data/eval.jsonl
data/test.jsonl
```

### 5. Train Your Model
```bash
conda activate ml-training
python src/train.py --config config/training_config.yaml
```

## Example: Creating Multiple Specialized Models

### Financial Model
```bash
cp -r "Llama 3.1 Fine-Tune" "Llama-Financial"
cd "Llama-Financial"
# Add financial data to data/
# Update config: run_name: "llama-3.1-8b-financial"
python src/train.py
```

### Legal Model
```bash
cp -r "Llama 3.1 Fine-Tune" "Llama-Legal"
cd "Llama-Legal"
# Add legal data to data/
# Update config: run_name: "llama-3.1-8b-legal"
python src/train.py
```

### Marketing Model
```bash
cp -r "Llama 3.1 Fine-Tune" "Llama-Marketing"
cd "Llama-Marketing"
# Add marketing data to data/
# Update config: run_name: "llama-3.1-8b-marketing"
python src/train.py
```

## Important Notes

1. **Shared Base Model**: The base Llama 3.1 8B model (~15GB) at `~/.cache/huggingface/` is shared across all projects

2. **Environment**: Always use the same conda environment (`ml-training`) for all models

3. **Outputs**: Each project keeps its own:
   - LoRA adapters in `outputs/`
   - Training logs in `logs/`
   - Processed data in `data/`

4. **Git**: Each cloned project can be its own git repository:
   ```bash
   git init
   git add .
   git commit -m "Initial: [Domain] fine-tuning project"
   ```

## Storage Estimates

- Base template: ~100MB (without model)
- Per specialized model: ~500MB (LoRA adapters only)
- Shared Llama base: 15GB (one-time download)

You can train dozens of specialized models with minimal storage overhead!