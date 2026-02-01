# CSV Data Processing Guide for Fine-Tuning

## Overview

This guide explains how to use CSV data for fine-tuning Llama 3.1 8B. The CSV processor automatically converts various CSV formats into the required JSONL format.

## Quick Start

### Basic Usage
```bash
# Simple Q&A format
python src/csv_data_processor.py --input data/example_qa.csv

# Domain-specific processing
python src/csv_data_processor.py --input data/example_financial.csv --domain financial

# Custom column mapping
python src/csv_data_processor.py --input data/custom.csv \
    --instruction_col "question" \
    --output_col "answer"
```

## Supported CSV Formats

### 1. Question-Answer Format (2 columns)

**CSV Structure:**
```csv
question,answer
"What is AI?","AI is artificial intelligence..."
"How does ML work?","Machine learning works by..."
```

**Auto-detected columns:**
- question, query, prompt → instruction
- answer, response, reply → output

### 2. Instruction-Input-Output Format (3 columns)

**CSV Structure:**
```csv
instruction,context,response
"Summarize this text","Long article text...","Summary: Key points are..."
"Translate to Spanish","Hello world","Hola mundo"
```

**Auto-detected columns:**
- instruction, task, prompt → instruction
- context, input, data → input (optional)
- response, output, result → output

### 3. Conversation Format

**CSV Structure:**
```csv
role,content
user,"What is the weather?"
assistant,"I cannot check real-time weather..."
user,"How about tomorrow?"
assistant,"I cannot predict future weather..."
```

**Processing:**
- Pairs user/assistant messages
- Creates instruction-output pairs

### 4. Domain-Specific Formats

#### Financial Data
```csv
ticker,metric,value,analysis
AAPL,P/E Ratio,28.5,"Analysis of Apple's valuation..."
```

**Converts to:**
```json
{
  "instruction": "Analyze P/E Ratio for AAPL",
  "input": "Ticker: AAPL\nMetric: P/E Ratio\nValue: 28.5",
  "output": "Analysis of Apple's valuation..."
}
```

#### Medical Data
```csv
symptoms,diagnosis,treatment
"Fever, cough","Common cold","Rest and fluids"
```

**Converts to:**
```json
{
  "instruction": "Provide a diagnosis based on the symptoms",
  "input": "Symptoms: Fever, cough",
  "output": "Diagnosis: Common cold\nRecommended Treatment: Rest and fluids"
}
```

#### Customer Service
```csv
customer_query,category,response
"Refund request",billing,"I'll help you process..."
```

## Advanced Features

### 1. Multi-Column Handling

If your CSV has many columns, the processor will:
1. Detect instruction/output columns
2. Combine other columns as context

**Example:**
```csv
question,date,user_id,department,answer
"Budget question",2024-01-15,u123,finance,"The budget is..."
```

**Becomes:**
```json
{
  "instruction": "Budget question",
  "input": "date: 2024-01-15\nuser_id: u123\ndepartment: finance",
  "output": "The budget is..."
}
```

### 2. Custom Column Mapping

Specify exact columns to use:

```bash
python src/csv_data_processor.py \
    --input data/custom.csv \
    --instruction_col "user_question" \
    --input_col "background_info" \
    --output_col "expert_response"
```

### 3. Data Splitting Configuration

Control train/validation/test splits:

```bash
python src/csv_data_processor.py \
    --input data/large_dataset.csv \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15
```

### 4. Domain-Specific Processing

Use predefined templates for common domains:

```bash
# Financial domain
python src/csv_data_processor.py --input financial.csv --domain financial

# Medical domain
python src/csv_data_processor.py --input medical.csv --domain medical

# Customer service
python src/csv_data_processor.py --input support.csv --domain customer_service
```

## Data Preparation Best Practices

### 1. Data Quality

**Good Example:**
```csv
question,answer
"What is compound interest?","Compound interest is interest calculated on the initial principal and accumulated interest from previous periods, leading to exponential growth over time."
```

**Poor Example:**
```csv
question,answer
"Interest?","It's complicated"
```

### 2. Consistency

Maintain consistent:
- Formatting style
- Response length
- Technical accuracy
- Terminology usage

### 3. Diversity

Include varied:
- Question types (what, how, why, when)
- Complexity levels
- Topics within domain
- Response formats

### 4. Data Cleaning

The processor automatically:
- Removes empty entries
- Trims excessive whitespace
- Validates minimum lengths
- Filters extremely long entries

## Output Structure

After processing, you'll have:

```
data/
├── train.jsonl       # 80% of data
├── eval.jsonl        # 10% of data
├── test.jsonl        # 10% of data
└── sample_processed.json  # First 5 examples for review
```

## Examples by Use Case

### 1. Financial Advisory Training

**Input CSV:**
```csv
client_question,advisor_response
"Should I invest in bonds now?","Given current interest rates..."
"What's a good P/E ratio?","A good P/E ratio varies by industry..."
```

**Command:**
```bash
python src/csv_data_processor.py --input financial_qa.csv
```

### 2. Technical Documentation

**Input CSV:**
```csv
command,description,example
"git push","Uploads local commits to remote","git push origin main"
"git pull","Downloads and merges remote changes","git pull origin main"
```

**Command:**
```bash
python src/csv_data_processor.py \
    --input commands.csv \
    --instruction_col "command" \
    --input_col "description" \
    --output_col "example"
```

### 3. Product Support

**Input CSV:**
```csv
issue_type,customer_message,support_response
"Refund","Product arrived damaged","I apologize for the damaged product. I'll process a full refund immediately..."
"Technical","App keeps crashing","Let's troubleshoot the app crashes. First, please try..."
```

**Command:**
```bash
python src/csv_data_processor.py \
    --input support.csv \
    --instruction_col "customer_message" \
    --input_col "issue_type" \
    --output_col "support_response"
```

## Integration with Training

After processing your CSV:

```bash
# 1. Process CSV to JSONL
python src/csv_data_processor.py --input your_data.csv

# 2. Review sample output
cat data/sample_processed.json

# 3. Start training
python src/train.py --config config/training_config.yaml
```

## Troubleshooting

### Issue: Columns not detected

**Solution:** Use explicit mapping:
```bash
python src/csv_data_processor.py \
    --input data.csv \
    --instruction_col "your_question_column" \
    --output_col "your_answer_column"
```

### Issue: Unicode/Encoding errors

**Solution:** Ensure CSV is UTF-8 encoded:
```python
# In Python
df = pd.read_csv('file.csv', encoding='utf-8')
df.to_csv('file_utf8.csv', encoding='utf-8', index=False)
```

### Issue: Data imbalance

**Solution:** The processor automatically balances splits, but you can adjust:
```bash
python src/csv_data_processor.py \
    --input data.csv \
    --train_ratio 0.6 \
    --val_ratio 0.2 \
    --test_ratio 0.2
```

### Issue: Mixed formats in one CSV

**Solution:** Preprocess to standardize:
```python
import pandas as pd

# Standardize your CSV first
df = pd.read_csv('mixed.csv')
df['instruction'] = df['question'].fillna(df['query'])
df['output'] = df['answer'].fillna(df['response'])
df[['instruction', 'input', 'output']].to_csv('standardized.csv', index=False)
```

## Performance Tips

1. **Optimal Dataset Size**: 1,000-10,000 examples for domain adaptation
2. **Quality over Quantity**: 1,000 high-quality > 10,000 mediocre examples
3. **Balance Categories**: Ensure even distribution of question types
4. **Validate Output**: Always review `sample_processed.json`
5. **Iterate**: Start with small dataset, evaluate, then expand

## Complete Workflow Example

```bash
# Step 1: Prepare your CSV data
# Ensure it has clear question/answer or instruction/output columns

# Step 2: Process CSV
python src/csv_data_processor.py \
    --input data/financial_qa.csv \
    --domain financial \
    --output_dir data

# Step 3: Review processed sample
cat data/sample_processed.json

# Step 4: Check data statistics
wc -l data/*.jsonl

# Step 5: Start fine-tuning
python src/train.py --config config/training_config.yaml

# Step 6: Test with your domain
python src/inference.py \
    --model_path outputs \
    --instruction "What is a good P/E ratio?" \
    --interactive
```

## CSV Template Downloads

Create these templates for your data:

### Basic Q&A Template
```csv
question,answer
"Your question here","Detailed answer here"
"Another question","Another detailed answer"
```

### Full Instruction Template
```csv
instruction,input,output
"Task description","Optional context","Expected output"
"Summarize this","Long text...","Summary: ..."
```

### Conversation Template
```csv
role,content
user,"User's question"
assistant,"Assistant's response"
user,"Follow-up question"
assistant,"Follow-up response"
```

Save any template and start adding your domain-specific data!

---

*The CSV processor handles most common formats automatically. For complex cases, use custom column mapping or preprocess your data into one of the standard formats above.*