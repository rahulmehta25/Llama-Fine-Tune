#!/bin/bash

# Initialize a new fine-tuning project from this template
# Usage: ./scripts/init_new_project.sh <project_name> <domain>
# Example: ./scripts/init_new_project.sh financial-llama financial

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check arguments
if [ "$#" -ne 2 ]; then
    echo -e "${RED}Usage: $0 <project_name> <domain>${NC}"
    echo -e "${YELLOW}Example: $0 financial-llama financial${NC}"
    echo -e "${YELLOW}Available domains: financial, legal, marketing, medical, technical, custom${NC}"
    exit 1
fi

PROJECT_NAME=$1
DOMAIN=$2
PARENT_DIR=$(dirname "$(pwd)")
NEW_PROJECT_PATH="$PARENT_DIR/$PROJECT_NAME"

echo -e "${GREEN}Creating new fine-tuning project: $PROJECT_NAME${NC}"
echo -e "${GREEN}Domain: $DOMAIN${NC}"
echo -e "${GREEN}Location: $NEW_PROJECT_PATH${NC}"

# Create new project directory
if [ -d "$NEW_PROJECT_PATH" ]; then
    echo -e "${RED}Error: Directory $NEW_PROJECT_PATH already exists${NC}"
    exit 1
fi

# Copy template structure
echo -e "${YELLOW}Copying template files...${NC}"
mkdir -p "$NEW_PROJECT_PATH"

# Copy essential directories and files
cp -r src "$NEW_PROJECT_PATH/"
cp -r scripts "$NEW_PROJECT_PATH/"
cp -r config "$NEW_PROJECT_PATH/"
cp requirements.txt "$NEW_PROJECT_PATH/"
cp .gitignore "$NEW_PROJECT_PATH/"

# Create empty directories with .gitkeep
mkdir -p "$NEW_PROJECT_PATH/data"
mkdir -p "$NEW_PROJECT_PATH/outputs"
mkdir -p "$NEW_PROJECT_PATH/logs"
mkdir -p "$NEW_PROJECT_PATH/docs"

touch "$NEW_PROJECT_PATH/data/.gitkeep"
touch "$NEW_PROJECT_PATH/outputs/.gitkeep"
touch "$NEW_PROJECT_PATH/logs/.gitkeep"

# Create domain-specific configuration
echo -e "${YELLOW}Creating domain-specific configuration...${NC}"

# Copy and customize config based on domain
case $DOMAIN in
    financial)
        cp config/training_config.yaml "$NEW_PROJECT_PATH/config/training_config.yaml"
        # Update config for financial domain
        sed -i '' "s/run_name: \"llama-3.1-8b-finetune\"/run_name: \"llama-3.1-8b-financial\"/g" "$NEW_PROJECT_PATH/config/training_config.yaml"

        cat > "$NEW_PROJECT_PATH/config/domain_info.yaml" << EOF
domain: financial
description: "Fine-tuning for financial analysis, market predictions, and financial advisory"
suggested_data:
  - Financial reports and analysis
  - Market data and trends
  - Investment strategies
  - Risk assessment documents
  - Regulatory compliance texts
training_tips:
  - Include diverse financial instruments
  - Add numerical reasoning examples
  - Include regulatory context
  - Balance technical and layman explanations
EOF
        ;;

    legal)
        cp config/training_config.yaml "$NEW_PROJECT_PATH/config/training_config.yaml"
        sed -i '' "s/run_name: \"llama-3.1-8b-finetune\"/run_name: \"llama-3.1-8b-legal\"/g" "$NEW_PROJECT_PATH/config/training_config.yaml"

        cat > "$NEW_PROJECT_PATH/config/domain_info.yaml" << EOF
domain: legal
description: "Fine-tuning for legal document analysis, case law, and legal advisory"
suggested_data:
  - Legal contracts and agreements
  - Case law summaries
  - Legal opinions and briefs
  - Statutory interpretations
  - Legal Q&A pairs
training_tips:
  - Include jurisdiction-specific content
  - Add citation examples
  - Focus on precise legal language
  - Include disclaimer templates
EOF
        ;;

    marketing)
        cp config/training_config.yaml "$NEW_PROJECT_PATH/config/training_config.yaml"
        sed -i '' "s/run_name: \"llama-3.1-8b-finetune\"/run_name: \"llama-3.1-8b-marketing\"/g" "$NEW_PROJECT_PATH/config/training_config.yaml"

        cat > "$NEW_PROJECT_PATH/config/domain_info.yaml" << EOF
domain: marketing
description: "Fine-tuning for marketing copy, campaign strategies, and brand communication"
suggested_data:
  - Marketing campaigns and copy
  - Brand guidelines and voice
  - Social media content
  - SEO-optimized content
  - Customer persona analyses
training_tips:
  - Include various tones and styles
  - Add platform-specific content
  - Include A/B testing examples
  - Focus on conversion-oriented language
EOF
        ;;

    medical)
        cp config/training_config.yaml "$NEW_PROJECT_PATH/config/training_config.yaml"
        sed -i '' "s/run_name: \"llama-3.1-8b-finetune\"/run_name: \"llama-3.1-8b-medical\"/g" "$NEW_PROJECT_PATH/config/training_config.yaml"

        cat > "$NEW_PROJECT_PATH/config/domain_info.yaml" << EOF
domain: medical
description: "Fine-tuning for medical information, clinical decision support, and health education"
suggested_data:
  - Medical literature and research
  - Clinical guidelines
  - Patient education materials
  - Symptom-diagnosis mappings
  - Treatment protocols
training_tips:
  - Include disclaimer requirements
  - Focus on evidence-based content
  - Add citation of medical sources
  - Balance technical and patient-friendly language
EOF
        ;;

    technical)
        cp config/training_config.yaml "$NEW_PROJECT_PATH/config/training_config.yaml"
        sed -i '' "s/run_name: \"llama-3.1-8b-finetune\"/run_name: \"llama-3.1-8b-technical\"/g" "$NEW_PROJECT_PATH/config/training_config.yaml"

        cat > "$NEW_PROJECT_PATH/config/domain_info.yaml" << EOF
domain: technical
description: "Fine-tuning for technical documentation, code generation, and engineering support"
suggested_data:
  - Technical documentation
  - Code examples and explanations
  - API documentation
  - Troubleshooting guides
  - Best practices and patterns
training_tips:
  - Include multiple programming languages
  - Add error handling examples
  - Focus on clear explanations
  - Include performance considerations
EOF
        ;;

    custom)
        cp config/training_config.yaml "$NEW_PROJECT_PATH/config/training_config.yaml"
        sed -i '' "s/run_name: \"llama-3.1-8b-finetune\"/run_name: \"llama-3.1-8b-$PROJECT_NAME\"/g" "$NEW_PROJECT_PATH/config/training_config.yaml"

        cat > "$NEW_PROJECT_PATH/config/domain_info.yaml" << EOF
domain: custom
description: "Custom fine-tuning project"
suggested_data:
  - Define your data requirements here
training_tips:
  - Customize based on your specific needs
EOF
        ;;

    *)
        echo -e "${RED}Unknown domain: $DOMAIN${NC}"
        echo -e "${YELLOW}Using custom configuration${NC}"
        cp config/training_config.yaml "$NEW_PROJECT_PATH/config/training_config.yaml"
        ;;
esac

# Create project-specific README
cat > "$NEW_PROJECT_PATH/README.md" << EOF
# $PROJECT_NAME - Llama 3.1 8B Fine-Tuning

This project is a $DOMAIN-specific fine-tuning implementation based on the Llama 3.1 8B model.

## Project Information
- **Domain**: $DOMAIN
- **Base Model**: Llama 3.1 8B
- **Method**: QLoRA (Quantized Low-Rank Adaptation)
- **Hardware**: Optimized for Apple Silicon M4 Max

## Quick Start

### 1. Activate Environment
\`\`\`bash
conda activate ml-training
\`\`\`

### 2. Prepare Your Data
Place your training data in \`data/\` directory in JSONL format:
\`\`\`json
{"instruction": "...", "input": "...", "output": "..."}
\`\`\`

### 3. Process Data
\`\`\`bash
python src/data_processor.py --input data/your_data.jsonl
\`\`\`

### 4. Start Training
\`\`\`bash
python src/train.py --config config/training_config.yaml
\`\`\`

### 5. Run Inference
\`\`\`bash
python src/inference.py --model_path outputs --interactive
\`\`\`

## Configuration
- Training config: \`config/training_config.yaml\`
- Domain info: \`config/domain_info.yaml\`

## Output
- Trained models: \`outputs/\`
- Training logs: \`logs/\`

Created from template on $(date)
EOF

# Create sample data file for the domain
cat > "$NEW_PROJECT_PATH/data/sample_$DOMAIN.jsonl" << EOF
{"instruction": "Provide a sample $DOMAIN response", "input": "", "output": "This is a sample response for $DOMAIN domain fine-tuning."}
EOF

# Initialize git repository
cd "$NEW_PROJECT_PATH"
git init
git add .
git commit -m "Initial commit: $PROJECT_NAME for $DOMAIN domain"

echo -e "${GREEN}✅ Project created successfully!${NC}"
echo -e "${GREEN}📁 Location: $NEW_PROJECT_PATH${NC}"
echo -e "${YELLOW}Next steps:${NC}"
echo -e "  1. cd $NEW_PROJECT_PATH"
echo -e "  2. Add your training data to data/"
echo -e "  3. Customize config/training_config.yaml"
echo -e "  4. Run: python src/data_processor.py"
echo -e "  5. Run: python src/train.py"