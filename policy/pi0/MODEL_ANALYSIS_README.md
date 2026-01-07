# Pi0 Model Difference Analysis

This repository contains scripts to analyze the differences between two Pi0 model checkpoints. The analysis helps understand where the main differences between trained models are located.

## Overview

The analysis compares two model checkpoints and computes various metrics to understand:
- Which parameters have changed the most
- Which parameter groups (LoRA, attention, MLP, etc.) show the largest differences
- Overall similarity between the models
- Detailed statistics for each parameter

## Files

- `analyze_model_differences.py`: Main analysis script that loads checkpoints and computes differences
- `run_model_analysis.py`: Simple wrapper script to run the analysis with predefined checkpoint paths
- `README.md`: This file with usage instructions

## Usage

### Quick Start

To analyze the differences between the two specific checkpoints mentioned in your request:

```bash
cd /home/gerigkja/RoboTwin/policy/pi0
python run_model_analysis.py
```

This will analyze:
- `checkpoints/pi0_base_aloha_robotwin_lora/place_empty_cup_demo_lim_obj_rand/15000/`
- `checkpoints/pi0_base_aloha_robotwin_lora/demo_clean_place_empty_cup/15000/`

### Custom Analysis

To analyze different checkpoints:

```bash
python analyze_model_differences.py <checkpoint1_path> <checkpoint2_path> [--output-dir <output_directory>]
```

Example:
```bash
python analyze_model_differences.py \
    checkpoints/model1/15000 \
    checkpoints/model2/15000 \
    --output-dir ./my_analysis_results
```

## Output

The analysis generates:

1. **`model_difference_report.txt`**: Comprehensive text report with:
   - Overall statistics
   - Top 20 parameters with largest differences
   - Group-level analysis (LoRA, attention, MLP, etc.)
   - Summary insights

2. **`weight_differences_overview.png`**: Overview plots showing:
   - Distribution of L2 norms
   - Distribution of cosine similarities
   - Distribution of relative differences
   - Top 20 parameters by L2 norm

3. **`group_analysis.png`**: Group-level analysis plots showing:
   - Parameter counts by group
   - Average metrics by group
   - Scatter plots comparing different metrics

## Metrics Computed

For each parameter, the analysis computes:

- **L2 Norm**: Euclidean distance between parameter vectors
- **L1 Norm**: Manhattan distance between parameter vectors
- **Cosine Similarity**: Cosine of the angle between parameter vectors
- **Relative Difference**: L2 norm normalized by average parameter magnitude
- **Max Absolute Difference**: Maximum element-wise absolute difference
- **Mean Absolute Difference**: Mean element-wise absolute difference
- **Std Absolute Difference**: Standard deviation of element-wise absolute differences

## Parameter Groups

The analysis automatically groups parameters into categories:

- **LoRA**: Parameters containing "lora" in their name
- **Attention**: Parameters containing "attn" or "attention"
- **MLP**: Parameters containing "mlp"
- **Embedding**: Parameters containing "embed"
- **Projection**: Parameters containing "proj"
- **Other**: All remaining parameters

## Requirements

The script requires the following Python packages:
- numpy
- jax
- matplotlib
- seaborn
- pandas
- flax
- orbax-checkpoint (for loading checkpoints)

Make sure you have the Pi0 environment set up with all dependencies installed.

## Example Output

After running the analysis, you'll see output like:

```
OVERALL STATISTICS
----------------------------------------
Total parameters analyzed: 1234
L2 Norm - Mean: 0.001234, Std: 0.005678
Cosine Similarity - Mean: 0.987654, Std: 0.012345
Relative Difference - Mean: 0.002345, Std: 0.008765

TOP 20 PARAMETERS WITH LARGEST DIFFERENCES
--------------------------------------------------
 1. PaliGemma/llm/lora_0/linear/value
    L2 Norm: 0.123456
    Cosine Similarity: 0.876543
    Relative Difference: 0.234567
    Max Abs Diff: 0.045678

GROUP-LEVEL ANALYSIS
------------------------------
LORA GROUP:
  Parameter count: 456
  Average L2 norm: 0.002345
  Max L2 norm: 0.123456
  Average cosine similarity: 0.987654
```

This helps identify that LoRA parameters show the largest differences, suggesting that the main differences between the models are in the LoRA adapters.
