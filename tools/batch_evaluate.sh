#!/bin/bash
"""
Batch evaluation script for all recent experiments with optimized thresholds.
"""

# Configuration
EXPERIMENTS_DIR="./experiments"
TEST_DATA="data/processed/comparison_dataset_clean.csv"
GRID_METRICS_FILE="grid_metrics.csv"
N_LAST_EXPERIMENTS=12
RESULTS_DIR="./batch_evaluation_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${RESULTS_DIR}/${TIMESTAMP}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting Batch Model Evaluation${NC}"
echo "=================================================="
echo "📁 Experiments dir: ${EXPERIMENTS_DIR}"
echo "📊 Test data: ${TEST_DATA}"
echo "📈 Grid metrics: ${GRID_METRICS_FILE}"
echo "🔢 Evaluating last ${N_LAST_EXPERIMENTS} experiments"
echo "📂 Output dir: ${OUTPUT_DIR}"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Check if grid metrics file exists
if [[ ! -f "${GRID_METRICS_FILE}" ]]; then
    echo -e "${RED}❌ Grid metrics file not found: ${GRID_METRICS_FILE}${NC}"
    exit 1
fi

# Check if test data exists
if [[ ! -f "${TEST_DATA}" ]]; then
    echo -e "${RED}❌ Test data file not found: ${TEST_DATA}${NC}"
    exit 1
fi

# Get the last N experiments from grid metrics
echo -e "${YELLOW}🔍 Reading experiment list from ${GRID_METRICS_FILE}...${NC}"

# Create temporary Python script to parse CSV and get experiment folders
cat > "${OUTPUT_DIR}/parse_experiments.py" << 'EOF'
import pandas as pd
import sys
import os

# Read grid metrics
df = pd.read_csv(sys.argv[1])

# Get last N experiments
n_last = int(sys.argv[2])
experiments_dir = sys.argv[3]

last_experiments = df.tail(n_last)

print(f"Found {len(last_experiments)} experiments to evaluate:")
print()

# Write experiment info to file
results = []
for _, row in last_experiments.iterrows():
    exp_id = row['ID']
    folder = row['Folder']
    threshold = row['Threshold']
    f1 = row['F1']
    accuracy = row['Accuracy']
    
    # Full path to experiment
    exp_path = os.path.join(experiments_dir, folder)
    
    if os.path.exists(exp_path):
        model_path = os.path.join(exp_path, "models", "model.pth")
        preprocessor_path = os.path.join(exp_path, "preprocessor.pkl")
        
        if os.path.exists(model_path) and os.path.exists(preprocessor_path):
            results.append({
                'id': exp_id,
                'folder': folder,
                'threshold': threshold,
                'f1': f1,
                'accuracy': accuracy,
                'model_path': model_path,
                'preprocessor_path': preprocessor_path,
                'exp_path': exp_path
            })
            print(f"✅ {exp_id}: {folder} (threshold={threshold:.3f}, F1={f1:.4f})")
        else:
            print(f"❌ {exp_id}: Missing model files in {folder}")
    else:
        print(f"❌ {exp_id}: Folder not found: {folder}")

print(f"\nValid experiments: {len(results)}")

# Save experiment list
import json
with open(sys.argv[4], 'w') as f:
    json.dump(results, f, indent=2)
EOF

# Parse experiments
python "${OUTPUT_DIR}/parse_experiments.py" "${GRID_METRICS_FILE}" "${N_LAST_EXPERIMENTS}" "${EXPERIMENTS_DIR}" "${OUTPUT_DIR}/experiments_list.json"

# Check if experiments were found
if [[ ! -f "${OUTPUT_DIR}/experiments_list.json" ]]; then
    echo -e "${RED}❌ Failed to parse experiments${NC}"
    exit 1
fi

# Count valid experiments
VALID_COUNT=$(python -c "import json; data=json.load(open('${OUTPUT_DIR}/experiments_list.json')); print(len(data))")

if [[ "${VALID_COUNT}" -eq 0 ]]; then
    echo -e "${RED}❌ No valid experiments found${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}📋 Found ${VALID_COUNT} valid experiments to evaluate${NC}"
echo ""

# Create evaluation script
cat > "${OUTPUT_DIR}/evaluate_single.py" << 'EOF'
import json
import sys
import os
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path

def evaluate_with_custom_threshold(model_path, preprocessor_path, test_data, threshold, output_dir):
    """Evaluate model with custom threshold."""
    
    # Create evaluation command
    eval_cmd = [
        "python", "scripts/evaluate_model.py",
        "--model", model_path,
        "--preprocessor", preprocessor_path, 
        "--test_data", test_data,
        "--output_dir", output_dir
    ]
    
    try:
        # Run evaluation
        result = subprocess.run(eval_cmd, capture_output=True, text=True, check=True)
        
        # Load results
        results_file = os.path.join(output_dir, "evaluation_results.json")
        predictions_file = os.path.join(output_dir, "detailed_predictions.csv")
        
        if not os.path.exists(results_file):
            return None, f"Results file not found: {results_file}"
            
        with open(results_file, 'r') as f:
            metrics = json.load(f)
        
        # Apply custom threshold if we have detailed predictions
        if os.path.exists(predictions_file) and threshold != 0.5:
            pred_df = pd.read_csv(predictions_file)
            
            # Recalculate with custom threshold
            y_true = (pred_df['true_gender'] == 'W').astype(int)
            y_prob = pred_df['prob'].values
            y_pred_custom = (y_prob >= threshold).astype(int)
            
            # Recalculate metrics
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
            
            accuracy = accuracy_score(y_true, y_pred_custom)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_custom, average='binary')
            
            # Bias metrics
            cm = confusion_matrix(y_true, y_pred_custom)
            tn, fp, fn, tp = cm.ravel()
            
            male_error_rate = fp / (tn + fp) if (tn + fp) > 0 else 0
            female_error_rate = fn / (tp + fn) if (tp + fn) > 0 else 0
            bias_ratio = male_error_rate / female_error_rate if female_error_rate > 0 else float('inf')
            
            # Update metrics with custom threshold results
            metrics.update({
                'threshold_used': float(threshold),
                'accuracy_custom': float(accuracy),
                'precision_custom': float(precision), 
                'recall_custom': float(recall),
                'f1_custom': float(f1),
                'bias_ratio_custom': float(bias_ratio),
                'male_error_rate_custom': float(male_error_rate),
                'female_error_rate_custom': float(female_error_rate)
            })
        
        return metrics, None
        
    except subprocess.CalledProcessError as e:
        return None, f"Evaluation failed: {e.stderr}"
    except Exception as e:
        return None, f"Error: {str(e)}"

if __name__ == "__main__":
    exp_info = json.loads(sys.argv[1])
    test_data = sys.argv[2]
    output_base = sys.argv[3]
    
    exp_id = exp_info['id']
    threshold = exp_info['threshold']
    model_path = exp_info['model_path']
    preprocessor_path = exp_info['preprocessor_path']
    
    # Create output directory for this experiment
    exp_output_dir = os.path.join(output_base, f"eval_{exp_id}")
    os.makedirs(exp_output_dir, exist_ok=True)
    
    print(f"Evaluating {exp_id} with threshold {threshold:.3f}...")
    
    # Run evaluation
    metrics, error = evaluate_with_custom_threshold(
        model_path, preprocessor_path, test_data, threshold, exp_output_dir
    )
    
    if error:
        print(f"❌ Error: {error}")
        result = {"id": exp_id, "error": error}
    else:
        print(f"✅ Success: F1={metrics.get('f1_custom', metrics['f1']):.4f}, Bias={metrics.get('bias_ratio_custom', metrics['bias_ratio']):.4f}")
        result = {"id": exp_id, "metrics": metrics, "threshold": threshold}
    
    # Save individual result
    result_file = os.path.join(exp_output_dir, "evaluation_result.json")
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
EOF

# Initialize results tracking
echo "[]" > "${OUTPUT_DIR}/all_results.json"
SUCCESSFUL_EVALS=0
TOTAL_EVALS=0

# Evaluate each experiment
echo -e "${BLUE}🔄 Starting individual evaluations...${NC}"
echo ""

while IFS= read -r exp_json; do
    if [[ -n "$exp_json" && "$exp_json" != "[]" ]]; then
        TOTAL_EVALS=$((TOTAL_EVALS + 1))
        
        # Extract experiment ID for progress
        EXP_ID=$(echo "$exp_json" | python -c "import json, sys; print(json.loads(sys.stdin.read())['id'])")
        
        echo -e "${YELLOW}📊 Evaluating experiment ${TOTAL_EVALS}/${VALID_COUNT}: ${EXP_ID}${NC}"
        
        # Run evaluation
        python "${OUTPUT_DIR}/evaluate_single.py" "$exp_json" "$TEST_DATA" "$OUTPUT_DIR"
        
        if [[ $? -eq 0 ]]; then
            SUCCESSFUL_EVALS=$((SUCCESSFUL_EVALS + 1))
            echo -e "${GREEN}   ✅ Completed${NC}"
        else
            echo -e "${RED}   ❌ Failed${NC}"
        fi
        echo ""
    fi
done < <(python -c "
import json
with open('${OUTPUT_DIR}/experiments_list.json', 'r') as f:
    experiments = json.load(f)
for exp in experiments:
    print(json.dumps(exp))
")

echo ""
echo -e "${BLUE}📊 Collecting and summarizing results...${NC}"

# Create summary script
cat > "${OUTPUT_DIR}/create_summary.py" << 'EOF'
import json
import pandas as pd
import os
import sys
from pathlib import Path

def collect_results(results_dir):
    """Collect all evaluation results."""
    results = []
    
    # Find all evaluation result files
    for eval_dir in Path(results_dir).glob("eval_*"):
        result_file = eval_dir / "evaluation_result.json"
        
        if result_file.exists():
            try:
                with open(result_file, 'r') as f:
                    result = json.load(f)
                
                if 'error' not in result and 'metrics' in result:
                    exp_id = result['id']
                    metrics = result['metrics']
                    threshold = result['threshold']
                    
                    # Extract key metrics
                    row = {
                        'experiment_id': exp_id,
                        'threshold': threshold,
                        'accuracy': metrics.get('accuracy_custom', metrics.get('accuracy')),
                        'f1': metrics.get('f1_custom', metrics.get('f1')),
                        'precision': metrics.get('precision_custom', metrics.get('precision')),
                        'recall': metrics.get('recall_custom', metrics.get('recall')),
                        'bias_ratio': metrics.get('bias_ratio_custom', metrics.get('bias_ratio')),
                        'male_error_rate': metrics.get('male_error_rate_custom', metrics.get('male_error_rate')),
                        'female_error_rate': metrics.get('female_error_rate_custom', metrics.get('female_error_rate')),
                        'total_samples': metrics.get('total_samples', 0),
                        'male_samples': metrics.get('male_samples', 0),
                        'female_samples': metrics.get('female_samples', 0)
                    }
                    
                    results.append(row)
                    
            except Exception as e:
                print(f"Error processing {result_file}: {e}")
    
    return results

def create_summary_report(results, output_dir):
    """Create comprehensive summary report."""
    
    if not results:
        print("No results to summarize")
        return
    
    df = pd.DataFrame(results)
    
    # Sort by F1 score descending
    df = df.sort_values('f1', ascending=False)
    
    print(f"\n{'='*80}")
    print(f"📋 BATCH EVALUATION SUMMARY ({len(results)} experiments)")
    print(f"{'='*80}")
    
    print(f"\n🏆 TOP 5 EXPERIMENTS BY F1 SCORE:")
    print("-" * 60)
    top5 = df.head(5)
    for _, row in top5.iterrows():
        print(f"{row['experiment_id']:>8}: F1={row['f1']:.4f}, Acc={row['accuracy']:.4f}, Bias={row['bias_ratio']:.4f}")
    
    print(f"\n⚖️  TOP 5 EXPERIMENTS BY BIAS RATIO (closest to 1.0):")
    print("-" * 60)
    df['bias_distance'] = abs(df['bias_ratio'] - 1.0)
    fair5 = df.sort_values('bias_distance').head(5)
    for _, row in fair5.iterrows():
        print(f"{row['experiment_id']:>8}: Bias={row['bias_ratio']:.4f}, F1={row['f1']:.4f}, Acc={row['accuracy']:.4f}")
    
    print(f"\n📊 OVERALL STATISTICS:")
    print("-" * 40)
    print(f"Best F1 Score:      {df['f1'].max():.4f}")
    print(f"Best Accuracy:      {df['accuracy'].max():.4f}")
    print(f"Best Bias Ratio:    {df.loc[df['bias_distance'].idxmin(), 'bias_ratio']:.4f}")
    print(f"Mean F1 Score:      {df['f1'].mean():.4f} ± {df['f1'].std():.4f}")
    print(f"Mean Bias Ratio:    {df['bias_ratio'].mean():.4f} ± {df['bias_ratio'].std():.4f}")
    
    # Find Pareto-optimal solutions (good F1 and good bias)
    print(f"\n🎯 PARETO-OPTIMAL EXPERIMENTS (F1 > {df['f1'].quantile(0.7):.4f} AND Bias ratio 0.85-1.15):")
    print("-" * 70)
    
    pareto_mask = (df['f1'] > df['f1'].quantile(0.7)) & (df['bias_ratio'] >= 0.85) & (df['bias_ratio'] <= 1.15)
    pareto_experiments = df[pareto_mask].sort_values('f1', ascending=False)
    
    if len(pareto_experiments) > 0:
        for _, row in pareto_experiments.iterrows():
            print(f"{row['experiment_id']:>8}: F1={row['f1']:.4f}, Acc={row['accuracy']:.4f}, Bias={row['bias_ratio']:.4f} ⭐")
    else:
        print("No experiments meet both performance and fairness criteria")
        # Show best compromise
        print("\n🤝 BEST COMPROMISE (weighted score):")
        df['composite_score'] = df['f1'] * (1 / (1 + df['bias_distance']))
        best_compromise = df.loc[df['composite_score'].idxmax()]
        print(f"{best_compromise['experiment_id']:>8}: F1={best_compromise['f1']:.4f}, Bias={best_compromise['bias_ratio']:.4f}")
    
    # Save detailed CSV
    csv_file = os.path.join(output_dir, "batch_evaluation_summary.csv")
    df.to_csv(csv_file, index=False)
    print(f"\n💾 Detailed results saved to: {csv_file}")
    
    # Save JSON summary
    summary = {
        'timestamp': sys.argv[2] if len(sys.argv) > 2 else '',
        'total_experiments': len(results),
        'best_f1': float(df['f1'].max()),
        'best_accuracy': float(df['accuracy'].max()),
        'best_bias_ratio': float(df.loc[df['bias_distance'].idxmin(), 'bias_ratio']),
        'mean_f1': float(df['f1'].mean()),
        'mean_bias_ratio': float(df['bias_ratio'].mean()),
        'pareto_optimal': pareto_experiments.to_dict('records') if len(pareto_experiments) > 0 else [],
        'top_5_f1': top5.to_dict('records'),
        'top_5_fairness': fair5.to_dict('records')
    }
    
    json_file = os.path.join(output_dir, "batch_evaluation_summary.json")
    with open(json_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📄 Summary JSON saved to: {json_file}")

if __name__ == "__main__":
    results_dir = sys.argv[1]
    results = collect_results(results_dir)
    create_summary_report(results, results_dir)
EOF

# Create summary
python "${OUTPUT_DIR}/create_summary.py" "${OUTPUT_DIR}" "${TIMESTAMP}"

echo ""
echo -e "${GREEN}🎉 Batch evaluation completed!${NC}"
echo "=================================="
echo -e "📊 Evaluated: ${SUCCESSFUL_EVALS}/${VALID_COUNT} experiments"
echo -e "📁 Results directory: ${OUTPUT_DIR}"
echo -e "📋 Summary: ${OUTPUT_DIR}/batch_evaluation_summary.csv"
echo -e "📄 JSON Summary: ${OUTPUT_DIR}/batch_evaluation_summary.json"
echo ""

# Clean up temporary files
rm -f "${OUTPUT_DIR}/parse_experiments.py"
rm -f "${OUTPUT_DIR}/evaluate_single.py" 
rm -f "${OUTPUT_DIR}/create_summary.py"

echo -e "${BLUE}✨ Done! Check the summary files for detailed results.${NC}"
