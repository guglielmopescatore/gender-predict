#!/usr/bin/env python3
"""
Find ultra-fair threshold with bias ratio as close to 1.0 as possible.
For rigorous academic standards where even 4.6% bias is too much.
"""

import pandas as pd
import numpy as np
import json

def find_ultra_fair_threshold(detailed_csv, max_bias_deviation=0.02):
    """
    Find threshold with bias ratio closest to 1.0.
    
    Args:
        detailed_csv: CSV with threshold analysis
        max_bias_deviation: Maximum allowed deviation from 1.0 (e.g., 0.02 = ±2%)
    """
    
    print(f"🔬 Ultra-Fair Threshold Search")
    print(f"=" * 40)
    print(f"📏 Max bias deviation: ±{max_bias_deviation*100:.1f}%")
    
    # Load data
    df = pd.read_csv(detailed_csv)
    
    # Calculate bias distance from 1.0
    df['bias_distance'] = abs(df['bias_ratio'] - 1.0)
    
    # Find ultra-fair candidates (within max deviation)
    ultra_fair = df[df['bias_distance'] <= max_bias_deviation].copy()
    
    print(f"📊 Analysis Results:")
    print(f"   Total thresholds tested: {len(df)}")
    print(f"   Ultra-fair candidates (±{max_bias_deviation*100:.1f}%): {len(ultra_fair)}")
    
    if len(ultra_fair) == 0:
        print(f"❌ NO ULTRA-FAIR THRESHOLDS FOUND!")
        print(f"   Expanding search to ±{max_bias_deviation*2*100:.1f}%...")
        
        # Expand search
        expanded_fair = df[df['bias_distance'] <= max_bias_deviation*2].copy()
        
        if len(expanded_fair) == 0:
            # Find the fairest possible
            fairest_idx = df['bias_distance'].idxmin()
            fairest = df.loc[fairest_idx]
            
            print(f"📊 FAIREST POSSIBLE THRESHOLD:")
            print(f"   Threshold: {fairest['threshold']:.3f}")
            print(f"   Bias ratio: {fairest['bias_ratio']:.4f}")
            print(f"   Bias deviation: {fairest['bias_distance']*100:.2f}%")
            print(f"   F1 score: {fairest['f1']:.4f}")
            
            return {
                'status': 'limited_fairness',
                'threshold': fairest['threshold'],
                'metrics': fairest.to_dict(),
                'bias_deviation_percent': fairest['bias_distance']*100,
                'note': f'Best possible bias deviation: {fairest["bias_distance"]*100:.2f}%'
            }
        else:
            ultra_fair = expanded_fair
            max_bias_deviation = max_bias_deviation * 2
            print(f"   Found {len(ultra_fair)} candidates with ±{max_bias_deviation*100:.1f}%")
    
    # Among ultra-fair candidates, find best F1
    best_ultra_fair = ultra_fair.loc[ultra_fair['f1'].idxmax()]
    
    print(f"\n🎯 ULTRA-FAIR THRESHOLD:")
    print(f"   Threshold: {best_ultra_fair['threshold']:.3f}")
    print(f"   F1 Score: {best_ultra_fair['f1']:.4f}")
    print(f"   Accuracy: {best_ultra_fair['accuracy']:.4f}")
    print(f"   Bias Ratio: {best_ultra_fair['bias_ratio']:.4f}")
    print(f"   Bias Deviation: {best_ultra_fair['bias_distance']*100:.2f}%")
    print(f"   Male Error Rate: {best_ultra_fair['male_error_rate']:.4f}")
    print(f"   Female Error Rate: {best_ultra_fair['female_error_rate']:.4f}")
    
    # Academic assessment
    bias_dev_pct = best_ultra_fair['bias_distance'] * 100
    
    print(f"\n🎓 Ultra-Rigorous Academic Assessment:")
    
    if bias_dev_pct <= 1.0:
        bias_grade = "PERFECT"
        print(f"   Bias Fairness: {bias_grade} (deviation: {bias_dev_pct:.2f}%)")
    elif bias_dev_pct <= 2.0:
        bias_grade = "EXCELLENT"
        print(f"   Bias Fairness: {bias_grade} (deviation: {bias_dev_pct:.2f}%)")
    elif bias_dev_pct <= 3.0:
        bias_grade = "VERY GOOD"
        print(f"   Bias Fairness: {bias_grade} (deviation: {bias_dev_pct:.2f}%)")
    else:
        bias_grade = "MARGINAL"
        print(f"   Bias Fairness: {bias_grade} (deviation: {bias_dev_pct:.2f}%)")
    
    # F1 assessment
    f1_gap = 0.903 - best_ultra_fair['f1']
    print(f"   F1 Gap to Target: {f1_gap:.4f}")
    
    # Publication readiness for ultra-rigorous standards
    ultra_ready = bias_dev_pct <= 2.0 and f1_gap <= 0.015
    
    if ultra_ready:
        print(f"   📚 ULTRA-RIGOROUS READY: Meets highest academic fairness standards")
    else:
        print(f"   📋 May face scrutiny in ultra-rigorous venues")
    
    # Compare with other approaches
    print(f"\n📊 Comparison with Previous Results:")
    
    # Academic-optimal (previous result)
    academic_bias = 1.046
    academic_f1 = 0.898
    
    print(f"                    Ultra-Fair    Academic-Opt    Difference")
    print(f"   Threshold:       {best_ultra_fair['threshold']:.3f}         0.467         {best_ultra_fair['threshold'] - 0.467:+.3f}")
    print(f"   F1 Score:        {best_ultra_fair['f1']:.4f}       0.8980        {best_ultra_fair['f1'] - academic_f1:+.4f}")
    print(f"   Bias Ratio:      {best_ultra_fair['bias_ratio']:.4f}       {academic_bias:.4f}        {best_ultra_fair['bias_ratio'] - academic_bias:+.4f}")
    print(f"   Bias Deviation:  {bias_dev_pct:.2f}%         4.60%         {bias_dev_pct - 4.6:+.2f}%")
    
    # Recommendation
    print(f"\n💡 Recommendation:")
    if bias_dev_pct <= 2.0:
        print(f"   ✅ RECOMMENDED for ultra-rigorous academic use")
        print(f"   📝 Highlight: Bias deviation only {bias_dev_pct:.2f}% (virtually unbiased)")
    elif bias_dev_pct <= 3.0:
        print(f"   ✅ ACCEPTABLE for most academic venues")
        print(f"   📝 Frame as: High-fairness approach with minimal bias")
    else:
        print(f"   ⚠️  Consider if {bias_dev_pct:.2f}% bias is acceptable for your application")
    
    return {
        'status': 'success',
        'threshold': best_ultra_fair['threshold'],
        'metrics': best_ultra_fair.to_dict(),
        'bias_deviation_percent': bias_dev_pct,
        'bias_grade': bias_grade,
        'ultra_rigorous_ready': ultra_ready,
        'comparison': {
            'vs_academic_optimal': {
                'f1_difference': best_ultra_fair['f1'] - academic_f1,
                'bias_improvement': academic_bias - best_ultra_fair['bias_ratio'],
                'bias_deviation_improvement': 4.6 - bias_dev_pct
            }
        }
    }

def main():
    detailed_csv = "clean_threshold_optimization_detailed.csv"
    
    # Try increasingly strict fairness standards
    fairness_levels = [
        (0.01, "±1% (Ultra-Rigorous)"),
        (0.02, "±2% (Very Rigorous)"), 
        (0.03, "±3% (Rigorous)"),
        (0.05, "±5% (Standard Academic)")
    ]
    
    print(f"🔬 Ultra-Fair Threshold Analysis")
    print(f"=" * 50)
    
    best_results = []
    
    for max_dev, level_name in fairness_levels:
        print(f"\n🎯 Testing {level_name}:")
        
        result = find_ultra_fair_threshold(detailed_csv, max_dev)
        best_results.append((level_name, result))
        
        if result['status'] == 'success':
            print(f"   ✅ Found threshold: {result['threshold']:.3f}")
            print(f"   📊 F1: {result['metrics']['f1']:.4f}, Bias dev: {result['bias_deviation_percent']:.2f}%")
        else:
            print(f"   ❌ No suitable threshold found")
        
        print("-" * 30)
    
    # Summary
    print(f"\n📋 SUMMARY OF FAIRNESS LEVELS:")
    for level_name, result in best_results:
        if result['status'] == 'success':
            print(f"   {level_name}: t={result['threshold']:.3f}, F1={result['metrics']['f1']:.4f}, bias_dev={result['bias_deviation_percent']:.2f}%")
        else:
            print(f"   {level_name}: Not achievable")
    
    # Save best result
    if best_results[0][1]['status'] == 'success':
        result = best_results[0][1]
        
        # Convert numpy types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            else:
                return obj
        
        json_safe_result = convert_for_json(result)
        
        with open('ultra_fair_threshold.json', 'w') as f:
            json.dump(json_safe_result, f, indent=2)
        print(f"\n💾 Ultra-fair result saved: ultra_fair_threshold.json")

if __name__ == "__main__":
    main()
