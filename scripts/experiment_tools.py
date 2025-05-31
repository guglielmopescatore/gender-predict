#!/usr/bin/env python3
"""
Experiment analysis and comparison tools.
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gender_predict.experiments import (
    compare_experiments, compare_bias_metrics, generate_full_report, compare_learning_curves
)

def main():
    parser = argparse.ArgumentParser(description="Experiment analysis tools")
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare experiments')
    compare_parser.add_argument('--base_dir', default='.', help='Base directory')
    compare_parser.add_argument('--metric', default='test_accuracy', help='Metric to compare')
    compare_parser.add_argument('--round', type=int, help='Filter by round')
    compare_parser.add_argument('--output', help='Output file for plot')
    
    # Bias command
    bias_parser = subparsers.add_parser('bias', help='Compare bias metrics')
    bias_parser.add_argument('--base_dir', default='.', help='Base directory')
    bias_parser.add_argument('--round', type=int, help='Filter by round')
    bias_parser.add_argument('--output', help='Output file for plot')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate reports')
    report_parser.add_argument('--base_dir', default='.', help='Base directory')
    report_parser.add_argument('--output', help='Output file for report')
    
    args = parser.parse_args()
    
    if args.command == 'compare':
        filter_dict = {}
        if args.round is not None:
            filter_dict['round'] = args.round
        
        compare_experiments(
            base_dir=args.base_dir,
            filter_dict=filter_dict,
            metric=args.metric,
            save_path=args.output
        )
    
    elif args.command == 'bias':
        filter_dict = {}
        if args.round is not None:
            filter_dict['round'] = args.round
        
        compare_bias_metrics(
            base_dir=args.base_dir,
            filter_dict=filter_dict,
            save_path=args.output
        )
    
    elif args.command == 'report':
        generate_full_report(
            base_dir=args.base_dir,
            output_path=args.output
        )
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
