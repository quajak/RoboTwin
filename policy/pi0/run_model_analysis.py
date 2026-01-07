#!/usr/bin/env python3
"""
Simple script to run model difference analysis between two Pi0 checkpoints.

Usage:
    python run_model_analysis.py

This script will analyze the differences between:
- checkpoints/pi0_base_aloha_robotwin_lora/place_empty_cup_demo_lim_obj_rand/15000/
- checkpoints/pi0_base_aloha_robotwin_lora/demo_clean_place_empty_cup/15000/
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run the model difference analysis."""
    
    # Define the checkpoint paths
    script_dir = Path(__file__).parent
    checkpoint1 = script_dir / "checkpoints" / "pi0_base_aloha_robotwin_lora" / "place_empty_cup_demo_lim_obj_rand" / "15000"
    checkpoint2 = script_dir / "checkpoints" / "pi0_base_aloha_robotwin_lora" / "demo_clean_place_empty_cup" / "15000"
    
    # Check if checkpoints exist
    if not checkpoint1.exists():
        print(f"Error: Checkpoint 1 not found at {checkpoint1}")
        print("Please ensure the checkpoint directory exists.")
        return 1
    
    if not checkpoint2.exists():
        print(f"Error: Checkpoint 2 not found at {checkpoint2}")
        print("Please ensure the checkpoint directory exists.")
        return 1
    
    # Set output directory
    output_dir = script_dir / "model_analysis_results"
    
    print("=" * 80)
    print("PI0 MODEL DIFFERENCE ANALYSIS")
    print("=" * 80)
    print(f"Checkpoint 1: {checkpoint1}")
    print(f"Checkpoint 2: {checkpoint2}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    # Run the analysis
    try:
        cmd = [
            sys.executable, 
            "analyze_model_differences.py",
            str(checkpoint1),
            str(checkpoint2),
            "--output-dir", str(output_dir)
        ]
        
        print("Running analysis...")
        result = subprocess.run(cmd, cwd=script_dir, check=True)
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Results saved to: {output_dir}")
        print("\nGenerated files:")
        print(f"  - model_difference_report.txt (detailed text report)")
        print(f"  - weight_differences_overview.png (overview plots)")
        print(f"  - group_analysis.png (group-level analysis)")
        print("\nTo view the results:")
        print(f"  - Read the report: cat {output_dir}/model_difference_report.txt")
        print(f"  - View plots: open {output_dir}/*.png")
        
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"Error running analysis: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
