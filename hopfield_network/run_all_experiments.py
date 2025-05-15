#!/usr/bin/env python3
import os
import sys
import subprocess

def print_section(title):
    """Print a section header"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60 + "\n")

def run_script(script_path, description):
    """Run a Python script and return its exit code"""
    print(f"Running: {description}...")
    cmd = [sys.executable, script_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print output
    if result.stdout:
        print(f"\nOutput:\n{result.stdout}")
    
    # Check for errors
    if result.returncode != 0:
        print(f"\nErrors:\n{result.stderr}")
        print(f"\nExecution failed with code {result.returncode}")
        return False
    
    print(f"\nExecution completed successfully.\n")
    return True

def main():
    """Run all experiments"""
    # Print header
    print_section("HOPFIELD NETWORK - RUNNING ALL EXPERIMENTS")
    
    # Get the base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Base directory: {base_dir}")
    
    # Define scripts to run
    scripts = [
        ("verify_hopfield.py", "Basic verification test"),
        ("demo.py", "Hopfield Network demonstration"),
        ("run_final_experiments.py", "Comprehensive experiments")
    ]
    
    # Run each script
    all_successful = True
    for script_name, description in scripts:
        print_section(description.upper())
        script_path = os.path.join(base_dir, script_name)
        success = run_script(script_path, description)
        if not success:
            all_successful = False
    
    # Print summary
    print_section("EXPERIMENT SUMMARY")
    
    if all_successful:
        print("All experiments completed successfully.")
        print("\nResults are available in the following directories:")
        print("- Final experiments: final_results/")
        print("- Demonstration: demo_output/")
    else:
        print("Some experiments failed. See output above for details.")
    
    print("\nTo analyze the results, check the following files:")
    print("- EXPERIMENT_SUMMARY.md: Summary of experimental results")
    print("- FINAL_REPORT.md: Comprehensive project report")
    print("- README.md: Project documentation")

if __name__ == "__main__":
    main()
