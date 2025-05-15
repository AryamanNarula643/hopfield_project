"""
Demonstration of Hopfield Network for Pattern Storage and Recall

This script provides a complete example of how to:
1. Generate patterns
2. Train a Hopfield Network
3. Test pattern recall with corruption
4. Visualize results
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from hopfield import HopfieldNetwork
from pbm_utils import (
    generate_pbm_pattern, 
    corrupt_flip, 
    corrupt_crop,
    display_patterns,
    display_sequence
)

def main():
    """Main demonstration function"""
    # Create output directory
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")
    output_dir = os.path.join(cwd, "hopfield_network/demo_output")
    print(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {os.path.exists(output_dir)}")
    
    print("=== Hopfield Network Demonstration ===")
    
    # Step 1: Generate random patterns
    print("\nStep 1: Generating patterns...")
    n_patterns = 5
    patterns = []
    for i in range(n_patterns):
        patterns.append(generate_pbm_pattern())
    patterns = np.array(patterns)
    
    # Display the patterns
    display_patterns(
        patterns, 
        rows=1, cols=5,
        title="Training Patterns",
        save_path=os.path.join(output_dir, "training_patterns.png")
    )
    
    # Step 2: Create and train the network
    print("\nStep 2: Training the Hopfield Network...")
    network = HopfieldNetwork(256)  # 16x16 = 256 neurons
    network.train(patterns)
    
    # Step 3: Test pattern recall with corruption
    print("\nStep 3: Testing pattern recall with corruption...")
    
    # Select first pattern for testing
    original = patterns[0]
    
    # Create corrupted versions
    print("  Creating corrupted patterns...")
    noise_levels = [0.1, 0.3, 0.5]
    corrupted_patterns = []
    
    for noise in noise_levels:
        corrupted = corrupt_flip(original, prob=noise)
        corrupted_patterns.append(corrupted)
    
    # Also test with cropping
    cropped = corrupt_crop(original)
    corrupted_patterns.append(cropped)
    
    # Step 4: Test network recall
    print("\nStep 4: Testing network recall...")
    recalled_patterns = []
    update_types = []
    iterations = []
    similarities = []
    
    # Test with different noise levels using synchronous updates
    for i, corrupted in enumerate(corrupted_patterns):
        if i < len(noise_levels):
            print(f"  Testing with {noise_levels[i]*100:.0f}% noise (synchronous update)...")
        else:
            print("  Testing with cropping (synchronous update)...")
            
        recalled, iters = network.update_sync(corrupted)
        similarity = network.measure_similarity(recalled, original)
        
        recalled_patterns.append(recalled)
        update_types.append("Sync")
        iterations.append(iters)
        similarities.append(similarity)
    
    # Test with 30% noise using asynchronous update
    print("  Testing with 30% noise (asynchronous update)...")
    corrupted = corrupt_flip(original, prob=0.3)
    recalled, iters = network.update_async(corrupted)
    similarity = network.measure_similarity(recalled, original)
    
    corrupted_patterns.append(corrupted)
    recalled_patterns.append(recalled)
    update_types.append("Async")
    iterations.append(iters)
    similarities.append(similarity)
    
    # Step 5: Visualize results
    print("\nStep 5: Visualizing results...")
    
    # For each experiment, display original, corrupted, and recalled
    for i in range(len(corrupted_patterns)):
        if i < len(noise_levels):
            title = f"{noise_levels[i]*100:.0f}% Noise ({update_types[i]})"
        elif i == len(noise_levels):
            title = f"Cropped ({update_types[i]})"
        else:
            title = f"30% Noise ({update_types[i]})"
            
        display_sequence(
            [original, corrupted_patterns[i], recalled_patterns[i]],
            titles=[
                "Original", 
                f"Corrupted ({title})", 
                f"Recalled (iters={iterations[i]}, sim={similarities[i]:.2f})"
            ],
            save_path=os.path.join(output_dir, f"recall_{i+1}.png")
        )
    
    print("\n=== Demonstration Complete ===")
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()
