import numpy as np
import matplotlib.pyplot as plt
import os
from hopfield import HopfieldNetwork

def generate_random_pattern(size=256):
    """Generate a random binary pattern"""
    pattern = np.ones(size) * -1
    flip_indices = np.random.choice(size, size=size//2, replace=False)
    pattern[flip_indices] = 1
    return pattern

def generate_pattern_set(num_patterns=5, size=256):
    """Generate a set of random patterns"""
    patterns = []
    for _ in range(num_patterns):
        patterns.append(generate_random_pattern(size))
    return np.array(patterns)

def corrupt_pattern(pattern, noise_level=0.3):
    """Corrupt a pattern with random noise"""
    corrupted = pattern.copy()
    flip_indices = np.random.choice(
        len(pattern),
        size=int(len(pattern) * noise_level),
        replace=False
    )
    corrupted[flip_indices] *= -1
    return corrupted

def display_pattern(pattern, ax=None, title=None):
    """Display a 16x16 pattern"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
        
    display = np.ones((16, 16))
    pattern_2d = pattern.reshape(16, 16)
    display[pattern_2d == 1] = 0  # Black pixels
    
    ax.imshow(display, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    
    if title:
        ax.set_title(title)

def verify_network():
    """Run a simple verification test on the Hopfield Network"""
    print("Running Hopfield Network verification...")
    
    # Create a small pattern set
    print("Generating patterns...")
    patterns = generate_pattern_set(3)
    
    # Create and train network
    print("Training network...")
    network = HopfieldNetwork(256)
    network.train(patterns)
    
    # Test with a corrupted pattern
    print("Testing recall with corrupted pattern...")
    original = patterns[0]
    corrupted = corrupt_pattern(original, noise_level=0.3)
    
    # Recall using synchronous update
    recalled, iters = network.update_sync(corrupted)
    
    # Calculate similarity
    similarity = network.measure_similarity(original, recalled)
    print(f"Recall similarity: {similarity:.4f} after {iters} iterations")
    
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    display_pattern(original, axes[0], "Original")
    display_pattern(corrupted, axes[1], "Corrupted (30% noise)")
    display_pattern(recalled, axes[2], f"Recalled (sim={similarity:.2f})")
    
    plt.tight_layout()
    plt.savefig("/hopfield_network/final_results/verification.png")
    plt.close()
    
    print("Verification complete! Results saved to final_results/verification.png")

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs("/hopfield_network/final_results", exist_ok=True)
    
    # Run verification
    verify_network()
    