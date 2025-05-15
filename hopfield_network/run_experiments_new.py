import os
import numpy as np
import matplotlib.pyplot as plt
from hopfield import HopfieldNetwork
from pbm_utils import (
    load_patterns,
    corrupt_flip,
    corrupt_crop,
    display_patterns,
    display_sequence,
    save_patterns_to_pbm,
    write_pbm
)

def test_recall(network, patterns, pattern_idx=0, results_dir="hopfield_network/results"):
    """
    Test recall of a pattern with noise and cropping
    
    Args:
        network: Trained Hopfield Network
        patterns: Array of training patterns
        pattern_idx: Index of the pattern to test
        results_dir: Directory to save results
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Select pattern to test
    original = patterns[pattern_idx]
    
    # Create corrupted versions
    flipped = corrupt_flip(original, prob=0.3)
    cropped = corrupt_crop(original)
    
    print("\nTesting pattern recall with noise (30% pixels flipped)...")
    
    # Test asynchronous update
    print("Running asynchronous update on flipped pattern...")
    async_flipped, async_iters = network.update_async(flipped)
    
    # Test synchronous update
    print("Running synchronous update on flipped pattern...")
    sync_flipped, sync_iters = network.update_sync(flipped)
    
    # Calculate similarities
    async_sim = network.measure_similarity(async_flipped, original)
    sync_sim = network.measure_similarity(sync_flipped, original)
    
    # Check if patterns match originals
    async_match, async_idx, _ = network.find_best_match(async_flipped, patterns)
    sync_match, sync_idx, _ = network.find_best_match(sync_flipped, patterns)
    
    # Print results
    print(f"  Async update: {async_iters} iterations, similarity: {async_sim:.4f}")
    print(f"  - Matched to pattern {async_idx}, correct: {async_idx == pattern_idx}")
    print(f"  Sync update: {sync_iters} iterations, similarity: {sync_sim:.4f}")
    print(f"  - Matched to pattern {sync_idx}, correct: {sync_idx == pattern_idx}")
    
    # Display and save results
    display_sequence(
        [original, flipped, async_flipped, sync_flipped],
        titles=[
            "Original", 
            "Flipped (30%)", 
            f"Async ({async_iters} iter, sim={async_sim:.2f})",
            f"Sync ({sync_iters} iter, sim={sync_sim:.2f})"
        ],
        save_path=os.path.join(results_dir, "recall_flipped.png")
    )
    
    # Save PBM files
    write_pbm(os.path.join(results_dir, "original.pbm"), original)
    write_pbm(os.path.join(results_dir, "flipped.pbm"), flipped)
    write_pbm(os.path.join(results_dir, "async_flipped.pbm"), async_flipped)
    write_pbm(os.path.join(results_dir, "sync_flipped.pbm"), sync_flipped)
    
    print("\nTesting pattern recall with cropping...")
    
    # Test asynchronous update on cropped pattern
    print("Running asynchronous update on cropped pattern...")
    async_cropped, async_iters = network.update_async(cropped)
    
    # Test synchronous update on cropped pattern
    print("Running synchronous update on cropped pattern...")
    sync_cropped, sync_iters = network.update_sync(cropped)
    
    # Calculate similarities
    async_sim = network.measure_similarity(async_cropped, original)
    sync_sim = network.measure_similarity(sync_cropped, original)
    
    # Check if patterns match originals
    async_match, async_idx, _ = network.find_best_match(async_cropped, patterns)
    sync_match, sync_idx, _ = network.find_best_match(sync_cropped, patterns)
    
    # Print results
    print(f"  Async update: {async_iters} iterations, similarity: {async_sim:.4f}")
    print(f"  - Matched to pattern {async_idx}, correct: {async_idx == pattern_idx}")
    print(f"  Sync update: {sync_iters} iterations, similarity: {sync_sim:.4f}")
    print(f"  - Matched to pattern {sync_idx}, correct: {sync_idx == pattern_idx}")
    
    # Display and save results
    display_sequence(
        [original, cropped, async_cropped, sync_cropped],
        titles=[
            "Original", 
            "Cropped", 
            f"Async ({async_iters} iter, sim={async_sim:.2f})",
            f"Sync ({sync_iters} iter, sim={sync_sim:.2f})"
        ],
        save_path=os.path.join(results_dir, "recall_cropped.png")
    )
    
    # Save PBM files
    write_pbm(os.path.join(results_dir, "cropped.pbm"), cropped)
    write_pbm(os.path.join(results_dir, "async_cropped.pbm"), async_cropped)
    write_pbm(os.path.join(results_dir, "sync_cropped.pbm"), sync_cropped)
    
def test_corruption_levels(network, patterns, pattern_idx=0, results_dir="hopfield_network/results"):
    """
    Test pattern recall with different levels of corruption
    
    Args:
        network: Trained Hopfield Network
        patterns: Array of training patterns
        pattern_idx: Index of the pattern to test
        results_dir: Directory to save results
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Select pattern to test
    original = patterns[pattern_idx]
    
    # Corruption probabilities
    probs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    n_trials = 20
    
    # Results storage
    correct_recalls = []
    iterations_data = [[] for _ in probs]
    similarity_data = [[] for _ in probs]
    
    print("\nTesting recovery at different corruption levels...")
    
    for i, p in enumerate(probs):
        print(f"Testing corruption probability p = {p:.1f}...")
        
        correct = 0
        for trial in range(n_trials):
            # Create corrupted pattern
            corrupted = corrupt_flip(original, prob=p)
            
            # Run synchronous update
            recalled, iters = network.update_sync(corrupted)
            
            # Check if recall is correct
            match, match_idx, similarity = network.find_best_match(recalled, patterns)
            
            # Record results
            similarity_data[i].append(similarity)
            if match and match_idx == pattern_idx:
                correct += 1
                iterations_data[i].append(iters)
                
        # Calculate success rate
        success_rate = correct / n_trials
        correct_recalls.append(success_rate)
        
        # Print results
        avg_iters = np.mean(iterations_data[i]) if iterations_data[i] else 0
        avg_sim = np.mean(similarity_data[i])
        print(f"  Success rate: {success_rate:.2f}, Avg iterations: {avg_iters:.1f}, Avg similarity: {avg_sim:.4f}")
    
    # Plot success rates
    plt.figure(figsize=(10, 6))
    plt.bar(probs, correct_recalls, width=0.07)
    plt.xlabel("Corruption Probability")
    plt.ylabel("Fraction of Correct Recalls")
    plt.title("Network Performance vs Corruption Level")
    plt.ylim(0, 1.1)
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(results_dir, "corruption_performance.png"))
    plt.close()
    
    # Plot iterations for successful recalls
    plt.figure(figsize=(10, 6))
    plt.boxplot([data for data in iterations_data if data])
    plt.xlabel("Corruption Probability")
    plt.ylabel("Number of Iterations")
    plt.title("Update Iterations for Successful Recalls")
    plt.xticks(range(1, len(probs) + 1), [str(p) for p in probs])
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(results_dir, "corruption_iterations.png"))
    plt.close()
    
    # Save results
    np.savez(
        os.path.join(results_dir, "corruption_data.npz"),
        probs=probs,
        correct_recalls=correct_recalls,
        iterations_data=[np.array(data) for data in iterations_data],
        similarity_data=[np.array(data) for data in similarity_data]
    )

def main():
    """Main function to run Hopfield Network experiments"""
    # Paths
    results_dir = "hopfield_network/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load or generate patterns
    print("Generating patterns...")
    n_patterns = 10  # Use fewer patterns for better capacity
    patterns = load_patterns(None, n_patterns=n_patterns)
    
    # Display patterns
    print(f"Generated {n_patterns} patterns")
    display_patterns(
        patterns, 
        rows=2, cols=5,
        title="Training Patterns",
        save_path=os.path.join(results_dir, "patterns.png")
    )
    
    # Train network
    print("\nTraining Hopfield Network...")
    network = HopfieldNetwork(256)  # 16x16 = 256 neurons
    network.train(patterns)
    
    # Test pattern recall
    test_recall(network, patterns, pattern_idx=0, results_dir=results_dir)
    
    # Test different corruption levels
    test_corruption_levels(network, patterns, pattern_idx=0, results_dir=results_dir)
    
    print("\nAll experiments completed! Results saved to:", results_dir)

if __name__ == "__main__":
    main()
