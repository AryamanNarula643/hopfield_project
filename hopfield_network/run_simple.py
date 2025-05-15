import os
import numpy as np
import random
import matplotlib.pyplot as plt
from hopfield_simple import (
    HopfieldNetwork, 
    read_pbm, 
    write_pbm, 
    corrupt_flip, 
    corrupt_crop, 
    load_patterns,
    display_patterns,
    display_sequence
)

def test_recall(network, patterns, pattern_idx=0, results_dir="hopfield_network/results"):
    """Test pattern recall with flipped and cropped versions"""
    os.makedirs(results_dir, exist_ok=True)
    
    # Select a pattern
    original = patterns[pattern_idx]
    
    # Create corrupted versions
    flip_prob = 0.3  # 30% corruption
    flipped = corrupt_flip(original, flip_prob)
    cropped = corrupt_crop(original)
    
    # Recall using asynchronous updates
    print("Running asynchronous updates on flipped pattern...")
    async_flipped, async_flipped_iters = network.update_async(flipped)
    
    print("Running asynchronous updates on cropped pattern...")
    async_cropped, async_cropped_iters = network.update_async(cropped)
    
    # Recall using synchronous updates
    print("Running synchronous updates on flipped pattern...")
    sync_flipped, sync_flipped_iters = network.update_sync(flipped)
    
    print("Running synchronous updates on cropped pattern...")
    sync_cropped, sync_cropped_iters = network.update_sync(cropped)
    
    # Compute similarities
    async_flipped_match, async_flipped_idx, async_flipped_sim = network.best_match(async_flipped, patterns)
    sync_flipped_match, sync_flipped_idx, sync_flipped_sim = network.best_match(sync_flipped, patterns)
    async_cropped_match, async_cropped_idx, async_cropped_sim = network.best_match(async_cropped, patterns)
    sync_cropped_match, sync_cropped_idx, sync_cropped_sim = network.best_match(sync_cropped, patterns)
    
    # Display results
    print(f"\nAsynchronous update (flipped): {async_flipped_iters} iterations")
    print(f"  - Match: {async_flipped_match}, Pattern: {async_flipped_idx}, Similarity: {async_flipped_sim:.4f}")
    print(f"  - Correct recall: {async_flipped_idx == pattern_idx}")
    
    print(f"\nSynchronous update (flipped): {sync_flipped_iters} iterations")
    print(f"  - Match: {sync_flipped_match}, Pattern: {sync_flipped_idx}, Similarity: {sync_flipped_sim:.4f}")
    print(f"  - Correct recall: {sync_flipped_idx == pattern_idx}")
    
    print(f"\nAsynchronous update (cropped): {async_cropped_iters} iterations")
    print(f"  - Match: {async_cropped_match}, Pattern: {async_cropped_idx}, Similarity: {async_cropped_sim:.4f}")
    print(f"  - Correct recall: {async_cropped_idx == pattern_idx}")
    
    print(f"\nSynchronous update (cropped): {sync_cropped_iters} iterations")
    print(f"  - Match: {sync_cropped_match}, Pattern: {sync_cropped_idx}, Similarity: {sync_cropped_sim:.4f}")
    print(f"  - Correct recall: {sync_cropped_idx == pattern_idx}")
    
    # Save visualizations
    display_sequence(
        [original, flipped, async_flipped, sync_flipped],
        titles=[
            "Original", 
            f"Flipped ({flip_prob:.1f})", 
            f"Async ({async_flipped_iters} iter, sim={async_flipped_sim:.2f})",
            f"Sync ({sync_flipped_iters} iter, sim={sync_flipped_sim:.2f})"
        ],
        save_path=os.path.join(results_dir, "recall_flipped.png")
    )
    
    display_sequence(
        [original, cropped, async_cropped, sync_cropped],
        titles=[
            "Original", 
            "Cropped", 
            f"Async ({async_cropped_iters} iter, sim={async_cropped_sim:.2f})",
            f"Sync ({sync_cropped_iters} iter, sim={sync_cropped_sim:.2f})"
        ],
        save_path=os.path.join(results_dir, "recall_cropped.png")
    )
    
    # Save PBM files
    write_pbm(os.path.join(results_dir, "original.pbm"), original)
    write_pbm(os.path.join(results_dir, "flipped.pbm"), flipped)
    write_pbm(os.path.join(results_dir, "cropped.pbm"), cropped)
    write_pbm(os.path.join(results_dir, "async_flipped.pbm"), async_flipped)
    write_pbm(os.path.join(results_dir, "sync_flipped.pbm"), sync_flipped)
    write_pbm(os.path.join(results_dir, "async_cropped.pbm"), async_cropped)
    write_pbm(os.path.join(results_dir, "sync_cropped.pbm"), sync_cropped)
    
    return {
        "async_flipped_correct": async_flipped_idx == pattern_idx,
        "sync_flipped_correct": sync_flipped_idx == pattern_idx,
        "async_cropped_correct": async_cropped_idx == pattern_idx,
        "sync_cropped_correct": sync_cropped_idx == pattern_idx
    }

def test_corruption_levels(network, patterns, pattern_idx=0, results_dir="hopfield_network/results"):
    """Test recovery with different corruption probabilities"""
    os.makedirs(results_dir, exist_ok=True)
    
    # Select a pattern
    original = patterns[pattern_idx]
    
    # Corruption probabilities
    probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    n_trials = 20
    
    # Results storage
    correct_convergences = []
    update_steps = [[] for _ in range(len(probs))]
    similarities = [[] for _ in range(len(probs))]
    
    for i, p in enumerate(probs):
        print(f"Testing with corruption probability p = {p:.1f}...")
        correct = 0
        
        for trial in range(n_trials):
            # Generate corrupted version
            corrupted = corrupt_flip(original, p)
            
            # Run network recall
            final_state, iterations = network.update_sync(corrupted)
            
            # Check if recalled correctly
            match, matched_idx, similarity = network.best_match(final_state, patterns)
            correct_match = matched_idx == pattern_idx if match else False
            
            if correct_match:
                correct += 1
                update_steps[i].append(iterations)
            
            similarities[i].append(similarity)
        
        correct_convergences.append(correct / n_trials)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.bar(probs, correct_convergences, width=0.07)
    plt.xlabel('Corruption Probability (p)')
    plt.ylabel('Fraction of Correct Convergences')
    plt.title('Network Performance vs Corruption Probability')
    plt.grid(alpha=0.3)
    plt.ylim(0, 1.1)
    plt.savefig(os.path.join(results_dir, "corruption_performance.png"), dpi=150)
    plt.close()
    
    # Plot average similarity
    plt.figure(figsize=(10, 6))
    avg_similarities = [np.mean(sim) for sim in similarities]
    plt.plot(probs, avg_similarities, 'o-', linewidth=2)
    plt.xlabel('Corruption Probability (p)')
    plt.ylabel('Average Similarity to Original')
    plt.title('Pattern Similarity vs Corruption Probability')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(results_dir, "corruption_similarity.png"), dpi=150)
    plt.close()
    
    # Print information
    for i, p in enumerate(probs):
        avg_steps = np.mean(update_steps[i]) if update_steps[i] else 0
        avg_sim = np.mean(similarities[i])
        print(f"p = {p:.1f}: {correct_convergences[i] * 100:.1f}% correct, " +
              f"avg steps: {avg_steps:.1f}, avg similarity: {avg_sim:.4f}")
    
    return correct_convergences, update_steps, similarities

def main():
    # Paths
    images_dir = "hopfield_network/images"
    results_dir = "hopfield_network/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load all patterns
    print("Loading patterns...")
    all_patterns = load_patterns(images_dir)
    
    # Show all patterns
    print(f"Found {len(all_patterns)} patterns")
    display_patterns(all_patterns, 
                   rows=5, cols=5, 
                   title="All Training Patterns",
                   save_path=os.path.join(results_dir, "all_patterns.png"))
    
    # Select a subset of most distinct patterns for better performance
    print("\nSelecting distinct patterns...")
    num_patterns = 5  # Using fewer patterns to improve recall performance
    pattern_subset = []
    
    # Pre-compute similarity matrix
    n_patterns = len(all_patterns)
    similarity_matrix = np.zeros((n_patterns, n_patterns))
    for i in range(n_patterns):
        for j in range(n_patterns):
            if i != j:
                dot_product = np.dot(all_patterns[i], all_patterns[j])
                norm_i = np.linalg.norm(all_patterns[i])
                norm_j = np.linalg.norm(all_patterns[j])
                similarity_matrix[i, j] = abs(dot_product / (norm_i * norm_j))
    
    # Find patterns with lowest average similarity (most distinct)
    avg_similarities = np.mean(similarity_matrix, axis=1)
    distinct_indices = np.argsort(avg_similarities)[:num_patterns]
    
    # Use the selected patterns
    patterns = all_patterns[distinct_indices]
    pattern_idx = 0  # Use the first pattern for tests
    
    print(f"Selected {len(patterns)} distinct patterns (indices: {distinct_indices})")
    display_patterns(patterns, 
                   rows=1, cols=len(patterns),
                   title="Selected Distinct Patterns", 
                   save_path=os.path.join(results_dir, "selected_patterns.png"))
    
    # Create and train the network
    print("\nTraining Hopfield Network...")
    network = HopfieldNetwork(256)  # 16x16 = 256 neurons
    network.train(patterns)
    
    # Test pattern recall
    print("\nTesting pattern recall with corrupted inputs...")
    recall_results = test_recall(network, patterns, pattern_idx, results_dir)
    
    # Test different corruption levels
    print("\nTesting performance across corruption levels...")
    convergences, steps, similarities = test_corruption_levels(
        network, patterns, pattern_idx, results_dir)
    
    print("\nAll experiments completed. Results saved to:", results_dir)

if __name__ == "__main__":
    main()
