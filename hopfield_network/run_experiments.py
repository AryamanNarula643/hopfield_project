import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

from hopfield_network import (
    HopfieldNetwork,
    read_pbm,
    write_pbm,
    flip_pixels,
    crop_image,
    display_image,
    display_image_sequence,
    save_all_patterns,
    load_all_patterns
)

def run_experiment_1(network: HopfieldNetwork, patterns: np.ndarray, 
                    memory_index: int = 0, results_dir: str = "hopfield_network/results"):
    """
    Test the network with flipped and cropped versions of a memory
    
    Args:
        network: Trained Hopfield Network
        patterns: Training patterns
        memory_index: Index of the memory to test
        results_dir: Directory to save results
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Select a memory
    memory = patterns[memory_index]
    
    # Create corrupted versions
    flipped = flip_pixels(memory, p=0.3)
    cropped = crop_image(memory)
    
    # Run asynchronous updates on flipped memory
    print("Running asynchronous updates on flipped memory...")
    final_state_async_flipped, iterations_async_flipped = network.async_update(flipped.copy())
    
    # Run synchronous updates on flipped memory
    print("Running synchronous updates on flipped memory...")
    final_state_sync_flipped, iterations_sync_flipped = network.sync_update(flipped.copy())
    
    # Run asynchronous updates on cropped memory
    print("Running asynchronous updates on cropped memory...")
    final_state_async_cropped, iterations_async_cropped = network.async_update(cropped.copy())
    
    # Run synchronous updates on cropped memory
    print("Running synchronous updates on cropped memory...")
    final_state_sync_cropped, iterations_sync_cropped = network.sync_update(cropped.copy())
    
    # Check if the final states match the original memory using pattern_match method
    match_async_flipped, idx_async_flipped, sim_async_flipped = network.pattern_match(final_state_async_flipped, patterns)
    match_sync_flipped, idx_sync_flipped, sim_sync_flipped = network.pattern_match(final_state_sync_flipped, patterns)
    match_async_cropped, idx_async_cropped, sim_async_cropped = network.pattern_match(final_state_async_cropped, patterns)
    match_sync_cropped, idx_sync_cropped, sim_sync_cropped = network.pattern_match(final_state_sync_cropped, patterns)
    
    # Check if the retrieved pattern is the correct one
    correct_async_flipped = idx_async_flipped == memory_index if match_async_flipped else False
    correct_sync_flipped = idx_sync_flipped == memory_index if match_sync_flipped else False
    correct_async_cropped = idx_async_cropped == memory_index if match_async_cropped else False
    correct_sync_cropped = idx_sync_cropped == memory_index if match_sync_cropped else False
    
    # Display and save results
    display_image_sequence(
        [memory, flipped, final_state_async_flipped, final_state_sync_flipped],
        titles=["Original", "Flipped (p=0.3)", 
                f"Async ({iterations_async_flipped} iter, sim={sim_async_flipped:.2f})",
                f"Sync ({iterations_sync_flipped} iter, sim={sim_sync_flipped:.2f})"],
        save_path=os.path.join(results_dir, "experiment1_flipped.png")
    )
    
    display_image_sequence(
        [memory, cropped, final_state_async_cropped, final_state_sync_cropped],
        titles=["Original", "Cropped", 
                f"Async ({iterations_async_cropped} iter, sim={sim_async_cropped:.2f})",
                f"Sync ({iterations_sync_cropped} iter, sim={sim_sync_cropped:.2f})"],
        save_path=os.path.join(results_dir, "experiment1_cropped.png")
    )
    
    # Save the PBM files
    write_pbm(os.path.join(results_dir, "original.pbm"), memory)
    write_pbm(os.path.join(results_dir, "flipped.pbm"), flipped)
    write_pbm(os.path.join(results_dir, "cropped.pbm"), cropped)
    write_pbm(os.path.join(results_dir, "final_async_flipped.pbm"), final_state_async_flipped)
    write_pbm(os.path.join(results_dir, "final_sync_flipped.pbm"), final_state_sync_flipped)
    write_pbm(os.path.join(results_dir, "final_async_cropped.pbm"), final_state_async_cropped)
    write_pbm(os.path.join(results_dir, "final_sync_cropped.pbm"), final_state_sync_cropped)
    
    # Print information
    print(f"Asynchronous update (flipped): {iterations_async_flipped} iterations")
    print(f"  - Similarity: {sim_async_flipped:.4f}, Matched to pattern {idx_async_flipped}, Correct: {correct_async_flipped}")
    
    print(f"Synchronous update (flipped): {iterations_sync_flipped} iterations")
    print(f"  - Similarity: {sim_sync_flipped:.4f}, Matched to pattern {idx_sync_flipped}, Correct: {correct_sync_flipped}")
    
    print(f"Asynchronous update (cropped): {iterations_async_cropped} iterations")
    print(f"  - Similarity: {sim_async_cropped:.4f}, Matched to pattern {idx_async_cropped}, Correct: {correct_async_cropped}")
    
    print(f"Synchronous update (cropped): {iterations_sync_cropped} iterations")
    print(f"  - Similarity: {sim_sync_cropped:.4f}, Matched to pattern {idx_sync_cropped}, Correct: {correct_sync_cropped}")

def run_experiment_2(network: HopfieldNetwork, patterns: np.ndarray, memory_index: int = 0, 
                   results_dir: str = "hopfield_network/results"):
    """
    Analyze network performance across corruption probabilities
    
    Args:
        network: Trained Hopfield Network
        patterns: Training patterns
        memory_index: Index of the memory to test
        results_dir: Directory to save results
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Select a memory
    memory = patterns[memory_index]
    
    # Probabilities to test
    probs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    n_trials = 20
    
    # Results storage
    correct_convergences = []
    update_steps = [[] for _ in range(len(probs))]
    similarity_values = [[] for _ in range(len(probs))]
    
    for i, p in enumerate(probs):
        print(f"Testing with p = {p}...")
        correct = 0
        
        for trial in range(n_trials):
            # Generate corrupted version
            corrupted = flip_pixels(memory, p=p)
            
            # Run synchronous updates
            final_state, iterations = network.sync_update(corrupted.copy())
            
            # Check if matched a pattern and if it's the correct one
            match_found, matched_idx, similarity = network.pattern_match(final_state, patterns)
            correct_match = matched_idx == memory_index if match_found else False
            
            if correct_match:
                correct += 1
                update_steps[i].append(iterations)
                
            similarity_values[i].append(similarity)
        
        correct_convergences.append(correct / n_trials)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.bar(probs, correct_convergences)
    plt.xlabel('Corruption Probability (p)')
    plt.ylabel('Fraction of Correct Convergences')
    plt.title('Network Performance vs Corruption Probability')
    plt.ylim(0, 1.1)  # Set y-axis to show full range from 0 to 1
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(results_dir, "experiment2_convergences.png"), dpi=150)
    plt.close()
    
    # Plot histograms of update steps
    fig, axs = plt.subplots(2, 4, figsize=(15, 8))
    axs = axs.flatten()
    
    for i, p in enumerate(probs):
        if update_steps[i]:  # Only plot if there are data points
            axs[i].hist(update_steps[i], bins=max(5, min(20, len(set(update_steps[i])))), align='mid')
            axs[i].set_title(f'p = {p}')
            axs[i].set_xlabel('Update Steps')
            axs[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "experiment2_update_steps.png"), dpi=150)
    plt.close()
    
    # Plot similarity values for each probability
    plt.figure(figsize=(10, 6))
    boxplot_data = [sim_vals for sim_vals in similarity_values]
    plt.boxplot(boxplot_data, labels=[f"{p}" for p in probs])
    plt.xlabel('Corruption Probability (p)')
    plt.ylabel('Similarity to Original Pattern')
    plt.title('Pattern Similarity vs Corruption Probability')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(results_dir, "experiment2_similarities.png"), dpi=150)
    plt.close()
    
    # Print information
    for i, p in enumerate(probs):
        avg_steps = np.mean(update_steps[i]) if update_steps[i] else 0
        avg_sim = np.mean(similarity_values[i])
        print(f"p = {p}: {correct_convergences[i] * 100:.1f}% correct, avg steps: {avg_steps:.1f}, avg similarity: {avg_sim:.4f}")
    
    # Save data for future reference
    np.savez(
        os.path.join(results_dir, "experiment2_data.npz"),
        probs=probs,
        correct_convergences=correct_convergences,
        update_steps=[np.array(steps) for steps in update_steps],
        similarities=[np.array(sims) for sims in similarity_values]
    )

def show_all_patterns(patterns: np.ndarray, results_dir: str = "hopfield_network/results"):
    """
    Create a visualization of all patterns
    
    Args:
        patterns: All patterns
        results_dir: Directory to save results
    """
    os.makedirs(results_dir, exist_ok=True)
    
    n_patterns = patterns.shape[0]
    n_cols = 5
    n_rows = (n_patterns + n_cols - 1) // n_cols
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axs = axs.flatten()
    
    for i in range(n_patterns):
        # Reshape to 2D
        img = patterns[i].reshape(16, 16)
        
        # Convert -1 to white and 1 to black for display
        img_display = np.zeros((16, 16))
        img_display[img == -1] = 1  # White
        img_display[img == 1] = 0   # Black
        
        axs[i].imshow(img_display, cmap='gray')
        axs[i].set_title(f"Pattern {i}")
        axs[i].axis('off')
    
    # Hide empty subplots
    for i in range(n_patterns, len(axs)):
        axs[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "all_patterns.png"), dpi=150)
    plt.close()

def main():
    # Paths
    images_dir = "hopfield_network/images"
    results_dir = "hopfield_network/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Check if patterns exist, if not generate them
    if not os.path.exists(images_dir) or len(os.listdir(images_dir)) < 25:
        print("Patterns not found. Generating patterns...")
        from generate_patterns import main as generate_patterns
        generate_patterns()
    
    # Load all patterns
    print("Loading patterns...")
    all_patterns = load_all_patterns(images_dir)
    
    # Show all patterns
    print("Creating visualization of all patterns...")
    show_all_patterns(all_patterns, results_dir)
    
    # Calculate similarity matrix
    print("Calculating similarity matrix between patterns...")
    n_patterns = len(all_patterns)
    similarity_matrix = np.zeros((n_patterns, n_patterns))
    
    for i in range(n_patterns):
        for j in range(n_patterns):
            if i != j:
                similarity_matrix[i, j] = abs(np.dot(all_patterns[i], all_patterns[j]) / 
                                             (np.linalg.norm(all_patterns[i]) * np.linalg.norm(all_patterns[j])))
    
    # Select a subset of the most distinct patterns
    num_distinct_patterns = 10  # Use fewer patterns for better recall
    distinct_indices = []
    remaining_indices = list(range(n_patterns))
    
    # Start with the pattern that has the lowest average similarity to all others
    avg_similarities = np.mean(similarity_matrix, axis=1)
    first_idx = np.argmin(avg_similarities)
    distinct_indices.append(first_idx)
    remaining_indices.remove(first_idx)
    
    # Iteratively add the pattern that is most distinct from those already selected
    while len(distinct_indices) < num_distinct_patterns:
        best_idx = -1
        min_max_similarity = float('inf')
        
        for idx in remaining_indices:
            max_similarity = max([similarity_matrix[idx, j] for j in distinct_indices])
            if max_similarity < min_max_similarity:
                min_max_similarity = max_similarity
                best_idx = idx
                
        if best_idx != -1:
            distinct_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        else:
            break
    
    # Use the selected distinct patterns
    patterns = all_patterns[distinct_indices]
    print(f"Selected {len(patterns)} distinct patterns (indices: {distinct_indices})")
    
    # Choose one pattern for experiments
    best_pattern_idx = 0  # Use the first pattern since it's already distinct
    print(f"Using pattern #{distinct_indices[best_pattern_idx]} for experiments")
    
    # Save selected patterns for visualization
    selected_dir = os.path.join(results_dir, "selected_patterns")
    os.makedirs(selected_dir, exist_ok=True)
    
    for i, idx in enumerate(distinct_indices):
        src_path = os.path.join(images_dir, f"pattern_{idx:02d}.pbm")
        dst_path = os.path.join(selected_dir, f"selected_{i:02d}.pbm")
        write_pbm(dst_path, all_patterns[idx])
    
    # Show selected patterns
    print("Creating visualization of selected patterns...")
    show_all_patterns(patterns, selected_dir)
    
    # Initialize and train the network
    print("Training Hopfield Network...")
    network = HopfieldNetwork(patterns[0].size)
    network.train(patterns)
    
    # Run experiments
    print("\nExperiment 1: Testing with flipped and cropped memories")
    run_experiment_1(network, patterns, memory_index=best_pattern_idx, results_dir=results_dir)
    
    print("\nExperiment 2: Analyzing performance across corruption probabilities")
    run_experiment_2(network, patterns, memory_index=best_pattern_idx, results_dir=results_dir)
    
    print("\nAll experiments completed. Results saved to:", results_dir)

if __name__ == "__main__":
    main()
