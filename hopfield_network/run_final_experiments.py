import numpy as np
import matplotlib.pyplot as plt
import os
import random

# Define Hopfield Network for 16x16 images
class HopfieldNetwork:
    def __init__(self, size=256):
        self.size = size
        self.weights = np.zeros((size, size))
    
    def train(self, patterns):
        """Train network with patterns"""
        print(f"Training network with {len(patterns)} patterns...")
        self.weights = np.zeros((self.size, self.size))
        
        # Apply hebbian learning
        for pattern in patterns:
            pattern = pattern.reshape(self.size)
            self.weights += np.outer(pattern, pattern)
        
        # Remove self-connections
        np.fill_diagonal(self.weights, 0)
        
        # Normalize
        if len(patterns) > 0:
            self.weights /= len(patterns)
    
    def update(self, state, synchronous=True, max_iterations=100):
        """Update state until convergence"""
        state = state.copy().reshape(self.size)
        iterations = 0
        
        for i in range(max_iterations):
            old_state = state.copy()
            
            if synchronous:
                # Update all neurons at once
                h = np.dot(self.weights, state)
                state = np.sign(h)
                state[state == 0] = old_state[state == 0]
            else:
                # Update neurons one by one in random order
                indices = list(range(self.size))
                random.shuffle(indices)
                for idx in indices:
                    h = np.dot(self.weights[idx], state)
                    state[idx] = 1 if h > 0 else -1 if h < 0 else state[idx]
            
            iterations += 1
            
            # Check for convergence
            if np.array_equal(state, old_state):
                break
                
        return state, iterations
    
    def energy(self, state):
        """Calculate energy of the state"""
        state = state.reshape(self.size)
        return -0.5 * state.dot(self.weights).dot(state)
    
    def similarity(self, state1, state2):
        """Calculate cosine similarity between states"""
        state1 = state1.flatten()
        state2 = state2.flatten()
        return np.dot(state1, state2) / (np.linalg.norm(state1) * np.linalg.norm(state2))

# Image utility functions
def create_random_images(n, width=16, height=16):
    """Create random binary images (-1=white, 1=black)"""
    images = []
    for _ in range(n):
        # Create base image (all white)
        img = np.ones((height, width)) * -1
        
        # Fill with some patterns
        if random.random() < 0.5:
            # Create a border
            img[0, :] = 1  # Top
            img[-1, :] = 1  # Bottom
            img[:, 0] = 1  # Left
            img[:, -1] = 1  # Right
        else:
            # Create diagonal lines
            for i in range(min(width, height)):
                img[i, i] = 1  # Main diagonal
                if i < width-1 and i < height-1:
                    img[i, width-i-1] = 1  # Anti-diagonal
        
        # Add some random noise
        noise_indices = np.random.choice(
            width * height, 
            size=width * height // 4, 
            replace=False
        )
        img.flat[noise_indices] = 1
        
        images.append(img.copy())
    
    return images

def add_noise(image, noise_level=0.2):
    """Flip pixels randomly according to noise level"""
    corrupted = image.copy()
    indices = np.random.choice(
        corrupted.size,
        size=int(corrupted.size * noise_level),
        replace=False
    )
    corrupted.flat[indices] *= -1
    return corrupted

def crop_image(image, width=16, height=16, box_size=10):
    """Keep central area, set rest to white (-1)"""
    cropped = np.ones((height, width)) * -1
    
    # Calculate box position
    start_x = (width - box_size) // 2
    start_y = (height - box_size) // 2
    
    # Copy central area
    cropped[start_y:start_y+box_size, start_x:start_x+box_size] = \
        image[start_y:start_y+box_size, start_x:start_x+box_size]
    
    return cropped

def display_image(image, ax=None, title=None):
    """Display a binary image"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    
    # Convert -1 (white) to 1, 1 (black) to 0 for display
    display_img = np.zeros_like(image)
    display_img[image == -1] = 1
    display_img[image == 1] = 0
    
    ax.imshow(display_img, cmap='gray')
    ax.axis('off')
    
    if title:
        ax.set_title(title)

def display_images(images, titles=None, rows=1, cols=None, figsize=(15, 5)):
    """Display multiple images in a grid"""
    if cols is None:
        cols = len(images)
        
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < len(images):
            display_image(images[i], ax)
            if titles and i < len(titles):
                ax.set_title(titles[i])
        else:
            ax.axis('off')
            
    plt.tight_layout()
    return fig, axes

# Run experiments
def main():
    # Create results directory with relative path
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")
    results_dir = "final_results"
    print(f"Creating results directory: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results directory created: {os.path.exists(results_dir)}")
    
    # Generate patterns
    print("Generating patterns...")
    n_patterns = 5  # Using a small number for better recall
    patterns = create_random_images(n_patterns)
    
    # Display patterns
    print("Displaying patterns...")
    fig, _ = display_images(
        patterns,
        titles=[f"Pattern {i}" for i in range(n_patterns)],
        figsize=(15, 3)
    )
    fig.suptitle("Training Patterns", fontsize=16)
    plt.savefig(os.path.join(results_dir, "patterns.png"))
    plt.close()
    
    # Train network
    print("Training network...")
    network = HopfieldNetwork()
    network.train(patterns)
    
    # Test 1: Recall with noise
    print("\nTest 1: Pattern recall with noise")
    pattern_idx = 0
    original = patterns[pattern_idx]
    noisy = add_noise(original, noise_level=0.3)
    
    # Recall using sync and async updates
    print("Running synchronous update...")
    sync_recall, sync_iters = network.update(noisy, synchronous=True)
    sync_recall = sync_recall.reshape(original.shape)
    
    print("Running asynchronous update...")
    async_recall, async_iters = network.update(noisy, synchronous=False)
    async_recall = async_recall.reshape(original.shape)
    
    # Calculate similarities
    sync_sim = network.similarity(sync_recall, original)
    async_sim = network.similarity(async_recall, original)
    
    print(f"Synchronous update: {sync_iters} iterations, similarity: {sync_sim:.4f}")
    print(f"Asynchronous update: {async_iters} iterations, similarity: {async_sim:.4f}")
    
    # Display results
    fig, _ = display_images(
        [original, noisy, sync_recall, async_recall],
        titles=[
            "Original", 
            "Noisy (30%)", 
            f"Sync ({sync_iters} iter, sim={sync_sim:.2f})",
            f"Async ({async_iters} iter, sim={async_sim:.2f})"
        ],
        figsize=(16, 4)
    )
    plt.savefig(os.path.join(results_dir, "recall_noisy.png"))
    plt.close()
    
    # Test 2: Recall with cropping
    print("\nTest 2: Pattern recall with cropping")
    cropped = crop_image(original)
    
    # Recall using sync and async updates
    print("Running synchronous update...")
    sync_recall, sync_iters = network.update(cropped, synchronous=True)
    sync_recall = sync_recall.reshape(original.shape)
    
    print("Running asynchronous update...")
    async_recall, async_iters = network.update(cropped, synchronous=False)
    async_recall = async_recall.reshape(original.shape)
    
    # Calculate similarities
    sync_sim = network.similarity(sync_recall, original)
    async_sim = network.similarity(async_recall, original)
    
    print(f"Synchronous update: {sync_iters} iterations, similarity: {sync_sim:.4f}")
    print(f"Asynchronous update: {async_iters} iterations, similarity: {async_sim:.4f}")
    
    # Display results
    fig, _ = display_images(
        [original, cropped, sync_recall, async_recall],
        titles=[
            "Original", 
            "Cropped", 
            f"Sync ({sync_iters} iter, sim={sync_sim:.2f})",
            f"Async ({async_iters} iter, sim={async_sim:.2f})"
        ],
        figsize=(16, 4)
    )
    plt.savefig(os.path.join(results_dir, "recall_cropped.png"))
    plt.close()
    
    # Test 3: Recall with different noise levels
    print("\nTest 3: Pattern recall with varying noise levels")
    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    n_trials = 10
    
    # Results storage
    similarities = []
    iterations = []
    
    for noise in noise_levels:
        print(f"Testing noise level {noise:.1f}...")
        noise_sim = []
        noise_iter = []
        
        for _ in range(n_trials):
            # Create noisy pattern
            noisy = add_noise(original, noise_level=noise)
            
            # Recall with synchronous update
            recalled, iters = network.update(noisy, synchronous=True)
            recalled = recalled.reshape(original.shape)
            
            # Calculate similarity
            sim = network.similarity(recalled, original)
            noise_sim.append(sim)
            noise_iter.append(iters)
        
        # Store average results
        similarities.append(np.mean(noise_sim))
        iterations.append(np.mean(noise_iter))
        
        print(f"  Avg similarity: {np.mean(noise_sim):.4f}, Avg iterations: {np.mean(noise_iter):.1f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, similarities, 'o-', linewidth=2)
    plt.xlabel("Noise Level")
    plt.ylabel("Average Similarity to Original")
    plt.title("Pattern Recovery vs Noise Level")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(results_dir, "noise_similarity.png"))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, iterations, 'o-', linewidth=2)
    plt.xlabel("Noise Level")
    plt.ylabel("Average Iterations")
    plt.title("Convergence Speed vs Noise Level")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(results_dir, "noise_iterations.png"))
    plt.close()
    
    print("\nAll experiments completed. Results saved to:", results_dir)

if __name__ == "__main__":
    try:
        main()
        print("Script completed successfully!")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
