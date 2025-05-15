import numpy as np
import os
import random
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import glob

class HopfieldNetwork:
    def __init__(self, n_neurons: int):
        """
        Initialize a Hopfield Network with n_neurons neurons
        
        Args:
            n_neurons: Number of neurons in the network
        """
        self.n_neurons = n_neurons
        self.weights = np.zeros((n_neurons, n_neurons))
        
    def train(self, patterns: np.ndarray):
        """
        Train the network using Hebbian learning rule
        
        Args:
            patterns: Array of shape (num_patterns, n_neurons) where each row is a pattern
        """
        n_patterns = patterns.shape[0]
        
        # Initialize weights to zero
        self.weights = np.zeros((self.n_neurons, self.n_neurons))
        
        # Compute the weight matrix using Hebbian learning rule
        for i in range(self.n_neurons):
            for j in range(i+1, self.n_neurons):  # Only compute upper triangle (weights are symmetric)
                # Sum over all patterns
                w_ij = 0
                for mu in range(n_patterns):
                    w_ij += patterns[mu, i] * patterns[mu, j]
                
                # Store weight (symmetric matrix)
                self.weights[i, j] = w_ij / n_patterns
                self.weights[j, i] = w_ij / n_patterns
        
        # Verify through testing that each pattern is stable
        self._test_pattern_stability(patterns)
        
    def _test_pattern_stability(self, patterns: np.ndarray):
        """
        Test if each pattern is stable under the weight matrix,
        print warning if any pattern is unstable
        
        Args:
            patterns: The training patterns
        """
        n_patterns = patterns.shape[0]
        unstable_patterns = 0
        
        for mu in range(n_patterns):
            pattern = patterns[mu]
            # Calculate local field for each neuron
            h = np.dot(self.weights, pattern)
            
            # Check if pattern is stable
            new_state = np.sign(h)
            new_state[new_state == 0] = pattern[new_state == 0]  # Keep original value if h=0
            
            if not np.array_equal(new_state, pattern):
                unstable_patterns += 1
                
        if unstable_patterns > 0:
            print(f"Warning: {unstable_patterns}/{n_patterns} patterns are unstable under the current weights.")
            # This indicates potential issues with the number of patterns vs. capacity of the network
    
    def compute_energy(self, state: np.ndarray) -> float:
        """
        Compute the energy of the current state
        
        Args:
            state: Current state of the network
        
        Returns:
            Energy value
        """
        return -0.5 * np.dot(np.dot(state, self.weights), state)
    
    def async_update(self, state: np.ndarray, max_iterations: int = None) -> Tuple[np.ndarray, int]:
        """
        Perform asynchronous update until convergence or max_iterations
        
        Args:
            state: Initial state
            max_iterations: Maximum number of iterations
        
        Returns:
            Tuple of (final state, number of iterations)
        """
        if max_iterations is None:
            max_iterations = 1000 * self.n_neurons
            
        current_state = state.copy()
        iterations = 0
        unchanged_count = 0
        
        # Keep track of the sequence of states and their energies
        energy = self.compute_energy(current_state)
        energy_history = [energy]
        
        while iterations < max_iterations:
            old_state = current_state.copy()
            
            # Create a random order to update neurons
            neuron_indices = list(range(self.n_neurons))
            random.shuffle(neuron_indices)
            
            # Update each neuron once in a random order
            for i in neuron_indices:
                # Compute local field
                h_i = np.dot(self.weights[i], current_state)
                
                # Update the neuron's state
                if h_i > 0:
                    current_state[i] = 1
                elif h_i < 0:
                    current_state[i] = -1
                # If h_i == 0, keep current state
            
            iterations += 1
            
            # Calculate energy of new state
            new_energy = self.compute_energy(current_state)
            energy_history.append(new_energy)
            
            # Check for convergence
            if np.array_equal(current_state, old_state):
                unchanged_count += 1
                if unchanged_count >= 2:  # Wait for stability over multiple cycles
                    break
            else:
                unchanged_count = 0
                
            # Ensure energy is decreasing (system should always move to lower energy)
            if new_energy > energy:
                print(f"Warning: Energy increased from {energy:.4f} to {new_energy:.4f}")
                
            energy = new_energy
                
        return current_state, iterations
    
    def sync_update(self, state: np.ndarray, max_iterations: int = 100) -> Tuple[np.ndarray, int]:
        """
        Perform synchronous update until convergence or max_iterations
        
        Args:
            state: Initial state
            max_iterations: Maximum number of iterations
        
        Returns:
            Tuple of (final state, number of iterations)
        """
        current_state = state.copy()
        iterations = 0
        unchanged_count = 0
        
        # Keep track of energies and cycle detection
        energy = self.compute_energy(current_state)
        energy_history = [energy]
        state_history = [current_state.copy()]
        
        for _ in range(max_iterations):
            iterations += 1
            
            # Compute local fields for all neurons
            h = np.dot(self.weights, current_state)
            
            # Create new state based on local fields
            new_state = np.ones_like(current_state)
            new_state[h < 0] = -1
            new_state[h == 0] = current_state[h == 0]  # Keep original value if h=0
            
            # Calculate new energy
            new_energy = self.compute_energy(new_state)
            energy_history.append(new_energy)
            
            # Check for convergence
            if np.array_equal(new_state, current_state):
                unchanged_count += 1
                if unchanged_count >= 2:  # Ensure stability
                    break
            else:
                unchanged_count = 0
                
            # Check for cycles (limit of 2-cycles for simplicity)
            for past_state in state_history[-2:]:
                if np.array_equal(new_state, past_state):
                    # We've entered a cycle
                    print(f"Detected a cycle at iteration {iterations}")
                    return new_state, iterations
            
            # Add state to history
            state_history.append(new_state.copy())
            
            # Update current state
            current_state = new_state
            energy = new_energy
            
        return current_state, iterations
    
    def pattern_match(self, state: np.ndarray, patterns: np.ndarray, threshold: float = 0.6) -> Tuple[bool, int, float]:
        """
        Check if state matches any of the patterns
        
        Args:
            state: Current state of the network
            patterns: Training patterns
            threshold: Similarity threshold for considering a match
            
        Returns:
            Tuple of (match_found, best_match_index, best_match_similarity)
        """
        # Calculate similarities with all patterns
        similarities = []
        for i, pattern in enumerate(patterns):
            # Calculate normalized dot product as similarity measure
            similarity = np.dot(state, pattern) / (np.linalg.norm(state) * np.linalg.norm(pattern))
            similarities.append(similarity)
            
        # Find best match
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        
        # Check if similarity exceeds threshold
        match_found = best_similarity > threshold
        
        return match_found, best_idx, best_similarity

# Functions for reading and writing PBM files
def read_pbm(file_path: str) -> np.ndarray:
    """
    Read a PBM file and convert it to a vector of -1 and 1
    
    Args:
        file_path: Path to the PBM file
    
    Returns:
        Vector representation of the image (-1 for white, 1 for black)
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Skip comments
    lines = [line for line in lines if not line.startswith('#')]
    
    # Parse header
    magic_number = lines[0].strip()
    if magic_number != "P1":
        raise ValueError(f"Unsupported PBM format: {magic_number}. Expected P1.")
    
    dimensions = lines[1].strip().split()
    width = int(dimensions[0])
    height = int(dimensions[1])
    
    # Parse pixel data
    pixel_data = ''.join(lines[2:]).replace('\n', '').replace(' ', '').replace('.', '')
    # Filter out any non-digit characters
    pixel_data = ''.join(c for c in pixel_data if c in '01')
    pixels = np.array([int(p) for p in pixel_data])
    
    # Ensure we have the right number of pixels
    if len(pixels) != width * height:
        # Pad or truncate to correct size
        if len(pixels) < width * height:
            pixels = np.pad(pixels, (0, width * height - len(pixels)), constant_values=0)
        else:
            pixels = pixels[:width * height]
    
    # Convert 0 to -1 (white) and 1 to 1 (black)
    pixels = 2 * pixels - 1
    
    return pixels

def write_pbm(file_path: str, data: np.ndarray, width: int = 16, height: int = 16):
    """
    Write a vector to a PBM file
    
    Args:
        file_path: Path to save the PBM file
        data: Vector representation of the image (-1 for white, 1 for black)
        width: Width of the image
        height: Height of the image
    """
    # Convert -1 to 0 (white) and 1 to 1 (black)
    pixels = (data + 1) // 2
    
    with open(file_path, 'w') as f:
        # Write header
        f.write("P1\n")
        f.write(f"{width} {height}\n")
        
        # Write pixel data with line breaks for readability
        for i in range(0, len(pixels), width):
            row = pixels[i:i+width]
            f.write(' '.join(map(str, row)) + '\n')

# Functions for corrupting memories
def flip_pixels(pattern: np.ndarray, p: float) -> np.ndarray:
    """
    Flip pixels with probability p
    
    Args:
        pattern: Original pattern
        p: Probability of flipping each pixel
    
    Returns:
        Corrupted pattern
    """
    corrupted = pattern.copy()
    for i in range(len(corrupted)):
        if random.random() < p:
            corrupted[i] = -corrupted[i]
    return corrupted

def crop_image(pattern: np.ndarray, width: int = 16, height: int = 16, 
               bbox_x: int = 3, bbox_y: int = 3, bbox_width: int = 10, bbox_height: int = 10) -> np.ndarray:
    """
    Keep a bounding box and set outside pixels to -1 (white)
    
    Args:
        pattern: Original pattern in 1D form
        width, height: Original image dimensions
        bbox_x, bbox_y: Top-left coordinates of the bounding box
        bbox_width, bbox_height: Dimensions of the bounding box
    
    Returns:
        Cropped pattern
    """
    # Reshape to 2D for easier manipulation
    pattern_2d = pattern.reshape(height, width)
    
    # Create a new image with all white pixels (-1)
    cropped_2d = -np.ones((height, width))
    
    # Copy the bounding box region
    for y in range(bbox_y, min(bbox_y + bbox_height, height)):
        for x in range(bbox_x, min(bbox_x + bbox_width, width)):
            cropped_2d[y, x] = pattern_2d[y, x]
    
    # Flatten back to 1D
    return cropped_2d.flatten()

def display_image(data: np.ndarray, width: int = 16, height: int = 16, title: str = None):
    """
    Display an image from its vector representation
    
    Args:
        data: Vector representation of the image (-1 for white, 1 for black)
        width, height: Image dimensions
        title: Plot title
    """
    # Reshape to 2D
    img = data.reshape(height, width)
    
    # Convert -1 to white and 1 to black for display
    img_display = np.zeros((height, width))
    img_display[img == -1] = 1  # White
    img_display[img == 1] = 0   # Black
    
    plt.imshow(img_display, cmap='gray')
    if title:
        plt.title(title)
    plt.axis('off')

def display_image_sequence(sequences: List[np.ndarray], width: int = 16, height: int = 16, 
                          titles: Optional[List[str]] = None, save_path: Optional[str] = None):
    """
    Display a sequence of images
    
    Args:
        sequences: List of image vectors
        width, height: Image dimensions
        titles: List of titles for each image
        save_path: Path to save the figure
    """
    n_images = len(sequences)
    fig, axs = plt.subplots(1, n_images, figsize=(n_images * 3, 3))
    
    if n_images == 1:
        axs = [axs]
    
    for i, data in enumerate(sequences):
        # Reshape to 2D
        img = data.reshape(height, width)
        
        # Convert -1 to white and 1 to black for display
        img_display = np.zeros((height, width))
        img_display[img == -1] = 1  # White
        img_display[img == 1] = 0   # Black
        
        axs[i].imshow(img_display, cmap='gray')
        if titles and i < len(titles):
            axs[i].set_title(titles[i])
        axs[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

def save_all_patterns(patterns: np.ndarray, folder: str, prefix: str = "pattern"):
    """
    Save all patterns to PBM files
    
    Args:
        patterns: Array of patterns
        folder: Folder to save files
        prefix: Prefix for filenames
    """
    os.makedirs(folder, exist_ok=True)
    
    for i, pattern in enumerate(patterns):
        file_path = os.path.join(folder, f"{prefix}_{i:02d}.pbm")
        write_pbm(file_path, pattern)

def load_all_patterns(folder: str, pattern: str = "*.pbm") -> np.ndarray:
    """
    Load all PBM files from a folder
    
    Args:
        folder: Folder containing PBM files
        pattern: Glob pattern for files
    
    Returns:
        Array of patterns
    """
    file_paths = sorted(glob.glob(os.path.join(folder, pattern)))
    patterns = []
    
    for file_path in file_paths:
        patterns.append(read_pbm(file_path))
    
    return np.array(patterns)
