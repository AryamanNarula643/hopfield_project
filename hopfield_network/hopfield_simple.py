import numpy as np
import os
import random
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import glob
import copy

class HopfieldNetwork:
    """
    Simple implementation of the Hopfield Network for 16x16 binary images
    """
    def __init__(self, size=256):
        """Initialize a Hopfield Network with size neurons"""
        self.size = size
        self.weights = np.zeros((size, size))
        
    def train(self, patterns):
        """
        Train the network using the Hebbian learning rule
        
        Args:
            patterns: Training patterns of shape (num_patterns, size)
        """
        num_patterns = len(patterns)
        print(f"Training network with {num_patterns} patterns...")
        
        # Reset weights
        self.weights = np.zeros((self.size, self.size))
        
        # Apply Hebbian learning rule
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)
            
        # Set diagonal to zero and normalize
        np.fill_diagonal(self.weights, 0)
        self.weights /= self.size
        
        # Test pattern stability
        stable_patterns = 0
        for pattern in patterns:
            h = np.dot(self.weights, pattern)
            s_new = np.sign(h)
            s_new[s_new == 0] = pattern[s_new == 0]
            
            if np.array_equal(s_new, pattern):
                stable_patterns += 1
                
        print(f"{stable_patterns} out of {num_patterns} patterns are stable")
        
    def update_async(self, state, max_iterations=1000):
        """
        Perform asynchronous update on the state until convergence
        
        Args:
            state: Initial state of the network
            max_iterations: Maximum number of iterations
            
        Returns:
            final_state, num_iterations
        """
        state = state.copy()
        iteration = 0
        convergence = False
        
        while not convergence and iteration < max_iterations:
            old_state = state.copy()
            
            # Update neurons in random order
            update_order = np.random.permutation(self.size)
            for i in update_order:
                # Calculate local field
                h = np.dot(self.weights[i], state)
                
                # Update neuron
                if h > 0:
                    state[i] = 1
                elif h < 0:
                    state[i] = -1
                # If h == 0, state remains unchanged
                
            # Check for convergence
            if np.array_equal(old_state, state):
                convergence = True
                
            iteration += 1
            
        return state, iteration
    
    def update_sync(self, state, max_iterations=100):
        """
        Perform synchronous update on the state until convergence
        
        Args:
            state: Initial state of the network
            max_iterations: Maximum number of iterations
            
        Returns:
            final_state, num_iterations
        """
        state = state.copy()
        iteration = 0
        convergence = False
        
        # Keep track of states to detect cycles
        past_states = [state.copy()]
        
        while not convergence and iteration < max_iterations:
            # Calculate local fields for all neurons
            h = np.dot(self.weights, state)
            
            # Update all neurons simultaneously
            new_state = np.ones_like(state)
            new_state[h < 0] = -1
            new_state[h == 0] = state[h == 0]  # Maintain current state when h = 0
            
            # Check for convergence
            if np.array_equal(new_state, state):
                convergence = True
            
            # Check for cycles
            for past_state in past_states:
                if np.array_equal(new_state, past_state):
                    print(f"Cycle detected at iteration {iteration}")
                    return new_state, iteration
            
            # Update state
            state = new_state
            past_states.append(state.copy())
            
            # Only keep the last few states to check for cycles
            if len(past_states) > 5:
                past_states.pop(0)
                
            iteration += 1
            
        return state, iteration
    
    def energy(self, state):
        """
        Calculate the energy of a state
        
        Args:
            state: State of the network
            
        Returns:
            Energy value
        """
        return -0.5 * np.dot(np.dot(state, self.weights), state)
    
    def compute_similarity(self, state1, state2):
        """
        Compute similarity between two states
        
        Args:
            state1, state2: States to compare
            
        Returns:
            Similarity value between -1 and 1
        """
        return np.dot(state1, state2) / (np.linalg.norm(state1) * np.linalg.norm(state2))
    
    def best_match(self, state, patterns, threshold=0.6):
        """
        Find the best matching pattern for a state
        
        Args:
            state: Current state of the network
            patterns: Training patterns
            threshold: Similarity threshold for a match
            
        Returns:
            (is_match, match_index, similarity)
        """
        similarities = [self.compute_similarity(state, p) for p in patterns]
        best_idx = np.argmax(similarities)
        best_sim = similarities[best_idx]
        
        return best_sim > threshold, best_idx, best_sim

# Functions to handle PBM files
def read_pbm(filename):
    """Read a PBM (P1) file and convert it to a binary array"""
    with open(filename, 'r') as f:
        # Skip comments and read header
        line = f.readline().strip()
        while line.startswith('#') or line.startswith('//'):
            line = f.readline().strip()
        
        # Check magic number
        if line != 'P1':
            raise ValueError(f"Not a valid PBM P1 file: {filename}")
        
        # Skip comments and read dimensions
        line = f.readline().strip()
        while line.startswith('#'):
            line = f.readline().strip()
        
        width, height = map(int, line.split())
        
        # Read pixel data
        pixels = []
        for line in f:
            if line.startswith('#') or line.startswith('//'):
                continue
            # Handle both integer and floating point values
            for value in line.strip().split():
                if value in ['0', '1']:
                    pixels.append(value)
                elif value in ['0.0', '1.0']:
                    pixels.append('0' if value == '0.0' else '1')
                else:
                    # Try to convert to float and then to binary
                    try:
                        float_val = float(value)
                        pixels.append('0' if float_val == 0.0 else '1')
                    except ValueError:
                        pass  # Skip invalid values
        
        # Convert to binary array (-1 for white, 1 for black)
        binary = np.array([1 if p == '1' else -1 for p in pixels])
        
        # Handle missing data
        if len(binary) < width * height:
            print(f"Warning: PBM file {filename} has fewer pixels than expected: {len(binary)} vs {width * height}")
            binary = np.pad(binary, (0, width * height - len(binary)), 'constant', constant_values=-1)
        elif len(binary) > width * height:
            print(f"Warning: PBM file {filename} has more pixels than expected: {len(binary)} vs {width * height}")
            binary = binary[:width * height]
            
        return binary

def write_pbm(filename, array, width=16, height=16):
    """Write a binary array to a PBM (P1) file"""
    # Convert to PBM format (0 for white, 1 for black)
    pbm_data = ((array + 1) / 2).astype(int)
    
    with open(filename, 'w') as f:
        # Write header
        f.write("P1\n")
        f.write(f"{width} {height}\n")
        
        # Write pixel data with line breaks for readability
        for y in range(height):
            row = pbm_data[y * width: (y + 1) * width]
            f.write(' '.join(map(str, row)) + '\n')

def corrupt_flip(pattern, prob=0.1):
    """Corrupt a pattern by randomly flipping pixels"""
    corrupted = pattern.copy()
    for i in range(len(corrupted)):
        if random.random() < prob:
            corrupted[i] = -corrupted[i]
    return corrupted

def corrupt_crop(pattern, width=16, height=16, box_x=3, box_y=3, box_width=10, box_height=10):
    """Corrupt a pattern by cropping to a bounding box"""
    # Reshape to 2D
    pattern_2d = pattern.reshape(height, width)
    
    # Create a new pattern with all white pixels (-1)
    result_2d = np.ones((height, width)) * -1
    
    # Copy the bounding box
    for y in range(box_y, min(box_y + box_height, height)):
        for x in range(box_x, min(box_x + box_width, width)):
            result_2d[y, x] = pattern_2d[y, x]
            
    return result_2d.flatten()

def load_patterns(directory, pattern_glob="*.pbm"):
    """Load all PBM patterns from a directory"""
    pattern_files = sorted(glob.glob(os.path.join(directory, pattern_glob)))
    patterns = []
    
    for file in pattern_files:
        patterns.append(read_pbm(file))
        
    return np.array(patterns)

def display_patterns(patterns, rows=5, cols=5, title=None, save_path=None):
    """Display patterns in a grid"""
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < len(patterns):
            pattern = patterns[i].reshape(16, 16)
            # Convert -1 to white, 1 to black
            img = np.ones((16, 16))
            img[pattern == 1] = 0
            ax.imshow(img, cmap='gray')
            ax.set_title(f"Pattern {i}")
        ax.axis('off')
        
    plt.tight_layout()
    
    if title:
        fig.suptitle(title, fontsize=16)
        plt.subplots_adjust(top=0.9)
        
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def display_sequence(images, titles=None, save_path=None):
    """Display a sequence of images"""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(n * 3, 3))
    
    if n == 1:
        axes = [axes]
        
    for i, ax in enumerate(axes):
        img = images[i].reshape(16, 16)
        # Convert -1 to white, 1 to black
        display_img = np.ones((16, 16))
        display_img[img == 1] = 0
        ax.imshow(display_img, cmap='gray')
        if titles and i < len(titles):
            ax.set_title(titles[i])
        ax.axis('off')
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
