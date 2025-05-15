import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import random

def generate_pbm_pattern(width=16, height=16):
    """Generate a random pattern and return it as a binary array"""
    # Create a random pattern (-1 for white, 1 for black)
    pattern = np.ones(width * height) * -1  # All white
    
    # Set about 50% to black
    black_indices = random.sample(range(width * height), width * height // 2)
    pattern[black_indices] = 1
    
    # Add some structure (e.g., border)
    pattern_2d = pattern.reshape(height, width)
    
    # Add a border or diagonal pattern
    if random.random() > 0.5:
        pattern_2d[0, :] = 1  # Top border
        pattern_2d[-1, :] = 1  # Bottom border
        pattern_2d[:, 0] = 1  # Left border
        pattern_2d[:, -1] = 1  # Right border
    else:
        # Add a diagonal
        for i in range(min(width, height)):
            pattern_2d[i, i] = 1
    
    return pattern_2d.flatten()

def save_patterns_to_pbm(directory, n_patterns=10):
    """Generate and save multiple patterns to PBM files"""
    os.makedirs(directory, exist_ok=True)
    
    for i in range(n_patterns):
        pattern = generate_pbm_pattern()
        write_pbm(os.path.join(directory, f"pattern_{i:02d}.pbm"), pattern)
        
    print(f"Generated and saved {n_patterns} patterns to {directory}")

def write_pbm(filename, pattern, width=16, height=16):
    """Write a binary pattern to a PBM file"""
    # Convert from -1/1 to 0/1 format
    pbm_data = ((pattern + 1) // 2).astype(int)
    
    with open(filename, 'w') as f:
        f.write("P1\n")
        f.write(f"{width} {height}\n")
        
        # Write pixel data with line breaks for readability
        for y in range(height):
            row = pbm_data[y * width: (y + 1) * width]
            f.write(' '.join(map(str, row)) + '\n')

def load_patterns(directory, n_patterns=10):
    """Generate patterns and return them as a numpy array"""
    patterns = []
    
    for i in range(n_patterns):
        pattern = generate_pbm_pattern()
        patterns.append(pattern)
        
    return np.array(patterns)

def corrupt_flip(pattern, prob=0.3):
    """Corrupt a pattern by randomly flipping pixels"""
    corrupted = pattern.copy()
    for i in range(len(corrupted)):
        if random.random() < prob:
            corrupted[i] = -corrupted[i]
    return corrupted

def corrupt_crop(pattern, width=16, height=16, box_x=3, box_y=3, box_width=10, box_height=10):
    """Keep a 10x10 bounding box and set outside pixels to -1 (white)"""
    pattern_2d = pattern.reshape(height, width)
    
    # Create a new all-white pattern
    cropped_2d = np.ones((height, width)) * -1
    
    # Copy the bounding box
    for y in range(box_y, min(box_y + box_height, height)):
        for x in range(box_x, min(box_x + box_width, width)):
            cropped_2d[y, x] = pattern_2d[y, x]
    
    return cropped_2d.flatten()

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
