import numpy as np
import os
import random
import string
from hopfield_network import write_pbm

def generate_random_patterns(n_patterns: int, width: int, height: int, density: float = 0.5) -> np.ndarray:
    """
    Generate random patterns with a given density of black pixels
    
    Args:
        n_patterns: Number of patterns to generate
        width, height: Dimensions of each pattern
        density: Proportion of black pixels
    
    Returns:
        Array of shape (n_patterns, width*height)
    """
    patterns = []
    
    for _ in range(n_patterns):
        # Create a random pattern
        pattern = np.ones(width * height) * -1  # All white
        
        # Set random pixels to black based on density
        n_black = int(width * height * density)
        black_indices = random.sample(range(width * height), n_black)
        pattern[black_indices] = 1
        
        # Ensure pattern has distinct features
        # Reshape to 2D for easier manipulation
        pattern_2d = pattern.reshape(height, width)
        
        # Add a border or some distinct features
        if random.random() > 0.5:
            # Add a border
            pattern_2d[0, :] = 1
            pattern_2d[-1, :] = 1
            pattern_2d[:, 0] = 1
            pattern_2d[:, -1] = 1
        else:
            # Add a diagonal
            for i in range(min(width, height)):
                pattern_2d[i, i] = 1
        
        patterns.append(pattern_2d.flatten())
        
    return np.array(patterns)

def generate_simple_shapes(n_patterns: int, width: int, height: int) -> np.ndarray:
    """
    Generate simple shapes like letters, digits, and geometric patterns
    
    Args:
        n_patterns: Number of patterns to generate
        width, height: Dimensions of each pattern
    
    Returns:
        Array of shape (n_patterns, width*height)
    """
    patterns = []
    
    # Generate patterns for letters
    letters = string.ascii_uppercase[:min(n_patterns, 26)]
    
    for letter in letters:
        pattern = np.ones(width * height) * -1  # All white
        
        # Create a 2D array for easier manipulation
        pattern_2d = pattern.reshape(height, width)
        
        # Draw letter (very simplified)
        mid_x = width // 2
        mid_y = height // 2
        thickness = max(1, min(width, height) // 8)
        
        if letter == 'A':
            # Draw 'A'
            for y in range(mid_y, height):
                for x in range(mid_x - (y - mid_y) // 2, mid_x + (y - mid_y) // 2 + 1):
                    if 0 <= x < width:
                        pattern_2d[y, x] = 1
            # Horizontal line
            for x in range(mid_x - 2, mid_x + 3):
                if 0 <= x < width:
                    pattern_2d[mid_y + 2, x] = 1
        
        elif letter == 'B':
            # Draw 'B'
            for y in range(4, height-4):
                pattern_2d[y, 5] = 1
            for x in range(5, 12):
                pattern_2d[4, x] = 1
                pattern_2d[height-5, x] = 1
                pattern_2d[height//2, x] = 1
            for y in range(4, height//2):
                pattern_2d[y, 12] = 1
            for y in range(height//2, height-4):
                pattern_2d[y, 12] = 1
        
        elif letter == 'C':
            # Draw 'C'
            for y in range(4, height-4):
                pattern_2d[y, 5] = 1
            for x in range(5, 12):
                pattern_2d[4, x] = 1
                pattern_2d[height-5, x] = 1
        
        elif letter == 'X':
            # Draw 'X'
            for i in range(width):
                y1 = i * (height - 1) // (width - 1)
                y2 = (height - 1) - y1
                for t in range(-thickness//2, thickness//2+1):
                    if 0 <= y1+t < height:
                        pattern_2d[y1+t, i] = 1
                    if 0 <= y2+t < height:
                        pattern_2d[y2+t, i] = 1
        
        elif letter == 'O':
            # Draw 'O'
            center_x = width // 2
            center_y = height // 2
            radius = min(width, height) // 3
            
            for y in range(height):
                for x in range(width):
                    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    if radius - thickness/2 <= dist <= radius + thickness/2:
                        pattern_2d[y, x] = 1
        
        # Add more letters as needed...
        
        else:
            # For other letters, draw a simple pattern
            for y in range(3, height-3, height//4):
                for x in range(3, width-3, width//4):
                    pattern_2d[y, x] = 1
        
        patterns.append(pattern_2d.flatten())
        
        if len(patterns) >= n_patterns:
            break
    
    # If we need more patterns, generate geometric shapes
    shapes_needed = n_patterns - len(patterns)
    
    for i in range(shapes_needed):
        pattern = np.ones(width * height) * -1  # All white
        pattern_2d = pattern.reshape(height, width)
        
        shape_type = i % 4
        
        if shape_type == 0:
            # Square
            for y in range(height//4, 3*height//4):
                for x in range(width//4, 3*width//4):
                    pattern_2d[y, x] = 1
        
        elif shape_type == 1:
            # Cross
            for y in range(height):
                pattern_2d[y, width//2] = 1
            for x in range(width):
                pattern_2d[height//2, x] = 1
        
        elif shape_type == 2:
            # Diagonal
            for i in range(min(width, height)):
                pattern_2d[i, i] = 1
                pattern_2d[i, width-i-1] = 1
        
        else:
            # Checkered pattern
            for y in range(0, height, 2):
                for x in range(y % 4, width, 4):
                    pattern_2d[y, x] = 1
        
        patterns.append(pattern_2d.flatten())
    
    return np.array(patterns)

def save_patterns_to_pbm(patterns: np.ndarray, output_dir: str, width: int, height: int):
    """
    Save patterns to PBM files
    
    Args:
        patterns: Array of patterns
        output_dir: Directory to save the files
        width, height: Dimensions of each pattern
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, pattern in enumerate(patterns):
        file_path = os.path.join(output_dir, f"pattern_{i:02d}.pbm")
        write_pbm(file_path, pattern, width, height)

def main():
    # Parameters
    n_patterns = 25
    width = 16
    height = 16
    output_dir = "hopfield_network/images"
    
    # Generate patterns (mix of random and shapes)
    random_patterns = generate_random_patterns(10, width, height, 0.3)
    shape_patterns = generate_simple_shapes(15, width, height)
    
    # Combine patterns
    all_patterns = np.vstack([random_patterns, shape_patterns])
    
    # Save patterns to PBM files
    save_patterns_to_pbm(all_patterns, output_dir, width, height)
    
    print(f"Generated {n_patterns} patterns and saved to {output_dir}")

if __name__ == "__main__":
    main()
