import numpy as np
import random

class HopfieldNetwork:
    """
    Simple implementation of a Hopfield Network for pattern storage and retrieval
    """
    
    def __init__(self, size):
        """
        Initialize the network with a given size (number of neurons)
        
        Args:
            size: Number of neurons in the network
        """
        self.size = size
        self.weights = np.zeros((size, size))
        
    def train(self, patterns):
        """
        Train the network using the Hebbian learning rule
        
        Args:
            patterns: Array of patterns, shape (num_patterns, size)
        """
        num_patterns = len(patterns)
        print(f"Training network with {num_patterns} patterns...")
        
        # Reset weights
        self.weights = np.zeros((self.size, self.size))
        
        # Apply Hebbian learning rule
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)
            
        # Set diagonal to zero (no self-connections)
        np.fill_diagonal(self.weights, 0)
        
        # Normalize by the number of neurons
        self.weights /= self.size
        
        # Test pattern stability
        self._test_pattern_stability(patterns)
        
    def _test_pattern_stability(self, patterns):
        """
        Test if the patterns are stable under the current weights
        
        Args:
            patterns: Array of patterns
        """
        stable_count = 0
        for i, pattern in enumerate(patterns):
            # Apply one synchronous update
            h = np.dot(self.weights, pattern)
            s = np.sign(h)
            s[s == 0] = pattern[s == 0]  # Keep original state where h = 0
            
            # Check stability
            if np.array_equal(s, pattern):
                stable_count += 1
        
        print(f"{stable_count} out of {len(patterns)} patterns are stable")
        
    def update_async(self, state, max_iterations=100):
        """
        Perform asynchronous update on the state until convergence
        
        Args:
            state: Initial state of the network
            max_iterations: Maximum number of iterations
            
        Returns:
            final_state, num_iterations
        """
        state = state.copy()
        iterations = 0
        
        for iteration in range(max_iterations):
            old_state = state.copy()
            
            # Update all neurons in random order
            neuron_order = list(range(self.size))
            random.shuffle(neuron_order)
            
            for i in neuron_order:
                # Calculate local field
                h = np.dot(self.weights[i], state)
                
                # Update neuron state
                if h > 0:
                    state[i] = 1
                elif h < 0:
                    state[i] = -1
                # If h = 0, keep current state
            
            iterations = iteration + 1
            
            # Check for convergence
            if np.array_equal(old_state, state):
                break
                
        return state, iterations
        
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
        iterations = 0
        
        # Keep track of visited states to detect cycles
        visited_states = set()
        state_key = tuple(state)
        visited_states.add(state_key)
        
        for iteration in range(max_iterations):
            # Calculate local fields
            h = np.dot(self.weights, state)
            
            # Update all neurons simultaneously
            new_state = np.sign(h)
            new_state[h == 0] = state[h == 0]  # Keep original state where h = 0
            
            iterations = iteration + 1
            
            # Check for convergence
            if np.array_equal(new_state, state):
                break
                
            # Check for cycles
            state = new_state
            state_key = tuple(state)
            if state_key in visited_states:
                print("Cycle detected")
                break
            visited_states.add(state_key)
            
        return state, iterations
    
    def energy(self, state):
        """
        Calculate the energy of a state
        
        Args:
            state: Network state
            
        Returns:
            Energy value
        """
        return -0.5 * np.dot(state, np.dot(self.weights, state))
    
    def measure_similarity(self, state1, state2):
        """
        Measure the similarity between two states
        
        Args:
            state1, state2: States to compare
            
        Returns:
            Similarity value between -1 and 1
        """
        return np.dot(state1, state2) / (np.linalg.norm(state1) * np.linalg.norm(state2))
    
    def find_best_match(self, state, patterns, threshold=0.6):
        """
        Find the best matching pattern
        
        Args:
            state: Current network state
            patterns: Training patterns
            threshold: Minimum similarity for a match
            
        Returns:
            (match_found, pattern_index, similarity)
        """
        similarities = [self.measure_similarity(state, p) for p in patterns]
        best_idx = np.argmax(similarities)
        best_sim = similarities[best_idx]
        
        return best_sim > threshold, best_idx, best_sim
