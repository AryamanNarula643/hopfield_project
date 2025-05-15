# Hopfield Network for Image Pattern Recognition

This project implements a Hopfield Network to store and recall 16x16 pixel black and white images in PBM format. The implementation follows specific requirements for network training, memory corruption, update rules, and experimental analysis.

## Project Structure

```
hopfield_network/
├── images/                 # Directory for 16x16 PBM images
├── results/                # Directory for initial experimental results
├── final_results/          # Directory for final experimental results
├── hopfield_network.py     # Initial implementation of the Hopfield Network
├── hopfield.py             # Final optimized implementation 
├── pbm_utils.py            # Utilities for handling PBM files
├── generate_patterns.py    # Script to generate PBM images
├── run_experiments.py      # Initial experiment script
├── run_experiments_new.py  # Improved experiment script
└── run_final_experiments.py # Final experiments with visualization
```

## Implementation Details

### 1. Dataset Preparation

- 25 unique 16x16 pixel black and white images in PBM format
- Functions to read/write PBM files, converting between PBM format (0=white, 1=black) and the internal representation (-1=white, 1=black)
- Random pattern generation with structural elements (borders, diagonals)

### 2. Hopfield Network Training

- Implementation of Hebbian learning rule to compute the weight matrix
- Ensures that diagonal weights are set to zero (W_ii = 0)

### 3. Memory Corruption

- Pixel Flipping: Randomly flips pixels with probability p
- Cropping: Keeps a 10x10 pixel bounding box and sets outside pixels to white (-1)

### 4. Update Rules

- Asynchronous Update: Randomly selects a neuron and updates its state based on the local field
- Synchronous Update: Updates all neurons simultaneously based on their local fields

### 5 & 6. Experiments

1. Test with specific initializations:
   - One memory corrupted by flipping pixels with p=0.3
   - One memory corrupted by cropping to a 10x10 bounding box
   - Run both asynchronous and synchronous updates until convergence

2. Analysis across corruption probabilities:
   - Tests with p = 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
   - 20 trials per probability
   - Measures correct convergence rate and number of update steps
   - Results visualized as bar charts and histograms

## How to Run

1. Generate patterns:
   ```
   python generate_patterns.py
   ```

2. Run all experiments:
   ```
   python run_experiments.py
   ```

## Results

### Initial Experiments

Initial experiment results are saved to the `results/` directory:

- `all_patterns.png`: Visualization of all 25 original patterns
- `experiment1_flipped.png`: Results from testing with flipped memory
- `experiment1_cropped.png`: Results from testing with cropped memory
- `experiment2_convergences.png`: Bar chart of convergence rates vs. corruption probability
- `experiment2_update_steps.png`: Histograms of update steps for each probability

### Final Experiments

The final optimized experiments are saved to the `final_results/` directory:

- `patterns.png`: Visualization of the training patterns
- `recall_noisy.png`: Results from recalling patterns with 30% noise
- `recall_cropped.png`: Results from recalling patterns with cropping
- `noise_similarity.png`: Graph of pattern similarity vs noise level
- `noise_iterations.png`: Graph of iterations required vs noise level

### Key Findings

1. **Pattern Recall with Noise**:
   - The network successfully recalled patterns with up to 30% corrupted pixels
   - Both synchronous and asynchronous updates achieved perfect recall (similarity: 1.0)
   - Convergence was achieved in just 2 iterations for most cases

2. **Pattern Recall with Cropping**:
   - The network struggled to recall correctly from cropped patterns
   - This demonstrates a limitation in pattern completion with structural damage

3. **Noise Tolerance Analysis**:
   - Perfect recall was maintained up to 20% noise
   - Recall quality declined rapidly after 40% noise
   - At noise levels above 60%, the network consistently produced inverted patterns
   - The number of iterations required for convergence peaked at mid-range noise levels (40-60%)

4. **Update Rule Comparison**:
   - Both synchronous and asynchronous updates performed similarly for this network size
   - Asynchronous updates were slightly more efficient in some cases

## Technical Details

- The network uses the Hebbian learning rule: W_ij = (1/N) * Σ(ξ_i^μ * ξ_j^μ) for i ≠ j
- Energy function: E = -0.5 * s^T * W * s
- Update rules ensure states change only when the local field is non-zero
- Convergence is determined by no state changes after a full update cycle

## Final Implementation Notes

The final implementation (in `hopfield.py` and `run_final_experiments.py`) includes several improvements:

1. **Robust PBM Handling**: Fixed issues with reading floating-point values in PBM files
2. **Optimized Training**: Improved weight normalization and zero-diagonal handling
3. **Better Pattern Similarity**: Implemented cosine similarity for more accurate pattern matching
4. **Enhanced Visualization**: Created clearer visualizations with similarity metrics
5. **Cycle Detection**: Added detection of update cycles to prevent infinite loops
6. **Performance Analysis**: Comprehensive testing across different corruption levels

## Theoretical Context

The Hopfield Network has a theoretical capacity limit of approximately 0.14N patterns (where N is the number of neurons). For our 16x16 network (256 neurons), this means we could theoretically store about 35 patterns, though practical performance typically degrades well before reaching this limit.

The network's ability to recover from noise depends on the structure of the energy landscape, where stored patterns correspond to local minima. As noise increases, the corrupted pattern may fall into the basin of attraction of a different stored pattern or even an inverted pattern.

## Conclusion

This implementation successfully demonstrates the pattern storage and recall capabilities of Hopfield Networks. The network shows excellent resilience to random noise (up to 30%) but struggles with structural damage like cropping. These results align with theoretical expectations and highlight both the strengths and limitations of this type of associative memory system.

## Requirements

- Python 3.6+
- NumPy
- Matplotlib
