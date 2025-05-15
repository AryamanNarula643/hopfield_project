# Experiment Summary

## 1. Network Verification

A basic verification was conducted using a small set of 3 patterns. The network successfully:
- Stored the patterns using Hebbian learning
- Confirmed pattern stability after training
- Retrieved a pattern from a corrupted version with 30% noise
- Achieved perfect recall with a similarity score of 1.0000 after just 2 iterations

## 2. Pattern Recall with Noise

The network was tested with a pattern corrupted by random noise (30% of pixels flipped):

| Update Method | Iterations | Similarity to Original |
|---------------|------------|------------------------|
| Synchronous   | 2          | 1.0000                 |
| Asynchronous  | 2          | 1.0000                 |

This demonstrates excellent noise tolerance for random pixel flipping.

## 3. Pattern Recall with Cropping

The network was tested with a pattern that had only its central portion preserved:

| Update Method | Iterations | Similarity to Original |
|---------------|------------|------------------------|
| Synchronous   | 5          | 0.0000                 |
| Asynchronous  | 4          | 0.0000                 |

This shows that the network struggles with pattern completion when large portions are missing.

## 4. Noise Level Analysis

The network's recall performance was tested across different noise levels:

| Noise Level | Avg. Similarity | Avg. Iterations |
|-------------|----------------|-----------------|
| 10%         | 1.0000         | 2.0             |
| 20%         | 1.0000         | 2.0             |
| 30%         | 0.9719         | 2.1             |
| 40%         | 0.8156         | 2.9             |
| 50%         | -0.0305        | 2.9             |
| 60%         | -0.7695        | 3.1             |
| 70%         | -0.9680        | 2.2             |
| 80%         | -1.0000        | 2.0             |

Key observations:
1. Perfect recall up to 20% noise
2. Good recall (similarity > 0.8) up to 40% noise
3. At 50% noise, the network is at a transition point
4. Above 60% noise, the network consistently produces inverted patterns
5. Iterations peak at 40-60% noise levels

## Conclusions

1. **Noise Tolerance**: The Hopfield Network shows robust recall with up to 30-40% random noise.

2. **Pattern Completion Limitations**: The network struggles with structural damage like cropping.

3. **Transition Behavior**: At around 50% noise, the network transitions from correct to inverted recall.

4. **Convergence Speed**: The network converges quickly for both very clean and very noisy patterns, with slower convergence at the transition point.

5. **Update Rule Comparison**: Both synchronous and asynchronous updates perform similarly for this network size.

These results are consistent with theoretical expectations for Hopfield Networks and demonstrate both their capabilities and limitations as associative memory systems.
