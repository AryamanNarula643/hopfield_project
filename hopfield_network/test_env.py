import numpy as np
import os
import sys

print("Python version:", sys.version)
print("NumPy version:", np.__version__)
print("Current working directory:", os.getcwd())

# Create a simple test directory and file
test_dir = "hopfield_network/test_output"
os.makedirs(test_dir, exist_ok=True)

with open(os.path.join(test_dir, "test.txt"), "w") as f:
    f.write("This is a test")

print("Created test file at:", os.path.join(test_dir, "test.txt"))
print("All done!")
