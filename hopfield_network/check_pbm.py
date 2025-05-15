import os
from hopfield_simple import read_pbm

def check_pbm_files(directory):
    """Check if PBM files can be read properly"""
    # List all PBM files in the directory
    files = [f for f in os.listdir(directory) if f.endswith(".pbm")]
    
    print(f"Found {len(files)} PBM files in {directory}")
    
    for file in files:
        path = os.path.join(directory, file)
        print(f"Reading {path}...")
        try:
            pattern = read_pbm(path)
            print(f"  Success: pattern shape = {pattern.shape}")
        except Exception as e:
            print(f"  Error: {str(e)}")

if __name__ == "__main__":
    check_pbm_files("hopfield_network/images")
