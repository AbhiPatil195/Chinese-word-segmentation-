"""
Chinese Word Segmentation Submission Generator

This script allows you to generate submissions using different segmentation methods.
It provides a command line interface to select the method and customize parameters.

Usage:
    python generate_submission.py [--method METHOD] [--output OUTPUT]

Methods:
    - max_match_forward: Forward maximum matching algorithm
    - max_match_backward: Backward maximum matching algorithm
    - bi_directional: Bi-directional maximum matching algorithm
    - hmm: Hidden Markov Model (using jieba)
    - hybrid: Hybrid approach combining multiple methods (default)
"""

import os
import argparse
import time
from word_segmentation import ChineseWordSegmenter, process_test_data

# Get the directory of the current script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Set file paths - use absolute paths to avoid issues
TRAIN_PATH = os.path.join(CURRENT_DIR, 'train.csv')
TEST_PATH = os.path.join(CURRENT_DIR, 'test.csv')
FREQ_LIST_PATH = os.path.join(CURRENT_DIR, 'chinese_word_freq_list.txt')
SAMPLE_SUBMISSION_PATH = os.path.join(CURRENT_DIR, 'sample_submission.csv')

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate Chinese word segmentation submission')
    parser.add_argument('--method', type=str, default='hybrid', 
                        choices=['max_match_forward', 'max_match_backward', 'bi_directional', 'hmm', 'hybrid'],
                        help='Segmentation method to use')
    parser.add_argument('--output', type=str, default='my_submission.csv',
                        help='Output file name')
    args = parser.parse_args()
    
    # Set submission path
    submission_path = os.path.join(CURRENT_DIR, args.output)
    
    print(f"Starting Chinese word segmentation using {args.method} method...")
    print(f"Output will be saved to {submission_path}")
    
    start_time = time.time()
    
    # Create segmenter
    segmenter = ChineseWordSegmenter()
    
    # Load word dictionary
    segmenter.load_word_dict(FREQ_LIST_PATH)
    
    # Train from corpus
    segmenter.train_from_corpus(TRAIN_PATH)
    
    # Process test data with the specified method
    process_test_data(segmenter, TEST_PATH, submission_path, SAMPLE_SUBMISSION_PATH, method=args.method)
    
    # Print total time
    elapsed = time.time() - start_time
    print(f"Total execution time: {elapsed:.2f} seconds")
    print(f"Submission file created at {submission_path}")
    print(f"Don't forget to change the Team Name to your student ID and name format!")

if __name__ == "__main__":
    main() 