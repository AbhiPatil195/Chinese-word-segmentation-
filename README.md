# Chinese Word Segmentation Project

This project is part of the Natural Language Processing course at Central South University's School of Automation.

## Overview

Chinese text does not have spaces between words, making word segmentation a fundamental task in Chinese NLP. This project implements a Chinese word segmentation system that:

1. Trains a segmentation model using the provided training data
2. Tests the model on the test dataset
3. Generates a submission file with the segmented results in the required format

## Files

- `train.csv`: Contains training data with segmented Chinese sentences
- `test.csv`: Contains test data that needs to be segmented
- `chinese_word_freq_list.txt`: A list of Chinese words with their frequencies
- `sample_submission.csv`: Example submission file showing the required format
- `transfer.py`: Script that converts segmented text to the required index-based notation
- `word_segmentation.py`: The main script that contains the segmentation model implementation
- `generate_submission.py`: A user-friendly script to generate submissions with different methods

## Segmentation Methods

The word segmentation system implements multiple approaches:

1. **Maximum Matching Forward**: A dictionary-based approach that matches the longest possible word from left to right in a sentence.
2. **Maximum Matching Backward**: Similar to the forward approach, but matches from right to left.
3. **Bi-directional Maximum Matching**: Combines forward and backward approaches and chooses the better segmentation based on heuristics.
4. **HMM-based Segmentation**: Uses Jieba, a popular Chinese word segmentation library based on Hidden Markov Models.
5. **Hybrid Approach (Default)**: Combines multiple methods and chooses the best result based on word frequency, bi-gram statistics, and other factors.

## Usage

### Basic Usage

To generate a submission with the default hybrid method:

```
python generate_submission.py
```

### Custom Method and Output

To specify a segmentation method and output file:

```
python generate_submission.py --method [METHOD] --output [FILENAME]
```

Available methods:
- `max_match_forward`
- `max_match_backward`
- `bi_directional`
- `hmm`
- `hybrid` (default)

Example:
```
python generate_submission.py --method bi_directional --output bi_dir_submission.csv
```

## Implementation Details

### Dictionary Building
- Loads word frequencies from the provided dictionary file
- Extracts additional vocabulary from the training corpus
- Builds bi-gram statistics for better segmentation decisions

### Training
- Processes the training data to build a comprehensive word dictionary
- Extracts bi-gram statistics to understand word co-occurrences
- Tracks single character frequencies for handling unknown words

### Segmentation
- Implements multiple segmentation algorithms
- The hybrid approach evaluates segmentation quality using:
  - Word frequency scores
  - Bi-gram probability
  - Number of words in the segmentation
  - Ratio of single-character words

## Requirements

- Python 3.6+
- pandas
- numpy
- jieba

## Important Note

**Don't forget to change the Team Name to your student ID and name format (学号-姓名) when submitting!**

## Notes

- The script handles potential encoding issues with Chinese text files
- It implements robust error handling to deal with formatting inconsistencies
- Performance can be improved by adjusting the segmentation method and parameters 