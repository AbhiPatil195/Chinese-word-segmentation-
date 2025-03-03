import pandas as pd
import numpy as np
import re
import os
from collections import defaultdict, Counter
import jieba  # Using jieba as a backup and for comparison
import codecs  # For handling different encodings
import time

# Import the transfer function from the provided script
from transfer import transfer

# Get the directory of the current script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Set file paths - use absolute paths to avoid issues
TRAIN_PATH = os.path.join(CURRENT_DIR, 'train.csv')
TEST_PATH = os.path.join(CURRENT_DIR, 'test.csv')
FREQ_LIST_PATH = os.path.join(CURRENT_DIR, 'chinese_word_freq_list.txt')
SAMPLE_SUBMISSION_PATH = os.path.join(CURRENT_DIR, 'sample_submission.csv')
SUBMISSION_PATH = os.path.join(CURRENT_DIR, 'my_submission.csv')

class ChineseWordSegmenter:
    def __init__(self):
        self.word_dict = {}  # Dictionary for word frequencies
        self.max_word_len = 0  # Maximum word length in the dictionary
        self.bi_gram = defaultdict(int)  # Bi-gram statistics
        self.single_char_freq = defaultdict(int)  # Frequency of single characters
        
    def load_word_dict(self, freq_list_path):
        """Load word dictionary from frequency list"""
        print("Loading word dictionary...")
        try:
            # Try different encodings
            encodings = ['utf-8', 'gbk', 'gb18030', 'latin1', 'ISO-8859-1']
            loaded = False
            
            for encoding in encodings:
                try:
                    with codecs.open(freq_list_path, 'r', encoding=encoding) as f:
                        line_count = 0
                        for line in f:
                            line_count += 1
                            parts = line.strip().split()
                            if len(parts) >= 2:  # Make sure there are enough parts
                                try:
                                    # Try to extract word and frequency
                                    # The format might be: index word frequency or word frequency
                                    if len(parts) >= 3:
                                        word = parts[1]
                                        freq = int(parts[2])
                                    else:
                                        word = parts[0]
                                        freq = int(parts[1])
                                    
                                    # Skip punctuation and single characters for dictionary
                                    if len(word) > 1 and not all(c in '，。！？；：""''（）【】《》、' for c in word):
                                        self.word_dict[word] = freq
                                        self.max_word_len = max(self.max_word_len, len(word))
                                    elif len(word) == 1:
                                        self.single_char_freq[word] = freq
                                except (ValueError, IndexError):
                                    continue  # Skip lines with incorrect format
                            
                            # Print progress for large files
                            if line_count % 20000 == 0:
                                print(f"Processed {line_count} lines...")
                    
                    print(f"Loaded {len(self.word_dict)} words from frequency list using {encoding} encoding")
                    loaded = True
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"Error with {encoding} encoding: {e}")
                    continue
            
            if not loaded:
                print("Failed to load word dictionary with any encoding")
                
            # If dictionary is empty, add some common words as fallback
            if not self.word_dict:
                self.word_dict = {
                    "的": 10000, "是": 9000, "在": 8000, "有": 7000, "和": 6000,
                    "不": 5000, "了": 4000, "中": 3000, "人": 2000, "我": 1000
                }
                self.max_word_len = 1
                print("Used fallback dictionary with common words")
                
        except Exception as e:
            print(f"Error loading word dictionary: {e}")
    
    def train_from_corpus(self, train_path):
        """Train the segmenter from a corpus of segmented text"""
        print("Training from corpus...")
        try:
            # Try different encodings
            encodings = ['utf-8', 'gbk', 'gb18030', 'latin1', 'ISO-8859-1']
            loaded = False
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(train_path, encoding=encoding)
                    if 'sentence' in df.columns:
                        print(f"Successfully read with encoding: {encoding}")
                        loaded = True
                        break
                except:
                    continue
            
            if not loaded:
                print("Failed to load training data with any encoding")
                return
            
            # Extract and process sentences
            word_counter = Counter()
            for _, row in df.iterrows():
                if 'sentence' in row:
                    # Assuming the sentence column contains pre-segmented text (words separated by spaces)
                    segmented_sentence = row['sentence']
                    if isinstance(segmented_sentence, str):
                        # Clean the sentence
                        segmented_sentence = re.sub(r'[^\w\s]', ' ', segmented_sentence)
                        words = segmented_sentence.split()
                        
                        # Update word frequencies
                        for word in words:
                            if word and len(word.strip()) > 0:
                                word_counter[word] += 1
                                
                                # Update single character frequency
                                if len(word) == 1:
                                    self.single_char_freq[word] += 1
                        
                        # Build bi-gram model
                        for i in range(len(words) - 1):
                            if words[i] and words[i+1]:
                                self.bi_gram[(words[i], words[i+1])] += 1
            
            # Update word dictionary with corpus words
            for word, count in word_counter.items():
                if word not in self.word_dict:
                    self.word_dict[word] = 0
                self.word_dict[word] += count
                self.max_word_len = max(self.max_word_len, len(word))
            
            print(f"Trained on corpus, dictionary now has {len(self.word_dict)} words")
            print(f"Collected {len(self.bi_gram)} bi-grams and {len(self.single_char_freq)} unique characters")
            
            # If max_word_len is too large, limit it for efficiency
            if self.max_word_len > 10:
                self.max_word_len = 10
                print(f"Limited maximum word length to {self.max_word_len}")
                
        except Exception as e:
            print(f"Error training from corpus: {e}")
    
    def segment_by_max_match_forward(self, sentence):
        """Segment using forward maximum matching algorithm"""
        result = []
        i = 0
        while i < len(sentence):
            matched = False
            for j in range(min(self.max_word_len, len(sentence) - i), 0, -1):
                word = sentence[i:i+j]
                if word in self.word_dict:
                    result.append(word)
                    i += j
                    matched = True
                    break
            if not matched:  # If no match, treat one character as a word
                result.append(sentence[i])
                i += 1
        return result
    
    def segment_by_max_match_backward(self, sentence):
        """Segment using backward maximum matching algorithm"""
        result = []
        i = len(sentence)
        while i > 0:
            matched = False
            for j in range(min(self.max_word_len, i), 0, -1):
                word = sentence[i-j:i]
                if word in self.word_dict:
                    result.insert(0, word)
                    i -= j
                    matched = True
                    break
            if not matched:  # If no match, treat one character as a word
                result.insert(0, sentence[i-1])
                i -= 1
        return result
    
    def segment_by_bi_directional(self, sentence):
        """Segment using bi-directional maximum matching"""
        forward = self.segment_by_max_match_forward(sentence)
        backward = self.segment_by_max_match_backward(sentence)
        
        # Choose the better segmentation based on heuristics
        if len(forward) != len(backward):
            # Choose the one with fewer words
            return forward if len(forward) < len(backward) else backward
        
        # If same number of words, choose the one with more multi-character words
        forward_single = sum(1 for word in forward if len(word) == 1)
        backward_single = sum(1 for word in backward if len(word) == 1)
        
        if forward_single != backward_single:
            return forward if forward_single < backward_single else backward
        
        # If still tied, choose based on word frequency
        forward_score = sum(self.word_dict.get(word, 0) for word in forward)
        backward_score = sum(self.word_dict.get(word, 0) for word in backward)
        
        return forward if forward_score >= backward_score else backward
    
    def segment_by_hmm(self, sentence):
        """Segment using HMM (Hidden Markov Model) with jieba"""
        result = jieba.cut(sentence)
        return list(result)
    
    def segment_hybrid(self, sentence):
        """Use a hybrid approach combining multiple methods"""
        # Get results from different methods
        bi_dir = self.segment_by_bi_directional(sentence)
        hmm = self.segment_by_hmm(sentence)
        
        # For each method, calculate a confidence score
        bi_dir_score = sum(self.word_dict.get(word, 0) for word in bi_dir)
        hmm_score = sum(self.word_dict.get(word, 0) for word in hmm)
        
        # Use bi-gram to further evaluate
        bi_dir_bigram_score = 0
        for i in range(len(bi_dir) - 1):
            bi_dir_bigram_score += self.bi_gram.get((bi_dir[i], bi_dir[i+1]), 0)
            
        hmm_bigram_score = 0
        for i in range(len(hmm) - 1):
            hmm_bigram_score += self.bi_gram.get((hmm[i], hmm[i+1]), 0)
        
        # Normalize scores by length
        bi_dir_total = (bi_dir_score + bi_dir_bigram_score) / len(bi_dir)
        hmm_total = (hmm_score + hmm_bigram_score) / len(hmm)
        
        # Choose the method with the highest score
        return bi_dir if bi_dir_total >= hmm_total else hmm
    
    def segment(self, sentence, method='hybrid'):
        """Segment a sentence using the specified method"""
        if method == 'max_match_forward':
            return self.segment_by_max_match_forward(sentence)
        elif method == 'max_match_backward':
            return self.segment_by_max_match_backward(sentence)
        elif method == 'bi_directional':
            return self.segment_by_bi_directional(sentence)
        elif method == 'hmm':
            return self.segment_by_hmm(sentence)
        elif method == 'hybrid':
            return self.segment_hybrid(sentence)
        else:
            return self.segment_hybrid(sentence)  # Default to hybrid

def process_test_data(segmenter, test_path, submission_path, sample_submission_path=None, method='hybrid'):
    """Process test data and create submission file"""
    print(f"Processing test data using {method} segmentation method...")
    try:
        # Read test data
        test_df = pd.read_csv(test_path)
        
        # Try to read sample submission if provided (for reference)
        sample_submission = None
        if sample_submission_path and os.path.exists(sample_submission_path):
            try:
                sample_submission = pd.read_csv(sample_submission_path)
                print(f"Loaded sample submission with {len(sample_submission)} entries for reference")
            except:
                print("Could not load sample submission")
        
        # Initialize results
        results = []
        
        # Process each sentence
        start_time = time.time()
        total_sentences = len(test_df)
        
        for idx, row in test_df.iterrows():
            if idx > 0 and idx % 100 == 0:
                elapsed = time.time() - start_time
                estimated_total = elapsed / idx * total_sentences
                remaining = estimated_total - elapsed
                print(f"Processed {idx}/{total_sentences} sentences. Estimated time remaining: {remaining:.2f} seconds")
            
            sentence = row['sentence']
            
            # Segment using the specified method
            segmented = segmenter.segment(sentence, method=method)
            
            # Join with spaces for the transfer function
            segmented_str = ' '.join(segmented)
            
            # Apply the transfer function
            transferred = transfer(segmented_str)
            
            # Add to results
            results.append({
                'id': row['id'],
                'expected': transferred
            })
        
        # Create submission dataframe
        submission_df = pd.DataFrame(results)
        
        # Save to csv
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission file created at {submission_path}")
        
        # Print total processing time
        elapsed = time.time() - start_time
        print(f"Total processing time: {elapsed:.2f} seconds ({elapsed/total_sentences:.4f} seconds per sentence)")
        
    except Exception as e:
        print(f"Error processing test data: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("Starting Chinese word segmentation...")
    start_time = time.time()
    
    # Initialize jieba
    jieba.initialize()
    
    # Create segmenter
    segmenter = ChineseWordSegmenter()
    
    # Load word dictionary
    segmenter.load_word_dict(FREQ_LIST_PATH)
    
    # Train from corpus
    segmenter.train_from_corpus(TRAIN_PATH)
    
    # Process test data
    process_test_data(segmenter, TEST_PATH, SUBMISSION_PATH, SAMPLE_SUBMISSION_PATH)
    
    # Print total time
    elapsed = time.time() - start_time
    print(f"Total execution time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main() 