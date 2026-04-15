"""
DNA Sequence Preprocessing Module for eDNA Analysis

This module provides functions for reading FASTQ files, performing quality control,
and converting DNA sequences to one-hot encoded format for VAE training.
"""

import gzip
import numpy as np
import os
import json
from typing import List, Tuple, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_fastq_file(file_path: str) -> List[Tuple[str, str, str]]:
    """
    Read a gzipped FASTQ file and return sequences with quality scores.
    
    Args:
        file_path (str): Path to the gzipped FASTQ file
        
    Returns:
        List[Tuple[str, str, str]]: List of (header, sequence, quality) tuples
    """
    sequences = []
    
    try:
        with gzip.open(file_path, 'rt') as f:
            while True:
                header = f.readline().strip()
                if not header:
                    break
                sequence = f.readline().strip()
                f.readline()  # Skip the '+' line
                quality = f.readline().strip()
                
                if header and sequence and quality:
                    sequences.append((header, sequence, quality))
                    
    except Exception as e:
        logger.error(f"Error reading FASTQ file {file_path}: {e}")
        raise
        
    logger.info(f"Read {len(sequences)} sequences from {file_path}")
    return sequences

def calculate_quality_score(quality_string: str) -> float:
    """
    Calculate average quality score for a sequence using Phred+33 encoding.
    
    Args:
        quality_string (str): Quality string from FASTQ file
        
    Returns:
        float: Average quality score
    """
    quality_scores = [ord(char) - 33 for char in quality_string]
    return np.mean(quality_scores)

def filter_sequences(sequences: List[Tuple[str, str, str]], 
                    min_quality: float = 20.0,
                    min_length: int = 50,
                    max_length: int = 500) -> List[str]:
    """
    Filter sequences based on quality and length criteria.
    
    Args:
        sequences (List[Tuple[str, str, str]]): List of (header, sequence, quality) tuples
        min_quality (float): Minimum average quality score (default: 20.0)
        min_length (int): Minimum sequence length (default: 50)
        max_length (int): Maximum sequence length (default: 500)
        
    Returns:
        List[str]: List of filtered DNA sequences
    """
    filtered_sequences = []
    
    for header, sequence, quality in sequences:
        # Check sequence length
        if len(sequence) < min_length or len(sequence) > max_length:
            continue
            
        # Check quality score
        avg_quality = calculate_quality_score(quality)
        if avg_quality < min_quality:
            continue
            
        # Check for valid nucleotides (A, T, C, G)
        if not all(nucleotide in 'ATCG' for nucleotide in sequence.upper()):
            continue
            
        filtered_sequences.append(sequence.upper())
    
    logger.info(f"Filtered {len(filtered_sequences)} sequences from {len(sequences)} total")
    return filtered_sequences

def process_sample_directory(sample_dir: str) -> List[str]:
    """
    Process a sample directory containing reads.fastq.gz and metadata.json.
    
    Args:
        sample_dir (str): Path to the sample directory
        
    Returns:
        List[str]: List of cleaned DNA sequences
    """
    fastq_path = os.path.join(sample_dir, "reads.fastq.gz")
    metadata_path = os.path.join(sample_dir, "metadata.json")
    
    if not os.path.exists(fastq_path):
        raise FileNotFoundError(f"FASTQ file not found: {fastq_path}")
    
    # Read metadata if available
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logger.info(f"Loaded metadata: {metadata}")
    
    # Read and filter sequences
    sequences = read_fastq_file(fastq_path)
    filtered_sequences = filter_sequences(sequences)
    
    return filtered_sequences

def one_hot_encode_sequence(sequence: str, sequence_length: int = 200) -> np.ndarray:
    """
    Convert a DNA sequence to one-hot encoded format.
    
    Args:
        sequence (str): DNA sequence
        sequence_length (int): Fixed length for all sequences
        
    Returns:
        np.ndarray: One-hot encoded sequence of shape (4, sequence_length)
    """
    # Nucleotide to index mapping
    nucleotide_map = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    
    # Initialize one-hot matrix
    one_hot = np.zeros((4, sequence_length), dtype=np.float32)
    
    # Process sequence
    for i, nucleotide in enumerate(sequence[:sequence_length]):
        if nucleotide in nucleotide_map:
            one_hot[nucleotide_map[nucleotide], i] = 1.0
    
    return one_hot

def sequences_to_one_hot(sequences: List[str], sequence_length: int = 200) -> np.ndarray:
    """
    Convert a list of DNA sequences to one-hot encoded format.
    
    Args:
        sequences (List[str]): List of DNA sequences
        sequence_length (int): Fixed length for all sequences
        
    Returns:
        np.ndarray: One-hot encoded sequences of shape (n_sequences, 4, sequence_length)
    """
    one_hot_sequences = []
    
    for sequence in sequences:
        one_hot_seq = one_hot_encode_sequence(sequence, sequence_length)
        one_hot_sequences.append(one_hot_seq)
    
    return np.array(one_hot_sequences)

def load_all_samples(data_dir: str, sequence_length: int = 200) -> Tuple[np.ndarray, List[str]]:
    """
    Load and preprocess all samples from the data directory.
    
    Args:
        data_dir (str): Path to the data directory
        sequence_length (int): Fixed length for all sequences
        
    Returns:
        Tuple[np.ndarray, List[str]]: (one_hot_sequences, sample_names)
    """
    all_sequences = []
    sample_names = []
    
    # Find all sample directories
    for item in os.listdir(data_dir):
        sample_path = os.path.join(data_dir, item)
        if os.path.isdir(sample_path):
            try:
                sequences = process_sample_directory(sample_path)
                if sequences:  # Only add if we have sequences
                    all_sequences.extend(sequences)
                    sample_names.extend([item] * len(sequences))
                    logger.info(f"Processed {len(sequences)} sequences from {item}")
            except Exception as e:
                logger.error(f"Error processing sample {item}: {e}")
                continue
    
    if not all_sequences:
        raise ValueError("No sequences found in any sample directories")
    
    # Convert to one-hot encoding
    one_hot_sequences = sequences_to_one_hot(all_sequences, sequence_length)
    
    logger.info(f"Total sequences processed: {len(all_sequences)}")
    logger.info(f"One-hot array shape: {one_hot_sequences.shape}")
    
    return one_hot_sequences, sample_names

if __name__ == "__main__":
    # Test the data processing functions
    import sys
    
    # Test with a sample directory if provided
    if len(sys.argv) > 1:
        sample_dir = sys.argv[1]
        try:
            sequences = process_sample_directory(sample_dir)
            print(f"Processed {len(sequences)} sequences")
            
            if sequences:
                # Test one-hot encoding
                one_hot = sequences_to_one_hot(sequences[:5])  # Test with first 5 sequences
                print(f"One-hot shape: {one_hot.shape}")
                print("Sample one-hot sequence (first 10 positions):")
                print(one_hot[0, :, :10])
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python data_processor.py <sample_directory>")
        print("Example: python data_processor.py ../data/sample_01")
