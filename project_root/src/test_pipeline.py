"""
Test script for the eDNA analysis pipeline.

This script creates synthetic data to test the pipeline components
without requiring real FASTQ files.
"""

import os
import sys
import numpy as np
import torch
from typing import List
import logging

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processor import sequences_to_one_hot, one_hot_encode_sequence
from vae_model import create_vae_model, vae_loss
from main import EDNATrainingPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_synthetic_dna_sequences(n_sequences: int = 1000, 
                                   sequence_length: int = 200) -> List[str]:
    """
    Generate synthetic DNA sequences for testing.
    
    Args:
        n_sequences (int): Number of sequences to generate
        sequence_length (int): Length of each sequence
        
    Returns:
        List[str]: List of synthetic DNA sequences
    """
    nucleotides = ['A', 'T', 'C', 'G']
    sequences = []
    
    # Generate sequences with some patterns to create clusters
    for i in range(n_sequences):
        if i < n_sequences // 3:
            # First cluster: AT-rich sequences
            seq = np.random.choice(['A', 'T'], size=sequence_length, p=[0.6, 0.4])
        elif i < 2 * n_sequences // 3:
            # Second cluster: GC-rich sequences
            seq = np.random.choice(['G', 'C'], size=sequence_length, p=[0.5, 0.5])
        else:
            # Third cluster: mixed sequences
            seq = np.random.choice(nucleotides, size=sequence_length, p=[0.25, 0.25, 0.25, 0.25])
        
        sequences.append(''.join(seq))
    
    return sequences

def create_synthetic_data_directory(data_dir: str = "../data"):
    """
    Create synthetic data directory structure for testing.
    
    Args:
        data_dir (str): Path to data directory
    """
    # Create sample directories
    sample_dirs = ['sample_01', 'sample_02', 'sample_03']
    
    for sample_dir in sample_dirs:
        sample_path = os.path.join(data_dir, sample_dir)
        os.makedirs(sample_path, exist_ok=True)
        
        # Generate synthetic sequences
        sequences = generate_synthetic_dna_sequences(n_sequences=200, sequence_length=200)
        
        # Create synthetic FASTQ file
        fastq_path = os.path.join(sample_path, "reads.fastq.gz")
        import gzip
        
        with gzip.open(fastq_path, 'wt') as f:
            for i, seq in enumerate(sequences):
                # Generate synthetic quality scores (high quality)
                quality = ''.join(['I'] * len(seq))  # High quality score
                
                f.write(f"@read_{i}\n")
                f.write(f"{seq}\n")
                f.write("+\n")
                f.write(f"{quality}\n")
        
        # Create metadata file
        metadata = {
            "sample_id": sample_dir,
            "location": f"test_location_{sample_dir}",
            "depth": "1000m",
            "collection_date": "2024-01-01"
        }
        
        import json
        metadata_path = os.path.join(sample_path, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Created synthetic data for {sample_dir}")

def test_data_processor():
    """Test the data processing functions."""
    logger.info("Testing data processor...")
    
    # Test one-hot encoding
    test_sequence = "ATCGATCG"
    one_hot = one_hot_encode_sequence(test_sequence, sequence_length=10)
    
    assert one_hot.shape == (4, 10), f"Expected shape (4, 10), got {one_hot.shape}"
    logger.info("✓ One-hot encoding test passed")
    
    # Test batch processing
    sequences = ["ATCG", "GCTA", "AAAA"]
    batch_one_hot = sequences_to_one_hot(sequences, sequence_length=10)
    
    assert batch_one_hot.shape == (3, 4, 10), f"Expected shape (3, 4, 10), got {batch_one_hot.shape}"
    logger.info("✓ Batch processing test passed")

def test_vae_model():
    """Test the VAE model."""
    logger.info("Testing VAE model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_vae_model(sequence_length=200, latent_dim=32, device=device)
    
    # Test forward pass
    batch_size = 4
    test_input = torch.randn(batch_size, 4, 200).to(device)
    
    reconstructed, mu, log_var = model(test_input)
    
    assert reconstructed.shape == test_input.shape, "Reconstruction shape mismatch"
    assert mu.shape == (batch_size, 32), "Mu shape mismatch"
    assert log_var.shape == (batch_size, 32), "Log var shape mismatch"
    
    # Test loss calculation
    loss = vae_loss(reconstructed, test_input, mu, log_var)
    assert loss.item() > 0, "Loss should be positive"
    
    logger.info("✓ VAE model test passed")

def test_pipeline_integration():
    """Test the complete pipeline with synthetic data."""
    logger.info("Testing complete pipeline...")
    
    # Create synthetic data
    create_synthetic_data_directory()
    
    # Create pipeline with reduced parameters for testing
    config = {
        'data_dir': '../data',
        'sequence_length': 200,
        'latent_dim': 16,  # Reduced for faster testing
        'batch_size': 16,
        'learning_rate': 1e-3,
        'num_epochs': 5,  # Very few epochs for testing
        'device': None
    }
    
    pipeline = EDNATrainingPipeline(**config)
    
    try:
        # Test data loading
        data_tensor, sample_names = pipeline.load_and_preprocess_data()
        logger.info(f"✓ Data loading test passed: {data_tensor.shape}")
        
        # Test model initialization
        pipeline.initialize_model()
        logger.info("✓ Model initialization test passed")
        
        # Test training (just a few epochs)
        training_losses = pipeline.train_model(data_tensor)
        logger.info(f"✓ Training test passed: {len(training_losses)} epochs")
        
        # Test latent extraction
        latent_vectors = pipeline.extract_latent_representations(data_tensor)
        logger.info(f"✓ Latent extraction test passed: {latent_vectors.shape}")
        
        # Test clustering
        cluster_labels = pipeline.perform_clustering(eps=1.0, min_samples=5)
        logger.info(f"✓ Clustering test passed: {len(np.unique(cluster_labels))} clusters")
        
        # Test abundance calculation
        abundances = pipeline.calculate_abundances(sample_names)
        logger.info("✓ Abundance calculation test passed")
        
        # Test visualization
        pipeline.visualize_latent_space()
        logger.info("✓ Visualization test passed")
        
        # Test report generation
        report_path = pipeline.generate_report(abundances)
        logger.info(f"✓ Report generation test passed: {report_path}")
        
        logger.info("🎉 All pipeline tests passed!")
        
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        raise

def main():
    """Run all tests."""
    logger.info("Starting pipeline tests...")
    
    try:
        # Test individual components
        test_data_processor()
        test_vae_model()
        
        # Test complete pipeline
        test_pipeline_integration()
        
        print("\n" + "="*50)
        print("🎉 ALL TESTS PASSED!")
        print("="*50)
        print("\nYour eDNA analysis pipeline is ready to use!")
        print("\nTo run the pipeline with real data:")
        print("1. Place your FASTQ files in the data/ directory")
        print("2. Run: python src/main.py")
        print("\nTo run with custom parameters, modify the config in main.py")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
