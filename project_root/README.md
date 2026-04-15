# eDNA Analysis Pipeline using Variational Autoencoder

This project implements an AI-driven pipeline for deep-sea environmental DNA (eDNA) analysis using a Variational Autoencoder (VAE) to identify eukaryotic taxa and assess biodiversity without relying on complete reference databases.

## 🧬 Project Overview

The pipeline processes raw eDNA reads from FASTQ files, learns meaningful representations using a VAE, and performs unsupervised clustering to identify taxonomic groups and assess biodiversity.

### Key Features

- **Quality Control**: Filters sequences based on quality scores and length
- **One-Hot Encoding**: Converts DNA sequences to numerical format
- **VAE Architecture**: Uses 1D convolutional layers for sequence analysis
- **Unsupervised Clustering**: DBSCAN clustering on latent representations
- **Biodiversity Assessment**: Calculates diversity indices and relative abundances
- **Comprehensive Reporting**: Generates detailed analysis reports and visualizations

## 📁 Project Structure

```
project_root/
├── data/                          # Input data directory
│   ├── sample_01/
│   │   ├── reads.fastq.gz        # Gzipped FASTQ file
│   │   └── metadata.json         # Sample metadata
│   ├── sample_02/
│   │   └── ...
│   └── ...
├── src/                          # Source code
│   ├── data_processor.py         # Data preprocessing functions
│   ├── vae_model.py             # VAE model implementation
│   ├── main.py                  # Main pipeline script
│   └── test_pipeline.py         # Test script
├── outputs/                      # Generated outputs (created during run)
│   ├── edna_analysis_report.json
│   ├── edna_analysis_summary.txt
│   └── latent_space_clustering.png
├── checkpoints/                  # Model checkpoints (created during training)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data

Place your eDNA samples in the `data/` directory with the following structure:

```
data/
├── sample_01/
│   ├── reads.fastq.gz
│   └── metadata.json
├── sample_02/
│   ├── reads.fastq.gz
│   └── metadata.json
└── ...
```

**FASTQ Format**: Standard FASTQ format with quality scores
**Metadata Format**: JSON file with sample information (optional)

Example metadata.json:
```json
{
  "sample_id": "sample_01",
  "location": "Pacific Ocean",
  "depth": "2000m",
  "collection_date": "2024-01-15"
}
```

### 3. Run the Pipeline

```bash
cd src
python main.py
```

### 4. Test the Pipeline

To test with synthetic data:

```bash
cd src
python test_pipeline.py
```

## 🔧 Configuration

You can modify the pipeline parameters in `src/main.py`:

```python
config = {
    'data_dir': '../data',           # Path to data directory
    'sequence_length': 200,          # Fixed sequence length
    'latent_dim': 32,               # Latent space dimension
    'batch_size': 32,               # Training batch size
    'learning_rate': 1e-3,          # Learning rate
    'num_epochs': 50,               # Number of training epochs
    'device': None                  # Auto-detect device (cuda/cpu)
}
```

## 📊 Output Files

The pipeline generates several output files in the `outputs/` directory:

### 1. Analysis Report (`edna_analysis_report.json`)
Comprehensive JSON report containing:
- Analysis metadata and parameters
- Data summary statistics
- Cluster analysis results
- Sample-wise abundance data
- Biodiversity metrics (Shannon and Simpson indices)

### 2. Summary Report (`edna_analysis_summary.txt`)
Human-readable summary of the analysis results

### 3. Visualization (`latent_space_clustering.png`)
2D t-SNE plot showing the clustering results in latent space

## 🧪 Pipeline Components

### 1. Data Preprocessing (`data_processor.py`)

**Functions:**
- `read_fastq_file()`: Reads gzipped FASTQ files
- `filter_sequences()`: Quality control and length filtering
- `one_hot_encode_sequence()`: Converts DNA to numerical format
- `load_all_samples()`: Processes all samples in data directory

**Quality Control Parameters:**
- Minimum quality score: 20.0 (Phred+33)
- Minimum sequence length: 50 bp
- Maximum sequence length: 500 bp

### 2. VAE Model (`vae_model.py`)

**Architecture:**
- **Encoder**: 1D convolutional layers with batch normalization
- **Latent Space**: Gaussian distribution with reparameterization trick
- **Decoder**: 1D transposed convolutional layers
- **Loss Function**: Reconstruction loss + KL divergence

**Key Features:**
- Handles variable sequence lengths through padding/truncation
- Uses softmax activation for nucleotide probability output
- Implements proper weight initialization

### 3. Main Pipeline (`main.py`)

**EDNATrainingPipeline Class:**
- Orchestrates the complete analysis workflow
- Handles training, clustering, and reporting
- Provides comprehensive error handling and logging

**Workflow:**
1. Load and preprocess data
2. Initialize and train VAE model
3. Extract latent representations
4. Perform DBSCAN clustering
5. Calculate abundances and diversity metrics
6. Generate visualizations and reports

## 🔬 Scientific Methodology

### VAE for eDNA Analysis

The Variational Autoencoder learns a compressed representation of DNA sequences that captures:
- **Sequence patterns**: Nucleotide composition and motifs
- **Taxonomic signatures**: Features that distinguish different taxa
- **Biodiversity signals**: Variations that indicate species diversity

### Clustering Strategy

- **DBSCAN**: Identifies dense regions in latent space
- **Noise Points**: Represent potential novel taxa or rare sequences
- **Cluster Centers**: Characterize dominant taxonomic groups

### Biodiversity Metrics

- **Shannon Diversity Index**: Measures species richness and evenness
- **Simpson Diversity Index**: Emphasizes dominant species
- **Relative Abundances**: Quantifies taxonomic composition

## 🛠️ Advanced Usage

### Custom Model Architecture

Modify the VAE architecture in `vae_model.py`:

```python
# Change hidden dimensions
hidden_dims = [64, 128, 256, 512]  # Deeper network

# Adjust latent dimension
latent_dim = 64  # Higher dimensional latent space
```

### Clustering Parameters

Adjust DBSCAN parameters in `main.py`:

```python
# More sensitive clustering
cluster_labels = pipeline.perform_clustering(eps=0.3, min_samples=3)

# More conservative clustering
cluster_labels = pipeline.perform_clustering(eps=1.0, min_samples=10)
```

### GPU Acceleration

The pipeline automatically detects and uses GPU if available. For CUDA-specific optimizations:

```bash
# Install CUDA-specific PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 📈 Performance Optimization

### Memory Management
- Adjust `batch_size` based on available memory
- Use `num_workers=0` for Windows compatibility
- Monitor GPU memory usage during training

### Training Efficiency
- Use learning rate scheduling (already implemented)
- Save checkpoints every 20 epochs
- Early stopping based on validation loss

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size`
   - Use CPU: set `device='cpu'`

2. **No Sequences Found**
   - Check FASTQ file format
   - Verify file paths in data directory
   - Adjust quality filtering parameters

3. **Poor Clustering Results**
   - Increase `num_epochs` for better training
   - Adjust DBSCAN parameters
   - Check data quality and diversity

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 References

- Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.
- Ester, M., et al. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise.
- Smart India Hackathon 2024 - Deep Sea eDNA Analysis Problem Statement

## 🤝 Contributing

This pipeline is designed for the Smart India Hackathon. For improvements or bug fixes:

1. Test changes with `test_pipeline.py`
2. Ensure compatibility with existing data formats
3. Update documentation for new features

## 📄 License

This project is developed for educational and research purposes as part of the Smart India Hackathon 2024.

---

**Happy eDNA Analysis! 🧬🔬**
