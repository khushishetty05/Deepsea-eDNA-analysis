"""
Main Pipeline for eDNA Analysis using Variational Autoencoder

This script orchestrates the complete pipeline:
1. Data loading and preprocessing
2. VAE training
3. Latent space clustering
4. Biodiversity assessment and reporting
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Any

# Import our custom modules
from data_processor import load_all_samples
from vae_model import create_vae_model, vae_loss

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('edna_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EDNATrainingPipeline:
    """
    Main pipeline class for eDNA analysis using VAE.
    """
    
    def __init__(self, 
                 data_dir: str = "../data",
                 sequence_length: int = 200,
                 latent_dim: int = 32,
                 batch_size: int = 32,
                 learning_rate: float = 1e-3,
                 num_epochs: int = 100,
                 device: str = None):
        """
        Initialize the training pipeline.
        
        Args:
            data_dir (str): Path to data directory
            sequence_length (int): Fixed sequence length
            latent_dim (int): Latent space dimension
            batch_size (int): Training batch size
            learning_rate (float): Learning rate for optimizer
            num_epochs (int): Number of training epochs
            device (str): Device to use ('cuda' or 'cpu')
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Pipeline initialized with device: {self.device}")
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.data_loader = None
        self.latent_vectors = None
        self.cluster_labels = None
        self.cluster_centers = None
        
    def load_and_preprocess_data(self) -> Tuple[torch.Tensor, List[str]]:
        """
        Load and preprocess all eDNA samples.
        
        Returns:
            Tuple[torch.Tensor, List[str]]: (processed_data, sample_names)
        """
        logger.info("Loading and preprocessing data...")
        
        try:
            # Load all samples
            one_hot_sequences, sample_names = load_all_samples(
                self.data_dir, 
                self.sequence_length
            )
            
            # Convert to PyTorch tensors
            data_tensor = torch.FloatTensor(one_hot_sequences)
            
            # Create data loader
            dataset = TensorDataset(data_tensor)
            self.data_loader = DataLoader(
                dataset, 
                batch_size=self.batch_size, 
                shuffle=True,
                num_workers=0  # Set to 0 for Windows compatibility
            )
            
            logger.info(f"Data loaded: {data_tensor.shape[0]} sequences")
            logger.info(f"Data shape: {data_tensor.shape}")
            
            return data_tensor, sample_names
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def initialize_model(self):
        """Initialize the VAE model and optimizer."""
        logger.info("Initializing VAE model...")
        
        # Create model
        self.model = create_vae_model(
            sequence_length=self.sequence_length,
            latent_dim=self.latent_dim,
            device=self.device
        )
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=10,
            verbose=True
        )
        
        logger.info("Model and optimizer initialized")
    
    def train_model(self, data_tensor: torch.Tensor) -> List[float]:
        """
        Train the VAE model.
        
        Args:
            data_tensor (torch.Tensor): Training data
            
        Returns:
            List[float]: Training losses per epoch
        """
        logger.info(f"Starting training for {self.num_epochs} epochs...")
        
        self.model.train()
        training_losses = []
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (batch_data,) in enumerate(self.data_loader):
                batch_data = batch_data.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                reconstructed, mu, log_var = self.model(batch_data)
                
                # Calculate loss
                loss = vae_loss(reconstructed, batch_data, mu, log_var)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    logger.info(f'Epoch {epoch+1}/{self.num_epochs}, '
                              f'Batch {batch_idx}/{len(self.data_loader)}, '
                              f'Loss: {loss.item():.4f}')
            
            avg_epoch_loss = epoch_loss / num_batches
            training_losses.append(avg_epoch_loss)
            
            # Update learning rate
            self.scheduler.step(avg_epoch_loss)
            
            logger.info(f'Epoch {epoch+1}/{self.num_epochs} completed. '
                       f'Average Loss: {avg_epoch_loss:.4f}')
            
            # Save checkpoint every 20 epochs
            if (epoch + 1) % 20 == 0:
                self.save_checkpoint(epoch + 1, avg_epoch_loss)
        
        logger.info("Training completed!")
        return training_losses
    
    def extract_latent_representations(self, data_tensor: torch.Tensor) -> np.ndarray:
        """
        Extract latent representations for all sequences.
        
        Args:
            data_tensor (torch.Tensor): Input data
            
        Returns:
            np.ndarray: Latent representations
        """
        logger.info("Extracting latent representations...")
        
        self.model.eval()
        latent_vectors = []
        
        with torch.no_grad():
            for batch_idx, (batch_data,) in enumerate(self.data_loader):
                batch_data = batch_data.to(self.device)
                latent_batch = self.model.get_latent_representations(batch_data)
                latent_vectors.append(latent_batch.cpu().numpy())
                
                if batch_idx % 10 == 0:
                    logger.info(f'Processed batch {batch_idx}/{len(self.data_loader)}')
        
        self.latent_vectors = np.vstack(latent_vectors)
        logger.info(f"Latent representations extracted: {self.latent_vectors.shape}")
        
        return self.latent_vectors
    
    def perform_clustering(self, eps: float = 0.5, min_samples: int = 5) -> np.ndarray:
        """
        Perform DBSCAN clustering on latent representations.
        
        Args:
            eps (float): DBSCAN eps parameter
            min_samples (int): DBSCAN min_samples parameter
            
        Returns:
            np.ndarray: Cluster labels
        """
        logger.info("Performing DBSCAN clustering...")
        
        if self.latent_vectors is None:
            raise ValueError("Latent representations not extracted. Run extract_latent_representations first.")
        
        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        self.cluster_labels = dbscan.fit_predict(self.latent_vectors)
        
        # Calculate cluster centers
        unique_labels = np.unique(self.cluster_labels)
        self.cluster_centers = {}
        
        for label in unique_labels:
            if label != -1:  # Skip noise points
                mask = self.cluster_labels == label
                center = np.mean(self.latent_vectors[mask], axis=0)
                self.cluster_centers[label] = center
        
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = np.sum(self.cluster_labels == -1)
        
        logger.info(f"Clustering completed: {n_clusters} clusters, {n_noise} noise points")
        
        return self.cluster_labels
    
    def calculate_abundances(self, sample_names: List[str]) -> Dict[str, Any]:
        """
        Calculate relative abundances for each cluster.
        
        Args:
            sample_names (List[str]): Names of samples
            
        Returns:
            Dict[str, Any]: Abundance statistics
        """
        logger.info("Calculating cluster abundances...")
        
        if self.cluster_labels is None:
            raise ValueError("Clustering not performed. Run perform_clustering first.")
        
        unique_labels = np.unique(self.cluster_labels)
        abundances = {}
        
        for label in unique_labels:
            if label == -1:
                abundances['noise'] = np.sum(self.cluster_labels == label)
            else:
                abundances[f'cluster_{label}'] = np.sum(self.cluster_labels == label)
        
        # Calculate relative abundances
        total_sequences = len(self.cluster_labels)
        relative_abundances = {
            cluster: count / total_sequences 
            for cluster, count in abundances.items()
        }
        
        # Per-sample analysis
        sample_abundances = {}
        unique_samples = list(set(sample_names))
        
        for sample in unique_samples:
            sample_mask = np.array(sample_names) == sample
            sample_labels = self.cluster_labels[sample_mask]
            
            sample_abundances[sample] = {}
            for label in unique_labels:
                if label == -1:
                    sample_abundances[sample]['noise'] = np.sum(sample_labels == label)
                else:
                    sample_abundances[sample][f'cluster_{label}'] = np.sum(sample_labels == label)
        
        return {
            'absolute_abundances': abundances,
            'relative_abundances': relative_abundances,
            'sample_abundances': sample_abundances,
            'total_sequences': total_sequences,
            'n_clusters': len(unique_labels) - (1 if -1 in unique_labels else 0),
            'n_noise': abundances.get('noise', 0)
        }
    
    def visualize_latent_space(self, output_dir: str = "outputs"):
        """
        Create 2D visualization of the latent space.
        
        Args:
            output_dir (str): Directory to save plots
        """
        logger.info("Creating latent space visualization...")
        
        if self.latent_vectors is None or self.cluster_labels is None:
            raise ValueError("Latent representations or cluster labels not available.")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Reduce dimensionality using t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        latent_2d = tsne.fit_transform(self.latent_vectors)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot clusters
        unique_labels = np.unique(self.cluster_labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                # Noise points
                mask = self.cluster_labels == label
                plt.scatter(latent_2d[mask, 0], latent_2d[mask, 1], 
                           c='black', marker='x', s=20, alpha=0.6, label='Noise')
            else:
                mask = self.cluster_labels == label
                plt.scatter(latent_2d[mask, 0], latent_2d[mask, 1], 
                           c=[color], s=30, alpha=0.7, label=f'Cluster {label}')
        
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.title('VAE Latent Space Clustering')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, 'latent_space_clustering.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Latent space visualization saved to {plot_path}")
    
    def generate_report(self, abundances: Dict[str, Any], output_dir: str = "outputs") -> str:
        """
        Generate comprehensive analysis report.
        
        Args:
            abundances (Dict[str, Any]): Abundance statistics
            output_dir (str): Directory to save report
            
        Returns:
            str: Path to the generated report
        """
        logger.info("Generating analysis report...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate report content
        report_content = {
            'analysis_info': {
                'timestamp': datetime.now().isoformat(),
                'pipeline_version': '1.0.0',
                'device_used': str(self.device),
                'model_parameters': {
                    'sequence_length': self.sequence_length,
                    'latent_dim': self.latent_dim,
                    'batch_size': self.batch_size,
                    'learning_rate': self.learning_rate,
                    'num_epochs': self.num_epochs
                }
            },
            'data_summary': {
                'total_sequences': abundances['total_sequences'],
                'n_clusters_found': abundances['n_clusters'],
                'n_noise_points': abundances['n_noise']
            },
            'cluster_analysis': {
                'absolute_abundances': abundances['absolute_abundances'],
                'relative_abundances': abundances['relative_abundances']
            },
            'sample_analysis': abundances['sample_abundances'],
            'biodiversity_metrics': {
                'shannon_diversity': self._calculate_shannon_diversity(abundances['relative_abundances']),
                'simpson_diversity': self._calculate_simpson_diversity(abundances['relative_abundances'])
            }
        }
        
        # Save JSON report
        json_path = os.path.join(output_dir, 'edna_analysis_report.json')
        with open(json_path, 'w') as f:
            json.dump(report_content, f, indent=2)
        
        # Generate text summary
        text_report = self._generate_text_summary(report_content)
        txt_path = os.path.join(output_dir, 'edna_analysis_summary.txt')
        with open(txt_path, 'w') as f:
            f.write(text_report)
        
        logger.info(f"Report generated: {json_path}")
        logger.info(f"Summary generated: {txt_path}")
        
        return json_path
    
    def _calculate_shannon_diversity(self, relative_abundances: Dict[str, float]) -> float:
        """Calculate Shannon diversity index."""
        # Exclude noise from diversity calculation
        abundances = [p for cluster, p in relative_abundances.items() if cluster != 'noise']
        if not abundances:
            return 0.0
        
        return -sum(p * np.log(p) for p in abundances if p > 0)
    
    def _calculate_simpson_diversity(self, relative_abundances: Dict[str, float]) -> float:
        """Calculate Simpson diversity index."""
        # Exclude noise from diversity calculation
        abundances = [p for cluster, p in relative_abundances.items() if cluster != 'noise']
        if not abundances:
            return 0.0
        
        return 1 - sum(p**2 for p in abundances)
    
    def _generate_text_summary(self, report_content: Dict[str, Any]) -> str:
        """Generate human-readable text summary."""
        summary = f"""
eDNA Analysis Report
===================

Analysis Date: {report_content['analysis_info']['timestamp']}
Pipeline Version: {report_content['analysis_info']['pipeline_version']}

Data Summary:
- Total sequences analyzed: {report_content['data_summary']['total_sequences']}
- Number of clusters found: {report_content['data_summary']['n_clusters_found']}
- Noise points (potential novel taxa): {report_content['data_summary']['n_noise_points']}

Biodiversity Metrics:
- Shannon Diversity Index: {report_content['biodiversity_metrics']['shannon_diversity']:.4f}
- Simpson Diversity Index: {report_content['biodiversity_metrics']['simpson_diversity']:.4f}

Cluster Abundances (Relative):
"""
        
        for cluster, abundance in report_content['cluster_analysis']['relative_abundances'].items():
            summary += f"- {cluster}: {abundance:.4f} ({abundance*100:.2f}%)\n"
        
        summary += f"\nSample-wise Analysis:\n"
        for sample, abundances in report_content['sample_analysis'].items():
            summary += f"\n{sample}:\n"
            for cluster, count in abundances.items():
                summary += f"  - {cluster}: {count} sequences\n"
        
        return summary
    
    def save_checkpoint(self, epoch: int, loss: float, output_dir: str = "checkpoints"):
        """Save model checkpoint."""
        os.makedirs(output_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'model_config': {
                'sequence_length': self.sequence_length,
                'latent_dim': self.latent_dim
            }
        }
        
        checkpoint_path = os.path.join(output_dir, f'vae_checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def run_complete_pipeline(self) -> str:
        """
        Run the complete eDNA analysis pipeline.
        
        Returns:
            str: Path to the generated report
        """
        logger.info("Starting complete eDNA analysis pipeline...")
        
        try:
            # Step 1: Load and preprocess data
            data_tensor, sample_names = self.load_and_preprocess_data()
            
            # Step 2: Initialize model
            self.initialize_model()
            
            # Step 3: Train model
            training_losses = self.train_model(data_tensor)
            
            # Step 4: Extract latent representations
            self.extract_latent_representations(data_tensor)
            
            # Step 5: Perform clustering
            self.perform_clustering()
            
            # Step 6: Calculate abundances
            abundances = self.calculate_abundances(sample_names)
            
            # Step 7: Generate visualizations
            self.visualize_latent_space()
            
            # Step 8: Generate report
            report_path = self.generate_report(abundances)
            
            logger.info("Pipeline completed successfully!")
            return report_path
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

def main():
    """Main function to run the eDNA analysis pipeline."""
    
    # Configuration
    config = {
        'data_dir': '../data',
        'sequence_length': 200,
        'latent_dim': 32,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'num_epochs': 50,  # Reduced for testing
        'device': None  # Auto-detect
    }
    
    # Create and run pipeline
    pipeline = EDNATrainingPipeline(**config)
    
    try:
        report_path = pipeline.run_complete_pipeline()
        print(f"\nAnalysis completed successfully!")
        print(f"Report saved to: {report_path}")
        print(f"Check the 'outputs' directory for visualizations and detailed results.")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
