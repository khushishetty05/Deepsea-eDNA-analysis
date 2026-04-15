# Deep-Sea eDNA AI-Driven Analysis Pipeline
**Smart India Hackathon 2024-25 | Problem Statement ID: SIH-1715**

## Overview
Deep-sea ecosystems harbor vast biodiversity that remains largely undiscovered. Traditional analysis of Environmental DNA (eDNA) relies heavily on reference databases like SILVA or NCBI. However, these databases often lack comprehensive data for deep-sea taxa, leading to significant "unassigned reads" and an underestimation of biodiversity.

This project implements an **AI-driven pipeline** using **Unsupervised Deep Learning (Variational Autoencoders)** to classify eukaryotic taxa and assess biodiversity directly from raw sequencing reads, significantly reducing reliance on incomplete reference databases.

## Key Features
* **Unsupervised Feature Learning:** Learns latent genomic signatures from raw eDNA sequences without needing taxonomic labels.
* **Novel Taxa Discovery:** Employs **DBSCAN Clustering** in the latent space to identify potentially undiscovered species.
* **End-to-End Pipeline:** Handles everything from raw FASTQ preprocessing to community abundance reporting.
* **PyTorch Core:** Leverages PyTorch's dynamic computation graphs for efficient VAE training and feature extraction.

## Project Structure
```text
project_root/
├── data/               # Raw sequencing reads (FASTQ.GZ) and metadata (JSON)
├── src/                # Core pipeline source code
│   ├── data_processor.py  # Quality filtering & One-hot encoding
│   ├── vae_model.py       # PyTorch VAE Architecture
│   └── main.py            # Pipeline orchestration and training
├── notebooks/          # Exploratory Data Analysis (EDA)
├── README.md           # Project documentation
└── requirements.txt    # Python dependencies
