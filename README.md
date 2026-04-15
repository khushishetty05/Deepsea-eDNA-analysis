# 🌊 Deep-Sea eDNA Analysis using AI
🎯 Focus: Discovering hidden biodiversity using unsupervised learning

**Smart India Hackathon 2024–25 | SIH-1715**

⭐ Using deep learning to uncover hidden marine biodiversity from DNA data

---

## 📌 Problem Statement

Deep-sea ecosystems contain vast biodiversity, but:

* Many species are **not present in reference databases** (SILVA, NCBI)
* Traditional methods fail to classify large portions of DNA (**“unassigned reads”**)
* This leads to **underestimation of biodiversity**

---

## 💡 Solution

This project builds an **AI-driven pipeline** that analyzes environmental DNA (eDNA) using:

* **Variational Autoencoders (VAE)** for feature learning
* **Clustering (DBSCAN)** to detect unknown species

👉 Reduces dependence on incomplete biological databases
👉 Enables discovery of **novel taxa**

---

## 🚀 Key Features

* 🧠 **Unsupervised Learning**
  Learns patterns directly from raw DNA sequences (no labels required)

* 🧬 **Latent Feature Extraction**
  Encodes genomic data into meaningful representations using VAE

* 🔍 **Novel Species Detection**
  Identifies unknown taxa via clustering in latent space

* ⚙️ **End-to-End Pipeline**
  From raw FASTQ data → preprocessing → modeling → insights

* ⚡ **Efficient Training**
  Built using PyTorch for scalable deep learning workflows

---

## 🛠️ Tech Stack

* Python
* PyTorch
* NumPy / Pandas
* Bioinformatics Data (FASTQ format)

---

## ⚙️ How It Works

1. Preprocess raw DNA sequences (quality filtering + encoding)
2. Convert sequences into numerical format (one-hot encoding)
3. Train Variational Autoencoder (VAE)
4. Extract latent representations
5. Apply DBSCAN clustering
6. Identify known vs unknown taxa

---

## 📂 Project Structure

```text
project_root/
├── data/                  # Raw eDNA sequencing data
├── src/                   # Core pipeline
│   ├── data_processor.py  # Preprocessing & encoding
│   ├── vae_model.py       # VAE architecture
│   └── main.py            # Pipeline execution
├── notebooks/             # EDA & experimentation
├── requirements.txt
└── README.md
```

---

## 📊 Key Learnings

* Applied **deep learning to real-world biological data**
* Built an **unsupervised ML pipeline from scratch**
* Understood challenges of **incomplete datasets in bioinformatics**
* Worked with **sequence data and high-dimensional feature spaces**

---

## 🚀 Future Improvements

* Integrate larger biological datasets
* Improve clustering accuracy
* Deploy as a research tool or API

---

## 🤝 Acknowledgment

Developed as part of **Smart India Hackathon 2024–25**

---
