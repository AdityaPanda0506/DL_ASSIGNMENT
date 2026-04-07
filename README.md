# 🧠 Content-Disentangled Continual Learning for Medical Imaging

A deep learning framework designed to tackle **catastrophic forgetting** in sequential medical imaging tasks by separating **content (disease)** and **style (scanner/domain)** representations.

---

## 🚀 Overview

In real-world medical applications, models often encounter data from different hospitals, scanners, or imaging conditions. Traditional deep learning models fail in such scenarios due to **catastrophic forgetting** — where learning new tasks erases previously learned knowledge.

This project proposes a **Content-Disentangled Elastic Weight Consolidation (CD-EWC)** approach that:
- Preserves important disease-related features (**content**)
- Allows flexibility in domain-specific variations (**style**)

---

## 🎯 Key Contributions

- ✅ **Content-Style Disentanglement**
  - Separates disease information from scanner/domain variations

- ✅ **Selective EWC Regularization**
  - Applies Fisher-based constraints only to content parameters

- ✅ **Multi-Objective Learning**
  - Combines classification, reconstruction, adversarial, and disentanglement losses

- ✅ **Memory-Efficient Continual Learning**
  - No replay buffer required (unlike DER++)

---

## 📂 Datasets Used

The project uses multiple MedMNIST datasets:

- PathMNIST  
- DermaMNIST  
- BloodMNIST  
- OrganAMNIST  

Each dataset simulates a **different domain/task**.

---

## 🔄 Domain Shift Simulation

To mimic real-world variability, artificial shifts are introduced:

- Gaussian Noise  
- Contrast Scaling  
- Histogram Equalization  
- Poisson Noise  

This ensures the model learns under **non-stationary conditions**.

---

## 🏗️ Model Architecture

### 🔹 Backbone
- ResNet18 encoder (pretrained)

### 🔹 Latent Representation
- **Content vector (z₍c₎)** → disease features  
- **Style vector (z₍s₎)** → domain/scanner features  

### 🔹 Components
- Variational Autoencoder (VAE) heads  
- PatchGAN discriminator  
- Multi-head classifiers (task-specific)  
- Reconstruction & segmentation decoders  

---

## ⚙️ Training Strategy

### Phase 1: Dataset Preparation & EDA
- Data loading, preprocessing, visualization
- Domain shift analysis

### Phase 2: Naive Fine-Tuning
- Sequential learning without constraints
- Demonstrates catastrophic forgetting

### Phase 3: CD-EWC (Proposed Method)
- Fisher-based regularization on content parameters only

### Phase 4: Baseline Comparisons
- EWC  
- DER++ (Replay-based)  
- Domain Adaptation  

### Phase 5: Evaluation & Analysis
- Performance metrics
- Visualization (t-SNE, heatmaps, radar charts)

---

## 📊 Evaluation Metrics

- **Final Accuracy**
- **Backward Transfer (BWT)** → measures forgetting
- **Forward Transfer (FWT)** → knowledge transfer
- **Memory Cost**

---

## 📈 Key Results

| Method           | Accuracy | BWT  | Memory |
|----------------|--------|------|--------|
| Naive FT       | Low    | ❌ High Forgetting | 0 |
| EWC            | Medium | ⚠️ Moderate | 0 |
| CD-EWC (Ours)  | High   | ✅ Improved | 0 |
| DER++          | High   | ✅ Good | 800 images |
| Domain Adapt   | Best   | ✅ No forgetting | Full data |

👉 **Insight:**  
CD-EWC achieves strong performance **without storing past data**, making it efficient and scalable.

---

## 🔍 Visualization Insights

- **t-SNE (Content Space):**
  - Clusters by disease → good feature preservation

- **t-SNE (Style Space):**
  - Clusters by dataset → successful domain separation

---

## ⚠️ Limitations

- Performance depends on EWC regularization strength (λ)
- Mutual Information penalty is an approximation
- Low-resolution images (28×28) limit clinical applicability

---

## 🧪 How to Run

```bash
# Install dependencies
pip install medmnist umap-learn torch torchvision

# Run the script
python dl_lab_23bai1297.py
