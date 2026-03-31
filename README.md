# CC-DRL

Official implementation of **CC-DRL**, a confusion-prone-class-aware discriminative representation learning framework for imbalanced DGA-based botnet detection.

This repository contains the code for data preprocessing, multi-channel feature construction, model training, downstream classification, evaluation, and experiment reproduction reported in our paper.

---

## Overview

DGA-based botnet detection often suffers from severe class imbalance across malware families. Existing studies mainly address this issue through resampling or cost-sensitive optimization. In contrast, CC-DRL focuses on a more fundamental bottleneck: some DGA families remain persistently difficult to distinguish because of severe representation overlap in the feature space.

To address this problem, CC-DRL introduces confusion-prone-class-aware discriminative representation learning. The framework consists of three main stages:

1. **MC (Multi-channel Representation Construction)**  
   Extracts complementary features from multiple views of domain names, including character sequence, bi-gram sequence, TLD-related information, and statistical features.

2. **CDR (CC-aware Discriminative Representation Learning)**  
   Dynamically identifies confusion-prone classes and enhances their discriminative representation learning through dedicated branches.

3. **IMB (Imbalance-aware Downstream Classification)**  
   Applies imbalance-aware classification strategies on learned representations for improved family-level recognition.
