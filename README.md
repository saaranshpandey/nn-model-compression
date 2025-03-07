# Neural Network Model Compression & Transfer Learning
**ESE 539: HW/SW Co-Design for ML (Fall 2024)**  
Final Project – Model Pruning & Compression

---

## 1. Overview
This project investigates multiple **neural network pruning** strategies (global pruning, layer-wise pruning, channel pruning) to reduce model size and computational overhead while retaining acceptable accuracy. It further explores **iterative pruning** to progressively prune and retrain models for better performance, and demonstrates how compressed networks can be adapted to new tasks via **transfer learning** (e.g., melanoma classification).

---

## 2. Project Notebooks
- **`proj_pt1.ipynb`**  
  - Implements:
    - Global Weight Pruning 
    - Layer-wise Weight Pruning 
    - Layer-wise Output Channel Pruning 
  - Evaluates performance (accuracy) on a pretrained VGG16 (ImageNet validation set).
- **`proj_pt2.ipynb`**  
  - Focuses on **iterative pruning** and extends it to **transfer learning**.
  - Shows how pruned models can be fine-tuned on a new dataset (e.g., melanoma classification).
- **`/results/` folder**  
  - Contains images/plots illustrating pruning outcomes (already included in the report).

*For additional details, refer to the* **ESE_539_Project_Report_.pdf** and **ESE 539 Project Presentation.pptx**.

---

## 3. Key Methods
1. **Global Weight Pruning**  
   - Removes the smallest-magnitude weights network-wide (L1 norm), preserving accuracy well at moderate sparsity levels.
2. **Layer-wise Weight Pruning**  
   - Prunes each layer independently; can maintain performance at low sparsity but declines faster at higher pruning rates.
3. **Layer-wise Channel Pruning**  
   - Removes entire output channels (structured pruning). Improves runtime/size but can sharply reduce accuracy if done aggressively.
4. **Hardened Channel Pruning**  
   - Rebuilds the model architecture to remove pruned channels, cutting actual computation and memory usage.
5. **Iterative Pruning**  
   - Repeatedly prune and retrain. Allows higher compression rates before accuracy noticeably degrades.
6. **Transfer Learning Extension**  
   - Fine-tunes a pruned model on a new task (e.g., melanoma classification), demonstrating compressed models’ adaptability.

---

## 4. Key Results
- **Global vs. Layer-wise Pruning:** Global pruning tended to preserve accuracy better at higher pruning percentages.  
- **Channel Pruning (Structured):** Notable runtime and model size improvements, but larger accuracy drop without retraining.  
- **Iterative Pruning:** Achieved significant weight reduction (~50–80% pruned) with minimal accuracy loss by retraining after each prune step.  
- **Transfer Learning:** Compressed models adapted well to the melanoma dataset, showing minimal performance impact and faster fine-tuning.

---

## 5. Conclusion
Pruning strategies offer significant **model compression** for large architectures like VGG16 while preserving accuracy. **Iterative pruning** with subsequent retraining yields particularly strong results, and **transfer learning** remains feasible with these compressed models. This approach suits **resource-constrained deployments** where both efficiency and performance are critical.

---
_**Contributors**: Saaransh Pandey, Aakash Agarwal._
