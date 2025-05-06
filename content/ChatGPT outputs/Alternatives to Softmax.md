
Here’s a concise overview of the main families of alternatives to the standard Softmax layer in Transformer architectures, organized by their core idea:

In essence, many proposals aim to either **induce sparsity** in the attention weights, **linearize** or **approximate** the Softmax computation for efficiency, or **replace** the exponential normalization altogether with other similarity measures or activation functions. The choice often trades off between computational complexity, ease of implementation, and performance on downstream tasks.

## 1. Sparse Normalization Schemes

These methods replace Softmax with a normalization that produces _sparse_ attention distributions—assigning exact zeros to many positions.

- **Sparsemax**  
    Transforms raw scores into a probability simplex but yields exact zeros for low-scoring entries, promoting interpretability and efficiency in downstream sparsity [arXiv](https://arxiv.org/abs/1602.02068?utm_source=chatgpt.com).
    
- **α-Entmax (including EntMax1.5)**  
    A generalization of Softmax that interpolates between Softmax (α→1) and Sparsemax (α=2), allowing learned sparsity levels per head [NeurIPS Proceedings](https://proceedings.neurips.cc/paper/2020/file/f0b76267fbe12b936bd65e203dc675c1-Paper.pdf?utm_source=chatgpt.com).
    
- **Adaptively Sparse Transformer**  
    Learns the α parameter per head, letting each decide its own sparsity pattern on the fly [arXiv](https://arxiv.org/abs/1909.00015?utm_source=chatgpt.com).
    
- **Sparse Transformer (explicit segment selection)**  
    Chooses only the top-k most relevant segments, improving concentration on global context [OpenReview](https://openreview.net/forum?id=Hye87grYDH&utm_source=chatgpt.com).
    
- **MultiMax**  
    Explores variants like “Sparse SoftMax” and EntMax-1.5 for vision tasks, though with limitations under label smoothing [arXiv](https://arxiv.org/html/2406.01189v1?utm_source=chatgpt.com).
    

## 2. Kernel- and Feature-Map-Based Linearizations

Aim to approximate the Softmax kernel in linear time, reducing the quadratic attention cost.

- **Performer (FAVOR+ / RFA)**  
    Uses random feature maps to approximate the exponential kernel of Softmax, achieving linear time and memory [Medium](https://medium.com/data-science/linearizing-attention-204d3b86cc1e?utm_source=chatgpt.com).
    
- **Linformer**  
    Projects keys and values into a low-rank subspace before Softmax, cutting complexity to linear in sequence length [Medium](https://medium.com/data-science/linearizing-attention-204d3b86cc1e?utm_source=chatgpt.com).
    
- **CosFormer**  
    Replaces Softmax with a cosine-based reweighting operator, preserving nonnegativity and re-weighting properties in linear time [OpenReview](https://openreview.net/pdf?id=Bl8CQrx2Up4&utm_source=chatgpt.com).
    
- **Cottention**  
    Directly substitutes Softmax with cosine similarity and reformulates attention as a recurrent process for constant memory usage [arXiv](https://arxiv.org/abs/2409.18747?utm_source=chatgpt.com).
    
- **The Hedgehog**  
    Learns simple feature maps via small MLPs to mimic Softmax’s “spiky” and monotonic behavior while keeping linear complexity [arXiv](https://arxiv.org/abs/2402.04347?utm_source=chatgpt.com).
    

## 3. Direct Softmax-Free Attention Blocks

Eliminate the normalization step entirely, replacing it with other activations or norms.

- **SOFT (Gaussian-kernel Transformer)**  
    Uses a Gaussian kernel in place of dot-product + Softmax, enabling low-rank approximation and full linear complexity [NeurIPS Papers](https://papers.nips.cc/paper/2021/file/b1d10e7bafa4421218a51b1e1f1b0ba2-Paper.pdf?utm_source=chatgpt.com).
    
- **SimA (ℓ₁-norm normalization)**  
    Normalizes queries and keys with an ℓ₁-norm and multiplies matrices directly, achieving softmax-free attention in vision transformers [arXiv](https://arxiv.org/abs/2206.08898?utm_source=chatgpt.com).
    
- **Rectified Linear Attention (ReLA)**  
    Replaces Softmax with ReLU, inducing sparsity naturally; stabilized via layer norms or gating [arXiv](https://arxiv.org/abs/2104.07012?utm_source=chatgpt.com).
    
- **ConSmax (Constant Softmax)**  
    A hardware-friendly, learnable normalization that removes the max and sum computations in Softmax for massive parallelism [arXiv](https://arxiv.org/html/2402.10930v2?utm_source=chatgpt.com).
    
- **Scalable-Softmax (SSMax)**  
    Adapts to varying input sizes with a scalable formulation, improving convergence and long-context retrieval [Reddit](https://www.reddit.com/r/singularity/comments/1ihe2ni/scalablesoftmax_is_superior_for_attention/?utm_source=chatgpt.com).
    

## 4. Alternative Activation Functions

Replace the exponential in Softmax with other nonlinearities to approximate its effect or introduce new inductive biases.

- **Polynomial Activations**  
    Design polynomial functions that mimic Softmax’s implicit regularization on the attention matrix [OpenReview](https://openreview.net/forum?id=PMf2Dg1TAA&utm_source=chatgpt.com).
    
- **Sigmoid- or ReLU-Based Attention**  
    Early explorations showed that simple element-wise activations can approximate attention behavior under proper normalization and gating [Apple Machine Learning Research](https://machinelearning.apple.com/research/attention-free-transformer?utm_source=chatgpt.com).
    

## 5. Practical Considerations and Trade-Offs

- **Efficiency vs. Accuracy**: Linearized and sparse methods often cut cost but may underperform vanilla Softmax in some tasks unless carefully tuned [Medium](https://medium.com/data-science/linearizing-attention-204d3b86cc1e?utm_source=chatgpt.com)[ACL Anthology](https://aclanthology.org/2022.spnlp-1.7.pdf?utm_source=chatgpt.com).
    
- **Hardware Constraints**: ConSmax and Quantized Softmax variants target FPGA/ASIC implementations by simplifying operations [arXiv](https://arxiv.org/html/2402.10930v2?utm_source=chatgpt.com).
    
- **Interpretability**: Sparsemax and EntMax yield more interpretable attention maps by zeroing irrelevant positions [arXiv](https://arxiv.org/abs/1602.02068?utm_source=chatgpt.com)[NeurIPS Proceedings](https://proceedings.neurips.cc/paper/2020/file/f0b76267fbe12b936bd65e203dc675c1-Paper.pdf?utm_source=chatgpt.com).
    
- **Implementation Complexity**: Some approaches (e.g., Performer’s random features or learnable α-entmax) add algorithmic or hyperparameter overhead.