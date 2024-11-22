# Multi Scale Texture Loss Function for CT denoising with GANs



<img width="969" alt="Screenshot 2024-11-22 at 20 54 10" src="https://github.com/user-attachments/assets/83c7abcf-286b-4198-ac0e-58cb13774f4b">


Overall framework of MSTLF. (a) MSTLF mainly includes two essential components, i.e., the Multi-Scale Texture Extractor (MSTE) and the Aggregation Module (AM).
We denote the selection of the aggregation rule with the logic operators demux and mux.
(b) MSTE module extracting a textural representation from the input images using a texture descriptor extracted from the GLCMs at different spatial and angular scales. (c) Dynamic aggregation by Self-Attention mechanism (SA) that combines
the extracted representation into a scalar loss function

