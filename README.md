# Multi Scale Texture Loss Function for CT denoising with GANs



<img width="1299" alt="Screenshot 2024-11-26 at 12 35 18" src="https://github.com/user-attachments/assets/73bc2904-719f-45e8-a0d4-83a167d79c1e">





Overall framework of MSTLF. (a) MSTLF mainly includes two essential components, i.e., the Multi-Scale Texture Extractor (MSTE) and the Aggregation Module (AM).
We denote the selection of the aggregation rule with the logic operators demux and mux.
(b) MSTE module extracting a textural representation from the input images using a texture descriptor extracted from the GLCMs at different spatial and angular scales. (c) Dynamic aggregation by Self-Attention mechanism (SA) that combines
the extracted representation into a scalar loss function

