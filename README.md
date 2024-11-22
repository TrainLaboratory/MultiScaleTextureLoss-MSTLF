# Multi Scale Texture Loss Function for CT denoising with GANs

<img width="1133" alt="Screenshot 2024-11-22 at 20 39 44" src="https://github.com/user-attachments/assets/95f63c93-60bf-4c50-a989-1f91404d45b3">



Overall framework of MSTLF. (a) MSTLF mainly includes two essential components, i.e., the Multi-Scale Texture Extractor (MSTE) and the Aggregation Module (AM).
We denote the selection of the aggregation rule with the logic operators demux and mux.
(b) MSTE module extracting a textural representation from the input images using a texture descriptor extracted from the GLCMs at different spatial and angular scales. (c) Dynamic aggregation by Self-Attention mechanism (SA) that combines
the extracted representation into a scalar loss function

