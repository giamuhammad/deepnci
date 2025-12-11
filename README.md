# deepnci



A reproduction model for Non-Covalent Interaction prediction from DeepNCI research (Li, 2021)



This project is reproduce a research from journal article "DeepNCI: DFT Noncovalent Interaction Correction with Transferable Multimodal Three-Dimensional Convolutional Neural Networks"



The research also create and share a source code from https://github.com/wenzelee/DeepNCI but this project recreate the model with PyTorch and preprocessing by Psi4, not by Gaussian09



We are create a same architecture (Multimodal Model)



Different Key from DeepNCI (Li, 2021)

* We only use 29 sample from >1000 sample
* We use Psi4 than Gaussian09 for DFT process for extracting Density Charge feature and 21 Quantum Descriptor due to license.
* We not use all same feature in 21 Quantum Descriptor due to calculation limitation feature from Psi4



The Result

* Model convergence even only 29 sample
* Only evaluate an accuracy, R-square, RMSE and MAE.
