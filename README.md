# NTK-study
Understand the spectral bias of deep learning through the study of NTK

## NTK of a Two Layer ReLU Network

we investigate several activation functions, including the rectified linear unit (ReLU) and some variants where we raise ReLU(x) to a constant power k (ReLU^k)

We also build NTK activated by certain trigonometric functions, namely sine and cosine. 

We implemented our neural tangent kernel based off the formulas given in [this paper](https://papers.nips.cc/paper/2019/file/c4ef9c39b300931b69a36fb3dbb8d60e-Paper.pdf)


We characterize the RKHS of the NTK for two-layer ReLU networks by providing a spectral decomposition of the kernel (approximate the kernel function with a matrix and then conduct eigenvalue decomposition on the matrix) and studying its spectral decay.


TODO:

1. figure out the relationship between the alpha in laplacian kernels and the k in ReLU^k NTK

2. Implement NTK for a three-layer ReLU network