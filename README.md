# NTK-study
Understand the spectral bias of deep learning through the study of NTK

We implemented our neural tangent kernel based off the formulas given in [this paper](https://papers.nips.cc/paper/2019/file/c4ef9c39b300931b69a36fb3dbb8d60e-Paper.pdf)

### Activation functions
we investigate several activation functions, including the rectified linear unit (ReLU) and some variants where we raise ReLU(x) to a constant power k (ReLU^k)

We also build NTK activated by certain trigonometric functions, namely sine and cosine. 

### Different Input Dimensions
The earliest model that we implemented only had one dimension, i.e. the input vector x was taken from a segment on the number line under the range [-1, 1]. 

### NTK of a Two Layer Fully Connected Network

We take 2-dimensional inputs. Uniformly take 200 points on the unit circle gives us an input vector `x` with shape (2, 200)

With the dot product operation and a fast scientic computing library in Python, we are able to follow the entry-wise kernel definitions shown in the paper and massively speed up the computation with vectorized linear algebra tricks. 

For the fully connected neural networks, we implemented the analytic solution for ReLU NTK with definitions of arc-cosine kernels (**reference needed here**), as well as numerical approximations of the kernels using Monte Carlo sampling of the weights.
For trigonometrically activated networks we only have numerical implementations. 


### Studying the Reproducing Kernel Hilbert Space (RKHS)
We characterize the RKHS of the NTK for two-layer ReLU networks by providing a spectral decomposition of the kernel and studying its spectral decay.

The kernel function is a function of two input variables, thus we can discretize the function with a 2-dimensional matrix where each entry K(x, x') is a discretized point in function f(x, x'). With this discrete approximation, we then conduct eigenvalue decomposition on the matrix and calculate its eigenvalues. Upon plotting the sorted eigenvalues we obtain the eigenvalue decay plot or spectral decay. 

We observe that for ReLU^k networks, the decay on log-log plots appear polynomial; the asymptotic decay actually appears linear - [insert picture]

We also plotted Laplacian kernels, Gaussian kernels, and NTK activated by sine and cosine. It turns out the Laplacian kernels have similar spectral decay rates as the ReLU^k kernels, whereas the sine and cosine-activated Neural Tangent Kernels have more resemblance to the Gaussian kernels.

### Experiment 1: Looking for Generalizations between ReLU^k kernel and Laplacian kernels

We calculated the slope of the spectral decay by taking two end points (x = 30 and x = 100) representing the 30th and the 100th sorted eigenvalues. Between these two end points we obtain 70 eigenvalues sorted in descending order. Since we know the decay rate is in polynomial order, we solve a linear regression line to fit the points under log-log scale. Concretely we take the log of both x and y values and fit a line log(y) = m log(x) + b. This `m` is the slope (or rate) of the decay.  

Our goal was that given a fixed value k (where we have ReLU^k as our activation function) we could find a pseudo laplacian kernel (where ||x - y|| is raised to a power alpha) that has approximately the same decay rate. 

The laplacian kernel is defined as $Laplacian(x, y) = exp(\frac{-||x - y||}{\sigma})$. With some numeirical confirmation, we noticed that changes in sigma does not affect the decay rate. We introduce the pseudo-laplacian kernel where we add a parameter `alpha` to the definition, now we have $Laplacian(x, y) = exp(\frac{-||x - y||^\alpha}{\sigma})$, where `alpha` takes real value from [1, 2). 

The Gaussian kernel is defined as $Gaussian(x, y) = exp(\frac{-||x - y||^2}{2(\sigma^2)})$. And since we know that the constant `sigma` does not contribute to the rate of decay, we note that when we take `alpha=2` that pseudo-laplacian kernel would have the same spetral decay as the Gaussian kernel. So `alpha` takes values from 1 to 2. 


#### Results for Experiment 1
Unfortunately, through the numerical experiments we found out that the decay rate of pseudo-laplacian kernel cannot match most of k values greater than 1, unless the value of `alpha` is very close to 2. NTK for ReLU network has a decay rate of approximately -1.9, but for ReLU^2 it becomes below -4. We noticed that the slope with respect to `k` and `alpha` values both decrease linearly, but with `k` value it drops a lot faster. [insert plots here]

Therefore we are to try another approach. To make the pseudo-laplacian kernel "smoother" and potentially slow down the eigenvalue decay rate, we add a new term to the kernel definition and introduce another parameter `beta`. Now we define our kernel to be $$Laplacian(x, y) = e^{(-||x - y||^\alpha)} ||x - y||^\beta$$ Since the $\sigma$ value holds no significance, we won't include it in the formulas anymore. 

With the extended pseudo-laplacian we conducted another series of numerical experiments. We fixed alpha value to be 1.0, and for each k value in ReLU^k networks we tuned beta value so that the difference in the rate of spetral decay between the modified laplacian kernel and the NTK is minimized. We found that for either `alpha=1` (where the kernel is a modified laplacian) or `alpha=2` (where the kernel becomes a scaled Gaussian kernel) we were able to find appropriate `beta` values that make the kernel decay rate the same as the ReLU^k activated NTKs, for each k value from 1 to 5. Moreover, such found beta values and the k values appear to have a linear relationship, i.e. as k increases, beta linearly increases as well. [insert plots here]

 
### Experiment 2: Implement multi-layer NTK and observe the behaviors of multiple activation functions acting upon each other

### TODO:
 - redo the experiments now tune both alpha and beta
 - Implement NTK for three-layer fully connected networks. Must be able to take both ReLU^k and sine/cosine as activation. Still implement the analytic solution so we can verify correctness
