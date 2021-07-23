# NTK-study
Understand the spectral bias of deep learning through the study of NTK

We implemented our neural tangent kernel based off the formulas given in [this paper](https://papers.nips.cc/paper/2019/file/c4ef9c39b300931b69a36fb3dbb8d60e-Paper.pdf)

### Theory
MLPs have difficulty learning high frequency functions, a phenomenon referred to in the literature as "spectral bias"

NTK theory suggests that this is because standard coordinate-based MLPs correspond to kernels with a rapid frequency falloff, which effectively prevents them from being able to represent the high-frequency components in functions and content present in natural images and scenes [reference](https://arxiv.org/pdf/2006.10739.pdf)

The outputs of a network throughout gradient descent remain close to those of a linear dynamical system whose convergence rate is governed by the eigenvalues of the NTK matrix. Analysis of the NTK’s eigendecomposition shows that its eigenvalue spectrum decays rapidly as a function of frequency, which explains the widely-observed "spectral bias" of deep networks towards learning low-frequency functions.

### Activation functions
we investigate several activation functions, including the rectified linear unit (ReLU) and some variants where we raise ReLU(x) to a constant power k (ReLU^k)

We also build NTK activated by certain trigonometric functions, namely sine and cosine. 

### Different Input Dimensions
The most rudimentery model that we implemented only had one dimension, i.e. the input vector x was taken from a segment on the number line under the range [-1, 1]. 

High dimensional inputs are taken from higher-dimensional spheres. We take 2-dimensional inputs. Here is an example in practice: uniformly drawing 200 points on the unit circle gives us an input vector `x` of shape (2, 200), where each column is a pair of coordinates (cos(theta), sin(theta))

### NTK of a Two Layer Fully Connected Network

With the dot product operation and a fast scientic computing library in Python, we are able to follow the entry-wise kernel definitions shown in the paper and massively speed up the computation with vectorized linear algebra tricks. 

For the fully connected neural networks, we implemented the analytic solution for ReLU NTK with definitions of arc-cosine kernels (**reference needed here**), as well as numerical approximations of the kernels using Monte Carlo sampling of the weights.
For trigonometrically activated networks we only have numerical implementations. 


### Studying the Reproducing Kernel Hilbert Space (RKHS)
We characterize the RKHS of the NTK for two-layer ReLU networks by providing a spectral decomposition of the kernel and studying its spectral decay.

The kernel function is a function of two input variables, thus we can discretize the function with a 2-dimensional matrix where each entry K(x, x') is a discretized point in function f(x, x'). With this discrete approximation, we then conduct eigenvalue decomposition on the matrix and calculate its eigenvalues. Upon plotting the sorted eigenvalues we obtain the eigenvalue decay plot or spectral decay. 

We observe that for ReLU^k networks, the decay on log-log plots appear polynomial; the asymptotic decay actually appears linear - [insert picture]

We also plotted Laplacian kernels, Gaussian kernels, and NTK activated by sine and cosine. It turns out the Laplacian kernels have the same magnitute of spectral decay rates as the ReLU^k kernels, whereas the sine and cosine-activated Neural Tangent Kernels have more resemblance to the Gaussian kernels.

### Experiment 1: Looking for Generalizations between ReLU^k kernel and Laplacian kernels

The first part of this project is to quantify the learnability of the kernel method based on NTK. This is expected to be done by investigating the reproducing 
kernel Hilbert space (RKHS) associated to the NTK. Concretely, we are interested in a quantified relationship between the ReLU^k activated NTK and laplacian kernels. 

We calculated the rate of the spectral decay by taking two end points (x = 30 and x = 100) representing the 30th and the 100th sorted eigenvalues. Between these two end points we obtain 70 eigenvalues sorted in descending order. Since we know the decay rate is in polynomial order, we solve a linear regression line to fit the points under log-log scale. Concretely we take the log of both x and y values and fit a line log(y) = m log(x) + b. This `m` is the slope (or rate) of the decay. In case of numerical issues caused by solving the singular value deposition (SVD) while doing the linear regression, I also preserved a naive implementation while we simply take two fixed endpoints and calculate the slope of the line passing through those two endpoints. 

Our goal was that given a fixed value k (where we have ReLU^k as our activation function) we could find a pseudo laplacian kernel (where ||x - y|| is raised to a power alpha) that has approximately the same decay rate. 

The laplacian kernel is defined as $Laplacian(x, y) = exp(\frac{-||x - y||}{\sigma})$. With some numeirical confirmation, we noticed that changes in sigma does not affect the decay rate. We introduce the pseudo-laplacian kernel where we add a parameter `alpha` to the definition, now we have $Laplacian(x, y) = exp(\frac{-||x - y||^\alpha}{\sigma})$, where `alpha` takes real value from [1, 2). 

The Gaussian kernel is defined as $Gaussian(x, y) = exp(\frac{-||x - y||^2}{2(\sigma^2)})$. And since we know that the constant `sigma` does not contribute to the rate of decay, we note that when we take `alpha=2` that pseudo-laplacian kernel would have the same spetral decay as the Gaussian kernel. So `alpha` takes values from 1 to 2. 


#### Results for Experiment 1
Unfortunately, through the numerical experiments we found out that the decay rate of pseudo-laplacian kernel cannot match most of k values greater than 1, unless the value of `alpha` is very close to 2. NTK for ReLU network has a decay rate of approximately -1.9, but for ReLU^2 it becomes below -4. We noticed that the slope with respect to `k` and `alpha` values both decrease linearly, but with `k` value it drops a lot faster. [insert plots here]

Therefore we are to try another approach. To make the pseudo-laplacian kernel "smoother" and potentially slow down the eigenvalue decay rate, we add a new term to the kernel definition and introduce another parameter `beta`. Now we define our kernel to be $$Laplacian(x, y) = e^{(-||x - y||^\alpha)} ||x - y||^\beta$$ Since the $\sigma$ value holds no significance, we won't include it in the formulas anymore. 

With the extended pseudo-laplacian we conducted another series of numerical experiments. We fixed alpha value to be 1.0, and for each k value in ReLU^k networks we tuned beta value so that the difference in the rate of spetral decay between the modified laplacian kernel and the NTK is minimized. We found that for either `alpha=1` (where the kernel is a modified laplacian) or `alpha=2` (where the kernel becomes a scaled Gaussian kernel) we were able to find appropriate `beta` values that make the kernel decay rate the same as the ReLU^k activated NTKs, for each k value from 1 to 5. Moreover, such found beta values and the k values appear to have a linear relationship, i.e. as k increases, beta linearly increases as well. [insert plots here]

 
### Experiment 2: Implement multi-layer NTK and observe the behaviors of the kernel with multiple activation functions stacking upon each other

Deep neural network with multiple activation functions can sometimes achieve much better approximation power than the ones with a single activation. Can we show that the spectral bias of NTK can be alleviated if one uses multiple activation functions?

To verify this idea, we had to extend our kernel construction from two layers to three layers, i.e. now we have an input layer and two hidden layers in our model. Because the sampling is now multivariate instead of the previous case, we had to build a new model from scratch to keep the compuatation coherent.

The paper again provides kernel definition for each entry assuming multi-dimensional inputs. To compute the expected values in the equation we again need to sample from a multivariate Gaussian distribution, but by doing a large number of sampling on each entry introduces a lot more computation and more numerical unstability (**not sure if it is correct, ask prof**). Similar to the two-layer case we have an analytial solution for the ReLU network, as well as numerical approximation to a network with a general nonlinear activation, with both an entry-wise version and a vectorized version. 

**Smoothness of activation functions** We know that the sine and cosine functions are smooth, i.e infinitely differentiable as any order of derivative is still a continuous. On the other hand, the rectified linear unit (ReLU) is a non-smooth function as even the first order of derivative is not continuous. As we raise ReLU to some power k, the resulting function ReLU^k becomes somewhat "smoother" due to more differentiability.

We have known facts about the relationship between the smoothness of the kernel functiona and their RKHS. As observed in our 2-layer NTKs, smoother kernels (Gaussian kernel and sine/cosine-activated NTK) have a much higher spectral decay rate (exponentially as we observed) whereas the other non-smooth kernels such as laplace and ReLU-activated NTK counterparts have a slower eigenvalue decay, linearly as observed. 

For the multi-layer NTK with different activation functions, our initial speculation was that the smoothness of the kernel is governed by whatever activation function that comes first, i.e. if we have a sine activation function on the first hidden layer and ReLU on the second, then the kernel would exhibit a faster spectral decay rate than the one if we switched the order of the activation functions. However our numerical experiments tells otherwise. 

With a three-layer NTK, we observed the decay by numerically computing the NTK and plotting the sorted eigenvalues. The result suggests that if we mix up the activation function (having multiple different activation functions on each hidden layer), the resulting spectral decays are similar, or at least they decay at the same rate. In fact, with ReLU + sine combination, the resulting eigendecay is always similar to the one with only ReLU activation, regardless of the order of placement. This overthrows our speculation and confirms that the non-smooth activation function actually plays a bigger role in determining the RKHS of the NTK in the multi-layer context.


### Experiment 3: Fitting multi-scale functions with ReLU and sine activation functions

We trained a three-layer fully connected neural network hoping that we would get some observation to acccomondate our previous results. We first constructed a function where we have a sum of sine waves of different frequencies. We also picked the fourier series of square wave as our second experiement subject. What these two functions have in common is that they have a range of different frequency components, and we can observe how different activation functions react to them. 

We constructed a three-layer fully connected neural network with 500 neurons in each layer. The weights of each linear layer is initialized from a Normal distribution centered at 0 and a standard deviation of $\frac{1}{\sqrt{m}}$. We have that under this specific initialization, the neural network training dynamic will be under the NTK regime. Note that in this kernel regime where the network layer is ultra-wide, the neuron weights don't change significantly during training, a phenomenon described as "lazy training" in literature (reference needed). Without the specific weight initialization where we scale the standard deviation, the training is said to be another regime called the mean field regime. 

The trainer of this neural network uses Mean Square Loss and the cost function is optimized by full batch gradient descent. 

We apply activation functions to each hidden layer separately. From the standard weight initialization, we observe that ReLU+ReLU combination perform rather poorly on fitting the data as the predicted curve is rigid on the turns. Sine + ReLU combination works somewhat better, as the predicted curves are smoother and generally closer to the data points. Sine + Sine activation was able to learn to fit the target data points fairly well, but the high-frequency components were still not perfectly fitted. The best performing activation combination is ReLU + Sine, as the network was able to achieve zero training loss on the given data. 

With the NTK weight initialization, we have different emperical results. With the fourier square wave as the target function, the experiments reveals that only sine+sine is a good performing combination of activation functions. All the other fitted curves could not fit any of the high frequency component but sine+sine had zero loss on the training data. 

In addition, we found that practically it is very hard to train the network with exponentiated ReLU as we experience gradient problem too often. During back propagation the loss would easily become too large numerically and compromise the whole fitting process. Therefore all of our numerical experiments were done with regular first order ReLU. The results we obtained with ReLU^k activation under the NTK regime is still valuable nevertheless, as their theoretical behavior and relations with Laplacian kernels are still very interesting to study.

Discussion: It is understandble how having sine as the last activation function is essential especially when our function is contructed by a composite of sine functions. 


### TODO:
insert images/plots and more latex formulas about NTK into the doc