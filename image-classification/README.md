# Image classification
 > Victor Ludvig, November 2023

## Part 1: read data
See [read_cifar.py](read_cifar.py). <br><br>
All paths are Posix path. The code will run only on Unix-like operating systems (e.g. Linux, MacOS). <br>
The code can easily be adapted to run with Windows using the Pathlib python module and a few more lines of codes with if/else statements to check the operating system, however it is not asked and not implemented.<br><br>

## Part II: implementation of knn classifier in python.
See [knn.py](knn.py). <br><br>
Functions contain minimal code, as requested. <br> Some code could have been reduced even more, however I made sure no line exceeded my laptop's screen with normal zoom.
<br><br>

## Part III: implementation of MLP in python.
See [derivatives.md](./derivatives.md) for answers to questions 2-9. <br>
See [mlp.py](mlp.py) for the rest.<br><br>
I used Einstein's notation to improve readability. <br>
I used the gradient notation $\nabla$ when a scalar is derived with respect to (w.r.t) a vector, and jacobian J when a vector is derived w.r.t another vector, instead of the partial derivative notation $\partial$ which is not precise. <br>
I also used the J notation when a scalar is derived w.r.t a matrix, because in our case the result is a matrix. <br><br>
The equation presented in the introduction is not the one implemented in the code section. Hence, I used the equation implemented to compute the derivatives: 
$\begin{equation} 
Z^{L+1}=A^{L}W^{L+1}+B^{L+1}
\end{equation}$
This changes equation 4) and 8).
<br><br>
The questions does not ask to compute the average of gradients on the batch. I tried to use an average gradient for the Jacobian matrix of the matrix weight, and simultaneously increased the learning rate because the coefficients are then much smaller. However, it does not change the results. We always get a 10% accuracy of the MLP on the test set. To have better performances, a decaying learning rate could be used, so that it is higher at the beginning of the training and the MLP does not converge to a local minimum. I did not implement this decaying learning rate version.