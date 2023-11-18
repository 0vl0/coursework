## Answers to questions 2-9 of the neural network part


2) With $C = \frac{1}{N_{out}}(\hat{y_i}-y_i)^2$, it follows:
$\begin{equation}
\frac{\partial C}{\partial a_i}=\frac{2}{N_{out}}(\hat{y_i}-y_i)
\end{equation}$ 
Hence:
$\begin{equation}
\nabla_{A^2}(C)=\frac{2}{N_{out}}(A^{2}-Y)
\end{equation}$ 
<br>

3) Using the chain rule: 
$\begin{equation}
\nabla_{Z^2}(C)=\nabla_{A^2}(C)J_{Z^2}(A^2)
\end{equation}$ 
With $A^2=(\sigma(z_1),...,\sigma(z_n))$:
$\begin{equation}
J_{Z^2}(A^2)=diag(\sigma'(Z^2))
\end{equation}$ 
Using 1., one obtains:
$\begin{equation}
\nabla_{Z^2}(C)=\nabla_{A^2}(C)diag(A^2(1_n-A^2))
\end{equation}$ 
<br>

4. Using the chain rule:
$\begin{equation}
\frac{\partial C}{\partial w_{jk}}=\frac{\partial C}{\partial z_{j}^2}\frac{\partial z_j^2}{\partial w_{jk}}
\end{equation}$ 
As $w_{jk}$ only appears in the j-th coefficient of $z^2$:
$\begin{equation}
z_j^2=w_{jk}a_k^1 + b_j^1
\end{equation}$
Thus,
$\begin{equation}
\frac{\partial z_j^2}{\partial w_{jk}} = a_k^1
\end{equation}$ 
Hence,
$\begin{equation}
\frac{\partial C}{\partial w_{jk}}=\frac{\partial C}{\partial z_{j}^2}a_k^2
\end{equation}$ 
Which leads to:
$\begin{equation}
J_{W^2}(C)=\nabla_{Z^2}(C)A_1^T
\end{equation}$ 
<br>
5. Using the chain rule and $J_{B^2}(Z^2)=I$
$\begin{equation}
\nabla_{B^2}(C) = \nabla_{Z^2}(C)J_{B^2}(Z^2) = \nabla_{Z^2}(C)
\end{equation}$

6. Using the chain rule:
$\begin{equation}
\nabla_{A^1}(C)=\nabla_{Z^2}(C)\nabla_{A^1}(Z^2)
\end{equation}$
Considering the k-th element of $A^1W^2$:
$\begin{equation}
(A^1W^2)_k=a_jw_{kj}
\end{equation}$
Hence by derivation:
$\begin{equation}
\frac{\partial (A^1W^2)_k}{\partial a_j}=w_{kj}
\end{equation}$
Which leads to:
$\begin{equation}
J_{A^1}(W^2)=W_2^T
\end{equation}$
So:
$\begin{equation}
\nabla_{A^1}(C)=\nabla_{Z^2}(C)W_2^T
\end{equation}$

7. Using 3. arguments:
$\begin{equation}
\nabla_{Z^1}(C)=\nabla_{A^1}(C)diag(A^1(1_n-A^1))
\end{equation}$ 

8. Using 4. arguments:
$\begin{equation}
J_{W^1}(C)=\nabla_{Z^1}(C)A_0^T
\end{equation}$ 

9. Using 5. arguments:
$\begin{equation}
\nabla_{B^2}(C) = \nabla_{Z^1}(C)
\end{equation}$