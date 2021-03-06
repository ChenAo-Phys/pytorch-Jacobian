# pytorch-Jacobian


## Target
Pytorch only provides autograd methods to calculate the gradient of scalars, but sometimes Jacobian - the gradient of vectors, are also significant in researches. In concrete, for N inputs $x^{(n)}, n=0,1,...,N-1$ feeded into a network as a batch and giving outputs $z^{(n)}$ and loss $l^{(n)}$, there is no method to calculate the Jacobian matrix with elements $J_{ni} = \frac{\partial z^{(n)}}{\partial W_i}$ or $\frac{\partial l^{(n)}}{\partial W_i}$ in a single forward and backward pass. Naively one can backpropagate N times and combine N gradients together to form the Jacobian matrix, but it's too slow.

There is already a package "backpack" aiming at this problem, but its BatchGrad - synonymous to Jacobian - only supports nn.Linear and nn.Conv2d layers. Here is the website <https://backpack.pt/> and paper <https://openreview.net/pdf?id=BJlrF24twB> of backpack.

Based on the main idea of backpack, this repository provides a more general interface for fast Jacobian calculations in pytorch networks. In example.py, it works more than 100 times faster than naive for loops when the batch size is 1000.


## Applicable Range
This package is available for networks satisfying the following conditions.

- All layers with parameters in the network should be 'quasi-linear'. 'Quasi-linear' includes linear layers and convolutional layers in different dimensions. The concrete math description is that the relation between weight $W_i$ bias $b_i$ input $x_j^{(n)}$ output $y_k^{(n)}$ can be written as $y_k^{(n)} = \sum_{ij} M_{kij} W_i x_j^{(n)} + \sum_i N_{ki} b_i$ where M and N are both tensors with elements 1 or 0. If one creates a layer like $y = W^2 x$ then this package will fail.

- Parameters in a module can be reached by module.weight or module.bias. This is the default behavior of pytorch layers like nn.Linear and nn.Conv2d. If you define a module by yourself please put your parameters accordingly.

- All layers with parameters should be immediate children modules. For example,
``` 
# This works
net = nn.Sequential(nn.Conv2d(1,1,2), nn.ReLU(), nn.Conv2d(1,1,2), nn.ReLU()) 

# This doesn't work
combine = nn.Sequential(nn.Conv2d(1,1,2), nn.ReLU())
net = nn.Sequential(combine, combine)
```

- Apart from the batch dimension, all other input dimensions fixed, or at least there only exists a few input shapes. For example, the input x can be of shape (1000, 1, 10, 10) as well as (10, 1, 10, 10), while (100, 1, 10, 10) and (100, 1, 9, 9) is not perfect (takes more time but still ok), and (100, 1, ?, ?) where ? can be any integer definitely can't work.

- This method is still very memory-consuming. It can only be used in very small networks.


## Usage
See example.py. The provided jacobian method has an input as_tuple default to False. If False, the output is an N by D tensor where N and D are the batch size and the number of parameters, respectively. If True, the output is a tuple aranged in the same order as net.parameters(). Every element in the tuple is a tensor with shape (N, shape of corresponding parameters).


## Theory
Assume the network gives a batch of output $z^{(n)}$, and the layer inputs and outputs are $x_j^{(n)}$ and $y_k^{(n)}$ respectively. The Jacobian can be written into two parts

$ \frac{\partial z^{(n)}}{\partial W_i} = \sum_k \frac{\partial z^{(n)}}{\partial y_k^{(n)}} \frac{\partial y_k^{(n)}}{\partial W_i} $

The first part is trivial. Assume $ z = \sum_n z^{(n)} $, then

$ \frac{\partial z}{\partial y_k^{(n)}} = \frac{\partial z^{(n)}}{\partial y_k^{(n)}} $

A single backward pass with suitable backward hook can solve it.

The second part is much more complicated so we need to invoke the 'quasi-linear' assumption

$y_k^{(n)} = \sum_{ij} M_{kij} W_i x_j^{(n)} + \sum_i N_{ki} b_i$

The tensors $M_{kij}$ and $N_{ki}$ can be calculated with some tricks. Consider weights W first. Let $b_i = 0$, $W_i = \delta_{i,a}$, $x_j^{(n)} = \delta_{n,j}$, then $y_k^{(n)} = M_{k,a,n}$. One can change $a=0,1,...,D-1$ (D is the number of weights in this layer) and forward pass D times to get all $M_{kij}$. This takes some time, but it's prepared before calculation and $M_{kij}$ is stored. $M_{kij}$ is very sparse so only the position of non-zero elements is stored to reduce the memory complexity. The room used in the calculation is roughly in the same magnitude as the number of connections in the layer (e.g. $N \times D$ in a fully connected network).

The bias part goes similarly. $x_j^{(n)} = 0$, $b_i = \delta_{i,a}$ so that $y_k^{(n)} = N_{ka}$.

After going through all these preparations, the Jacobian can be expressed as

$ \frac{\partial z^{(n)}}{\partial W_i} = \sum_k \frac{\partial z}{\partial y_k^{(n)}} \sum_j M_{kij} x_j^{(n)}$

$ \frac{\partial z^{(n)}}{\partial b_i} = \sum_k \frac{\partial z}{\partial y_k^{(n)}} N_{ki}$
