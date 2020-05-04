# pytorch-Jacobian


## Target
Pytorch only provides autograd methods to calculate the gradient of scalars, but sometimes Jacobian - the gradient of vectors, are also significant in researches. In concrete, for N inputs $x^{(n)} n=0,1,...,N-1$ feeded into a network with parameters $W_i$ as a batch and giving outputs $z^{(n)}$ and loss $l^{(n)}$, there is no method to calculate the Jacobian matrix with elements $J_{ni} = \frac{\partial z^{(n)}}{\partial W_i}$ or $\frac{\partial l^{(n)}}{\partial W_i}$ in a single forward and backward pass. Naively one can backpropagate N times and combine N gradients together to form the Jacobian matrix, but it's too slow.

There is already a package "backpack" aiming at this problem, but its BatchGrad - synonymous to Jacobian - only supports nn.Linear and nn.Conv2d layers. Here is the website <https://backpack.pt/> and paper <https://openreview.net/pdf?id=BJlrF24twB> of backpack.

Based on the main idea of backpack, this repository provides an interface for fast Jacobian calculations in pytorch networks for most pytorch layers.


## Aplicable Range
This package is available for networks satisfying the following conditions.

- All layers with parameters in the network should be 'quasi-linear'. 'Quasi-linear' includes linear layers and convolutional layers in different dimensions. The concrete math description is that the relation between weight $W_i$, bias $b_i$, input $x_j$, output $y_k$ can be written as $y_k = \sum_{ij} M_{kij} W_i x_j + \sum_i N_{ki} b_i$ where M_{kij} and N_{ki} are both tensors with elements 1 or 0. If one creates a layer like $y = W^2 x$ then this package will fail.

- All layers with parameters should be immediate children modules. For example,
``` 
# This works
net = nn.Sequential(nn.Conv2d(1,1,2), nn.ReLU(), nn.Conv2d(1,1,2), nn.ReLU()) 

# This doesn't work
combine = nn.Sequential(nn.Conv2d(1,1,2), nn.ReLU())
net = nn.Sequential(combine, combine)
```

- The input shape other than batch dimension should be fixed, or at least there only exists a few input shapes. For example, the input x can be of shape (1000, 1, 10, 10) as well as (10, 1, 10, 10), while (100, 1, 10, 10) and (100, 1, 9, 9) is not perfect (but still ok), and (100, 1, ?, ?) where ? can be any integer definitely can't work.
