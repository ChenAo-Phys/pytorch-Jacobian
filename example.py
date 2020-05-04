import torch
from torch import nn
from jacobian import extend, JacobianMode

# define network
net = nn.Sequential(
    nn.Conv3d(1,2,2, bias=False),
    nn.ReLU(),
    nn.Conv3d(2,1,2),
    nn.ReLU(),
    nn.AvgPool3d(4),
    nn.Flatten(start_dim=1)
)

# extended for Jacobian calculation
extend(net, (1,6,6,6))

x = torch.randn(1000,1,6,6,6)

# Jacobian computed by the improved method
# On Colab CPU 0.16s, K80 GPU 0.14s
with JacobianMode(net):
    out = net(x)
    out.sum().backward()
    jac = net.jacobian()

# Jacobian computed by naive for loops
# On Colab CPU 18.50s, K80 GPU 17.66s
out = net(x)
jac_loop = []
for o in out.view(-1):
    net.zero_grad()
    grad = []
    o.backward(retain_graph=True)
    for param in net.parameters():
        grad.append(param.grad.reshape(-1))
    jac_loop.append(torch.cat(grad))
jac_loop = torch.stack(jac_loop)

# the two Jacobian matrices are identical
print((jac-jac_loop).abs().max().item())
