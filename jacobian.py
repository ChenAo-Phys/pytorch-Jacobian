import torch
from torch import nn
import types


def extend(model, input_shape):
    if not isinstance(model, nn.Module):
        raise TypeError('model should be a nn.Module')
    if not isinstance(input_shape, tuple):
        raise TypeError('input_shape should be a tuple')

    device = next(model.parameters()).device
    M_list = []
    N_list = []
    x = torch.zeros((1,) + input_shape, device=device)
    with torch.no_grad():
        for module in model.children():
            y = module(x)
            if sum(p.numel() for p in module.parameters()):
                # for all layers with parameters

                # store parameters and clear bias for future calculation
                if module.weight is not None:
                    initial_weight = module.weight.data.clone()
                if module.bias is not None:
                    initial_bias = module.bias.data.clone()
                    module.bias.data = torch.zeros_like(module.bias)

                if module.weight is not None:
                    Nweight = module.weight.numel()
                    M = torch.empty(y.numel(), Nweight, x.numel(), dtype=torch.bool, device=device)
                    Xeye = torch.eye(x.numel()).reshape((-1,)+x.shape[1:])
                    for i in range(Nweight):
                        weight = torch.zeros(Nweight, device=device)
                        weight[i] = 1.
                        module.weight.data = weight.reshape(module.weight.shape)
                        # output of module is of dimension (j,k), transposed to (k,j)
                        out = module(Xeye).reshape(x.numel(), y.numel()).t()
                        if (out[out.abs()>1e-5] - 1.).abs().max() > 1e-5:
                            raise RuntimeError('detect factors before weight')
                        M[:,i,:] = torch.abs(out) > 0.5
                    M_list.append(M)
                    module.weight.data = initial_weight
                else:
                    M_list.append(None)
                
                if module.bias is not None:
                    Nbias = module.bias.numel()
                    N = torch.empty(y.numel(), module.bias.numel(), dtype=torch.bool, device=device)
                    for i in range(Nbias):
                        bias = torch.zeros(Nbias, device=device)
                        bias[i] = 1.
                        module.bias.data = bias.reshape(module.bias.shape)
                        out = module(x).reshape(-1)
                        if (out[out.abs()>1e-5] - 1.).abs().max() > 1e-5:
                            raise RuntimeError('detect factors before bias')
                        N[:,i] = torch.abs(out) > 0.5
                    N_list.append(N)
                    module.bias.data = initial_bias
                else:
                    N_list.append(None)
                    
            x = torch.zeros_like(y)
        
    if not hasattr(model, '_Jacobian_shape_dict'):
        model._Jacobian_shape_dict = {}
    model._Jacobian_shape_dict[input_shape] = (M_list, N_list)


    # assign jacobian method to model
    def jacobian(self, as_tuple=False):
        self.gradient.reverse()
        shape = self.input_shape
        if hasattr(self, '_Jacobian_shape_dict') and shape in self._Jacobian_shape_dict:
            M_list, N_list = self._Jacobian_shape_dict[shape]
        else:
            raise RuntimeError('model or specific input shape is not extended for jacobian calculation')

        jac = []
        layer = 0
        for module in self.children():
            if sum(p.numel() for p in module.parameters()):
                M = M_list[layer]
                N = N_list[layer]
                x = self.x_in[layer]
                n = x.shape[0]
                k,i,j = M.shape[0:3]
                dz_dy = self.gradient[layer].reshape(n,k)

                if M is not None:
                    dy_dW = torch.einsum('kij,nj->nki', M.type_as(x), x.reshape(n,j))
                    dz_dW = torch.einsum('nk,nki->ni', dz_dy, dy_dW)
                    jac.append(dz_dW.reshape((n,) + module.weight.shape))
                if N is not None:
                    dz_db = torch.einsum('nk,ki->ni', dz_dy, N.type_as(dz_dy))
                    jac.append(dz_db.reshape((n,) + module.bias.shape))

                layer += 1

        if as_tuple:
            return tuple(jac)
        else:
            return torch.cat([j.reshape(n,-1) for j in jac], dim=1)

    model.jacobian = types.MethodType(jacobian, model)


    
class JacobianMode():
    
    def __init__(self, model):
        self.model = model
        if not isinstance(model, nn.Module):
            raise TypeError('model should be a nn.Module')


    def __enter__(self):
        model = self.model
        model.x_in = []
        model.gradient = []
        self.forward_pre_hook = []
        self.backward_hook = []
        
        def record_input_shape(self, input):
            model.input_shape = input[0].shape[1:]

        def record_forward(self, input):
            model.x_in.append(input[0].detach())

        def record_backward(self, grad_input, grad_output):
            model.gradient.append(grad_output[0])

        module0 = next(model.children())
        self.first_forward_hook = module0.register_forward_pre_hook(record_input_shape)

        for module in model.children():
            if sum(p.numel() for p in module.parameters()):
                self.forward_pre_hook.append(module.register_forward_pre_hook(record_forward))
                self.backward_hook.append(module.register_backward_hook(record_backward))


    def __exit__(self, exc_type, exc_val, exc_tb):
        self.first_forward_hook.remove()
        for hook in self.forward_pre_hook:
            hook.remove()
        for hook in self.backward_hook:
            hook.remove()
        
        del self.model.input_shape
        del self.model.x_in
        del self.model.gradient
