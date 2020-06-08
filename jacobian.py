import torch
from torch import nn
import types
from functools import partial


def extend(model, input_shape):
    if not isinstance(model, nn.Module):
        raise TypeError("model should be a nn.Module")
    if not isinstance(input_shape, tuple):
        raise TypeError("input_shape should be a tuple")

    device = next(model.parameters()).device

    weight_input_list = []
    weight_output_list = []
    weight_repeat_list = []
    bias_output_list = []
    bias_repeat_list = []

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
                    weight_input = []
                    weight_output = []
                    weight_repeat = torch.zeros(
                        Nweight, dtype=torch.long, device=device
                    )
                    Xeye = torch.eye(x.numel(), device=device).reshape(
                        (-1,) + x.shape[1:]
                    )
                    for i in range(Nweight):
                        weight = torch.zeros(Nweight, device=device)
                        weight[i] = 1.0
                        module.weight.data = weight.reshape(module.weight.shape)
                        # output of module is of dimension (j,k)
                        out = module(Xeye).reshape(x.numel(), y.numel())
                        if (out[out.abs() > 1e-5] - 1.0).abs().max() > 1e-5:
                            raise RuntimeError(
                                "the network is not written in the standard form, see https://github.com/ChenAo-Phys/pytorch-Jacobian"
                            )
                        nonzero = torch.nonzero(out > 0.5, as_tuple=False)
                        weight_input.append(nonzero[:, 0])
                        weight_output.append(nonzero[:, 1])
                        weight_repeat[i] = nonzero.shape[0]
                    weight_input_list.append(torch.cat(weight_input, dim=0))
                    weight_output_list.append(torch.cat(weight_output, dim=0))
                    weight_repeat_list.append(weight_repeat)
                    module.weight.data = initial_weight
                else:
                    weight_input_list.append(None)
                    weight_output_list.append(None)
                    weight_repeat_list.append(None)

                if module.bias is not None:
                    Nbias = module.bias.numel()
                    bias_output = []
                    bias_repeat = torch.zeros(Nbias, dtype=torch.long, device=device)
                    for i in range(Nbias):
                        bias = torch.zeros(Nbias, device=device)
                        bias[i] = 1.0
                        module.bias.data = bias.reshape(module.bias.shape)
                        out = module(x).reshape(-1)
                        if (out[out.abs() > 1e-5] - 1.0).abs().max() > 1e-5:
                            raise RuntimeError(
                                "the network is not written in the standard form, see https://github.com/ChenAo-Phys/pytorch-Jacobian"
                            )
                        nonzero = torch.nonzero(out > 0.5, as_tuple=False)
                        bias_output.append(nonzero[:, 0])
                        bias_repeat[i] = nonzero.shape[0]
                    bias_output_list.append(torch.cat(bias_output, dim=0))
                    bias_repeat_list.append(bias_repeat)
                    module.bias.data = initial_bias
                else:
                    bias_output_list.append(None)
                    bias_repeat_list.append(None)

            x = torch.zeros_like(y)

    if not hasattr(model, "_Jacobian_shape_dict"):
        model._Jacobian_shape_dict = {}
    model._Jacobian_shape_dict[input_shape] = (
        weight_input_list,
        weight_output_list,
        weight_repeat_list,
        bias_output_list,
        bias_repeat_list,
    )

    # assign jacobian method to model
    def jacobian(self, as_tuple=False):
        shape = self.input_shape
        if hasattr(self, "_Jacobian_shape_dict") and shape in self._Jacobian_shape_dict:
            (
                weight_input_list,
                weight_output_list,
                weight_repeat_list,
                bias_output_list,
                bias_repeat_list,
            ) = self._Jacobian_shape_dict[shape]
        else:
            raise RuntimeError(
                "model or specific input shape is not extended for jacobian calculation"
            )

        device = next(model.parameters()).device
        jac = []
        layer = 0
        for module in self.children():
            if sum(p.numel() for p in module.parameters()):
                weight_input = weight_input_list[layer]
                weight_output = weight_output_list[layer]
                weight_repeat = weight_repeat_list[layer]
                bias_output = bias_output_list[layer]
                bias_repeat = bias_repeat_list[layer]
                x = self.x_in[layer]
                N = x.shape[0]
                dz_dy = self.gradient[layer].reshape(N, -1)

                if weight_repeat is not None:
                    Nweight = weight_repeat.shape[0]
                    dz_dy_select = dz_dy[:, weight_output]
                    x_select = x.reshape(N, -1)[:, weight_input]
                    repeat = torch.repeat_interleave(weight_repeat)
                    dz_dW = torch.zeros(N, Nweight, device=device).index_add_(
                        1, repeat, dz_dy_select * x_select
                    )
                    if as_tuple:
                        dz_dW = dz_dW.reshape((N,) + module.weight.shape)
                    jac.append(dz_dW)
                if bias_repeat is not None:
                    Nbias = bias_repeat.shape[0]
                    dz_dy_select = dz_dy[:, bias_output]
                    repeat = torch.repeat_interleave(bias_repeat)
                    dz_db = torch.zeros(N, Nbias, device=device).index_add_(
                        1, repeat, dz_dy_select
                    )
                    if as_tuple:
                        dz_db = dz_db.reshape((N,) + module.bias.shape)
                    jac.append(dz_db)
                layer += 1

        if as_tuple:
            return tuple(jac)
        else:
            return torch.cat(jac, dim=1)

    if not hasattr(model, "jacobian"):
        model.jacobian = types.MethodType(jacobian, model)


class JacobianMode:
    def __init__(self, model):
        self.model = model
        if not isinstance(model, nn.Module):
            raise TypeError("model should be a nn.Module")

    def __enter__(self):
        model = self.model
        model.x_in = []
        model.gradient = []
        self.forward_pre_hook = []
        self.backward_hook = []

        def record_input_shape(self, input):
            model.input_shape = input[0].shape[1:]

        def record_forward(self, input, layer):
            model.x_in[layer] = input[0].detach()

        def record_backward(self, grad_input, grad_output, layer):
            model.gradient[layer] = grad_output[0]

        module0 = next(model.children())
        self.first_forward_hook = module0.register_forward_pre_hook(record_input_shape)

        layer = 0
        for module in model.children():
            if sum(p.numel() for p in module.parameters()):
                model.x_in.append(None)
                model.gradient.append(None)
                self.forward_pre_hook.append(
                    module.register_forward_pre_hook(
                        partial(record_forward, layer=layer)
                    )
                )
                self.backward_hook.append(
                    module.register_backward_hook(partial(record_backward, layer=layer))
                )
                layer += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.first_forward_hook.remove()
        for hook in self.forward_pre_hook:
            hook.remove()
        for hook in self.backward_hook:
            hook.remove()

        del self.model.input_shape
        del self.model.x_in
        del self.model.gradient
