import torch
import torch.nn as nn
import math

#实现了一个局部线性层
class LocallyConnected(nn.Module):
    """Local linear layer, i.e. Conv1dLocal() with filter size 1.

    Args:
        num_linear: num of local linear layers, i.e.局部线性层的数量。
        in_features: m1 输入特征的数量（m1）。
        out_features: m2 输出特征的数量（m2）。
        bias: whether to include bias or not

    Shape:
        - Input: [n, d, m1]
        - Output: [n, d, m2]

    Attributes:
        weight: [d, m1, m2]权重参数，形状为 [num_linear, input_features, output_features]
        bias: [d, m2]偏置参数，形状为 [num_linear, output_features]。
    """

    def __init__(self, num_linear, input_features, output_features, bias=True):
        super(LocallyConnected, self).__init__()
        self.num_linear = num_linear
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(num_linear,
                                                input_features,
                                                output_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_linear, output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        self.reset_parameters()#初始化权重和偏置。

    @torch.no_grad()
    #使用均匀分布初始化权重和偏置，范围为 [-bound, bound]。
    def reset_parameters(self):
        k = 1.0 / self.input_features
        bound = math.sqrt(k)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)
    #输入形状为 [n, d, m1]，输出形状为 [n, d, m2]。
    def forward(self, input: torch.Tensor):
        # [n, d, 1, m2] = [n, d, 1, m1] @ [1, d, m1, m2]
        #使用矩阵乘法计算每个位置的线性变换。
        out = torch.matmul(input.unsqueeze(dim=2), self.weight.unsqueeze(dim=0))
        out = out.squeeze(dim=2)
        #如果有偏置项，则将其加到输出中。
        if self.bias is not None:
            # [n, d, m2] += [d, m2]
            out += self.bias
        return out

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'num_linear={}, in_features={}, out_features={}, bias={}'.format(
            self.num_linear, self.in_features, self.out_features,
            self.bias is not None
        )

#测试代码
def main():
    n, d, m1, m2 = 2, 3, 5, 7

    # numpy使用 NumPy 进行计算
    import numpy as np
    #随机生成的输入数据和权重。
    input_numpy = np.random.randn(n, d, m1)
    weight = np.random.randn(d, m1, m2)
    output_numpy = np.zeros([n, d, m2])
    for j in range(d):
        # [n, m2] = [n, m1] @ [m1, m2]
        output_numpy[:, j, :] = input_numpy[:, j, :] @ weight[j, :, :]

    # torch 使用 PyTorch 进行计算
    torch.set_default_dtype(torch.double)
    input_torch = torch.from_numpy(input_numpy)
    locally_connected = LocallyConnected(d, m1, m2, bias=False)
    locally_connected.weight.data[:] = torch.from_numpy(weight)
    output_torch = locally_connected(input_torch)

    # compare比较 NumPy 和 PyTorch 的结果
    print(torch.allclose(output_torch, torch.from_numpy(output_numpy)))


if __name__ == '__main__':
    main()
