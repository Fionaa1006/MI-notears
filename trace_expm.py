import torch
import numpy as np
import scipy.linalg as slin
#计算矩阵指数的迹

class TraceExpm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # detach so we can cast to NumPy
        E = slin.expm(input.detach().numpy())# 计算矩阵指数
        f = np.trace(E)# 计算矩阵指数的迹
        E = torch.from_numpy(E)# 转换回 PyTorch 张量
        ctx.save_for_backward(E)# 保存中间结果用于反向传播
        return torch.as_tensor(f, dtype=input.dtype) # 返回迹的值

    @staticmethod
    def backward(ctx, grad_output):
        E, = ctx.saved_tensors
        grad_input = grad_output * E.t()# 计算输入的梯度
        return grad_input


trace_expm = TraceExpm.apply


def main():
    input = torch.randn(20, 20, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(trace_expm, input)

    input = torch.tensor([[1, 2], [3, 4.]], requires_grad=True)
    tre = trace_expm(input)
    f = 0.5 * tre * tre
    print('f\n', f.item())
    f.backward()
    print('grad\n', input.grad)


if __name__ == '__main__':
    main()
