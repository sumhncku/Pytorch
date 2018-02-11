from torch.autograd import Variable, Function
import torch

# 生成 Variable
# requires_grad表示在backward是否計算其梯度
x = Variable(torch.Tensor([1]), requires_grad=True)
w = Variable(torch.Tensor([2]), requires_grad=True)
b = Variable(torch.Tensor([3]), requires_grad=True)

# 生成一個計算圖 (computational graph)
y = w * x + b

# 計算梯度
y.backward()

# 印出 data
print("x.data: ", x.data)
print("w.data: ", w.data)
print("b.data: ", b.data)

# 印出梯度
print("x.grad: ", x.grad)
print("w.grad: ", w.grad)
print("b.grad: ", b.grad)

# 印出 grad_fn
print("x.grad_fn: ", x.grad_fn)
print("w.grad_fn: ", w.grad_fn)
print("b.grad_fn: ", b.grad_fn)

# 印出 y
print("y: ", y)