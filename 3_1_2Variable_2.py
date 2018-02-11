from torch.autograd import Variable, Function
import torch

# 矩陣求導範例
x = torch.randn(3)
x = Variable(x, requires_grad=True)

y = x * 2
y.backward(torch.FloatTensor([1, 0.1, 0.01]))

print("x.data: ", x.data)
print("y: ", y)
print("x.grad: ", x.grad)