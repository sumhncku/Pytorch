import torch
from sklearn.datasets import load_boston
from sklearn import preprocessing
from torch.autograd import Variable

# dtype = torch.FloatTensor
dtype = torch.cuda.FloatTensor
X, y = load_boston(return_X_y=True)
X = preprocessing.scale(X[:100,:])
y = preprocessing.scale(y[:100].reshape(-1, 1))

data_size, D_input, D_output, D_hidden = X.shape[0], X.shape[1], 1, 50
X = Variable(torch.Tensor(X).type(dtype), requires_grad=False)
y = Variable(torch.Tensor(y).type(dtype), requires_grad=False)
w1 = Variable(torch.randn(D_input, D_hidden).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(D_hidden, D_output).type(dtype), requires_grad=True)

lr = 1e-5
epoch = 200000
for i in range(epoch):
    
    # forward
    h = torch.mm(X, w1)
    h_relu = h.clamp(min=0)
    y_pred = torch.mm(h_relu, w2)
    loss = (y_pred - y).pow(2).sum()
    if i % 10000 == 0:
        print('epoch: {} loss: {}'.format(i, loss.data[0]))    # 使用loss.data[0]，可以输出Tensor的值，而不是Tensor信息
    
    # backward 我们直接使用Variable.backward()，就能根据forward构建的计算图进行反向传播
    loss.backward()
      

    w1.data -= lr * w1.grad.data                       
    w2.data -= lr * w2.grad.data

    w1.grad.data.zero_()
    w2.grad.data.zero_()