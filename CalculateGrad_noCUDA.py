import torch
from sklearn.datasets import load_boston
from sklearn import preprocessing
from torch.autograd import Variable

# X shape = (506,13); X.size = 6578
# y shape = (506,); y.size = 506
X, y = load_boston(return_X_y = True)

# 只取 X 前 100 筆資料 (X shape = (100,13); X.size = 1300) 進行正規化
X = preprocessing.scale(X[:100,:])
# 只取 y 前 100 筆資料 (y shape = (100,1); y.size = 100) 進行正規化
y = preprocessing.scale(y[:100].reshape(-1, 1))
# data_size = 506, D_input = 13, D_output = 1, D_hidden = 50
data_size, D_input, D_output, D_hidden = X.shape[0], X.shape[1], 1, 50

X = Variable(torch.Tensor(X), requires_grad=False)
y = Variable(torch.Tensor(y), requires_grad=False)
# w1.shape = torch.Size([13, 50]); w2.shape = torch.Size([50, 1])
w1 = Variable(torch.randn(D_input, D_hidden), requires_grad=True)
w2 = Variable(torch.randn(D_hidden, D_output), requires_grad=True)

lr = 1e-5
epoch = 200000
for i in range(epoch):
    # ================== forward ==================
    # torch.mm(X,w1) 會將 X 和 w1 兩個矩陣進行 product 計算
    # h.shape = torch.Size([100,50])
    h = torch.mm(X, w1)
    # clamp(min=0) is exactly ReLU
    h_relu = h.clamp(min=0)
    # y_pred.shape = ([100,1])
    y_pred = torch.mm(h_relu, w2)
    # loss.shape = torch.Size([1])
    loss = (y_pred - y).pow(2).sum()
    if i % 10000 == 0:
        # 使用loss.data[0]，可以輸出Tensor的值，而不是Tensor資訊
        print('epoch: {} loss: {}'.format(i, loss.data[0]))
    
    # ================== backward ================== 
    # 我們直接使用Variable.backward()，就能根據forward構建的計算圖進行反向傳播
    loss.backward()
    # Update weights using gradient descent; w1.data and w2.data are Tensors,
    # w1.grad and w2.grad are Variables and w1.grad.data and w2.grad.data are
    # Tensors.
    w1.data -= lr * w1.grad.data
    w2.data -= lr * w2.grad.data
    # Manually zero the gradients after updating weights
    w1.grad.data.zero_()
    w2.grad.data.zero_()