# #LINEAR REGRESSION SCRATCH----------------
# import numpy as np

# #f = w*x
# X = np.array([1,2,3,4], dtype = np.float)
# Y = np.array([2,4,6,8], dtype = np.float)

# #initialise weight w
# w=0
# #model prediction
# def forwardPass(x):
#     return w*x

# #loss
# def loss(y,y_predicted):
#     return ((y_predicted-y)**2).mean()

# #gradient
# #dJ/dw = 1/N(2x)(wx-y)
# def gradient(x,y,y_predicted):
#     return np.dot(2*x, y_predicted-y).mean()


# print(f'prediction before training: f(5) = {forwardPass(5):.3f}')

# #training

# learning_rate = 0.01
# n_iter = 20

# for epoch in range(n_iter):
#     y_pred = forwardPass(X)
#     l = loss(Y,y_pred)
#     dw = gradient(X,Y,y_pred)

#     #update weights
#     w = w-learning_rate*dw

#     #if epoch % 1 == 0:
#     print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

# print(f'prediction after training: f(5) = {forwardPass(10):.3f}')


#LINEAR REGRESSION SCRATCH USING SOME PYTORCH----------------
import torch
import torch.nn as nn

#f = w*x
X = torch.tensor([[1],[2],[3],[4]], dtype = torch.float).cuda()
Y = torch.tensor([[2],[4],[6],[8]], dtype = torch.float).cuda()
X_test = torch.tensor([5], dtype=torch.float).cuda()
n_samples, n_features = X.shape


#initialise weight w
input_size = n_features
output_size = n_features
model = nn.Linear(input_size, output_size).cuda()

#gradient
#dJ/dw = 1/N(2x)(wx-y)


print(f'prediction before training: f(5) = {model(X_test).item():.3f}')

#training

learning_rate = 0.01
n_iter = 500

loss = nn.MSELoss()
optimiser = torch.optim.SGD(model.parameters(), lr = learning_rate)
for epoch in range(n_iter):
    #prediction (forward pass)
    y_pred = model(X)

    #loss
    l = loss(Y,y_pred)

    #gradients
    
    l.backward() #backward pass

    #update weights
    optimiser.step()

    #empty the gradients
    optimiser.zero_grad()
    [w,b] = model.parameters()
    print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

print(f'prediction after training: f(5) = {model(X_test).item():.3f}')
