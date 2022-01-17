import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# prepare data
X_numpy, Y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
X = torch.from_numpy(X_numpy.astype(np.float32))
Y = torch.from_numpy(Y_numpy.astype(np.float32))

#reshape T from row vector to column vector
Y = Y.view(Y.shape[0], 1)
n_samples, n_features = X.shape

# define model
model = nn.Linear(n_features, 1)

# define loss and optimiser
learning_rate=0.01
criterion = nn.MSELoss()
optim = torch.optim.SGD(model.parameters(), lr=learning_rate)

# training looop
n_iter = 100
for epoch in range(n_iter):
    #forward pass
    Y_pred = model(X)
    loss = criterion(Y_pred, Y)

    #backward pass
    loss.backward()

    #update weights
    optim.step()

    optim.zero_grad()
    print(f'epoch {epoch+1}: loss = {loss.item():.3f}')


#plot
predicted = model(X).detach().numpy()
plt.plot(X_numpy, Y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()