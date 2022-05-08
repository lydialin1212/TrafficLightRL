import numpy as np

class blackBox(object):
    def __init__(self):
        max_epoch = 300
        batch_size = 30
        hidden_size = 10
        learning_rate = 1.0
        
        self.model = TwoLayerNet(input_size=7, hidden_size=hidden_size, output_size=1)
        self.optimizer = SGD(lr=learning_rate)
        
        self.total_loss = 0
        self.loss_count = 0
        self.loss_list = []
        self.replay_buffer = []

        
    def predict(self, features):

        Q = self.model.predict(features)[0]
        return Q
        

    def update(self, batch_x, batch_y):

        loss = self.model.forward(batch_x, batch_y)


        self.model.backward()
        self.optimizer.update(self.model.params, self.model.grads)
        
        self.total_loss += loss
        self.loss_count += 1
        self.loss_list += [loss]
        print(self.loss_list[-20:-1])


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        W1 = 0.01 * np.random.randn(I, H)
        b1 = np.zeros(H)
        W2 = 0.01 * np.random.randn(H, O)
        b2 = np.zeros(O)

        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]
        self.loss_layer = Loss()

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
            

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x, t):
        score = self.predict(x)

        loss = self.loss_layer.forward(score, t)
        self.loss = loss
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None
        
    def forward(self, x):   
        W, b = self.params

        out = np.dot(x, W) + b
        self.x = x

        return out
        
    def backward(self, dout):
        W, b = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis = 0)
        
        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx
        
class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None
        
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out
        
    def backward(self, dout):
        dx = dout * ( 1.0 - self.out)
        return dx
        
def mean_squared_error(x, y):
    m = len(x)
    res = 0
    for i in range(m):

        res += (x[i]-y[i])*(x[i]-y[i])
    res /= m


    return res
        
class Loss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None  
        self.t = None  

    def forward(self, x, t):
        self.t = t
        self.y = x

        #if self.t.size == self.y.size:
         #   self.t = self.t.argmax(axis=1)

        loss = mean_squared_error(self.y.reshape(-1), self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = self.y.copy()

        #dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size

        return dx
        
        
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]
    
