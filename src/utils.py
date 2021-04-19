import torch
def squared_loss(ground_truth,predictions):
    loss = 0.5*(ground_truth-predictions.reshape(ground_truth.shape))**2
    return loss

def sgd(batch_size,lr,params):
    with torch.no_grad():
        for param in params:
            param -= lr*param.grad/batch_size
            param.grad.zero_()
def linreg(X,w,b):
    return torch.matmul(X,w)+b
