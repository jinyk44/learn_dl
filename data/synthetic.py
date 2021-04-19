import torch

def make_linear_data(w,b,num_example):
    """
    生成 y = Xw + b + 噪声
    """
    X = torch.normal(0,1,(num_example,len(w)))
    y = torch.matmul(X,w)+b
    y += torch.normal(0,0.01,y.shape)
    return X,y.reshape((-1,1))
    
