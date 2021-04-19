import random
import torch
def data_iter(features,labels,batch_size):
    num_example = len(features)
    indices = list(range(num_example))
    random.shuffle(indices)
    for i in range(0,num_example, batch_size):
        batch_indices = torch.tensor(indices[i:min(i+batch_size,num_example)])
        batch_features = features[batch_indices]
        batch_labels = labels[batch_indices]
        yield batch_features,batch_labels
