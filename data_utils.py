import numpy as np
from torch.utils.data import Subset

def split_noniid(train_idcs, train_labels, alpha, n_clients):
    '''
    Splits a list of data indices with corresponding labels
    into subsets according to a dirichlet distribution with parameter
    alpha
    '''
    n_classes = train_labels.max()+1
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)

    class_idcs = [np.argwhere(train_labels[train_idcs]==y).flatten() 
           for y in range(n_classes)]

    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [train_idcs[np.concatenate(idcs)] for idcs in client_idcs]

    min_length = min(len(idcs) for idcs in client_idcs)
    client_idcs = [idcs[:min_length] for idcs in client_idcs]
  
    return client_idcs


def noniid_train(x_train, y_train, num_clients=2, alpha=1.0):
    train_idcs = np.random.permutation(len(x_train))

    train_labels = y_train[train_idcs]
    client_idcs = split_noniid(train_idcs, train_labels, alpha, num_clients)

    combined_client_idcs = np.concatenate(client_idcs)

    return x_train[combined_client_idcs], y_train[combined_client_idcs]
