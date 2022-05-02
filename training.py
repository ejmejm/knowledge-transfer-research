import tqdm
import torch
import numpy as np

def chu_train(X, weights, n_epochs, batch_size, learning_rate=2e-2, precision=1e-30,
              anti_hebbian_learning_strength=0.4, lebesgue_norm=2.0, rank=2, verbose=False):
    sample_sz = X.shape[1] # Number of input units
    n_hidden = weights.shape[0]
    # weights = copy.deepcopy(weights)
    if verbose:
        iter_range = tqdm(range(n_epochs))
    else:
        iter_range = range(n_epochs)
    for epoch in iter_range:    
        eps = learning_rate * (1 - epoch / n_epochs)
        shuffled_epoch_data = X[torch.randperm(X.shape[0]),:]
        for i in range(int(np.ceil(X.shape[0] // batch_size))):
            mini_batch = shuffled_epoch_data[i*batch_size:(i+1)*batch_size,:].cuda()            
            mini_batch = torch.transpose(mini_batch, 0, 1)            
            sign = torch.sign(weights)            
            W = sign * torch.abs(weights) ** (lebesgue_norm - 1)        
            tot_input = torch.mm(W, mini_batch)
            # tot_input is the output of the hidden units before the activation function or bias
            
            y = torch.argsort(tot_input, dim=0)            
            yl = torch.zeros((n_hidden, batch_size), dtype = torch.float).cuda()
            yl[y[n_hidden-1,:], torch.arange(batch_size)] = 1.0
            yl[y[n_hidden-rank], torch.arange(batch_size)] = -anti_hebbian_learning_strength  
            # yl is a matrix where each column is a vector with the same length as the number of hidden units
            # The index corresponding to the largest hidden unit output is 1
            # The index corresponding to the `rank` largest hidden unit output is -`anti_hebbian_learning_strength`
            # Every other entry is 0      
                    
            xx = torch.sum(yl * tot_input, 1)            
            xx = xx.unsqueeze(1)                    
            xx = xx.repeat(1, sample_sz)
            # xx is a matrix of size (n_hidden, sample_sz)
            # Each column is the sum of `yl`s over many samples     
            ds = torch.mm(yl, torch.transpose(mini_batch, 0, 1)) - xx * weights
            # ds is a matrix of size (n_hidden, sample_sz)

            # `yl` - The direction and magnitude each hidden unit should be adjusted for multiple samples
            # `xx` - The sum of the hidden unit outputs for each sample
            
            nc = torch.max(torch.abs(ds))            
            if nc < precision: nc = precision            
            weights += eps*(ds/nc)
    return weights