import torch
import pdb

dtype = torch.float 
device = torch.device("cpu")

#-- N is batch size; D_in is input dimension
#   H is hidden dimension; D_out is output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

#-- Create random input and output data
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

#-- Randomly initialize weights
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(500):
    #-- Forward pass: compute predicted y
    h = x.mm(w1) #-- mm probaby means matrix multiply ?
    

pdb.set_trace()