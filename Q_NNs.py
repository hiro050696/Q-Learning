import gym
import numpy as np
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

model = nn.Linear(16,4,bias=False)           # Tao mang neural
nn.init.uniform(model.weight,0,0.01)         # Khoi tao weight, do Q(s,a)<1
loss_fn = nn.MSELoss(size_average=False)     # Mean square error
lr = 1e-2                                    # Learning rate
e = 0.1                                      # He so su dung cho greedly
y = 0.99                                     # gamma, he so discount reward
optim = torch.optim.Adam(model.parameters(),lr=lr)
num_episodes = 2000                          # So game se choi
jList = []          # List chua so action thuc hien cua moi episode, ve bieu do quan sat
rList = []          # List chua reward nhan duoc cua moi episode               
lList = []          # List chua loss accumulate cua episode
for i in range(num_episodes):
    s = env.reset()                                      # 
    s = Variable(torch.Tensor(np.identity(16)[s:s+1]))   # chuyen s sang one-hot vector
    rAll = 0        # khoi tao reward 
    lAll = 0        # khoi tao loss
    d = False       # = True khi ket thuc episode
    j = 0           # So action thuc hien trong 1 episode
    while j<99:
        j+=1
        Q = model(s)    # Predict s -> Q(s,a)
        _,a = Q.max(1)  # Chon action a co Q(s,a) lon nhat
        a = a.data[0]   # Tensor -> integer
        if np.random.rand(1) < e: a = env.action_space.sample()   # Tao nhieu (nosie) 
        s1,r,d,_ = env.step(a)                                    # Thuc hien action a  
        s1 = Variable(torch.Tensor(np.identity(16)[s1:s1+1]))     # s1 (next state)->one-hot vector
        Q1 = model(s1)              # Predict s1-> Q(s,a)
        Qmax,_ = Q1.max(1)          # Chon gia tri Q(s1,a1) lon nhat
        Qmax = Qmax.data[0] 
        targetQ = Q.clone().data    # Clone Q sang Qmax, ko the thuc hien assign o day******
        targetQ[0][a] = r + y*Qmax  # Cap nhat Q(s,a) theo Bellman equation
        targetQ = Variable(targetQ, requires_grad=False) 
        loss = loss_fn(Q, targetQ)  # Loss bang gia tri Q(s,a) sau khi update tru Q(s,a) predict boi Neural Network 
        lAll += loss.data[0]        
        
        optim.zero_grad()           # Update
        loss.backward()             # backprop
        optim.step()
        s = s1                      # current state <- next state
        rAll +=r
        if d==True:
            e = 1.0/((i/50+10))     # giam noise
            lAll = lAll/j   
            break                   
    jList.append(j)
    rList.append(rAll)
    lList.append(lAll)
plt.plot(rList) 
plt.show()
print ('end')

            
        
        
