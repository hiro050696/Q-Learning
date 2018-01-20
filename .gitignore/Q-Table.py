# -*- coding: utf-8 -*-
import gym
import numpy as np
import matplotlib.pyplot as plt


env = gym.make('FrozenLake-v0')  # Load moi truong (enviroment) FrozenLake-v0
Q = np.zeros([16,4])             # Tao bang 16x4 voi gia tri khoi tao cho tat ca cac cell la 0
#--------------------------------# Cai dat thong so cho learning
lr = 0.8                         # Toc do hoc
y = 0.95                         # gamma, discount reward
num_episodes = 2000              # So game se choi
rList = []                       # tao 1 danh sach chua chuoi reward nhan duoc sau 2000 game
for i in range(num_episodes):
    s = env.reset()              # reset enviroment, bat dau game moi, s gan bang state ban dau
    rAll = 0                     # tong reward nhan duoc sau moi episode
    d = False                    # game ket thuc neu d=True
    j = 0                        # hoac ket thuc neu j=99, gioi han so action toi da agent thuc hien trong 1 episode
    while j<99:                  
        j+=1
        a = np.argmax(Q[s,:] + np.random.randn(1,4)*(1.0/(i+1)))  # Chon hanh dong (greedily) voi nhieu (nosie) tu bang Q
        s1,r,d,_ = env.step(a)                                    # Thuc hien hanh dong nhan lai state, reward, done, info(don't care)
        Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:])-Q[s,a])       # Cap nhat gia tri bang Q sau khi thuc hien 1 action
        rAll += r                                                 # Cong don (accomulate) reward nhan duoc tren ca episode
        s = s1                                                    # Gan trang thai hien tai bang trang thai moi
        if d==True:                                               # d=True neu agent toi vi tri Goal hoac roi xuong ho
            break
    rList.append(rAll)                                            # bo xung rAll vao rList
plt.plot(rList)
plt.show()
print ('end')
