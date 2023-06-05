import hickle as hkl
import numpy as np
import nnet as net
import matplotlib.pyplot as plt

def y_value(x1, x2):
    if np.abs(x1) >= 0.1 and np.abs(x2) >= 0.1:
        return np.sin(np.pi * x1 * x2)/(x1**2 + x2**2)
    if np.abs(x1) < 0.1:
        return np.sin(np.pi * 0.1 * x2)/(0.1 ** 2 + x2**2)
    if np.abs(x2) < 0.1:
        return np.sin(np.pi * 0.1 * x1)/(0.1 ** 2 + x1**2)


x1 = np.linspace(-1, 1, 10)
x2 = np.linspace(-1, 1, 10)

X1, X2 = np.meshgrid(x1, x2)
x1 = np.arange(-1, 1, 0.1)
x2 = np.arange(-1, 1, 0.1)

x = np.zeros([2, len(x1) * len(x2)])
y_t = np.zeros([1,len(x1)* len(x2)])

k=0
print(len(x1))
for i in range(0,len(x1)):
    for j in range(0,len(x2)):
        x[0,k]= x1[i]
        x[1,k]= x2[j]

        y_t[0,k]= y_value(x1[i], x2[j])

        k+=1
       

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.plot_surface(X1, X2, y_true, cmap='viridis')

# ax.set_xlabel('X1')
# ax.set_ylabel('X2')
# ax.set_zlabel('Y')

# plt.show()

# xx1= np.array([1,2,3,4,5])
# xx2= np.array([1,2,3,4,5])

# x = np.column_stack((xx1, xx2))

# print(x)
# pass

def layer3_momentum_adapt(x , y_t,
    max_epoch = 2000,
    err_goal = 1e-3 ,
    disp_freq = 1000 ,
    lr = 0.001 ,
    mc = 0.9,
    ksi_inc = 1.05,
    ksi_dec = 0.7,
    er = 1.04,
    L = x.shape[0] ,
    K1 = 100,
    K2 = 100,
    K3 = 100):

    SSE_vec = [] 

    w1, b1 = net.nwtan(K1, L)  
    w2, b2 = net.nwtan(K2, K1)  
    w3, b3 = net.rands(K3, K2)

    hkl.dump([w1,b1,w2,b2,w3,b3], 'wagi3w.hkl')
    # w1,b1,w2,b2,w3,b3= hkl.load('wagi3w.hkl')
    w1_t_1, b1_t_1, w2_t_1, b2_t_1, w3_t_1, b3_t_1 = w1, b1, w2, b2,w3, b3

    SSE = 0
    lr_vec = list()

    for epoch in range(1, max_epoch+1): 
        y1 = net.tansig( np.dot(w1, x),  b1) 
        y2 = net.tansig( np.dot(w2, y1),  b2) 
        y3 = net.purelin(np.dot(w3, y2), b3) 
        
        e = y_t - y3 
        
        
        SSE_t_1 = SSE
        SSE = net.sumsqr(e) 
        if np.isnan(SSE): 
            break
        else:
            if SSE > er * SSE_t_1:
                lr *= ksi_dec
            elif SSE < SSE_t_1:
                lr *= ksi_inc
        lr_vec.append(lr)
        
        
        d3 = net.deltalin(y3, e)
        d2 = net.deltatan(y2, d3, w3) 
        d1 = net.deltatan(y1, d2, w2) 

        
        dw1, db1 = net.learnbp(x,  d1, lr) 
        dw2, db2 = net.learnbp(y1, d2, lr)
        dw3, db3 = net.learnbp(y2, d3, lr)
        
        w1_temp, b1_temp, w2_temp, b2_temp, w3_temp, b3_temp = \
        w1.copy(), b1.copy(), w2.copy(), b2.copy() , w3.copy(), b3.copy()
        
        w1 += dw1 + mc * (w1 - w1_t_1)
        b1 += db1 + mc * (b1 - b1_t_1) 
        w2 += dw2 + mc * (w2 - w2_t_1) 
        b2 += db2 + mc * (b2 - b2_t_1)
        w3 += dw3 + mc * (w3 - w3_t_1) 
        b3 += db3 + mc * (b3 - b3_t_1)  
        
        w1_t_1, b1_t_1, w2_t_1, b2_t_1, w3_t_1, b3_t_1 = \
        w1_temp, b1_temp, w2_temp, b2_temp,w3_temp, b3_temp
        
        SSE = net.sumsqr(e) 
        if np.isnan(SSE): 
            break

        SSE_vec.append(SSE)
    
        if SSE < err_goal: 
            break 
    print("Epoch: %5d | SSE: %5.5e " % (epoch, SSE))
    hkl.dump([SSE_vec], 'SSE2w_adapt.hkl')
                
    plt.plot(SSE_vec) 
    plt.ylabel('SSE') 
    plt.yscale('linear') 
    plt.title('epoch') 
    plt.grid(True) 
    plt.show()





# for i in range(n_points):
#     for j in range(n_points):
#         CC_sym.input['AA'] = x[i, j]
#         CC_sym.input['BB'] = y[i, j]
#         CC_sym.compute()
#         z[i, j] = CC_sym.output['CC']

# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(x, y, z, cmap='viridis')
# ax.set_xlabel('AA')
# ax.set_ylabel('BB')
# ax.set_zlabel('CC')
# ax.view_init(30, 200)
# print("Najmniejsza możliwa otrzymana wartość CC: ",z.min())
# print("Największa możliwa otrzymana wartość CC: ",z.max())
# plt.show()

layer3_momentum_adapt(x,  y_t)
pass

