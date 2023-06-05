import hickle as hkl
import numpy as np
import nnet as net
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def y_value(x1, x2):
    if np.abs(x1) >= 0.1 and np.abs(x2) >= 0.1:
        return np.sin(np.pi * x1 * x2)/(x1**2 + x2**2)
    elif np.abs(x1) < 0.1:
        return np.sin(np.pi * 0.1 * x2)/(0.1 ** 2 + x2**2)
    elif np.abs(x2) < 0.1:
        return np.sin(np.pi * 0.1 * x1)/(0.1 ** 2 + x1**2)




x1 = np.arange(-1, 1.01, 0.21)
x2 = np.arange(-1, 1.01, 0.21)


x = np.zeros([2, len(x1) * len(x2)])
y_t = np.zeros([1,len(x1)* len(x2)])

k=0

for i in range(0,len(x1)):
    for j in range(0,len(x2)):
        x[0,k]= x1[i]
        x[1,k]= x2[j]
        y_t[0,k]= y_value(x1[i], x2[j])
        k+=1



x 
y_t
max_epoch =  2000
err_goal = 1e-10 
disp_freq = 100 
lr = 0.001 
mc = 0.95
ksi_inc = 1.05
ksi_dec = 0.7
er = 1.04
L = 2
K1 = 40
K2 = 20
K3 = 1
SSE_vec = [] 
w1, b1 = net.nwtan(K1, L)  
w2, b2 = net.nwtan(K2, K1)  
w3, b3 = net.rands(K3, K2)
hkl.dump([w1,b1,w2,b2,w3,b3], 'wagi3w.hkl')
# w1,b1,w2,b2,w3,b3= hkl.load('wagi3w.hkl')
w1_t_1, b1_t_1, w2_t_1, b2_t_1, w3_t_1, b3_t_1 = w1, b1, w2, b2,w3, b3
SSE = 0
lr_vec = list()

fig = plt.figure()
figa = plt.figure()
figb = plt.figure()
figc = plt.figure()

for epoch in range(1, max_epoch+1): 
    y1 = net.tansig( np.dot(w1, x),  b1) 
    y2 = net.tansig( np.dot(w2, y1),  b2) 
    y3 = net.purelin(np.dot(w3, y2), b3) 
    
    e = y_t - y3 
    
    
    SSE_t_1 = SSE
    SSE = net.sumsqr(e) 
    
    #adaptacyjne
    if np.isnan(SSE): 
        break
    else:
        if SSE > er * SSE_t_1:
            lr *= ksi_dec
        elif SSE < SSE_t_1:
            lr *= ksi_inc
    lr_vec.append(lr)
    #adaptacyjne
    
    
    d3 = net.deltalin(y3, e)
    d2 = net.deltatan(y2, d3, w3) 
    d1 = net.deltatan(y1, d2, w2) 
    
    dw1, db1 = net.learnbp(x,  d1, lr) 
    dw2, db2 = net.learnbp(y1, d2, lr)
    dw3, db3 = net.learnbp(y2, d3, lr)

    #momentum
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
    #momentum

    #odkomentuj jeśli nie używasz momentum
    # w1 += dw1
    # b1 += db1
    # w2 += dw2
    # b2 += db2
    # w3 += dw3
    # b3 += db3

    SSE = net.sumsqr(e) 
    if np.isnan(SSE): 
        break
    SSE_vec.append(SSE)
    
    if SSE < err_goal: 
        break 
    if (epoch % disp_freq) == 0:
            print("Epoch: %5d | SSE: %5.5e " % (epoch, SSE))
            
            
            fig.clf()
            figa.clf()
            figb.clf()
            figc.clf()


            lx = fig.add_subplot(111)
            lx.plot(SSE_vec)
            lx.set_ylabel('SSE')
            lx.set_yscale('linear')
            lx.set_title('epoch')
            lx.grid(True)
            
            X1, X2 = np.meshgrid(x1, x2)
            E = np.reshape(e, X1.shape)
            ax = figa.add_subplot(111, projection='3d')
            ax.plot_surface(X1, X2, E, cmap='viridis')
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.set_zlabel('e')
            X1, X2 = np.meshgrid(x1, x2)
            Y3 = np.reshape(y3, X1.shape)
            bx = figb.add_subplot(111, projection='3d')
            bx.plot_surface(X1, X2, Y3, cmap='viridis')
            bx.set_xlabel('x1')
            bx.set_ylabel('x2')
            bx.set_zlabel('y3')
            X1, X2 = np.meshgrid(x1, x2)
            Y_T = np.reshape(y_t, X1.shape)
           
            
            cx = figc.add_subplot(111, projection='3d')
            cx.plot_surface(X1, X2, Y_T, cmap='viridis')
            cx.set_xlabel('x1')
            cx.set_ylabel('x2')
            cx.set_zlabel('y_t')
              # Pauza, aby wykres się odświeżył
            
            plt.pause(0.1)
            plt.draw_all()
            
          
            
   

        
print("Epoch: %5d | SSE: %5.5e " % (epoch, SSE))
hkl.dump([SSE_vec], 'SSE2w_adapt.hkl')
while 1:
    pass
            
            


