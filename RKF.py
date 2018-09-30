import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


c1=np.array([1/4,3/8,12/13,1,1/2])
c2=np.array([[1/4,0,0,0,0,0],
             [3/32,9/32,0,0,0,0],
             [1932/2197,-7200/2197,7296/2197,0,0,0],
             [439/216,-8,3680/513,-845/4104,0,0],
             [-8/27,2,-3544/2565,1859/4104,-11/40,0]])

c3=np.array([16/135,0,6656/12825,28561/56430,-9/50,2/55]) #가중평균계수    

def f(t,y):
    
    return y-t**2+1

dt=0.1
tf=5

t0=0
x0=np.array([0.5]) #x'(0),x(0)

size=len(x0)
data=[]
data.append(x0)
ki=np.zeros((6,size))

c1=np.array([1/4,3/8,12/13,1,1/2])

while t0<tf:

    ki[0]=f(t0,x0)
    
    for i in range(5):
        ti=t0+c1[i]*dt
        xi=x0+c2[i,0:i+1].dot(ki[0:i+1])*dt
        ki[i+1]=f(ti,xi)
        
    dx=c3.T.dot(ki).T*dt    
   
    t0=t0+dt
    x0=x0+dx
    
    data.append(x0)
    
df=pd.DataFrame(data)
df.insert(0,'time',df.index*dt)
df.insert(2,'x_exact',df.time**2+2*df.time+1-0.5*np.exp(df.time))

df.columns=['time','x','x_exact']
df.plot(x='time')
plt.grid()


