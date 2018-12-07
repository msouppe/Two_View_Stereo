import numpy as np

I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
t = np.array([[0,0,0]])

t = np.transpose(t)
It = np.hstack((I,t))
print(It)

a = np.array([[1,2],[3,4]])  
b = np.array([[5,6],[7,8]]) 

print ('Horizontal stacking:') 
c = np.hstack((a,b)) 
print (c) 