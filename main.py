import numpy as np
import math
import matplotlib.pyplot as plt
import NeurNetsFromBeginNUMPY as NeurNFN

m = NeurNFN.Model(2, (4,6,1))
print(m.Forward((4,2)))
databs = [[(4,2),1],[(1,1),1],[(1.5,1.6),1],[(1.2,1.4),1],
          [(3.4,4),0],[(0.3,0.5),0],[(1.9,0.1),0],[(0.2,4.4),0],[(1.5,3.4),0],
          [(4,1.4),0],[(4,3),0]]
for dt in databs:
    dt[1] = [dt[1]]

m.Train(databs, 4000, 0.03)
plt.subplot(1,2,1)
for data in databs:
    colr = 'blue'
    if(data[1][0] == 1):
        colr = 'red'
    plt.plot(data[0][0],data[0][1],marker='o',color=colr)
plt.subplot(1,2,2)
X = np.linspace(-1,5,1000)
Y = np.random.uniform(0,4.5,1000)
c = []
for cint in range(0,1000):
    c.append(m.Forward((X[cint],Y[cint])))

print(m.Forward((4,2)))
plt.scatter(X,Y,c=c,cmap='coolwarm')
plt.show()
