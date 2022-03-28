import numpy as np 
from matplotlib import pyplot as plt 

RECORD_FREQ = 100

D_loss = np.load('./checkpoints_large/D_history.npy')
G_loss = np.load('./checkpoints_large/G_history.npy')

x = np.arange(D_loss.shape[0])*RECORD_FREQ

plt.plot(x, D_loss,x, G_loss)
plt.ylim(top=10)
plt.xlabel("Backprops")
plt.legend(['D_loss', 'G_loss'])


plt.show()