import numpy as np
import matplotlib.pyplot as plt

nt=2000;ng=400

f2=open('dobs.bin','rb')
dobs=np.fromfile(f2,dtype='float32')
print(dobs.size)
dobs=dobs.reshape(nt,ng)
plt.figure();plt.imshow(dobs,extent=[0,1,0,1],cmap='gray');
plt.title("observed");

f2=open('dcal.bin','rb')
dcal=np.fromfile(f2,dtype='float32')
print(dcal.size)
dcal=dcal.reshape(nt,ng)
plt.figure();plt.imshow(dcal,extent=[0,1,0,1],cmap='gray');
plt.title("Cal wavefield");

f2=open('corr.bin','rb')
corr=np.fromfile(f2,dtype='float32')
print(corr.size)
corr=corr.reshape(2*nt-1,ng)
plt.figure();plt.imshow(corr,extent=[0,1,0,1],cmap='gray');
plt.title("Cal wavefield");


plt.show()
