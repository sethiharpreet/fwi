import numpy as np
import matplotlib.pyplot as plt


ng=400;nt=2000;nb=60;nz=400;nx=400;



f1=open('vnew.bin','rb')
vnew=np.fromfile(f1,dtype='float32')
vnew=vnew.reshape(nz+nb,nx+2*nb,order='f');
print(vnew.size)
plt.figure();plt.imshow(vnew);plt.title("Extended Velocity Model");

f1=open('grad.bin','rb')
vnew=np.fromfile(f1,dtype='float32')
vnew=vnew.reshape(nz,nx,order='f');
print(vnew.size)
plt.figure();plt.imshow(vnew);plt.title("Gradient");

f1=open('gsnap.bin','rb')
snap=np.fromfile(f1,dtype='float32')
snap=snap.reshape((nz+nb,nx+2*nb),order='f');
print(snap.size)
plt.figure();plt.imshow(snap,cmap='gray');plt.title("Backpropagated wavefield snapshot");

f1=open('rsnap.bin','rb')
snap=np.fromfile(f1,dtype='float32')
snap=snap.reshape((nz+nb,nx+2*nb),order='f');
print(snap.size)
plt.figure();plt.imshow(snap,cmap='gray');plt.title("Reconstructed wavefield snapshot");

f1=open('snap.bin','rb')
snap=np.fromfile(f1,dtype='float32')
snap=snap.reshape((nz+nb,nx+2*nb),order='f');
print(snap.size)
plt.figure();plt.imshow(snap,cmap='gray');plt.title("Forward wavefield snapshot");

f1=open('source.bin','rb')
src=np.fromfile(f1,dtype='float32')
print(src.size)
plt.figure();plt.plot(src);plt.title("Ricker Source");

#f2=open('spertz.bin','rb')
#spz=np.fromfile(f2,dtype='float32')
#print(spz.size)
#spz=spz.reshape(nt,ng)
#plt.figure();plt.imshow(spz,extent=[0,1,0,1],cmap='gray');
#plt.title("perturbation wavefield z");

#f2=open('spertx.bin','rb')
#spx=np.fromfile(f2,dtype='float32')
#print(spx.size)
#spx=spx.reshape(nt,ng)
#plt.figure();plt.imshow(spx,extent=[0,1,0,1],cmap='gray');
#plt.title("perturbation wavefield x");

f2=open('resd.bin','rb')
res=np.fromfile(f2,dtype='float32')
print(res.size)
res=res.reshape(nt,ng)
plt.figure();plt.imshow(res,extent=[0,1,0,1],cmap='gray');
plt.title("residuals");
plt.show()
