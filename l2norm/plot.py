import numpy as np
import matplotlib.pyplot as plt


ng=400;nt=2000;nb=80;nz=400;nx=400;
ns=2


f1=open('gvel.bin','rb')
gvel=np.fromfile(f1,dtype='float32')
gvel=gvel.reshape(nz,nx,order='f');
print(gvel.size)
plt.figure();plt.imshow(gvel);plt.title("Original Velocity Model");

f1=open('vel.bin','rb')
vel=np.fromfile(f1,dtype='float32')
vel=vel.reshape(nz,nx,order='f');
print(vel.size)
plt.figure();plt.imshow(vel);plt.title("Initial Velocity Model");

f1=open('source.bin','rb')
src=np.fromfile(f1,dtype='float32')
print(src.size)
plt.figure();plt.plot(src);plt.title("Ricker Source");


f2=open('dobs.bin','rb')
dobs=np.fromfile(f2,dtype='float32')
print(dobs.size)
dobs=dobs.reshape(ns,nt,ng)
plt.figure();plt.imshow(dobs[0,:,:],extent=[0,1,0,1],cmap='gray');
plt.title("observed data");

f2=open('resd.bin','rb')
derr=np.fromfile(f2,dtype='float32')
print(derr.size)
derr=derr.reshape(ns,nt,ng)
plt.figure();plt.imshow(derr[0,:,:],extent=[0,1,0,1],cmap='gray');
plt.title("residuals");

f1=open('rsnap.bin','rb')
snap=np.fromfile(f1,dtype='float32')
snap=snap.reshape((nz+nb,nx+2*nb),order='f');
print(snap.size)
plt.figure();plt.imshow(snap[:nz,nb:nx+nb],cmap='gray');plt.title("Source Wavefield Reconstucted");

f1=open('snap.bin','rb')
snap=np.fromfile(f1,dtype='float32')
snap=snap.reshape((nz+nb,nx+2*nb),order='f');
print(snap.size)
plt.figure();plt.imshow(snap[:nz,nb:nx+nb],cmap='gray');plt.title("Forward Wavefield");

f1=open('spgrad.bin','rb')
spgrad=np.fromfile(f1,dtype='float32')
spgrad=spgrad.reshape(nz,nx,order='f');
print(spgrad.size)
plt.figure();plt.imshow(spgrad);plt.title("Source Gradient");

f1=open('grad.bin','rb')
grad=np.fromfile(f1,dtype='float32')
grad=grad.reshape(nz,nx,order='f');
print(grad.size)
plt.figure();plt.imshow(grad);plt.title("Gradient");

f1=open('upvel.bin','rb')
vnew=np.fromfile(f1,dtype='float32')
vnew=vnew.reshape(nz,nx,order='f');
print(vnew.size)
plt.figure();plt.imshow(vnew);plt.title("New Model");

plt.show()
