import numpy as np
import matplotlib.pyplot as plt


ng=400;nt=2000;nb=80;nz=400;nx=400;


f1=open('gvel.bin','rb')
gvel=np.fromfile(f1,dtype='float32')
gvel=gvel.reshape(nz,nx,order='f');
print(gvel.size)
plt.figure();plt.imshow(gvel);plt.title("Original Velocity Model");

f1=open('vel.bin','rb')
vnew=np.fromfile(f1,dtype='float32')
vnew=vnew.reshape(nz,nx,order='f');
print(vnew.size)
plt.figure();plt.imshow(vnew);plt.title("Extended Velocity Model");

f1=open('grad.bin','rb')
grad=np.fromfile(f1,dtype='float32')
grad=grad.reshape(nz,nx,order='f');
print(grad.size)
plt.figure();plt.imshow(grad);plt.title("Gradient");

f1=open('spgrad.bin','rb')
spgrad=np.fromfile(f1,dtype='float32')
spgrad=spgrad.reshape(nz,nx,order='f');
print(spgrad.size)
plt.figure();plt.imshow(spgrad);plt.title("Source Gradient");

f1=open('gsnap.bin','rb')
snap=np.fromfile(f1,dtype='float32')
snap=snap.reshape((nz+nb,nx+2*nb),order='f');
print(snap.size)
plt.figure();plt.imshow(snap,cmap='gray');plt.title("Backpropagated wavefield snapshot");

f1=open('rsnap.bin','rb')
snap=np.fromfile(f1,dtype='float32')
snap=snap.reshape((nz+nb,nx+2*nb),order='f');
snap=snap[0:nz,nb:nx+nb]
print(snap.size)
plt.xlabel('X in km');plt.ylabel('Z in km')
plt.figure();plt.imshow(snap,cmap='gray');plt.title("Reconstructed wavefield snapshot");

#f1=open('snap.bin','rb')
#snap=np.fromfile(f1,dtype='float32')
#snap=snap.reshape((nz+nb,nx+2*nb),order='f');
#print(snap.size)
#plt.figure();plt.imshow(snap,cmap='gray');plt.title("Forward wavefield snapshot");

f1=open('source.bin','rb')
src=np.fromfile(f1,dtype='float32')
print(src.size)
plt.figure();plt.plot(src);plt.title("Ricker Source");


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

f1=open('scagrad.bin','rb')
scgrad=np.fromfile(f1,dtype='float32')
scgrad=scgrad.reshape(nz,nx,order='f');
print(scgrad.size)
plt.figure();plt.imshow(scgrad);plt.title("Scaled Gradient");

f1=open('illum.bin','rb')
illum=np.fromfile(f1,dtype='float32')
illum=illum.reshape(nz+nb,nx+2*nb,order='f');
print(illum.size)
plt.figure();plt.imshow(illum);plt.title("Illumination");

f1=open('upvel.bin','rb')
vnew=np.fromfile(f1,dtype='float32')
vnew=vnew.reshape(nz,nx,order='f');
print(vnew.size)
plt.figure();plt.imshow(vnew);plt.title("New Model");

plt.show()
