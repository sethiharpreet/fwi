# From rsf format to bin : requires Madagascar api , the tough way
# Marmousi 2 model nz by nx ( 210 by 1024 )
# Harpreet Singh
# Please note madagascar binary require float32 format : change it accordingly if writing for madagascar
import rsf.api as rsf
import numpy as np
import matplotlib.pyplot as plt

# P-wave-velocity
inp = rsf.Input('gp-velocity.rsf')
n1=inp.int("n1")
n2=inp.int("n2")
vp=np.zeros((n2,n1),'f')
inp.read(vp)
vp=vp.T
plt.imshow(vp);plt.title("P-wave-velocity model");
vp.reshape((n2,n1),'c').T.tofile('gp.bin')

## S-wave-velocity
#ins = rsf.Input('vssmall.rsf')
#n1=ins.int("n1")
#n2=ins.int("n2")
#vs=np.zeros((n2,n1),'f')
#ins.read(vs)
#vs=vs.T
#plt.figure();plt.imshow(vs);plt.title("S-wave-velocity model");
#vs.tofile('vs.bin')

## Density
#ind = rsf.Input('densitysmall.rsf')
#n1=ind.int("n1")
#n2=ind.int("n2")
#d=np.zeros((n2,n1),'f')
#ind.read(d)
#d=d.T
#plt.figure();plt.imshow(d);plt.title("Density model");
#d.tofile('density.bin')

plt.show()

## If Need to read then following is the code ##
#  f1=open('vp.bin','rb')
#  vp=np.fromfile(f1)
#  vp=vp.reshape(n1,n2)  #(n1=210 , n2=1024)
#
########
