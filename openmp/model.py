import numpy as np
import matplotlib.pyplot as plt

vel=2.0*np.ones((400,400));
plt.imshow(vel);plt.title('velocity model');plt.colorbar()
plt.show();

vel.T.astype('float32').tofile('vel.bin')


