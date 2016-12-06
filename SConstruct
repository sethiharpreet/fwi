import os

env = Environment(ENV = os.environ,CC='gcc')
env.Append(CCFLAGS=['-fopenmp'])
env.Prepend(LIBS=['m','fftw3f'])
ldflags='-fopenmp'

mod=env.Program('modeling.exe','modeling.c',LINKFLAGS=ldflags)
fwi=env.Program('crossfwi.exe','crossfwi.c',LINKFLAGS=ldflags)

Depends(mod,['gvel.bin','corrfunc.c'])
#Depends(fwi,['gp.bin','dobs.bin','func.c'])
