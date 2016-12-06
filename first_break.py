import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

def fb_sta_lta(trace, stab, thresh, times):
	env=np.abs(hilbert(trace))
	nshort=max(round(np.size(env)/(10.0*times)),2)
	nlong=min(times*nshort,np.size(env))
	shortones=np.ones(int(nshort));
	longones=np.ones(int(nlong));
	sta=np.convolve(env,shortones,'full')/nshort;
	lta=np.convolve(env,longones,'full')/nlong;
	ltamax=np.max(lta);
	lta=lta[:np.size(sta)]
	ratio=sta/(lta+stab*ltamax);
	ind=np.where(ratio>thresh);
	if(not all(ind[0]) and ind[0][0]!=1):
		tp=ind[0][0]
	else:
		tp=np.nan
	return tp,sta,lta,ratio
		
		

trace=np.loadtxt('seis.asc')
thresh=0.9
times=10
stab=0.1
tp,sta,lta,ratio=fb_sta_lta(trace,stab,thresh,times)
plt.figure(1);
plt.plot(trace/np.max(trace))
plt.hold(True)
plt.plot(ratio,'r')
der=np.array([-1,1])
der_ratio=np.convolve(ratio,der,'full')
plt.plot(der_ratio,'k');
#amp,first_break = max(abs(der_ratio))
plt.show()
