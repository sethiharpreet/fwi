#include <stdio.h>
#include <string.h>
#include "corrfunc.c"

#include <fftw3.h>

int main(int argcs, char *argv[])
{
	int i, na=150, nb=100;
	float *a, *b, *corr;
	a=(float*)malloc(sizeof(float)*na);
	b=(float*)malloc(sizeof(float)*nb);
	corr=(float*)malloc(sizeof(float)*(na+nb-1));
	readbin(a,"a.bin");
	readbin(b,"b.bin");
	cross_corr(a,b,corr,na,nb);
	writebin(corr,na+nb-1,"corr.bin");
	

	}
