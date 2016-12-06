#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

#include "corrfunc.c"

#ifdef _OPENMP
#include <omp.h>
#include "ompfunc.c"
#endif


int main(int argc, char *argv[])
{
	
	int i,j=0;
	float tmax,max=0,E=0;
	int ig,ns,ng,nt;
	float *dobs, *dcal, *corr;
	float *trdobs, *trdcal, *trcorr;
	nt=2000 , ns=1, ng=400;
	dcal=(float*)malloc(ns*ng*nt*sizeof(float));
	dobs=(float*)malloc(ns*ng*nt*sizeof(float));
	trdcal=(float*)malloc(ns*ng*nt*sizeof(float));
	trdobs=(float*)malloc(ns*ng*nt*sizeof(float));
	corr=(float*)malloc(ng*(2*nt-1)*sizeof(float));
	trcorr=(float*)malloc(ng*(2*nt-1)*sizeof(float));
	
	readbin(dobs,"dobs.bin");
	readbin(dcal,"dcal.bin");
	matrix_transpose(dobs,trdobs,ng,nt);			
	matrix_transpose(dcal,trdcal,ng,nt);
	for(ig=0;ig<ng;ig++){
		cross_corr(&trdobs[ig*nt],&trdcal[ig*nt],&trcorr[ig*2*(nt-1)],nt,nt);
		for(i=0;i<2*nt-1;i++){
			max=MAX(max,trcorr[i]);
			j=(max>trcorr[i]) ? i : j;
		printf("val is %d \n",j);
		}
		}
	matrix_transpose(trcorr,corr,2*nt-1,ng);	
	writebin(corr,ng*(2*nt-1),"corr.bin");

		
	}
