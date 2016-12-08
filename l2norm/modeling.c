#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include "func.c"



#ifdef _OPENMP
#include <omp.h>
#include "ompfunc.c"
#endif


int main(int argc, char *argv[])
{
	
	int is, it;
	int nz, nx, nt, ns, ng, nb, hfop, ompchunk ;
	int sxbeg, szbeg, gxbeg, gzbeg, jsx, jsz, jgx, jgz ; /* acquisiton parameters */
	int *sxz, *gxz;
  	float dx, dz, fm, dt, amp, totaltime=0	;
	float *wlt, *dobs, **v, **vv, **u0, **u1, **wfl, **lap, **ptr=NULL;
;
	fdm2d fdm;
	float start, end;

#ifdef _OPENMP
   omp_init();
#endif

	// Variable Initialization
	nz=400, nx=400, nt=2000, ns=2, ng=400 ;
	sxbeg=100, szbeg=200, gxbeg=0, gzbeg=4, jsx=100, jsz=40, jgx=1, jgz=0 ;
	dx=0.005, dz=0.005,fm=10, dt=0.0005, amp=1.0, nb=80, hfop=2, ompchunk=1 ;
	// End
	
	
	fdm=fd_init(nx, nz, dx, dz, nb, hfop, ompchunk);
	
	wlt=(float*)malloc(nt*sizeof(float));
	dobs=(float*)malloc(ns*ng*nt*sizeof(float));
	sxz=(int*)malloc(ns*sizeof(int));
	gxz=(int*)malloc(ng*sizeof(int));


	v=alloc2d(fdm->nz,fdm->nx);
	vv=alloc2d(fdm->nzpad,fdm->nxpad);
	u0=alloc2d(fdm->nzpad,fdm->nxpad);
	u1=alloc2d(fdm->nzpad,fdm->nxpad);
	lap=alloc2d(fdm->nzpad,fdm->nxpad);
	
	
	memset(dobs,0,ns*ng*nt*sizeof(float));
	printf("Reading Velocity Model\n");
	readbin(v[0],"gvel.bin");
	//printf("val is %f\n",v[0][4]);
	expand2d(v,vv,fdm);
	//writebin(vv[0],fdm->nzpad*fdm->nxpad,"gp.bin");
	memset(u0[0],0,fdm->nzpad*fdm->nxpad*sizeof(float));
	memset(u1[0],0,fdm->nzpad*fdm->nxpad*sizeof(float));
	memset(lap[0],0,fdm->nzpad*fdm->nxpad*sizeof(float));
	
	/* setup */
	if (!(sxbeg>=0 && szbeg>=0 && sxbeg+(ns-1)*jsx<nx && szbeg+(ns-1)*jsz<nz))	
	{ fprintf(stderr,"sources exceeds the computing zone!\n"); exit(1);}
	sg_init(sxz, szbeg, sxbeg, jsz, jsx, ns, nz);
	if (!(gxbeg>=0 && gzbeg>=0 && gxbeg+(ng-1)*jgx<nx && gzbeg+(ng-1)*jgz<nz))	
	{ fprintf(stderr,"geophones exceeds the computing zone!\n"); exit(1);}
	sg_init(gxz, gzbeg, gxbeg, jgz, jgx, ng, nz);
	/* source */
	ricker_wavelet(wlt,amp,fm,dt,nt);
	printf("Writing source in binary file\n");
	writebin(wlt,nt,"source.bin");
	
	
	for(is=0; is<ns; is++){
		memset(u0[0],0,fdm->nzpad*fdm->nxpad*sizeof(float));
		memset(u1[0],0,fdm->nzpad*fdm->nxpad*sizeof(float));
		start=omp_get_wtime();
		/* forward modeling */
		for(it=0; it<nt; it++){
			add_source(u1, &wlt[it], &sxz[is], fdm->nz, 1, nb, true);
			step_forward(u1, u0, vv, fdm, dt);
			ptr=u0;u0=u1;u1=ptr;
			//Boundaries
			paraxbound(u0,u1,vv,dt,fdm);
			sponge(u0,fdm);
			sponge(u1,fdm);
	
		
			//
			if(it==1000) writebin(u1[0],fdm->nxpad*fdm->nzpad,"snap.bin");
			record_seis(&dobs[is*ng*nt+it*ng], gxz, u1, ng, nz, nb);
			}
			end = omp_get_wtime();
			printf("shot %d finished: %f seconds\n", is+1,(end-start));
		
			}
			printf("Writing shots in binary file\n");
			writebin(dobs,ns*ng*nt,"dobs.bin");
			
			
			
	free(sxz); free(gxz);
	free(dobs);
	free(wlt);
	free(*v); free(v);
	free(*vv); free(vv);
	free(*u0); free(u0);
	free(*u1); free(u1);
	free(*lap); free(lap);

    exit(0);
		
		
				
		
		
	}
