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
	bool precon=true;
	int is, it;
	int nz, nx, nt, ns, ng, nb, hfop, ompchunk ;
	int sxbeg, szbeg, gxbeg, gzbeg, jsx, jsz, jgx, jgz ; /* acquisiton parameters */
	int *sxz, *gxz;
  	float dx, dz, fm, dt, amp, totaltime=0	;
	float *wlt, *dobs, *dcal, *derr, **v, **vv, **vtmp, **u0, **u1, **lap, **ptr=NULL, **sptr=NULL;
	float **gp0, **gp1, **gd0, **gd1, **cg, **illum ;
	float *objval, *alpha1, *alpha2, *bndr;
	float epsil,beta,alpha;
	float **spgd0, **spgd1, **spcg, **sptmp;
	float spepsil,spbeta,spalpha;
	fdm2d fdm;
	int niter,iter;
	float obj,obj1;
	float start, end;

#ifdef _OPENMP
   omp_init();
#endif

	// Variable Initialization
	nz=400, nx=400, nt=2000, ns=2, ng=400 ;
	sxbeg=100, szbeg=200, gxbeg=0, gzbeg=4, jsx=100, jsz=40, jgx=1, jgz=0 ;
	dx=0.005, dz=0.005,fm=10, dt=0.0005, amp=1.0, nb=80, hfop=2, ompchunk=1 ;
	niter=17,beta=0.0,epsil=0.0,alpha=0.0;
	// End
	
	
	fdm=fd_init(nx, nz, dx, dz, nb, hfop, ompchunk);
	
	wlt=(float*)malloc(nt*sizeof(float));/* source wavelet */
	dobs=(float*)malloc(ns*ng*nt*sizeof(float));/* observed data */
	dcal=(float*)malloc(ng*sizeof(float));/* calculated data */
	derr=(float*)malloc(ns*ng*nt*sizeof(float));//residuals
	
	sxz=(int*)malloc(ns*sizeof(int));
	gxz=(int*)malloc(ng*sizeof(int));
	alpha1=(float*)malloc(ng*sizeof(float));
	alpha2=(float*)malloc(ng*sizeof(float));
	objval=(float*)malloc(niter*sizeof(float));/* objective/misfit function */
	bndr=(float*)malloc(nt*(4*nz+2*nx)*sizeof(float));/* saving boundary for reconstruction */

	v=alloc2d(fdm->nz,fdm->nx);
	vtmp=alloc2d(fdm->nz,fdm->nx);
	sptmp=alloc2d(fdm->nz,fdm->nx);
	vv=alloc2d(fdm->nzpad,fdm->nxpad);
	u0=alloc2d(fdm->nzpad,fdm->nxpad);
	u1=alloc2d(fdm->nzpad,fdm->nxpad);
	gp0=alloc2d(fdm->nzpad,fdm->nxpad);//adjoint wavefield
	gp1=alloc2d(fdm->nzpad,fdm->nxpad);
	gd0=alloc2d(fdm->nz,fdm->nx);//velocity gradient
	gd1=alloc2d(fdm->nz,fdm->nx);
	cg=alloc2d(fdm->nz,fdm->nx);//conjugate gradient
	spcg=alloc2d(fdm->nz,fdm->nx);//source conjugate gradient
	lap=alloc2d(fdm->nzpad,fdm->nxpad);//laplacian
	illum=alloc2d(fdm->nzpad,fdm->nxpad);//illumination
	spgd0=alloc2d(fdm->nz,fdm->nx);//source gradient
	spgd1=alloc2d(fdm->nz,fdm->nx);

		
	printf("Reading Shotgather Data\n");
	readbin(dobs,"dobs.bin");
	
	printf("Reading Velocity Model\n");
	readbin(v[0],"vel.bin");
	expand2d(v,vv,fdm);
	//writebin(vv[0],fdm->nzpad*fdm->nxpad,"vnew.bin");
	
	memset(dcal,0,ng*sizeof(float));
	memset(derr,0,ns*ng*nt*sizeof(float));
	memset(objval,0,niter*sizeof(float));
	memset(lap[0],0,fdm->nzpad*fdm->nxpad*sizeof(float));
	memset(illum[0],0,fdm->nzpad*fdm->nxpad*sizeof(float));
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
	
	for(iter=0; iter<niter; iter++){
		memcpy(gd1[0],gd0[0],fdm->nzpad*fdm->nxpad);
		memset(illum[0],0,fdm->nzpad*fdm->nxpad*sizeof(float));
		memset(gd0[0],0,fdm->nzpad*fdm->nxpad*sizeof(float));
		for(is=0; is<ns; is++){
			start=omp_get_wtime();
				memset(u0[0],0,fdm->nzpad*fdm->nxpad*sizeof(float));
				memset(u1[0],0,fdm->nzpad*fdm->nxpad*sizeof(float));
			/* forward modeling */
			for(it=0; it<nt; it++){
				add_source(u1, &wlt[it], &sxz[is], fdm->nz, 1, nb, true);
				step_forward(u1, u0, vv, fdm, dt);	
				ptr=u0;u0=u1;u1=ptr;
				//Boundaries
				paraxbound(u0,u1,vv,dt,fdm);
				sponge(u0,fdm);
				sponge(u1,fdm);				
				/*saving the boundaries*/
				boundary_rw(u0,&bndr[it*(4*nz+2*nx)],false,fdm);
				/*snapshot*/				
				if(it==500){ 
					printf("saving snapshot at t=%f s\n",it*dt);
					writebin(u1[0],fdm->nxpad*fdm->nzpad,"snap.bin");
					};
				
				record_seis(dcal, gxz, u1, ng, nz, nb);
				cal_residuals(dcal,&dobs[is*ng*nt+it*ng],&derr[is*ng*nt+it*ng],ng); //residuals
					
					}
				printf("writing residual wavefield \n");
				writebin(derr,ns*ng*nt,"resd.bin");
				writebin(u1[0],fdm->nzpad*fdm->nxpad,"wav.bin");
				memset(gp0[0],0,fdm->nzpad*fdm->nxpad*sizeof(float));
				memset(gp1[0],0,fdm->nzpad*fdm->nxpad*sizeof(float));
				printf("Backpropagation begins\n");
				/* backward propagation */
		
				for(it=nt-1; it>-1; it--){
					
					/* source backpropagation */
					
					boundary_rw(u0,&bndr[it*(4*nz+2*nx)],true,fdm);
					ptr=u0;u0=u1;u1=ptr;
					step_backward(illum,lap,u1,u0,vv,fdm,dt);
					add_source(u1,&wlt[it],&sxz[is],nz,1,nb,false);
					if(it==500){ 
						printf("saving snapshot at t=%f s\n",it*dt);
						writebin(u1[0],fdm->nxpad*fdm->nzpad,"rsnap.bin");
					};
					/* extrapolate residual wavefield */
					add_source(gp1, &derr[is*ng*nt+it*ng],gxz,nz,ng,nb,true);
					step_forward(gp1,gp0,vv,fdm,dt);
					ptr=gp0;gp0=gp1;gp1=ptr;					
					paraxbound(gp0,gp1,vv,dt,fdm);
					sponge(gp0,fdm);
					sponge(gp1,fdm);
					if(it==500){ 
						printf("saving snapshot at t=%f s\n",it*dt);
						writebin(gp1[0],fdm->nxpad*fdm->nzpad,"gsnap.bin");
					};
					/* velocity gradient */
					cal_gradient(gd0,lap,gp1,fdm);
					src_spat_gradient(spgd0,gp1,&wlt[it],fdm);
					}							
				}

			/* calculate the value of the objective function */
			cal_objective(&obj,derr,ns*ng*nt);
			//printf("value of objective function is %f\n",obj);
			
			/* scale the gradient with preconditioning */
			scale_gradient(gd0,v,illum,fdm,true);
			writebin(gd0[0],fdm->nx*fdm->nz,"grad.bin");
			writebin(spgd0[0],fdm->nx*fdm->nz,"spgrad.bin");
			/* calculate the factor beta in conjugate gradient method */
			if (iter>0) { cal_beta(&beta,gd1,gd0,cg,fdm);}
			/* compute the conjugate gradient */
			cal_conjgrad(gd0,cg,beta,fdm);	
				
			/* estimate epsilon */
			cal_epsilon(v,cg,&epsil,fdm);
		
			/* obtain a tentative model to estimate a good stepsize alpha */
			cal_vtmp(vtmp,v,cg,epsil,fdm);
			
			readbin(dobs,"dobs.bin");
			expand2d(vtmp,vv,fdm);
			for(is=0; is<ns; is++){
				memset(u0[0],0,fdm->nzpad*fdm->nxpad*sizeof(float));
				memset(u1[0],0,fdm->nzpad*fdm->nxpad*sizeof(float));
			    /* forward modeling */
			    for(it=0; it<nt; it++){
					add_source(u1, &wlt[it], &sxz[is], fdm->nz, 1, nb, true);
					step_forward(u1, u0, vv, fdm, dt);
					ptr=u0;u0=u1;u1=ptr;
					//Boundaries
					paraxbound(u0,u1,vv,dt,fdm);
					sponge(u0,fdm);
					sponge(u1,fdm);
					record_seis(dcal, gxz, u1, ng, nz, nb);
					/* compute the numerator and the denominator of alpha */
					sum_alpha12(alpha1,alpha2,dcal,&dobs[is*ng*nt+it*ng],&derr[is*ng*nt+it*ng],ng);
					}
				}
			
			/* find a good stepsize alpha */
			cal_alpha(&alpha,alpha1,alpha2,epsil,ng);
			/* update velocity model */
			update_vel(v,cg,alpha,fdm);
			writebin(v[0],fdm->nx*fdm->nz,"upvel.bin");
			expand2d(v,vv,fdm);
			end = omp_get_wtime();
			/* compute the normalized objective function */
			if(iter==0) 	{obj1=obj; objval[iter]=1.0;}
			else	objval[iter]=obj/obj1;
			printf("------------------------------------------------------\n");
			printf("  obj=%f  beta=%f epsil=%f alpha=%f \n",obj,beta,epsil,alpha);
			printf("  iteration %d finished: %f (s)\n",iter+1, (end-start));
			printf("------------------------------------------------------\n");
	
		}
	}
