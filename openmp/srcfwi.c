#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

#include "functions.c"

#ifdef _OPENMP
#include <omp.h>
#include "ompfunc.c"
#endif


int main(int argc, char *argv[])
{
	bool precon;
	int is, it;
	int nz, nx, nt, ns, ng, nb, hfop, ompchunk ;
	int sxbeg, szbeg, gxbeg, gzbeg, jsx, jsz, jgx, jgz ; /* acquisiton parameters */
	int *sxz, *gxz;
  	float dx, dz, fm, dt, amp, totaltime=0	;
	float *wlt, *dobs, *dcal, *derr, **v, **vv, **u0, **u1, **lap, **ptr=NULL, **sptr=NULL;
	float **gp0, **gp1, **gd0, **gd1, **cg, **illum ;
	float *dspertx, *dspertz, *objval, *alpha1, *alpha2, *bndr;
	float gdsx0, gdsz0, gdsx1, gdsz1;
	float betax, betaz, cgsx, cgsz, epsilx, epsilz;
	float alphax, alphaz ;
	float **usx0, **usx1, **usz0, **usz1;
	fdm2d fdm;
	int sxtmp,sztmp;
	int niter,iter;
	float obj,obj1;
	float start, end;

#ifdef _OPENMP
   omp_init();
#endif

	// Variable Initialization
	nz=400, nx=400, nt=2000, ns=1, ng=400 ;
	sxbeg=200, szbeg=150, gxbeg=0, gzbeg=4, jsx=0, jsz=0, jgx=1, jgz=0 ;
	dx=0.005, dz=0.005,fm=10, dt=0.0005, amp=1.0, nb=80, hfop=2, ompchunk=1 ;
	niter=9;
	// End
	
	
	fdm=fd_init(nx, nz, dx, dz, nb, hfop, ompchunk);
	
	wlt=(float*)malloc(nt*sizeof(float));/* source wavelet */
	dobs=(float*)malloc(ng*nt*sizeof(float));/* observed data */
	dcal=(float*)malloc(ng*sizeof(float));/* calculated data */
	derr=(float*)malloc(ns*ng*nt*sizeof(float));//residuals
	dspertx=(float*)malloc(ng*nt*sizeof(float));//source perturb x
	dspertz=(float*)malloc(ng*nt*sizeof(float));//source perturb z
	sxz=(int*)malloc(ns*sizeof(int));
	gxz=(int*)malloc(ng*sizeof(int));
	alpha1=(float*)malloc(ng*sizeof(float));
	alpha2=(float*)malloc(ng*sizeof(float));
	objval=(float*)malloc(niter*sizeof(float));/* objective/misfit function */
	bndr=(float*)malloc(nt*(4*nz+2*nx));/* saving boundary for reconstruction */

	v=alloc2d(fdm->nz,fdm->nx);
	vv=alloc2d(fdm->nzpad,fdm->nxpad);
	u0=alloc2d(fdm->nzpad,fdm->nxpad);
	u1=alloc2d(fdm->nzpad,fdm->nxpad);
	gp0=alloc2d(fdm->nzpad,fdm->nxpad);//adjoint wavefield
	gp1=alloc2d(fdm->nzpad,fdm->nxpad);
	gd0=alloc2d(fdm->nzpad,fdm->nxpad);//velocity gradient
	gd1=alloc2d(fdm->nzpad,fdm->nxpad);
	cg=alloc2d(fdm->nzpad,fdm->nxpad);//conjugate gradient
	lap=alloc2d(fdm->nzpad,fdm->nxpad);//laplacian
	illum=alloc2d(fdm->nzpad,fdm->nxpad);//illumination
	usx0=alloc2d(fdm->nzpad,fdm->nxpad);
	usx1=alloc2d(fdm->nzpad,fdm->nxpad);
	usz0=alloc2d(fdm->nzpad,fdm->nxpad);
	usz1=alloc2d(fdm->nzpad,fdm->nxpad);

	
	
	printf("Reading Shotgather Data\n");
	readbin(dobs,"shots.bin");
	
	printf("Reading Velocity Model\n");
	readbin(v[0],"gp.bin");
	expand2d(v,vv,fdm);
	writebin(vv[0],fdm->nzpad*fdm->nxpad,"vnew.bin");
	
	memset(dcal,0,ng*sizeof(float));
	memset(derr,0,ng*nt*sizeof(float));
	memset(dspertx,0,ng*nt*sizeof(float));
	memset(dspertz,0,ng*nt*sizeof(float));
	memset(objval,0,niter*sizeof(float));
	memset(lap[0],0,fdm->nzpad*fdm->nxpad*sizeof(float));
	
	gdsx0=0; gdsz0=0; gdsx1=0; gdsz1=0;
	betax=0; betaz=0; cgsx=0; cgsz=0; epsilx=0; epsilz=0;
	sxtmp=0; sztmp=0; alphax=0; alphaz=0;
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
		
		gdsx0=gdsx1;gdsz0=gdsz1;
		gdsx1=0;gdsz1=0;
		for(is=0; is<ns; is++){
			start=omp_get_wtime();
				memset(u0[0],0,fdm->nzpad*fdm->nxpad*sizeof(float));
				memset(u1[0],0,fdm->nzpad*fdm->nxpad*sizeof(float));
				memset(usx0[0],0,fdm->nzpad*fdm->nxpad*sizeof(float));
				memset(usx1[0],0,fdm->nzpad*fdm->nxpad*sizeof(float));
				memset(usz0[0],0,fdm->nzpad*fdm->nxpad*sizeof(float));
				memset(usz1[0],0,fdm->nzpad*fdm->nxpad*sizeof(float));
	
			/* forward modeling */
			for(it=0; it<nt; it++){
				add_source(u1, &wlt[it], &sxz[is], nz, 1, nb, true);
				step_forward_spertb(u1, u0, usx1, usx0, usz1, usz0, lap, vv, fdm, dt );		
				//Boundaries
				paraxbound(u0,u1,vv,dt,fdm);
				sponge(u0,fdm);
				sponge(u1,fdm);
				paraxbound(usx0,usx1,vv,dt,fdm);
				sponge(usx0,fdm);
				sponge(usx1,fdm);
				paraxbound(usz0,usx1,vv,dt,fdm);
				sponge(usz0,fdm);
				sponge(usz1,fdm);
				
				ptr=u0;u0=u1;u1=ptr;
				sptr=usx0;usx0=usx1;usx1=sptr;
				sptr=usz0;usz0=usz1;usz1=sptr;

				boundary_rw(u0,&bndr[it*(4*nz+2*nx)],false,fdm);
				//
				
				if(it==1000){ 
					printf("saving snapshot at t=%f s\n",it*dt);
					writebin(u1[0],fdm->nxpad*fdm->nzpad,"snap.bin");
					};
				record_seis(dcal, gxz, u1, ng, nz, nb);
				record_seis(&dspertx[it*ng], gxz, usx1, ng, nz, nb);
				record_seis(&dspertz[it*ng], gxz, usz1, ng, nz, nb);
				cal_residuals(dcal,&dobs[it*ng],&derr[is*ng*nt+it*ng],ng); //residuals
		
				}

				
				printf("writing perturbation wavefield x\n");
				writebin(dspertx,ng*nt,"spertx.bin");
				
				printf("writing perturbation wavefield z\n");
				writebin(dspertz,ng*nt,"spertz.bin");
				
				printf("writing residual wavefield \n");
				writebin(derr,ng*nt,"resd.bin");
				
				memset(gp0[0],0,fdm->nzpad*fdm->nxpad*sizeof(float));
				memset(gp1[0],0,fdm->nzpad*fdm->nxpad*sizeof(float));
				printf("Backpropagation begins\n");
				/* backward propagation */
				gdsx0=0;gdsz0=0;
				for(it=nt-1; it>-1; it--){
					
					/* source backpropagation */
					boundary_rw(u0,&bndr[it*(4*nz+2*nx)],true,fdm);
					ptr=u0;u0=u1;u1=ptr;
					step_forward(u0,u1,vv,fdm,dt);
					add_source(u0,&wlt[it],&sxz[is],nz,1,nb,false);
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
					/* source gradient */
					src_gradient(&gdsx1,&gdsz1,&dspertx[it*ng],&dspertz[it*ng],&derr[is*ng*nt+it*ng],gp1,&wlt[it],sxz,ng,fdm);
					
					}
				
				/* scale gradient */
				scale_src_gradient(&gdsx1,&gdsz1);	
				//printf("source x-gradient is %f\n",gdsx1);
				//printf("source z-gradient is %f\n",gdsz1);
			
				}
			//printf("source conjugate x-gradient is %f\n",cgsx);
			//printf("source conjugaet z-gradient is %f\n",cgsz);
			/* calculate the value of the objective function */
			cal_objective(&obj,derr,ns*ng*nt);
			//printf("value of objective function is %f\n",obj);
			
			/* calculate the factor beta in conjugate gradient method */
			if (iter>0) {
					src_cal_beta(&betax,gdsx0,gdsx1,cgsx);
					src_cal_beta(&betaz,gdsz0,gdsz1,cgsz);
					}
			/* compute the conjugate gradient */
			src_cal_conjgrad(gdsx1,&cgsx,betax);
			src_cal_conjgrad(gdsz1,&cgsz,betaz);
			//printf("source conjugate x-gradient is %f\n",cgsx);
			//printf("source conjugaet z-gradient is %f\n",cgsz);
			
			/* estimate epsilon */
			src_cal_epsilon(sxbeg,cgsx,&epsilx);
			src_cal_epsilon(szbeg,cgsz,&epsilz);
	
			
			/* obtain a tentative model to estimate a good stepsize alpha */
			cal_sptmp(&sxtmp,sxbeg,cgsx,epsilx);
			cal_sptmp(&sztmp,szbeg,cgsz,epsilz);
			sg_init(sxz,sztmp,sxtmp,jsz,jsx,ns,fdm->nz);
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
					sum_alpha12(alpha1,alpha2,dcal,&dobs[it*ng],&derr[is*ng*nt+it*ng],ng);				
					}
				}
			/* find a good stepsize alpha */
			cal_alpha(&alphax,&alphaz,alpha1,alpha2,epsilx,epsilz,ng);
			/* update source position */
			update_src(&sxbeg,cgsx,alphax);
			update_src(&szbeg,cgsz,alphaz);

			end = omp_get_wtime();
			/* compute the normalized objective function */
			if(iter==0) 	{obj1=obj; objval[iter]=1.0;}
			else	objval[iter]=obj/obj1;
			printf("------------------------------------------------------\n");
			printf("  sx=%d  sz=%d  obj=%f\n",sxbeg,szbeg,obj);
			printf("  betax=%f betaz=%f epsilx=%f epsilz=%f  alphax=%f alphaz=%f\n", betax, betaz,epsilx, epsilz, alphax, alphaz);
			printf("  iteration %d finished: %f (s)\n",iter+1, (end-start));
			printf("------------------------------------------------------\n");
		}
	
	}
