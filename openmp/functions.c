#include <stdlib.h>
#include <float.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define PI 3.14159265358979323846
#define EPS FLT_EPSILON 

#define MAX(x, y) (((x) > (y)) ? (x) : (y))

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#define A1  0.66666666666666666666 /*  2/3  */  
#define A2 -0.08333333333333333333 /* -1/12 */



/*4th order in space*/
#define DX(a,ix,iz,s) (A2*(a[ix+2][iz] - a[ix-2][iz]) +  \
                       A1*(a[ix+1][iz] - a[ix-1][iz])  )*s
                       
#define DZ(a,ix,iz,s) (A2*(a[ix][iz+2] - a[ix][iz-2]) +  \
                       A1*(a[ix][iz+1] - a[ix][iz-1])  )*s




/*----------------Data types and intialization----------------*/



typedef struct fdm2 {
	
	int nb;
    int nz,nzpad;
    int nx,nxpad;
    float dz,dx;
    bool free;
    int hfop,ompchunk;
    
	} *fdm2d ;


fdm2d fd_init( int nx, int nz, float dx, float dz, int nb, int hfop, int ompchunk)
/*< Initialize the fdm structure >*/
{	fdm2d fdm ;
	fdm = (fdm2d)malloc(sizeof(*fdm));
	
	fdm->nb=nb;
	
	fdm->nz=nz;
	fdm->nx=nx;
	
	fdm->dz=dz;
	fdm->dx=dx;
			
	fdm->nzpad=nz+fdm->nb;
	fdm->nxpad=nx+2*fdm->nb;
	

	fdm->hfop = hfop;
	fdm->ompchunk = ompchunk ;
	
	return fdm ;
	}


/*<-----------------General Utilities-------------------------->*/

float **alloc2d(int nz, int nx)
/*< Allocate 2D array in a contiguous memory >*/
{ 	
	int i;
	float **mat = (float **)malloc(nx*sizeof(float *));
	mat[0]= (float *)malloc(nz*nx*sizeof(float *));
	for (i=1;i<nx;i++){
		mat[i]=mat[0]+i*nz;
		}
	return mat;
	}
	
float ***alloc3d(int nz, int nx, int nt)
/*< Allocate 3D array in a contiguous memory >*/
{
	int i;
	float ***cell= (float ***)malloc(nt*sizeof(float*));
    cell[0] = alloc2d(nz,nx*nt);
    for (i=1; i< nt; i++) {
	cell[i] = cell[0]+i*nz;
    }
    return cell;	
	}
	
void readbin(float *data,char *name)
/*< Read Binary File >*/
{
	FILE *file;
	long fsize;
	size_t result;
	file=fopen(name,"rb");
	if (file==NULL) {fputs ("File error \n",stderr); exit(1);}
	// obtain file size
	fseek(file , 0 , SEEK_END);
	fsize = ftell(file);
	rewind(file);
	//Memory check
	if (data==NULL) { fputs("Memory error \n",stderr); exit(2);}
	// copy file into pointers
	result = fread(data,1,fsize,file);
	if (result != fsize){fputs("Reading error \n",stderr); exit(2);}
	printf("Loading file complete \n");
	fclose(file);	
	
	}
	
	
void writebin(float *data,int size, char *name)
/*< Write Binary File  >*/
{
	FILE *file;
	file=fopen(name,"wb");
	fwrite(data,sizeof(float),size,file);
	printf("Writing complete \n");
	fclose(file);
	}	
	
	
///*	
//float max(float a, float b)
///*< Maximum >*/
//{
	//a=(a>b)?a:b;
	//return a ;
	//}

//float min(float a, float b)
///*< Minimum >*/
//{
	//a=(a<b)?a:b ;
	//return a ;
	//}	
	
//float **gradx(float **u, fdm2d fdm)
///*gradient in x-direction */
//{
	//int ix,iz;
	//float **gdx=alloc2d(fdm->nzpad,fdm->nxpad);
	//float c11,c12;
	//float idx=1.0/(fdm->dx);
	//c11=1.0/12.0,c12=2.0/3.0;
		
	//for(ix=0;ix<fdm->nxpad;ix++){
		//for(iz=0;ix<fdm->nzpad;iz++){
			//gdx[ix][iz]=(c11*(u[ix-2][iz]-u[ix+2][iz])+c12*(u[ix+1][iz]-u[ix-1][iz]))*idx;
			//}
		//}
	//return gdx;	
	//}

//float **gradz(float **u, fdm2d fdm)
///*gradient in x-direction */
//{
	//int ix,iz;
	//float **gdz=alloc2d(fdm->nzpad,fdm->nxpad);
	//float c11,c12;
	//float idz=1.0/(fdm->dz);
	//c11=1.0/12.0,c12=2.0/3.0;
		
	//for(ix=0;ix<fdm->nxpad;ix++){
		//for(iz=0;ix<fdm->nzpad;iz++){
			//gdz[ix][iz]=(c11*(u[ix][iz-2]-u[ix][iz+2])+c12*(u[ix][iz+1]-u[ix-1][iz-1]))*idz;
			//}
		//}
	//return gdz;	
	//}
		
/*<----------FDM Utilities------------->*/	


void expand2d(float** a, float** b, fdm2d fdm)
/*< expand domain of 'a' to 'b': source(a)-->destination(b) >*/
{
    int iz,ix;

#ifdef _OPENMP
#pragma omp parallel for default(none)	\
	private(ix,iz)			\
	shared(b,a,fdm)
#endif
    for(ix=0;ix<fdm->nx;ix++) {
		for(iz=0;iz<fdm->nz;iz++) {
			b[fdm->nb+ix][iz] = a[ix][iz];
		}
    }

    for(ix=0; ix<fdm->nxpad;ix++) {
		for(iz=0; iz<fdm->nb;iz++) {
			b[ix][fdm->nzpad-iz-1] = b[ix][fdm->nzpad-fdm->nb-1];/* bottom*/
			}
		}
    for(ix=0; ix<fdm->nb;    ix++) {
		for(iz=0; iz<fdm->nzpad; iz++) {
			b[ix][iz] = b[fdm->nb][iz];/* left */
			b[fdm->nxpad-ix-1 ][iz] = b[fdm->nxpad-fdm->nb-1][iz];/* right */
		}
    }
}
	

void window2d(float **a, float **b, fdm2d fdm)
/* Window the domain 'b' to 'a': source(b)-->destination(a)  */
{	
	int ix,iz;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
	private(ix,iz) 					   \
	shared(a,b,fdm)
#endif	
	for (ix=0;ix<fdm->nx;ix++) {
		for (iz=0;iz<fdm->nz;iz++) {
			a[ix][iz] = b[fdm->nb+ix][iz];
		}
	}
	}


void matrix_transpose(float *matrix, float *trans, int n1, int n2)
/*< matrix transpose: matrix tansposed to be trans >*/
{
	int i1, i2;

	for(i2=0; i2<n2; i2++){
		for(i1=0; i1<n1; i1++){
			trans[i2+n2*i1]=matrix[i1+n1*i2];
		}
	}	
	}

void ricker_wavelet(float *wlt, float amp, float fm, float dt, int nt)
/*< generate ricker wavelet with time delay >*/
{
	int it;
	
    for(it=0;it<nt;it++){
		float tmp = PI*fm*(it*dt-1.0/fm);
	    tmp *=tmp;
		wlt[it]=amp*(1.0-2.0*tmp)*expf(-tmp);
		}
	}
	
void add_source(float **u, float *source, int *sxz, int nz, int ns, int nb, bool add)
/*< add/subtract seismic sources >*/
{
	int is, sx, sz;
	if(add){
		for(is=0;is<ns; is++){
			sx=sxz[is]/nz + nb;
			sz=sxz[is]%nz;
			u[sx][sz]+=source[is];
		}
	}else{
		for(is=0;is<ns; is++){
			sx=sxz[is]/nz + nb;
			sz=sxz[is]%nz;
			u[sx][sz]-=source[is];
		}
	}
}

void record_seis(float *seis_it, int *gxz, float **u, int ng, int nz, int nb)
/*< record seismogram at time it into a vector length of ng >*/
{
	int ig, gx, gz;
	for(ig=0;ig<ng; ig++){
		gx=gxz[ig]/nz + nb;
		gz=gxz[ig]%nz;
		seis_it[ig]=u[gx][gz];
		}
}

void sg_init(int *sxz, int szbeg, int sxbeg, int jsz, int jsx, int ns, int nz)
/*< shot/geophone position initialize >*/
{
	int is, sz, sx;
	for(is=0; is<ns; is++){
		sz=szbeg+is*jsz;
		sx=sxbeg+is*jsx;
		sxz[is]=sz+nz*sx;
		}
}


/*----------------------Acoustic------------------------------*/

void step_forward(float **up,  float **ub, float **vv, fdm2d fdm, float dt)
/*< forward modeling step >*/
{
	int ix,iz;
	float idx,idz;
	float v,lap;
	idz= 1.0/(fdm->dz*fdm->dz);
	idx= 1.0/(fdm->dx*fdm->dx);

	/* 4th order coefficients */
	float c0,c1,c2;
	c0= -5.0/2.0,c1=4.0/3.0,c2=-1.0/12.0;

#ifdef _OPENMP
#pragma omp parallel for default(none) 		\
    private(ix,iz,v,lap)	schedule(dynamic,fdm->ompchunk)				\
    shared(up,ub,vv,fdm,c0,c1,c2,idx,idz,dt)  
#endif	
	for(ix=2;ix<fdm->nxpad-2;ix++){
		for(iz=2;iz<fdm->nzpad-2;iz++){
			v=(vv[ix][iz]*vv[ix][iz])*(dt*dt);
	
			lap=c0*up[ix][iz]*(idx+idz)+c1*(up[ix-1][iz]+up[ix+1][iz])*idx +
				c2*(up[ix-2][iz]+up[ix+2][iz])*idx + c1*(up[ix][iz-1]+up[ix][iz+1])*idz +
				c2*(up[ix][iz-2]+up[ix][iz+2])*idz;
			ub[ix][iz]=2*up[ix][iz]-ub[ix][iz]+v*lap;
			
			}
		}

	}
	

	
void step_forward_spertb(float **up, float **ub, float **usopx, float **usobx, float **usopz, float **usobz , float **lap, float **vv, fdm2d fdm, float dt)
/*< forward modeling with source perturbation >*/
{
	int ix,iz;
	float idx,idz;
	float v;

	idz= 1.0/(fdm->dz*fdm->dz);
	idx= 1.0/(fdm->dx*fdm->dx);

	/* 4th order coefficients */
	float c0,c1,c2;
	c0= -5.0/2.0,c1=4.0/3.0,c2=-1.0/12.0;

#ifdef _OPENMP
#pragma omp parallel for default(none) 		\
    private(ix,iz,v)	schedule(dynamic,fdm->ompchunk)				\
    shared(up,ub,lap,vv,fdm,c0,c1,c2,idx,idz,dt)  
#endif	
	for(ix=2;ix<fdm->nxpad-2;ix++){
		for(iz=2;iz<fdm->nzpad-2;iz++){
			v=(vv[ix][iz]*vv[ix][iz])*(dt*dt);
			lap[ix][iz]=c0*up[ix][iz]*(idx+idz)+c1*(up[ix-1][iz]+up[ix+1][iz])*idx +
						c2*(up[ix-2][iz]+up[ix+2][iz])*idx + c1*(up[ix][iz-1]+up[ix][iz+1])*idz +
						c2*(up[ix][iz-2]+up[ix][iz+2])*idz;
			ub[ix][iz]=2*up[ix][iz]-ub[ix][iz]+v*lap[ix][iz];
		
		}
	}
	
/*<----------source perturbation--------->*/	

	float srcx,srcz,isz,isx,lpx,lpz;
	isx=1.0/fdm->dx;
	isz=1.0/fdm->dz;
#ifdef _OPENMP
#pragma omp parallel for 	\
    private(ix,iz,v,srcx,srcz,lpx,lpz)	schedule(dynamic,fdm->ompchunk)		\
    shared(usopx,usobx,usopz,usobz,vv,lap,fdm,c0,c1,c2,idx,idz,dt,isx)  
#endif	
	for(ix=2;ix<fdm->nxpad-2;ix++){
		for(iz=2;iz<fdm->nzpad-2;iz++){
			/*-------------source term ---------------------------*/
			/* 		(dw/dx)/w(lap u) and (dw/dz)/w(lap u)     */
			/*----------------------------------------------------*/
			srcx=-(2.0*DX(vv,ix,iz,isx)/vv[ix][iz])*lap[ix][iz];
			srcz=-(2.0*DZ(vv,ix,iz,isz)/vv[ix][iz])*lap[ix][iz];
			/*---------------------------------------*/
			v=(vv[ix][iz]*vv[ix][iz])*(dt*dt);
			lpx=c0*usopx[ix][iz]*(idx+idz)+c1*(usopx[ix-1][iz]+usopx[ix+1][iz])*idx +
				c2*(usopx[ix-2][iz]+usopx[ix+2][iz])*idx + c1*(usopx[ix][iz-1]+usopx[ix][iz+1])*idz +
				c2*(usopx[ix][iz-2]+usopx[ix][iz+2])*idz;
			lpz=c0*usopz[ix][iz]*(idx+idz)+c1*(usopz[ix-1][iz]+usopz[ix+1][iz])*idx +
				c2*(usopz[ix-2][iz]+usopz[ix+2][iz])*idx + c1*(usopz[ix][iz-1]+usopz[ix][iz+1])*idz +
				c2*(usopz[ix][iz-2]+usopz[ix][iz+2])*idz;
			usobx[ix][iz]=2*usopx[ix][iz]-usobx[ix][iz]+v*(lpx+srcx);
			usobz[ix][iz]=2*usopz[ix][iz]-usobz[ix][iz]+v*(lpz+srcz);
			}
		}	
	}
	
void step_backward(float **illum, float **lap, float **up, float **ub, float **vv, fdm2d fdm, float dt)
/*< step backward >*/
{
	int ix,iz;
	float idx,idz;
	float v;
	idz= 1.0/(fdm->dz*fdm->dz);
	idx= 1.0/(fdm->dx*fdm->dx);

	/* 4th order coefficients */
	float c0,c1,c2;
	c0= -5.0/2.0,c1=4.0/3.0,c2=-1.0/12.0;

#ifdef _OPENMP
#pragma omp parallel for default(none) 		\
    private(ix,iz,v)	schedule(dynamic,fdm->ompchunk)				\
    shared(up,ub,vv,lap,illum,fdm,c0,c1,c2,idx,idz,dt)  
#endif	
	for(ix=2;ix<fdm->nxpad-2;ix++){
		for(iz=2;iz<fdm->nzpad-2;iz++){
			v=(vv[ix][iz]*vv[ix][iz])*(dt*dt);
	
			lap[ix][iz]=c0*up[ix][iz]*(idx+idz)+c1*(up[ix-1][iz]+up[ix+1][iz])*idx +
						c2*(up[ix-2][iz]+up[ix+2][iz])*idx + c1*(up[ix][iz-1]+up[ix][iz+1])*idz +
						c2*(up[ix][iz-2]+up[ix][iz+2])*idz;
			ub[ix][iz]=2*up[ix][iz]-ub[ix][iz]+v*lap[ix][iz];
			illum[ix][iz]+=up[ix][iz]*up[ix][iz];
			}
		}
	
	
	
	}
	
	
/*<----------------------Elastic ----------------------->*/

/*< 8th order in space >*/

void disp2strain(float **ubx, float **ubz, float **txx, float **tzx, float **tzz, fdm2d fdm)
/*< convert displacement to strain >*/
{
	int ix,iz;
	int idx = 1.0/(fdm->dx) ;
	int idz = 1.0/(fdm->dz) ;
	
	/*< FD operator coeffcients >*/
	float C1=0.8, C2=-0.2 , C3=0.038095, C4=-0.003571 ; 	

#ifdef _OPENMP
#pragma omp parallel for	    \
    schedule(dynamic,fdm->ompchunk)		\
    private(iz,ix)				\
    shared(fdm,tzz,tzx,txx,ubz,ubx,idz,idx)
#endif	
	for(ix=fdm->hfop;ix<fdm->nxpad-fdm->hfop; ix++) {
		for(iz=fdm->hfop;iz<fdm->nzpad-fdm->hfop; iz++) {
		
			txx[ix][iz] = (C4*(ubx[ix+4][iz]-ubx[ix-4][iz])+C3*(ubx[ix+3][iz]-ubx[ix-3][iz])+ 
						  C2*(ubx[ix+2][iz]-ubx[ix-2][iz])+C1*(ubx[ix+1][iz]-ubx[ix-1][iz]))*idx ;
					 
			tzz[ix][iz] = (C4*(ubz[ix][iz+4]-ubz[ix][iz-4])+C3*(ubz[ix][iz+3]-ubz[ix][iz-3]) + 
						  C2*(ubz[ix][iz+2]-ubz[ix][iz-2]) + C1*(ubz[ix][iz+1] -ubz[ix][iz-1]))*idz ;
					 
			tzx[ix][iz] = (C4*(ubz[ix+4][iz]-ubz[ix-4][iz])+C3*(ubz[ix+3][iz] - ubz[ix-3][iz]) +	
						   C2*(ubz[ix+2][iz]-ubz[ix-2][iz])+C1*(ubz[ix+1][iz] - ubz[ix-1][iz]))*idx + 
						  (C4*(ubx[ix][iz+4]-ubx[ix][iz-4])+ C3*(ubx[ix][iz+3]-ubx[ix][iz-3]) +	
						   C2*(ubx[ix][iz+2]-ubx[ix][iz-2]) + C1*(ubx[ix][iz+1]-ubx[ix][iz-1]))*idz ;


				}
			}			
	
	}
	
void str2acc_fwd(float **txx, float **tzx, float **tzz, float **rho, float **vp, float **vs, fdm2d fdm, float dt,
				float **upx, float **upz, float **ufx, float **ufz,float **ubx, float **ubz, float **uax, float **uaz)
/*< strain to acceleration and forward time >*/
{
	int ix,iz;
	int idx = 1.0/(fdm->dx) ;
	int idz = 1.0/(fdm->dz) ;
		
	/*< FD operator coeffcients >*/
	float C1=0.8, C2=-0.2 , C3=0.038095, C4=-0.003571 ; 	
	
	float sxx,szx,szz;
	
	/*< Tensor components >*/
	float c11,c13,c55,c33;
	
#ifdef _OPENMP
#pragma omp parallel for	   			 				\
    schedule(dynamic,fdm->ompchunk)		 				\
    private(iz,ix,szz,szx,sxx,c11,c55,c33,c13)			\
    shared(fdm,tzz,tzx,txx,vp,vs,rho)
#endif
	for    (ix=0; ix<fdm->nxpad; ix++) {
	    for(iz=0; iz<fdm->nzpad; iz++) {
			c11=rho[ix][iz]*vp[ix][iz]*vp[ix][iz];
			c55=rho[ix][iz]*vs[ix][iz]*vs[ix][iz];
			c33=c11;
			c13=c11-2*c55;
			
			sxx = c11*txx[ix][iz] + c13*tzz[ix][iz];			
			szz = c13*txx[ix][iz] + c33*tzz[ix][iz]; 	
			szx = c55*tzx[ix][iz];

			txx[ix][iz] = sxx;
			tzz[ix][iz] = szz;
			tzx[ix][iz] = szx;
			}
		}
	
#ifdef _OPENMP
#pragma omp parallel for			\
    schedule(dynamic,fdm->ompchunk)		\
    private(iz,ix)				\
    shared(fdm,tzz,tzx,txx,uaz,uax,idz,idx)
#endif
	for(ix=fdm->hfop; ix<fdm->nxpad-fdm->hfop; ix++){
		for(iz=fdm->hfop; iz<fdm->nzpad-fdm->hfop; iz++){
			
		uax[ix][iz] =(C4*(txx[ix+4][iz]-txx[ix-4][iz])+C3*(txx[ix+3][iz]-txx[ix-3][iz]) +	
					  C2*(txx[ix+2][iz]-txx[ix-2][iz]) + C1*(txx[ix+1][iz]-txx[ix-1][iz]))*idx + 
					 (C4*(tzx[ix][iz+4]-tzx[ix][iz-4])+C3*(tzx[ix][iz+3]-tzx[ix][iz-3]) +	
					 C2*(tzx[ix][iz+2]-tzx[ix][iz-2]) +C1*(tzx[ix][iz+1]-tzx[ix][iz-1]))*idz ;
					 
		uaz[ix][iz] = (C4*(tzx[ix+4][iz]-tzx[ix-4][iz]) + C3*(tzx[ix+3][iz]-tzx[ix-3][iz]) +	
					   C2*(tzx[ix+2][iz]-tzx[ix-2][iz]) + C1*(tzx[ix+1][iz]-tzx[ix-1][iz]))*idx +	
					  (C4*(tzz[ix][iz+4]-tzz[ix][iz-4]) + C3*(tzz[ix][iz+3]-tzz[ix][iz-3]) +		
					   C2*(tzz[ix][iz+2]-tzz[ix][iz-2]) + C1*(tzz[ix][iz+1]-tzz[ix][iz-1]))*idz;
		
		rho[ix][iz] = (dt*dt)/rho[ix][iz];	
		
			}
		}
		
		
#ifdef _OPENMP
#pragma omp parallel for				\
    schedule(dynamic,fdm->ompchunk)			\
    private(iz,ix)					\
    shared(fdm,ubz,ubx,upz,upx,ufz,ufx,uaz,uax,rho)
#endif
	for    (ix=0; ix<fdm->nxpad; ix++) {
	    for(iz=0; iz<fdm->nzpad; iz++) {
		
		ufz[ix][iz] = 2*ubz[ix][iz]-upz[ix][iz]+uaz[ix][iz]*rho[ix][iz]; 
		ufx[ix][iz] = 2*ubx[ix][iz]-upx[ix][iz]+uax[ix][iz]*rho[ix][iz]; 
			
			}
		}

	
	}
	
		



/*<-------------Boundaries----------------------------->*/	
void sponge(float **u, fdm2d fdm)
/*< apply absorbing boundary condition >*/
{
	int ix,iz,ib,ibx,ibz;
	float w;
		
#ifdef _OPENMP
#pragma omp parallel for		\
    private(ib,iz,ix,ibz,ibx,w)		\
    shared(u,fdm)
#endif
    for(ib=0; ib<fdm->nb; ib++) {
		float tmp=ib/(sqrt(2.0)*4*fdm->nb);;
		w=expf(-tmp*tmp);
		ibz = fdm->nzpad-ib-1;
		for(ix=0; ix<fdm->nxpad; ix++) {
			u[ix][ibz] *= w; /* bottom sponge */
			}

		ibx = fdm->nxpad-ib-1;
		for(iz=0; iz<fdm->nzpad; iz++) {
			u[ib ][iz] *= w; /*   left sponge */
			u[ibx][iz] *= w; /*  right sponge */
			}
    }
}

void paraxbound(float **up, float **ub, float **vv, float dt, fdm2d fdm)
/*< Paraxial Boundary Conditions : Clayton and Enquist >*/
{
	int iz,ix,iop;
    float d,w;
    
#ifdef _OPENMP
#pragma omp parallel for		\
    schedule(dynamic)			\
    private(iz,ix,iop,d,w)			\
    shared(fdm,ub,up)
#endif
	for(ix=0;ix<fdm->nxpad;ix++) {
	    for(iop=0;iop<fdm->hfop;iop++) {
		/* bottom BC */
		iz=fdm->nzpad-fdm->hfop+iop-1;
		d = vv[ix][fdm->nzpad-fdm->hfop-1]*dt/fdm->dz; 
		w = (1-d)/(1+d);
		ub[ix][iz]=up[ix][iz-1]+(up[ix][iz]-ub[ix][iz-1])*w;
	    }
	}

#ifdef _OPENMP
#pragma omp parallel for		\
    schedule(dynamic)			\
    private(iz,ix,iop,d,w)			\
    shared(fdm,ub,up)
#endif
	for(iz=0;iz<fdm->nzpad;iz++){
		for(iop=0;iop<fdm->hfop;iop++) {

		    /* left BC */
		    ix=fdm->hfop-iop;
		    d=vv[fdm->hfop][iz]*dt/fdm->dx; 
		    w=(1-d)/(1+d);
		    ub[ix][iz]=up[ix+1][iz]+(up[ix][iz]-ub[ix+1][iz])*w;

		    /* right BC */
		    ix= fdm->nxpad-fdm->hfop+iop-1;
		    d = vv[fdm->nxpad-fdm->hfop-1][iz]*dt/fdm->dx; 
		    w = (1-d)/(1+d);
		    ub[ix][iz]=up[ix-1][iz]+(up[ix][iz]-ub[ix-1][iz])*w;
			}
	    }
	
	}

/*----------------FWI routines--------------------*/


void cal_residuals(float *dcal, float *dobs, float *derr, int ng)
/*< calculate residual wavefield at the receiver positions
   dcal: d_{cal}
   dobs: d_{obs}
   derr: d_{err}=d_{cal}-d_{obs} >*/
{
	int ig;
	for(ig=0;ig<ng;ig++){
			derr[ig]=dcal[ig]-dobs[ig];
		}	
	}
	
float cal_objective(float *obj,float *err, int ng)
/*< calculate the L2 norm objective function >*/
{
	int ig;
	float result=0.0;
#ifdef _OPENMP
#pragma omp parallel for shared(err)  	\
		private(ig) reduction(+:result) 	
#endif	
	for(ig=0;ig<ng;ig++){
		result+=(err[ig]*err[ig]);
		}
	*obj=result ;
	
	}
	
void cal_gradient(float **gd, float **lap,float **gp, fdm2d fdm)
/*< calculate gradient  >*/
{
	int ix,iz;
/* Here, the second derivative of u has been replaced with laplace according to wave equation:	second_derivative{u}=v^2 lap{u}	*/

#ifdef _OPENMP
#pragma omp parallel for private(iz) \
		shared(gp,lap,gd)
#endif
	for(ix=0;ix<fdm->nx;ix++){
		for(iz=0;iz<fdm->nz;iz++){
					gd[ix][iz]+=gp[ix+fdm->nb][iz]*lap[ix+fdm->nb][iz] ;
						
			}
		}
	}
	
void scale_gradient(float **gd, float **v, float **illum, fdm2d fdm, bool precon)
/*< scale gradient >*/
{
	int ix,iz;
	for(ix=0;ix<fdm->nx;ix++){
		for(iz=0;iz<fdm->nz;iz++){
			float a=v[ix][iz];
			if (precon) a*=sqrtf(illum[ix+fdm->nb][iz]+EPS);/*precondition with residual wavefield illumination*/
			gd[ix][iz]*=2.0/a ;
						
			}
		}
	}
	
void src_gradient(float *gdsx, float *gdsz, float *dusopx, float *dusopz, float *derr, float **gp, float *wlt, int *sxz, int ng, fdm2d fdm)
/*< source position gradient >*/
{
	int ig,sx,sz;
	float idx,idz;
	idx=1.0/(fdm->dx*fdm->dx);
	idz=1.0/(fdm->dz*fdm->dz);
	sx=sxz[0]/fdm->nz + fdm->nb;
	sz=sxz[0]%fdm->nz;
	for(ig=0;ig<ng;ig++){
		*gdsx+=dusopx[ig]*derr[ig];
		*gdsz+=dusopz[ig]*derr[ig];		
		}
	*gdsx+=DX(gp,sx,sz,idx)*wlt[0];
	*gdsz+=DZ(gp,sx,sz,idz)*wlt[0];
		
	}
	
void scale_src_gradient(float *gdsx, float *gdsz)
/* Non dimensionalize the gradient with covariance matrix*/
{
	float a,b,c;
	a=*gdsx;
	b=*gdsz;
	c=sqrtf(a*a+b*b);
	*gdsx/=c;
	*gdsz/=c;
	
	}
	
void cal_beta(float *beta, float **gd0, float **gd1, float **cg, fdm2d fdm)
/*< calculate beta for nonlinear conjugate gradient algorithm for velocity >*/
{
	int ix,iz;
	float a,b,c;
	a=0.0,b=0.0,c=0.0;

#ifdef _OPENMP 		
#pragma omp parallel for default(none) \
		schedule(guided) private(iz) reduction(+:a,b,c)	\
		shared(gd0,gd1,cg,fdm)
#endif	
	
	for(ix=0;ix<fdm->nx;ix++){
		for(iz=0;iz<fdm->nz;iz++){
			a+=gd1[ix][iz]*(gd1[ix][iz]-gd0[ix][iz]);
			b+=cg[ix][iz]*(gd1[ix][iz]-gd0[ix][iz]);
			c+=gd1[ix][iz]*gd1[ix][iz];
	
			}
		}
	
	float beta_HS=0.0;
	float beta_DY=0.0;
	if(fabs(b)>EPS){	
		beta_HS=a/b; 
		beta_DY=c/b;	
		} 
	*beta=MAX(0.0, MIN(beta_HS, beta_DY));/* Hybrid HS-DY method combined with iteration restart */

	
	}
	
void src_cal_beta(float *beta, float gd0, float gd1, float cg)
/*< calculate beta for nonlinear conjugate gradient algorithm for source location >*/
{

	float a, b, c ;
	a=0.0,b=0.0,c=0.0 ;			

	a+=gd1*(gd1-gd0);
	b+=cg*(gd1-gd0);
	c+=gd1*gd1;		
			
	float beta_HS=0.0;
	float beta_DY=0.0;
	if(fabs(b)>EPS) 
	{
		beta_HS=a/b; 
		beta_DY=c/b;
		
		} 
	*beta=MAX(0.0, MIN(beta_HS, beta_DY));/* Hybrid HS-DY method combined with iteration restart */

}

void cal_conjgrad(float **gd1, float **cg, float beta, fdm2d fdm)
/*< calculate non-linear conjugate gradient for velocity >*/
{
	int ix,iz;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
		shared(gd1,cg,beta,fdm) private(ix,iz) 
#endif
	for(ix=0;ix<fdm->nx;ix++){
		for(iz=0;iz<fdm->nz;iz++){
			cg[ix][iz]=-gd1[ix][iz]+beta*cg[ix][iz];	
			}
		}
	}

void src_cal_conjgrad(float gd1, float *cg, float beta)
/*< calculate non-linear conjugate gradient for source location >*/
{
	
	cg[0]=-gd1+beta*cg[0];
				
	}
	
	

void cal_epsilon( float **v, float **cg, float *epsil, fdm2d fdm)
/*< calculated estimated stepsize for velocity >*/
{
	int ix,iz;
	float vdat,cdat,a,b;
	vdat=0.0,cdat=0.0;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
		private(ix,iz,a,b) shared(v,cg,fdm) reduction(max:vdat,cdat)
#endif
	for(ix=0;ix<fdm->nx;ix++){
		for(iz=0;iz<fdm->nz;iz++){
			a=fabsf(v[ix][iz]);
			b=fabsf(cg[ix][iz]);
			vdat=MAX(a,vdat);
			cdat=MAX(b,cdat);
			}
		}
	*epsil=(cdat>EPS) ? 0.01*(vdat/cdat) : 0.0 ;
	printf("cdat = %f",cdat);
	}
	
void src_cal_epsilon(float sx, float cg, float *epsil)
/*< calculate estimated stepsize (epsil) according to Taratola's method >*/
{

	*epsil=(fabsf(cg)>EPS) ? (0.01*fabsf(sx/cg)) : 0.0 ; 
	
	}
	
void cal_vtmp(float **vtmp, float **v, float **cg, float epsil, fdm2d fdm)
/*< calculate temporary velocity model> */
{
	int ix,iz;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
		private(iz) shared(v,cg,vtmp,epsil,fdm)
#endif	
	for(ix=0;ix<fdm->nx;ix++){
		for(iz=0;iz<fdm->nz;iz++){
			vtmp[ix][iz]=v[ix][iz]+epsil*cg[ix][iz];
			}
		}	
	}
	

void cal_sptmp(int *stmp, int sx, float cg, float epsil)
/*< calculate temporary source location >*/
{
	
	*stmp=sx+epsil*cg;
						
	}

void sum_alpha12(float *alpha1, float *alpha2, float *dcaltmp, float *dobs, float *derr, int ng)
/*< calculate the numerator and denominator of alpha
	alpha1: numerator; length=ng
	alpha2: denominator; length=ng >*/
{
	int ig;
	
	for(ig=0;ig<ng;ig++){	
		float c=derr[ig];
		float a=dobs[ig]+c;/* since f(mk)-dobs[id]=derr[id], thus f(mk)=b+c; */
		float b=dcaltmp[ig]-a;/* f(mk+epsil*cg)-f(mk) */
		alpha1[ig]-=b*c; alpha2[ig]+=b*b; 
		}
	
	}
	
void cal_alpha(float *alpha, float *alpha1, float *alpha2, float epsil, int ng)
/*< calculate searched stepsize (alpha) according to Taratola's method >*/
{
	int ig;
	float a1, a2 ;
	a1=0.0;a2=0.0 ;
#ifdef _OPENMP 		
#pragma omp parallel for     										\
	    schedule(static) shared(alpha1,alpha2)  					\
		private(ig) reduction(+:a1,a2) 									
#endif			
	for(ig=0;ig<ng;ig++){
		a1+=alpha1[ig];
		a2+=alpha2[ig];
		}	
	*alpha=(a2>EPS) ? epsil*a1/(a2+EPS) : 0.0;
	}
	
void src_cal_alpha(float *alphax, float *alphaz, float *alpha1, float *alpha2, float epsilx, float epsilz, int ng)
/*< calculate searched stepsize (alpha) according to Taratola's method for source perturbation >*/
{
	int ig;
	float a1, a2 ;
	a1=0.0;a2=0.0 ;
#ifdef _OPENMP 		
#pragma omp parallel for     										\
	    schedule(static) shared(alpha1,alpha2)  					\
		private(ig) reduction(+:a1,a2) 									
#endif			
	for(ig=0;ig<ng;ig++){
		a1+=alpha1[ig];
		a2+=alpha2[ig];
		}			
	*alphax=(a2>EPS) ? epsilx*a1/(a2+EPS) : 0 ;
	*alphaz=(a2>EPS) ? epsilz*a1/(a2+EPS) : 0 ;
	
	}
	
void update_vel(float **v, float **cg, float alpha, fdm2d fdm)
/*< update velocity model with obtained stepsize (alpha) >*/
{
		int ix,iz;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
		private(ix,iz) shared(v,cg,alpha,fdm)
#endif
		for(ix=0;ix<fdm->nx;ix++){
			for(iz=0;iz<fdm->nz;iz++){
				v[ix][iz]=v[ix][iz]+alpha*cg[ix][iz];
				}
		}
	
	}

void update_src(int *sx, float cg, float alpha)
/* update source position */
{
	*sx+=alpha*cg;
	
	}
	
/* ------------- Wavefield Reconstruction --------------*/

void boundary_rw(float **u, float *bndr, bool read, fdm2d fdm)
/* read/write using effective boundary saving strategy: 
   if read=true, read the boundary out; else save/write the boundary ,*/
{
	int ix,iz;

	if(read){
#ifdef _OPENMP
#pragma omp parallel for			\
		private(ix,iz)				\
		shared(u,bndr,fdm)  
#endif	
    for(ix=0;ix<fdm->nx;ix++){
		for(iz=0;iz<2;iz++){
			u[ix+fdm->nb][iz+fdm->nz]=bndr[iz+2*ix];
			}
		}
#ifdef _OPENMP
#pragma omp parallel for			\
		private(ix,iz)				\
		shared(u,bndr,fdm)  
#endif	
    for(iz=0;iz<fdm->nz; iz++){
		for(ix=0;ix<2; ix++){	
			u[ix-2+fdm->nb][iz]=bndr[2*fdm->nx+iz+fdm->nz*ix];
			u[ix+fdm->nx+fdm->nb][iz]=bndr[2*fdm->nx+iz+fdm->nz*(ix+2)];
			}
		}
	}
	else{
#ifdef _OPENMP
#pragma omp parallel for			\
		private(ix,iz)				\
		shared(u,bndr,fdm)  
#endif	
    for(ix=0;ix<fdm->nx;ix++){
		for(iz=0; iz<2; iz++){
			bndr[iz+2*ix]=u[ix+fdm->nb][iz+fdm->nz];/* bottom boundary */
			}
		}	
#ifdef _OPENMP
#pragma omp parallel for			\
		private(ix,iz)				\
		shared(u,bndr,fdm)  
#endif	
    for(iz=0; iz<fdm->nz; iz++){
		for(ix=0; ix<2; ix++){
			bndr[2*fdm->nx+iz+fdm->nz*ix]=u[ix-2+fdm->nb][iz];/*left boundary*/
			bndr[2*fdm->nx+iz+fdm->nz*(ix+2)]=u[ix+fdm->nx+fdm->nb][iz];/*right boundary*/
			}
		}
	}
}
//	  spo[4*nx+iz+nz*ix]=p[ix-2+nb][iz+nb];
//	  spo[4*nx+iz+nz*(ix+2)]=p[ix+nx+nb][iz+nb];
void save_wav(float **u , float **wfl, fdm2d fdm)
/*< save wavefield >*/
{
	int ix,iz;
	for(ix=0;iz<fdm->nx;ix++){
		for(iz=0;ix<fdm->nz;iz++){
			
			wfl[ix][iz]=u[ix+fdm->nb][iz];		
			
			}
		}	
	
	}	
