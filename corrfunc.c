#include <stdlib.h>
#include <float.h>
#include <fftw3.h>

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



/*-------------Binary Operations--------------*/

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
	
/*-----------------FFTW operations----------------------*/


void cross_corr(float *dobs, float *dcal, float *corr, int na, int nb)
/*<-----------cross correlation using fftw-------------->*/
{
	int i;
	fftwf_complex *dobs_cpx, *dcal_cpx ;
	fftwf_complex *fdobs_cpx, *fdcal_cpx, *fcorr_cpx, *ifft_corr;
	fftwf_plan fft_dobs, fft_dcal, fft_corr;
	
	dobs_cpx=(fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*(na+nb-1));
	dcal_cpx=(fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*(na+nb-1));
	fcorr_cpx=(fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*(na+nb-1));
	fdobs_cpx=(fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*(na+nb-1));
	fdcal_cpx=(fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*(na+nb-1));
	ifft_corr=(fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*(na+nb-1));
	
	for(i=0;i<na;i++){
		dobs_cpx[i][0]=dobs[i];
		dobs_cpx[i][1]=0.0;
		}
	for(i=0;i<nb;i++){
		dcal_cpx[i][0]=dcal[i];
		dcal_cpx[i][1]=0.0;
		}
		
	fft_dobs= fftwf_plan_dft_1d((na+nb)-1,dobs_cpx,fdobs_cpx,FFTW_FORWARD,FFTW_ESTIMATE); 
	fft_dcal= fftwf_plan_dft_1d((na+nb)-1,dcal_cpx,fdcal_cpx,FFTW_FORWARD,FFTW_ESTIMATE); 
	fft_corr= fftwf_plan_dft_1d((na+nb)-1,fcorr_cpx,ifft_corr,FFTW_BACKWARD,FFTW_ESTIMATE); 
	
	fftwf_execute(fft_dobs);
	fftwf_execute(fft_dcal);
	/* cross correlation */
	for(i=0;i<na+nb-1;i++){
		fcorr_cpx[i][0]=(fdobs_cpx[i][0]*fdcal_cpx[i][0]+fdobs_cpx[i][1]*fdcal_cpx[i][1]);
		fcorr_cpx[i][1]=(fdobs_cpx[i][1]*fdcal_cpx[i][0]-fdobs_cpx[i][0]*fdcal_cpx[i][1]);
		}
	fftwf_execute(fft_corr);
	/*Reordering*/
	for(i=0;i<(na+nb)-1;i++){
		corr[i]=ifft_corr[(i+na)%((na+nb)-1)][0]/((na+nb)-1);
		}
	
	}
	
void anal_hilbert(float *din, float *dout, int nt)
/*<----------Analytic Signal (or Python/Matlab Hilbert transform) using fftw--------------->*/
{
	int i;
	fftwf_complex *in_cpx0,*out_cpx0, *out_cpx1, *mid_cpx;
	fftwf_plan fft;
	fftwf_plan ifft;
	in_cpx0=(fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*nt);
	out_cpx0=(fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*nt);
	out_cpx1=(fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*nt);
	mid_cpx=(fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*nt);
	/*Formatting Input in complex*/
	for(i=0;i<nt;i++){
		in_cpx0[i][0]=din[i];
		in_cpx0[i][1]=0.0;
		}
	fft=fftwf_plan_dft_1d(nt,in_cpx0,out_cpx0,FFTW_FORWARD,FFTW_ESTIMATE); 
	fftwf_execute(fft);
	/* Deleting Negative Frequencies and Scaling by 2 b/w DC and Nyquist */
	for(i=1;i<nt/2;i++){
		mid_cpx[i][0]=2*out_cpx0[i][0];
		mid_cpx[i][1]=2*out_cpx0[i][1];
		mid_cpx[nt/2+i][0]=0.0;
		mid_cpx[nt/2+i][1]=0.0;
		 }
		mid_cpx[0][0]=out_cpx0[0][0];
		mid_cpx[0][1]=out_cpx0[0][1];
		mid_cpx[nt/2][0]=out_cpx0[nt/2][0];
		mid_cpx[nt/2][1]=out_cpx0[nt/2][1];
	/*Inverse Fourier Transform*/
	ifft=fftwf_plan_dft_1d(nt,mid_cpx,out_cpx1,FFTW_BACKWARD,FFTW_ESTIMATE); 
	fftwf_execute(ifft);
	/*Formatting output to float (complex form) */
	for(i=0;i<nt;i++){
		dout[2*i]=out_cpx1[i][0]/nt; //Real
		dout[2*i+1]=out_cpx1[i][1]/nt; //Imaginary
		}
	
	}

void convolve(float *dobs, float *dcal, float *conv, int na, int nb)
/*<-----------convolution using fftw-------------->*/
{
	int i;
	fftwf_complex *dobs_cpx, *dcal_cpx ;
	fftwf_complex *fdobs_cpx, *fdcal_cpx, *fconv_cpx, *ifft_conv;
	fftwf_plan fft_dobs, fft_dcal, fft_conv;

	
	dobs_cpx=(fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*((na+nb)-1));
	dcal_cpx=(fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*((na+nb)-1));
	fconv_cpx=(fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*((na+nb)-1));
	fdobs_cpx=(fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*((na+nb)-1));
	fdcal_cpx=(fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*((na+nb)-1));
	ifft_conv=(fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*((na+nb)-1));
	
		
	for(i=0;i<na;i++){
		dobs_cpx[i][0]=dobs[i];
		dobs_cpx[i][1]=0.0;
		}
	for(i=0;i<nb;i++){
		dcal_cpx[i][0]=dobs[i];
		dcal_cpx[i][1]=0.0;
		}	
	fft_dobs= fftwf_plan_dft_1d((na+nb)-1,dobs_cpx,fdobs_cpx,FFTW_FORWARD,FFTW_ESTIMATE); 
	fft_dcal= fftwf_plan_dft_1d((na+nb)-1,dcal_cpx,fdcal_cpx,FFTW_FORWARD,FFTW_ESTIMATE); 
	fft_conv= fftwf_plan_dft_1d((na+nb)-1,fconv_cpx,ifft_conv,FFTW_BACKWARD,FFTW_ESTIMATE); 
	
	fftwf_execute(fft_dobs);
	fftwf_execute(fft_dcal);
	/* convolution*/
	for(i=0;i<(na+nb)-1;i++){
		fconv_cpx[i][0]=(fdobs_cpx[i][0]*fdcal_cpx[i][0]-fdobs_cpx[i][1]*fdcal_cpx[i][1]);
		fconv_cpx[i][1]=(fdobs_cpx[i][0]*fdcal_cpx[1][0]+fdobs_cpx[i][1]*fdcal_cpx[i][0]);
		}
	fftwf_execute(fft_conv);
	/*Reordering*/
	for(i=0;i<(na+nb)-1;i++){
		conv[i]=ifft_conv[(i+na)%((na+nb)-1)][0]/((na+nb)-1);
		}
	
	}
	
void pick_fbreak(float *dobs, float *win_dobs, float stab,float thresh, int times, int nt)
/*<--------------Pick First Breaks----------------->*/
{
	int i;
	float nshort, nlong, ltamax=0.0;
	float *env, *hilb, *shortones, *longones, *ratio;
	float *sta, *lta;
	env=(float*)malloc(nt*sizeof(float));
	hilb=(float*)malloc(2*nt*sizeof(float));
	
	anal_hilbert(dobs,hilb,nt);
	for(i=0;i<nt;i++){
		env[i]=hypotf(hilb[2*i],hilb[2*i+1]);
		}
	nshort=MAX(round(nt/(10.0*times)),2);
	nlong=MIN(times*nshort,nt);
	
	ratio=(float*)malloc(((int)(nshort)+nt-1)*sizeof(float));
	shortones=(float*)malloc((int)(nshort)*sizeof(float));
	longones=(float*)malloc((int)(nlong)*sizeof(float));
	sta=(float*)malloc(((int)(nshort)+nt-1)*sizeof(float));
	lta=(float*)malloc(((int)(nlong)+nt-1)*sizeof(float));
	memset(shortones,1.0,(int)(nshort)*sizeof(float));
	memset(longones,1.0,(int)(nlong)*sizeof(float));
	convolve(env,shortones,sta,nt,(int)(nshort));
	convolve(env,longones,lta,nt,(int)(nlong));
	for(i=0;i<(int)(nlong)+nt-1;i++){
		lta[i]=lta[i]/nlong;
		ltamax=MAX(ltamax,lta[i]);
		}
	for(i=0;i<(int)(nshort)+nt-1;i++){
		sta[i]=sta[i]/nshort;
		ratio[i]=sta[i]/(lta[i]+stab*ltamax);
		}

	}
	

void cal_adj_source(float *dobs, float *dcal, float *adjsrc, int nt, float dt)
/*<-------------calculate adjoint sources---------->*/
{
	int i,j=0;
	float tmax,max,E=0;
	float *dobsft, *der_dobsft, *dder_dobsft, *corr;
	
	dobsft=(float*)malloc(nt*sizeof(float));
	der_dobsft=(float*)malloc(nt*sizeof(float));
	dder_dobsft=(float*)malloc(nt*sizeof(float));
	corr=(float*)malloc((2*nt-1)*sizeof(float));
	memset(dobsft,0,nt*sizeof(float));
	memset(der_dobsft,0,nt*sizeof(float));
	memset(dder_dobsft,0,nt*sizeof(float));
	memset(corr,0,(2*nt-1)*sizeof(float));
	cross_corr(dobs,dcal,corr,nt,nt);
	/* Max of cross-correlation */
	max=corr[j];
	for(i=0;i<2*nt-1;i++){
		if (corr[i] > max){ 
				max = corr[i];
				j = i;
			}
		}
	tmax=j*dt;
	/* Reordering/Shifting by max  */
	for(i=0;i<nt;i++){
		dobsft[i]=dobs[(j+i)%nt];
		}
	for(i=0;i<nt;i++){
		/*Derivative*/
		der_dobsft[i]=(i>1 && i<nt-1)?(dobsft[i+1]-dobsft[i-1])/(2*dt):0;
		/*Double Derivative*/
		dder_dobsft[i]=(i>1 && i<nt-1)?(dobsft[i+1]-2*dobsft[i]+dobsft[i-1])/(dt*dt):0;
		}
	
		/*Derivative*/
		der_dobsft[0]=(-dobsft[2]+4*dobsft[1]-3*dobsft[0])/(2*dt);
		der_dobsft[nt-1]=(dobsft[nt-3]-4*dobsft[nt-2]+3*dobsft[nt-1])/(2*dt);
		/*Double Derivative*/
		dder_dobsft[0]=(2*dobsft[0]-5*dobsft[1]+4*dobsft[2]-dobsft[3])/(dt*dt);
		dder_dobsft[nt-1]=(-dobsft[nt-4]+4*dobsft[nt-2]-5*dobsft[nt-2]+2*dobsft[nt-1])/(dt*dt);
		
		for(i=0;i<nt;i++){
			E+= dder_dobsft[i]*dcal[i];
			}
		for(i=0;i<nt;i++){
			adjsrc[i]= -tmax*der_dobsft[i]/(E+EPS);
			}
	}
	

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
	float a,gmax;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
		private(ix,iz,a) shared(v,gd,fdm) reduction(max:gmax)
#endif			
	for(ix=0;ix<fdm->nx;ix++){
		for(iz=0;iz<fdm->nz;iz++){
			a=fabsf(gd[ix][iz]);
			gmax=MAX(a,gmax);			
			}
		}
		//printf("gmax= %f",gmax);
	for(ix=0;ix<fdm->nx;ix++){
		for(iz=0;iz<fdm->nz;iz++){
			float a=v[ix][iz]*gmax;
			if (precon) a*=sqrtf(illum[ix+fdm->nb][iz]+EPS);/*precondition with residual wavefield illumination*/
			gd[ix][iz]*=2.0/a ;
						
			}
		}
	}

void src_spat_gradient(float **fxz, float **u, float *wlt, fdm2d fdm)
/* source spatial gradient */
{
	int ix,iz;
	for(ix=0;ix<fdm->nx;ix++){
		for(iz=0;iz<fdm->nz;iz++){
			fxz[ix][iz]+=u[ix][iz]*wlt[0];
			}
		}		
	}
	


/*------------------------Inversion -----------------------------------*/

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
	
