#include <stdio.h>
#include <string.h>
#include "binmod.c"
#include <fftw3.h>

int main(int argc, char *argv[])
{
	int i,nt=1000;
	float *data, *out ;
	fftwf_complex *in_cpx0,*out_cpx0, *out_cpx1, *mid_cpx;
	
	fftwf_plan fft;
	fftwf_plan ifft;

	in_cpx0=(fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*nt);
	out_cpx0=(fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*nt);
	out_cpx1=(fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*nt);
	mid_cpx=(fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*nt);
	data=(float*)malloc(sizeof(float)*nt);
	out=(float*)malloc(sizeof(float)*nt*2);
	memset(out,0,sizeof(float)*nt*2);
	readbin(data,"data.bin");
	/*Formatting Input in complex*/
	for(i=0;i<nt;i++){
		in_cpx0[i][0]=data[i];
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
		out[2*i]=out_cpx1[i][0]/nt; //Real
		out[2*i+1]=out_cpx1[i][1]/nt; //Imaginary
		}
	writebin(out,nt*2,"hilb.bin");
	
	fftwf_destroy_plan(fft);
	fftwf_destroy_plan(ifft);
	fftwf_free(out_cpx0);
	fftwf_free(out_cpx1);
	fftwf_free(mid_cpx);
	free(data);
	free(out);
		
	}
