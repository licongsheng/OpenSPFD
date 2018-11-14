#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuComplex.h"
#include "cusparse_v2.h"

#include <stdlib.h>
#include <stdio.h>
#include <complex>
#include <math.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <time.h>
#include <complex>
#include "common.h"
#include "cuComplex.h"
#include "cusparse_v2.h"

using namespace std;

#define MAXTHREADS 1024
int max_threads = 1024;
int Num_devs = 0;

__global__ void SetData(float *Ax,float *Ay)
{
	int i = blockIdx.x *MAXTHREADS + threadIdx.x;
	Ax[i] = Ax[i] + Ay[i];//dlx/temp*Amp*1e-7;
	Ay[i] = 2;//dly/temp*Amp*1e-7;
}

__global__ void CalculateMagneticVector(float cx,float cy,float cz,float dlx,float dly,float dlz,
		float *x_axis,int nx,float *y_axis,int ny,float *z_axis,int nz,float Amp,
		float *Ax,float *Ay,float *Az,float *Bx,float *By,float *Bz)
{
	int N = nx*ny*nz;
	int i = blockIdx.x *MAXTHREADS + threadIdx.x;
	if(i < N)
	{
		int z = (int)((float)i / (float)(nx*ny));
		int res = i - z*nx*ny;
		int y = (int)((float)res / (float)nx);
		int x = res - y*nx;

		float dx = x_axis[x] - cx;
		float dy = y_axis[y] - cy;
		float dz = z_axis[z] - cz;
		float temp = sqrt(dx*dx+dy*dy+dz*dz);

		Ax[i] = Ax[i] + dlx/temp*Amp*1e-7;
		Ay[i] = Ay[i] + dly/temp*Amp*1e-7;
		Az[i] = Az[i] + dlz/temp*Amp*1e-7;

		temp = temp*temp*temp;
		Bx[i] = Bx[i] + (dly*dz - dlz*dy)/temp*Amp*1e-7;
		By[i] = By[i] + (dlz*dx - dlx*dz)/temp*Amp*1e-7;
		Bz[i] = Bz[i] + (dlx*dy - dly*dx)/temp*Amp*1e-7;
	}
	__syncthreads();
}


__global__ void ConstructSPFDA(cuComplex *eps,int dims[3],float delt[3],cuComplex *Sx)
{
	cuComplex Zero = make_cuFloatComplex(10,0);
	int i,j,k;
	long N = dims[0]*dims[1]*dims[2];
	int idx = blockIdx.x *MAXTHREADS + threadIdx.x;
	if(idx < N)
	{
		Sx[idx] = eps[idx];
		//Sy[idx] = eps[idx];
		//Sz[idx] = eps[idx];
		/*
		k = (int)((float)idx / (float)(dims[0]*dims[1]));
		int res = idx - k*dims[0]*dims[1];
		j = (int)((float)res / (float)dims[0]);
		i = res - j*dims[0];
		cuComplex yz_x = make_cuFloatComplex(delt[1]*delt[2] / delt[0],0);
		cuComplex h_yz_x = cuCmulf(make_cuFloatComplex(0.5,0),yz_x);
		cuComplex xz_y = make_cuFloatComplex(delt[0]*delt[2] / delt[1],0);
		cuComplex h_xz_y = cuCmulf(make_cuFloatComplex(0.5,0),xz_y);
		cuComplex xy_z = make_cuFloatComplex(delt[0]*delt[1] / delt[2],0);
		cuComplex h_xy_z = cuCmulf(make_cuFloatComplex(0.5,0),xy_z);

		if(i == 0)
			Sx[idx] = cuCmulf(eps[idx],yz_x);
		else
			Sx[idx] = cuCmulf(cuCaddf(eps[idx],eps[idx-1]),h_yz_x);

		if(j == 0)
			Sy[idx] = cuCmulf(eps[idx],xz_y);
		else
			Sy[idx] = cuCmulf(cuCaddf(eps[idx],eps[idx-dims[0]]),h_xz_y);

		if(k == 0)
			Sz[idx] = cuCmulf(eps[idx],xy_z);
		else
			Sz[idx] = cuCmulf(cuCaddf(eps[idx],eps[idx-dims[0]*dims[1]]),h_xy_z);
		 */
	}
}


__global__ void ConstructSPFDB(float *Ax,float *Ay,float *Az,int dims[3],float delt[3],cuComplex jw,
		cuComplex *Sx,cuComplex *Sy,cuComplex *Sz,cuComplex *SA)
{
	cuComplex Zero = make_cuFloatComplex(0,0);
	cuComplex Air = cuCmulf(make_cuFloatComplex(EPSILON,0),jw); //sigma = 0 S/m, epsilon = 1.0
	cuComplex dx = make_cuFloatComplex(delt[0],0);
	cuComplex dy = make_cuFloatComplex(delt[1],0);
	cuComplex dz = make_cuFloatComplex(delt[2],0);
	long N = dims[0]*dims[1]*dims[2];
	int idx = blockIdx.x *MAXTHREADS + threadIdx.x;
	if(idx < N)
	{
		int k = (int)((float)idx / (float)(dims[0]*dims[1]));
		int res = idx - k*dims[0]*dims[1];
		int j = (int)((float)res / (float)dims[0]);
		int i = res - j*dims[0];
		cuComplex Sx_n,Sx_p,Ax_n,Ax_p,Sy_n,Sy_p,Ay_n,Ay_p,Sz_n,Sz_p,Az_n,Az_p;
		if(i == 0)
		{
			Sx_n = Air;
			Sx_p = Sx[idx];
			Ax_n = Zero;
			Ax_p = make_cuFloatComplex(Ax[idx+1],0);
		}
		else if(i == dims[0]-1)
		{
			Sx_n = Sx[idx-1];
			Sx_p = Air;
			Ax_n = make_cuFloatComplex(Ax[idx-1],0);
			Ax_p = Zero;
		}
		else
		{
			Sx_n = Sx[idx-1];
			Sx_p = Sx[idx];
			Ax_n = make_cuFloatComplex(Ax[idx-1],0);
			Ax_p = make_cuFloatComplex(Ax[idx+1],0);
		}

		if(j == 0)
		{
			Sy_n = Air;
			Sy_p = Sy[idx];
			Ay_n = Zero;
			Ay_p = make_cuFloatComplex(Ay[idx+dims[0]],0);
		}
		else if(j == dims[1]-1)
		{
			Sy_n = Sy[idx-dims[0]];
			Sy_p = Air;
			Ay_n = make_cuFloatComplex(Ay[idx-dims[0]],0);
			Ay_p = Zero;
		}
		else
		{
			Sy_n = Sy[idx-dims[0]];
			Sy_p = Sy[idx];
			Ay_n = make_cuFloatComplex(Ay[idx-dims[0]],0);
			Ay_p = make_cuFloatComplex(Ay[idx+dims[0]],0);
		}

		if(k == 0)
		{
			Sz_n = Air;
			Sz_p = Sz[idx];
			Az_n = Zero;
			Az_p = make_cuFloatComplex(Az[idx+dims[0]*dims[1]],0);
		}
		else if(k == dims[2]-1)
		{
			Sz_n = Sz[idx-dims[0]*dims[1]];
			Sz_p = Air;
			Az_n = make_cuFloatComplex(Az[idx-dims[0]*dims[1]],0);
			Az_p = Zero;
		}
		else
		{
			Sz_n = Sz[idx-dims[0]*dims[1]];
			Sz_p = Sz[idx];
			Az_n = make_cuFloatComplex(Az[idx-dims[0]*dims[1]],0);
			Az_p = make_cuFloatComplex(Az[idx+dims[0]*dims[1]],0);
		}

		cuComplex temp = cuCaddf(cuCaddf(cuCsubf(cuCmulf(cuCmulf(Ax_p,Sx_p),dx),cuCmulf(cuCmulf(Ax_n,Sx_n),dx)),
				cuCsubf(cuCmulf(cuCmulf(Ay_p,Sy_p),dy),cuCmulf(cuCmulf(Ay_n,Sy_n),dy))),
				cuCsubf(cuCmulf(cuCmulf(Az_p,Sz_p),dz),cuCmulf(cuCmulf(Az_n,Sz_n),dz)));
		SA[idx] = cuCmulf(jw,temp);
	}
}


__global__ void cdotcKernel(cuComplex *x, cuComplex *y, int L)
{
	int i = blockIdx.x *MAXTHREADS + threadIdx.x;
	if(i < L)
	{
		cuComplex xc = make_cuComplex(x[i].x, -x[i].y);
		y[i] = cuComplexDoubleToFloat(cuCmul(cuComplexFloatToDouble(xc), cuComplexFloatToDouble(y[i])));
	}
}

__global__ void dotcKernel(cuComplex *x, cuComplex *y, cuComplex *z, int L)
{
	int i = blockIdx.x *MAXTHREADS + threadIdx.x;
	if (i < L)
	{
		cuComplex xc = make_cuComplex(x[i].x, -x[i].y);
		z[i] = cuComplexDoubleToFloat(cuCmul(cuComplexFloatToDouble(xc), cuComplexFloatToDouble(y[i])));
	}
}

__global__ void vecAddKernel(cuComplex alpha, cuComplex *x, cuComplex beta, cuComplex *y, int L)
{
	int i = blockIdx.x *MAXTHREADS + threadIdx.x;
	if (i < L)
	{
		y[i] = cuComplexDoubleToFloat(cuCadd(cuCmul(cuComplexFloatToDouble(x[i]), cuComplexFloatToDouble(alpha)) ,
			cuCmul(cuComplexFloatToDouble(y[i]), cuComplexFloatToDouble(beta))));
	}
}

__global__ void vecCaddKernel(cuComplex alpha, cuComplex *x, cuComplex beta, cuComplex *y, cuComplex *z, int L)
{
	int i = blockIdx.x *MAXTHREADS + threadIdx.x;
	if (i < L)
	{
		z[i] = cuComplexDoubleToFloat(cuCadd(cuCmul(cuComplexFloatToDouble(x[i]), cuComplexFloatToDouble(alpha)),
			cuCmul(cuComplexFloatToDouble(y[i]), cuComplexFloatToDouble(beta))));
	}
}


__global__ void mvMulKernel(int *RowPtr, int *ColIdx, cuComplex*A, cuComplex *x, cuComplex *b, int N, long nnz)
{
	int i = blockIdx.x *MAXTHREADS + threadIdx.x;
	if (i < N)
	{
		b[i] = make_cuComplex(0,0);
		for(int j= RowPtr[i];j<RowPtr[i+1];j++)
			b[i] = cuComplexDoubleToFloat(cuCadd(cuComplexFloatToDouble(b[i]),
				cuCmul(cuComplexFloatToDouble(A[j]), cuComplexFloatToDouble(x[ColIdx[j]]))));
	}
}



extern "C"
{


/*
 *  Calculate Magnetic Vector and Flux ...
 * */
cudaError_t CUDAAnalysisMagneticField(float *Rx,float *Ry,float *Rz,float *dLx,float *dLy,float *dLz,int N_seg,
		float *x_axis,int nx,float *y_axis,int ny,float *z_axis,int nz,float Amp,
		float *Ax,float *Ay,float *Az,float *Bx,float *By,float *Bz,int devid)
{
	cudaError_t cudaStatus;
	float *x_axis_d = NULL;
	float *y_axis_d = NULL;
	float *z_axis_d = NULL;
	float *Ax_d = NULL;
	float *Ay_d = NULL;
	float *Az_d = NULL;
	float *Bx_d = NULL;
	float *By_d = NULL;
	float *Bz_d = NULL;
	int nblocks = 0;
	long N = nx*ny*nz;
	int i=0;

	cudaStatus = cudaSetDevice(devid);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!\n  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&x_axis_d, nx * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CUDA malloc failed!\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&y_axis_d, ny * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CUDA malloc failed!\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&z_axis_d, nz * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CUDA malloc failed!\n");
		goto Error;
	}

	cudaStatus = cudaMemcpy(x_axis_d, x_axis, nx * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy 'x_axis' failed!\n");
		goto Error;
	}

	cudaStatus = cudaMemcpy(y_axis_d, y_axis, ny * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy 'y_axis' failed!\n");
		goto Error;
	}

	cudaStatus = cudaMemcpy(z_axis_d, z_axis, nz * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy 'z_axis' failed!\n");
		goto Error;
	}

	//////////////////////////////////////////////////////////////////////////
	cudaStatus = cudaMalloc((void**)&Ax_d, N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CUDA malloc failed!\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&Ay_d, N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CUDA malloc failed!\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&Az_d, N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CUDA malloc failed!\n");
		goto Error;
	}

	cudaStatus = cudaMemcpy(Ax_d, Ax, N * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy 'Ax' failed!\n");
		goto Error;
	}

	cudaStatus = cudaMemcpy(Ay_d, Ay, N * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy 'Ay' failed!\n");
		goto Error;
	}

	cudaStatus = cudaMemcpy(Az_d, Az, N * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy 'Az' failed!\n");
		goto Error;
	}

	//////////////////////////////////////////////////////////////////////////
	cudaStatus = cudaMalloc((void**)&Bx_d, N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CUDA malloc failed!\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&By_d, N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CUDA malloc failed!\n");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&Bz_d, N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CUDA malloc failed!\n");
		goto Error;
	}

	cudaStatus = cudaMemcpy(Bx_d, Bx, N * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy 'Bx' failed!\n");
		goto Error;
	}

	cudaStatus = cudaMemcpy(By_d, By, N * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy 'By' failed!\n");
		goto Error;
	}

	cudaStatus = cudaMemcpy(Bz_d, Bz, N * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy 'Bz' failed!\n");
		goto Error;
	}

	nblocks = (int)ceil((double)N / (double)max_threads);
	/////////////////////////////////////////////////////////////////////////////////////
	for(i=0;i<N_seg;i++)
	{
		cout<< "Process for coil segmentation @ " <<i << "/" << N_seg << ". block = "<< nblocks << ", threads = " << MAXTHREADS <<endl;
		CalculateMagneticVector <<<nblocks, MAXTHREADS >>> (Rx[i],Ry[i],Rz[i],dLx[i],dLy[i],dLz[i],
			x_axis_d,nx,y_axis_d,ny,z_axis_d,nz,Amp,Ax_d,Ay_d,Az_d,Bx_d,By_d,Bz_d);
	}

	/////////////////////////////////////////////////////////////////////////////////////

	cudaStatus = cudaMemcpy(Ax, Ax_d, N * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		goto Error;
	}

	cudaStatus = cudaMemcpy(Ay, Ay_d, N * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		goto Error;
	}

	cudaStatus = cudaMemcpy(Az, Az_d, N * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		goto Error;
	}


	cudaStatus = cudaMemcpy(Bx, Bx_d, N * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		goto Error;
	}

	cudaStatus = cudaMemcpy(By, By_d, N * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		goto Error;
	}

	cudaStatus = cudaMemcpy(Bz, Bz_d, N * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		goto Error;
	}
	/////////////////////////////////////////////////////////////////////////////////////
Error:
	if(x_axis_d) cudaFree(x_axis_d);
	if(y_axis_d) cudaFree(y_axis_d);
	if(z_axis_d) cudaFree(z_axis_d);
	if(Ax_d) cudaFree(Ax_d);
	if(Ay_d) cudaFree(Ay_d);
	if(Az_d) cudaFree(Az_d);
	if(Bx_d) cudaFree(Bx_d);
	if(By_d) cudaFree(By_d);
	if(Bz_d) cudaFree(Bz_d);

	return cudaStatus;
}


cudaError_t CUDAAnalysisSxyz(complex<float> *cEps,float *Ax,float *Ay,float *Az,float f,
		complex<float> *Sx,complex<float> *Sy,complex<float> *Sz,
		complex<float> *SA,int dims[3],float spacing[3],int devid)
{
	cudaError_t cudaStatus;
	cuComplex *cEps_d = NULL;
	float *Ax_d = NULL;
	float *Ay_d = NULL;
	float *Az_d = NULL;
	cuComplex *Sx_d = NULL;
	cuComplex *Sy_d = NULL;
	cuComplex *Sz_d = NULL;
	cuComplex *SA_d = NULL;
	cuComplex jw = make_cuComplex(0,2*M_PI*f);
	int nblocks = 0;
	long N = dims[0]*dims[1]*dims[2];

	cudaStatus = cudaSetDevice(devid);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!\n  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	/////////////////////////////////////////////////////////////////////////////
	cudaStatus= cudaMalloc((void**)&cEps_d, N * sizeof(cuComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CUDA malloc failed!\n");
		goto Error;
	}
	cudaStatus = cudaMemcpy(cEps_d, cEps, N * sizeof(cuComplex), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy 'complex epsilon' failed!\n");
		goto Error;
	}

	/////////////////////////////////////////////////////////////////////////////
	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&Ax_d, N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CUDA malloc failed!\n");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&Ay_d, N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CUDA malloc failed!\n");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&Az_d, N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CUDA malloc failed!\n");
		goto Error;
	}
	cudaStatus = cudaMemcpy(Ax_d, Ax, N * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy 'Ax' failed!\n");
		goto Error;
	}
	cudaStatus = cudaMemcpy(Ay_d, Ay, N * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy 'Ay' failed!\n");
		goto Error;
	}
	cudaStatus = cudaMemcpy(Az_d, Az, N * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy 'Az' failed!\n");
		goto Error;
	}

	/////////////////////////////////////////////////////////////////////////////
	cudaStatus= cudaMalloc((void**)&Sx_d, N * sizeof(cuComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CUDA malloc failed!\n");
		goto Error;
	}
	cudaStatus= cudaMalloc((void**)&Sy_d, N * sizeof(cuComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CUDA malloc failed!\n");
		goto Error;
	}
	cudaStatus= cudaMalloc((void**)&Sz_d, N * sizeof(cuComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CUDA malloc failed!\n");
		goto Error;
	}
	cudaStatus= cudaMalloc((void**)&SA_d, N * sizeof(cuComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CUDA malloc failed!\n");
		goto Error;
	}
	cudaStatus = cudaMemset((void *)Sx_d, 0, sizeof(Sx_d[0]) * N);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy 'Sx' failed!\n");
		goto Error;
	}
	cudaStatus = cudaMemset((void *)Sy_d, 0, sizeof(Sy_d[0]) * N);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy 'Sy' failed!\n");
		goto Error;
	}
	cudaStatus = cudaMemset((void *)Sz_d, 0, sizeof(Sz_d[0]) * N);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy 'Sz' failed!\n");
		goto Error;
	}
	cudaStatus = cudaMemset((void *)SA_d, 0, sizeof(SA_d[0]) * N);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy 'SA' failed!\n");
		goto Error;
	}
	////////////////////////////////////////////////////////////////////////////////////////
	nblocks = (int)ceil((double)N / (double)max_threads);
	cout<<"Constructing SPFD .... "<<endl;
	ConstructSPFDA <<<nblocks, MAXTHREADS >>> (cEps_d,dims,spacing,Sx_d);
	//ConstructSPFDB <<<nblocks, MAXTHREADS >>> (Ax_d,Ay_d,Az_d,dims,spacing,jw,
	//				Sx_d,Sy_d,Sz_d,SA_d);
	////////////////////////////////////////////////////////////////////////////////////////
	cudaStatus = cudaMemcpy(Sx, Sx_d, N * sizeof(cuComplex), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy 'Sx_d' failed! %s\n",cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaMemcpy(Sy, Sy_d, N * sizeof(cuComplex), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy 'Sy_d' failed! %s\n",cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaMemcpy(Sz, Sz_d, N * sizeof(cuComplex), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy 'Sz_d' failed! %s\n",cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaMemcpy(SA, SA_d, N * sizeof(cuComplex), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy 'SA_d' failed! %s\n",cudaGetErrorString(cudaStatus));
		goto Error;
	}
	//////////////////////////////////////////////////////////////////////////////////////////
Error:
	if(cEps_d) cudaFree(cEps_d);
	if(Ax_d) cudaFree(Ax_d);
	if(Ay_d) cudaFree(Ay_d);
	if(Az_d) cudaFree(Az_d);
	if(Sx_d) cudaFree(Sx_d);
	if(Sy_d) cudaFree(Sy_d);
	if(Sz_d) cudaFree(Sz_d);
	if(SA_d) cudaFree(SA_d);
	return cudaStatus;
}


float CudaSquaredNorm(cuComplex *x, long N)
{
	complex<float> *x_host = new complex<float>[N];
	cudaMemcpy(x_host, x, N * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	float sum = 0;
	for (int i = 0; i < N; i++)
		sum = sum + x_host[i].real()*x_host[i].real() + x_host[i].imag()*x_host[i].imag();

	delete[]x_host;
	return sum;
}

complex<float> CudaSum(cuComplex *x, long N)
{
	complex<float> *x_host = new complex<float>[N];
	cudaMemcpy(x_host, x, N * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	complex<float> sum = 0;
	for (int i = 0; i < N; i++)
		sum = sum + x_host[i];

	delete[]x_host;
	return sum;
}

cudaError_t CudaVectorCdotc(complex<float> *x_host, complex<float> *y_host, complex<float> &dot, int L,int devid)
{
	cudaError_t cudaStatus;
	cuComplex *x_dev = NULL;
	cuComplex *y_dev = NULL;
	complex<float> *z_host = NULL;
	int nblocks = 0;

	cudaStatus = cudaSetDevice(devid);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&x_dev, L * sizeof(cuComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&y_dev, L * sizeof(cuComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(x_dev, x_host, L * sizeof(cuComplex), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(y_dev, y_host, L * sizeof(cuComplex), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	nblocks = (int)ceil((double)L / (double)max_threads);
	cdotcKernel <<<nblocks, MAXTHREADS >>> (x_dev,y_dev,L);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "MatrixMULTVector Kernal launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	z_host = new complex<float>[L];
	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(z_host, y_dev, L * sizeof(cuComplex), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	dot = 0;
	for (int i = 0; i < L; i++)
		dot = dot + z_host[i];


Error:
	if (x_dev) cudaFree(x_dev);
	if (y_dev) cudaFree(y_dev);
	if (z_host) delete[]z_host;
	return cudaStatus;
}

cudaError_t CudaVectorCadd(complex<float> alpha, complex<float> *x_host, complex<float> beta, complex<float> *y_host, int L, int devid)
{
	cudaError_t cudaStatus;
	cuComplex *x_dev = NULL;
	cuComplex *y_dev = NULL;
	cuComplex cu_alpha = make_cuComplex(alpha.real(), alpha.imag());
	cuComplex cu_beta = make_cuComplex(beta.real(), beta.imag());
	int nblocks = 0;

	cudaStatus = cudaSetDevice(devid);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&x_dev, L * sizeof(cuComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&y_dev, L * sizeof(cuComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(x_dev, x_host, L * sizeof(cuComplex), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(y_dev, y_host, L * sizeof(cuComplex), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	nblocks = (int)ceil((double)L / (double)max_threads);
	vecAddKernel <<<nblocks, MAXTHREADS >>> (cu_alpha,x_dev, cu_beta, y_dev, L);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "MatrixMULTVector Kernal launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(y_host, y_dev, L * sizeof(cuComplex), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	if (x_dev) cudaFree(x_dev);
	if (y_dev) cudaFree(y_dev);
	return cudaStatus;
}

cudaError_t CudamvMul(int *RowPtr_host, int *ColIdx_host, complex<float> *A_host, complex<float> *x_host, complex<float> *b_host, int N,long nnz, int devid)
{
	cudaError_t cudaStatus;
	int *RowPtr_dev = NULL;
	int *ColIdx_dev = NULL;
	cuComplex *A_dev = NULL;
	cuComplex *X_dev = NULL;
	cuComplex *b_dev = NULL;
	int nblocks = 0;

	cudaStatus = cudaSetDevice(devid);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&RowPtr_dev, (N+1) * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&ColIdx_dev, nnz * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&A_dev, nnz * sizeof(cuComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&X_dev, N * sizeof(cuComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&b_dev, N * sizeof(cuComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	// Copy input vectors from host memory to GPU buffers.

	cudaStatus = cudaMemcpy(RowPtr_dev, RowPtr_host, (N+1) * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(ColIdx_dev, ColIdx_host, nnz * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(A_dev, A_host, nnz * sizeof(cuComplex), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(X_dev, x_host, N * sizeof(cuComplex), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	nblocks = (int)ceil((double)N / (double)max_threads);
	mvMulKernel << <nblocks, MAXTHREADS >> > (RowPtr_dev, ColIdx_dev, A_dev, X_dev, b_dev, N,nnz);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "MatrixMULTVector Kernal launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(b_host, b_dev, N * sizeof(cuComplex), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
Error:
	if (RowPtr_dev) cudaFree(RowPtr_dev);
	if (ColIdx_dev) cudaFree(ColIdx_dev);
	if (A_dev) cudaFree(A_dev);
	if (X_dev) cudaFree(X_dev);
	if (b_dev) cudaFree(b_dev);
	return cudaStatus;
}


cudaError_t GPUBICGSTAB(int *RowPtr_host, int *ColIdx_host, complex<float> *A_host, complex<float> *b_host, complex<float> *x_host, int N, long nnz, vector<float> residuals, int max_steps, float tol,int devid)
{
	cudaError_t cudaStatus;
	int *RowPtr_dev = NULL;
	int *ColIdx_dev = NULL;
	cuComplex *A_dev = NULL;
	cuComplex *X_dev = NULL;
	cuComplex *b_dev = NULL;

	cuComplex *r = NULL;
	cuComplex *rh = NULL;
	cuComplex *v = NULL;
	cuComplex *p = NULL;
	cuComplex *temp = NULL;
	cuComplex *h = NULL;
	cuComplex *s = NULL;
	cuComplex *t = NULL;
	float nrmr, nrmr0;
	cuComplex rho0, rho, alpha, w, beta;
	float err = 1000;
	int interation = 0;
	cuComplex zero = make_cuComplex(0.0, 0.0);
	cuComplex one = make_cuComplex(1.0, 0.0);
	cuComplex none = make_cuComplex(-1.0, 0.0);
	int nblocks = (int)ceil((double)N / (double)max_threads);

	////////////////////////////////////////////////////////////////////////////
	cudaStatus = cudaSetDevice(devid);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}


	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&RowPtr_dev, (N + 1) * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&ColIdx_dev, nnz * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&A_dev, nnz * sizeof(cuComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&X_dev, N * sizeof(cuComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&b_dev, N * sizeof(cuComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.

	cudaStatus = cudaMemcpy(RowPtr_dev, RowPtr_host, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(ColIdx_dev, ColIdx_host, nnz * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(A_dev, A_host, nnz * sizeof(cuComplex), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(b_dev, b_host, N * sizeof(cuComplex), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(X_dev, x_host, N * sizeof(cuComplex), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	cudaStatus = cudaMalloc((void**)&r, N * sizeof(cuComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&rh, N * sizeof(cuComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&v, N * sizeof(cuComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&p, N * sizeof(cuComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&h, N * sizeof(cuComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&s, N * sizeof(cuComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&t, N * sizeof(cuComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&temp, N * sizeof(cuComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(r, b_host, N * sizeof(cuComplex), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(rh, b_host, N * sizeof(cuComplex), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemset((void *)v, 0, sizeof(v[0]) * N);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed!");
		goto Error;
	}
	cudaStatus = cudaMemset((void *)p, 0, sizeof(p[0]) * N);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed!");
		goto Error;
	}
	cudaStatus = cudaMemset((void *)h, 0, sizeof(h[0]) * N);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed!");
		goto Error;
	}
	cudaStatus = cudaMemset((void *)s, 0, sizeof(s[0]) * N);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed!");
		goto Error;
	}
	cudaStatus = cudaMemset((void *)t, 0, sizeof(t[0]) * N);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed!");
		goto Error;
	}
	cudaStatus = cudaMemset((void *)temp, 0, sizeof(temp[0]) * N);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed!");
		goto Error;
	}
	////////////////////////////////////////////////////////////////////////////////////////

	rho0 = alpha = w = one;
	nrmr0 = CudaSquaredNorm(r, N);
	cout << "BICGStab on GPU start iterating ..." << err << tol << interation << max_steps  <<endl;
	while (err > tol && interation < max_steps)
	{
		//1. ρi = (r̂0, ri−1)
		//cout << "BICGStab @ step 1"<< endl;
		dotcKernel <<<nblocks, MAXTHREADS >> > (rh, r,temp,N);
		complex<float> rho_host = CudaSum(temp, N);
		rho = make_cuComplex(rho_host.real(), rho_host.imag());
		cudaMemset((void *)temp, 0, sizeof(temp[0]) * N);

		//2. β = (ρi/ρi−1)(α/ωi−1)
		//cout << "BICGStab @ step 2" << endl;
		beta = cuComplexDoubleToFloat(cuCmul(cuCdiv(cuComplexFloatToDouble(rho), cuComplexFloatToDouble(rho0)),
			cuCdiv(cuComplexFloatToDouble(alpha), cuComplexFloatToDouble(w))));
		rho0 = rho;

		//3. pi = ri−1 + β(pi−1 − ωi−1vi−1)
		//cout << "BICGStab @ step 3" << endl;
		cuComplex nw = make_cuComplex(-w.x,-w.y);
		vecCaddKernel << <nblocks, MAXTHREADS >> > (one, p, nw, v, temp, N);
		vecCaddKernel << <nblocks, MAXTHREADS >> > (one, r, beta, temp, p, N);
		cudaMemset((void *)temp, 0, sizeof(temp[0]) * N);

		//4. vi = Api
		//cout << "BICGStab @ step 4" << endl;
		mvMulKernel << <nblocks, MAXTHREADS >> > (RowPtr_dev, ColIdx_dev, A_dev, p, v, N, nnz);

		//5. α = ρi/(r̂0, vi)
		//cout << "BICGStab @ step 5" << endl;
		dotcKernel << <nblocks, MAXTHREADS >> > (rh, v, temp, N);
		complex<float> rhv_host = CudaSum(temp, N);
		cuDoubleComplex rhv = make_cuDoubleComplex(rhv_host.real(), rhv_host.imag());
		cudaMemset((void *)temp, 0, sizeof(temp[0]) * N);
		alpha = cuComplexDoubleToFloat(cuCdiv(cuComplexFloatToDouble(rho),rhv));

		//6. h = xi−1 + αpi
		//cout << "BICGStab @ step 6" << endl;
		vecCaddKernel << <nblocks, MAXTHREADS >> > (one, X_dev, alpha, p, h, N);
		//7. If h is accurate enough, then set xi = h and quit

		//8. s = ri−1 − αvi
		//cout << "BICGStab @ step 8" << endl;
		cuComplex nalpha = make_cuComplex(-alpha.x, -alpha.y);
		vecCaddKernel << <nblocks, MAXTHREADS >> > (one, r, nalpha, v, s, N);

		//9. t = As
		//cout << "BICGStab @ step 9" << endl;
		mvMulKernel << <nblocks, MAXTHREADS >> > (RowPtr_dev, ColIdx_dev, A_dev, s, t, N, nnz);

		//10. ωi = (t, s)/(t, t)
		//cout << "BICGStab @ step 10" << endl;
		//cuDoubleComplex tt = make_cuDoubleComplex(0, 0);
		//cuDoubleComplex ts = make_cuDoubleComplex(0, 0);
		dotcKernel << <nblocks, MAXTHREADS >> > (t, t, temp, N);
		complex<float> tt_host = CudaSum(temp, N);
		cudaMemset((void *)temp, 0, sizeof(temp[0]) * N);

		dotcKernel << <nblocks, MAXTHREADS >> > (t, s, temp, N);
		complex<float> ts_host = CudaSum(temp, N);
		cudaMemset((void *)temp, 0, sizeof(temp[0]) * N);
		complex<float> w_host = ts_host / tt_host;
		w = make_cuComplex(w_host.real(),w_host.imag());

		//11. xi = h + ωis
		//cout << "BICGStab @ step 11" << endl;
		vecCaddKernel << <nblocks, MAXTHREADS >> > (one, h, w, s, X_dev, N);

		//12. If xi is accurate enough, then quit

		//13. ri = s − ωit
		//cout << "BICGStab @ step 13" << endl;
		nw = make_cuComplex(-w.x, -w.y);
		vecCaddKernel << <nblocks, MAXTHREADS >> > (one, s, nw, t, r, N);

		nrmr = CudaSquaredNorm(r, N);

		err = sqrt(nrmr / nrmr0);
		cout << "BICGStab @ " << interation;
		cout.setf(ios::scientific, ios::floatfield);
		cout << ", Residual = " << err <<endl;
		residuals.push_back(err);
		interation++;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(x_host, X_dev, N * sizeof(cuComplex), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	if (RowPtr_dev) cudaFree(RowPtr_dev);
	if (ColIdx_dev) cudaFree(ColIdx_dev);
	if (A_dev) cudaFree(A_dev);
	if (X_dev) cudaFree(X_dev);
	if (b_dev) cudaFree(b_dev);
	if (r) cudaFree(r);
	if (rh) cudaFree(rh);
	if (v) cudaFree(v);
	if (p) cudaFree(p);
	if (temp) cudaFree(temp);
	if (h) cudaFree(h);
	if (s) cudaFree(s);
	if (t) cudaFree(t);
	return cudaStatus;
}



void LowerTriSolver(complex<float> *A, complex<float> *b, complex<float> *X, int *R_ptr, int *C_idx, int N, long nnz)
{
	int i = 0;
	int j = 0;
	X[0] = b[0];

	for (i = 1; i < N; i++)
	{
		complex<float> s = 0;
		for (j = R_ptr[i]; j < R_ptr[i + 1]; j++)
			if (C_idx[j] < i)
				s = s + A[j] * X[C_idx[j]];
		X[i] = b[i] - s;
	}
}

void UpperTriSolver(complex<float> *A, complex<float> *b, complex<float> *X, int *R_ptr, int *C_idx, int N, long nnz)
{
	int i = 0;
	int j = 0;
	X[N - 1] = b[N - 1];

	for (i = N - 2; i >= 0; i--)
	{
		complex<float> s = 0;
		complex<float> dig = 1.0;
		for (j = R_ptr[i]; j < R_ptr[i + 1]; j++)
		{
			if (C_idx[j] == i)
				dig = A[j];
			if (C_idx[j] > i)
				s = s + A[j] * X[C_idx[j]];
		}

		X[i] = (b[i] - s) / dig;
	}
}



cudaError_t GPUPBICGSTAB(int *RowPtr_host, int *ColIdx_host, complex<float> *A_host, complex<float> *A_ilu_host, complex<float> *b_host, complex<float> *x_host, int N, long nnz,vector<float> residuals, int max_steps, float tol, int devid, const char *logfile)
{
	cudaEvent_t start, stop;
	float costtime;
	cudaError_t cudaStatus;
	int *RowPtr_dev = NULL;
	int *ColIdx_dev = NULL;
	cuComplex *A_dev = NULL;
	cuComplex *iluA_dev = NULL;
	cuComplex *X_dev = NULL;
	cuComplex *b_dev = NULL;

	cuComplex *r = NULL;
	cuComplex *rh = NULL;
	cuComplex *v = NULL;
	cuComplex *p = NULL;
	cuComplex *y = NULL;
	cuComplex *temp = NULL;
	cuComplex *h = NULL;
	cuComplex *z = NULL;
	cuComplex *s = NULL;
	cuComplex *t = NULL;

	complex<float> *p_host = NULL;
	complex<float> *temp_host = NULL;
	complex<float> *y_host = NULL;

	complex<float> *s_host = NULL;
	complex<float> *z_host = NULL;
	clock_t cup_start, cup_stop;
	ofstream file;

	float nrmr, nrmr0;
	cuComplex rho0, rho, alpha, w, beta;
	float err = 1000;
	int interation = 0;
	cuComplex zero = make_cuComplex(0.0, 0.0);
	cuComplex one = make_cuComplex(1.0, 0.0);
	cuComplex none = make_cuComplex(-1.0, 0.0);
	int nblocks = (int)ceil((double)N / (double)max_threads);
	time_t tt = time(NULL);
	struct tm* current_time = localtime(&tt);

	////////////////////////////////////////////////////////////////////////////
	cudaStatus = cudaSetDevice(devid);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&RowPtr_dev, (N + 1) * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&ColIdx_dev, nnz * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&A_dev, nnz * sizeof(cuComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&iluA_dev, nnz * sizeof(cuComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&X_dev, N * sizeof(cuComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&b_dev, N * sizeof(cuComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.

	cudaStatus = cudaMemcpy(RowPtr_dev, RowPtr_host, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(ColIdx_dev, ColIdx_host, nnz * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(A_dev, A_host, nnz * sizeof(cuComplex), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(iluA_dev, A_ilu_host, nnz * sizeof(cuComplex), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(b_dev, b_host, N * sizeof(cuComplex), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(X_dev, x_host, N * sizeof(cuComplex), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	cudaStatus = cudaMalloc((void**)&r, N * sizeof(cuComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&rh, N * sizeof(cuComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&v, N * sizeof(cuComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&p, N * sizeof(cuComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&temp, N * sizeof(cuComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&y, N * sizeof(cuComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&h, N * sizeof(cuComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&s, N * sizeof(cuComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&z, N * sizeof(cuComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&t, N * sizeof(cuComplex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	/////////////////////////////////////////////////////////////////////////////////////
	cudaStatus = cudaMemcpy(r, b_host, N * sizeof(cuComplex), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(rh, b_host, N * sizeof(cuComplex), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemset((void *)v, 0, sizeof(v[0]) * N);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed!");
		goto Error;
	}

	cudaStatus = cudaMemset((void *)p, 0, sizeof(p[0]) * N);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed!");
		goto Error;
	}

	cudaStatus = cudaMemset((void *)y, 0, sizeof(y[0]) * N);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed!");
		goto Error;
	}

	cudaStatus = cudaMemset((void *)h, 0, sizeof(h[0]) * N);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed!");
		goto Error;
	}

	cudaStatus = cudaMemset((void *)s, 0, sizeof(s[0]) * N);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed!");
		goto Error;
	}

	cudaStatus = cudaMemset((void *)z, 0, sizeof(z[0]) * N);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed!");
		goto Error;
	}

	cudaStatus = cudaMemset((void *)t, 0, sizeof(t[0]) * N);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed!");
		goto Error;
	}
	cudaStatus = cudaMemset((void *)temp, 0, sizeof(temp[0]) * N);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemset failed!");
		goto Error;
	}
	p_host = new complex<float>[N];
	memset(p_host,0,sizeof(p_host[0]) * N);
	temp_host = new complex<float>[N];
	memset(temp_host, 0, sizeof(temp_host[0]) * N);
	y_host = new complex<float>[N];
	memset(y_host, 0, sizeof(y_host[0]) * N);

	s_host = new complex<float>[N];
	memset(s_host, 0, sizeof(s_host[0]) * N);
	z_host = new complex<float>[N];
	memset(z_host, 0, sizeof(z_host[0]) * N);

	file.open(logfile);
	////////////////////////////////////////////////////////////////////////////////////
	rho0 = alpha = w = rho = one;
	nrmr0 = CudaSquaredNorm(r, N);
	cout << "BICGStab on GPU start iterating ..." << err << tol << interation << max_steps << endl;

	current_time = localtime(&tt);
	file << "current time is " << current_time->tm_hour<<":"<<current_time->tm_min<<":"<<current_time->tm_sec;

	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);
	printf("Memory cost: %6.4f MB.\n",
			(float)ceil((11.0*N*(double)sizeof(cuComplex)+
					nnz*(double)sizeof(cuComplex)+
					(N+1)*(double)sizeof(int)+
					nnz*(double)sizeof(int))/1.0e6));
	while (err > tol && interation < max_steps)
	{
		rho0 = rho;
		//1. ρi = (r̂0, ri−1)
		//cudaEventRecord(start,0);
		dotcKernel << <nblocks, MAXTHREADS >> > (rh, r, temp, N);
		//cudaEventRecord(stop,0);
		//cudaEventSynchronize(stop);
		//cudaEventElapsedTime(&costtime,start,stop);
		//printf("CUDA Vector Inner Product spend %6.4f ms.\n",costtime);
		complex<float> rho_host = CudaSum(temp, N);
		rho = make_cuComplex(rho_host.real(), rho_host.imag());
		cudaMemset((void *)temp, 0, sizeof(temp[0]) * N);

		//2. β = (ρi/ρi−1)(α/ωi−1)
		//cout << "BICGStab @ step 2" << endl;
		beta = cuComplexDoubleToFloat(cuCmul(
			cuCdiv(cuComplexFloatToDouble(rho), cuComplexFloatToDouble(rho0)),
			cuCdiv(cuComplexFloatToDouble(alpha), cuComplexFloatToDouble(w))));

		//3. pi = ri−1 + β(pi−1 − ωi−1vi−1)
		//cout << "BICGStab @ step 3" << endl;
		cuComplex nw = make_cuComplex(-w.x, -w.y);
		vecCaddKernel << <nblocks, MAXTHREADS >> > (one, p, nw, v, temp, N);
		vecCaddKernel << <nblocks, MAXTHREADS >> > (one, r, beta, temp, p, N);
		cudaMemset((void *)temp, 0, sizeof(temp[0]) * N);

		//4. y = K−1pi
		// K=LU K=LU , p = L temp, temp = Uy
		//10.1 temp = L-1 p
		//cudaEventRecord(start,0);
		//cup_start = clock();
		cudaStatus = cudaMemcpy(p_host, p, N * sizeof(cuComplex), cudaMemcpyDeviceToHost);
		memset(temp_host, 0, sizeof(temp_host[0]) * N);
		LowerTriSolver(A_ilu_host, p_host, temp_host, RowPtr_host, ColIdx_host, N, nnz);
		//10.2 y = U-1 temp
		UpperTriSolver(A_ilu_host, temp_host,y_host, RowPtr_host, ColIdx_host, N, nnz);
		cudaStatus = cudaMemcpy(y, y_host, N * sizeof(cuComplex), cudaMemcpyHostToDevice);
		//cup_stop = clock();
		//cudaEventRecord(stop,0);
		//cudaEventSynchronize(stop);
		//cudaEventElapsedTime(&costtime,start,stop);
		//printf("CUDA Vector / SparseM spend %6.4f ms.\n",costtime);
		//printf("CPU Vector / SparseM spend %6.4f ms.\n",float(cup_stop - cup_start));
		//5. vi = Ay
		//cudaEventRecord(start,0);
		mvMulKernel << <nblocks, MAXTHREADS >> > (RowPtr_dev, ColIdx_dev, A_dev, y, v, N, nnz);
		//cudaEventRecord(stop,0);
		//cudaEventSynchronize(stop);
		//cudaEventElapsedTime(&costtime,start,stop);
		//printf("CUDA SparseM * Vector spend %6.4f ms.\n",costtime);

		//6. α = ρi/(r̂0, vi)
		dotcKernel << <nblocks, MAXTHREADS >> > (rh, v, temp, N);
		complex<float> rhv_host = CudaSum(temp, N);
		cuDoubleComplex rhv = make_cuDoubleComplex(rhv_host.real(), rhv_host.imag());
		cudaMemset((void *)temp, 0, sizeof(temp[0]) * N);
		alpha = cuComplexDoubleToFloat(cuCdiv(cuComplexFloatToDouble(rho), rhv));

		//7. h = xi−1 + αy
		//vecCaddKernel << <nblocks, MAXTHREADS >> > (one, X_dev, alpha, p, h, N);

		//8. If h is accurate enough then xi = h and quit

		//9. s = ri−1 − αvi
		cuComplex nalpha = make_cuComplex(-alpha.x, -alpha.y);
		vecCaddKernel << <nblocks, MAXTHREADS >> > (one, r, nalpha, v, s, N);

		//10. z = K−1s
		//K=LU , s = L temp, temp = Uz
		//10.1 temp = L-1 s
		cudaStatus = cudaMemcpy(s_host, s, N * sizeof(cuComplex), cudaMemcpyDeviceToHost);
		memset(temp_host, 0, sizeof(temp_host[0]) * N);
		LowerTriSolver(A_ilu_host, s_host, temp_host, RowPtr_host, ColIdx_host, N, nnz);
		//10.2 z = U-1 temp
		UpperTriSolver(A_ilu_host, temp_host, z_host, RowPtr_host, ColIdx_host, N, nnz);
		cudaStatus = cudaMemcpy(z, z_host, N * sizeof(cuComplex), cudaMemcpyHostToDevice);

		//11. t = Az
		mvMulKernel << <nblocks, MAXTHREADS >> > (RowPtr_dev, ColIdx_dev, A_dev, z, t, N, nnz);

		//12. ωi = (K −1t, K −1s) / (K −1t, K −1t)
		dotcKernel << <nblocks, MAXTHREADS >> > (t, t, temp, N);
		complex<float> tt_host = CudaSum(temp, N);
		cudaMemset((void *)temp, 0, sizeof(temp[0]) * N);
		if (tt_host.real() > 0)
		{
			dotcKernel << <nblocks, MAXTHREADS >> > (t, s, temp, N);
			complex<float> ts_host = CudaSum(temp, N);
			cudaMemset((void *)temp, 0, sizeof(temp[0]) * N);
			complex<float> w_host = ts_host / tt_host;
			w = make_cuComplex(w_host.real(), w_host.imag());
		}
		else
			w = zero;

		//13. h = xi−1 + αy, xi = h + ωiz
		vecCaddKernel << <nblocks, MAXTHREADS >> > (one, X_dev, alpha, y, h, N);
		vecCaddKernel << <nblocks, MAXTHREADS >> > (one, h, w, z, X_dev, N);

		//14. If xi is accurate enough, then quit

		//15. ri = s − ωit
		//cout << "BICGStab @ step 13" << endl;
		nw = make_cuComplex(-w.x, -w.y);
		vecCaddKernel << <nblocks, MAXTHREADS >> > (one, s, nw, t, r, N);

		nrmr = CudaSquaredNorm(r, N);
		err = sqrt(nrmr / nrmr0);
		//err = nrmr / nrmr0;
		cout << "BICGStab @ " << interation;
		cout.setf(ios::scientific, ios::floatfield);
		cout << ", Residual = " << err << endl;
		file << "BICGStab @ " << interation;
		file.setf(ios::scientific, ios::floatfield);
		file << ", Residual = " << err << endl;
		residuals.push_back(err);
		interation++;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(x_host, X_dev, N * sizeof(cuComplex), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	tt = time(NULL);
	current_time = localtime(&tt);
	file << "current time is " << current_time->tm_hour << ":" << current_time->tm_min << ":" << current_time->tm_sec;
	file.close();

	//cudaEventDestroy(start);
	//cudaEventDestroy(stop);
Error:
	if (p_host) delete[]p_host;
	if (y_host) delete[]y_host;
	if (s_host) delete[]s_host;
	if (z_host) delete[]z_host;
	if (temp_host) delete[]temp_host;

	if (RowPtr_dev) cudaFree(RowPtr_dev);
	if (ColIdx_dev) cudaFree(ColIdx_dev);
	if (A_dev) cudaFree(A_dev);
	if (iluA_dev) cudaFree(iluA_dev);
	if (X_dev) cudaFree(X_dev);
	if (b_dev) cudaFree(b_dev);

	if (r) cudaFree(r);
	if (rh) cudaFree(rh);
	if (v) cudaFree(v);
	if (p) cudaFree(p);
	if (y) cudaFree(y);
	if (temp) cudaFree(temp);
	if (z) cudaFree(z);
	if (h) cudaFree(h);
	if (s) cudaFree(s);
	if (t) cudaFree(t);

	return cudaStatus;
}



}

