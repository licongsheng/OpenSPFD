/*
 * SPFD2LinearEquations.cpp
 *
 *  Created on: 2018年5月28日
 *      Author: root
 */

#include "SPFD2LinearEquations.h"
#include <complex>
#include <vector>
#include <iostream>
#include <string.h>
#include "stdlib.h"
#include "matio.h"
#include <cuda_runtime.h>

using namespace std;

extern "C"
{
cudaError_t CUDAAnalysisSxyz(complex<float> *cEps,float *Ax,float *Ay,float *Az,float f,
		complex<float> *Sx,complex<float> *Sy,complex<float> *Sz,
		complex<float> *SA,int dims[3],float spacing[3],int devid);
}

SPFD2LinearEquations::SPFD2LinearEquations() {
	// TODO Auto-generated constructor stub
	cEps = NULL;
	Ax = NULL;
	Ay = NULL;
	Az = NULL;
	Sx = NULL;
	Sy = NULL;
	Sz = NULL;
	SA = NULL;
	dims[0] = 0;
	dims[1] = 0;
	dims[2] = 0;
	frequency = 1000;
	RowsIndex.clear();
	ColsIndex.clear();
	Aval.clear();
	B.clear();
	dx = 0;
	dy = 0;
	dz = 0;
}

SPFD2LinearEquations::~SPFD2LinearEquations() {
	// TODO Auto-generated destructor stub
	cEps = NULL;
	Ax = NULL;
	Ay = NULL;
	Az = NULL;
	dims[0] = 0;
	dims[1] = 0;
	dims[2] = 0;
	RowsIndex.clear();
	ColsIndex.clear();
	Aval.clear();
	B.clear();
}

void SPFD2LinearEquations::SetSphicalVolume(complex<float> *pv,int dim[3],float spacing[3])
{
	cEps = pv;
	dims[0] = dim[0];
	dims[1] = dim[1];
	dims[2] = dim[2];
	dx = spacing[0] / 1000.0;  //unit: mm -> m
	dy = spacing[1] / 1000.0;  //unit: mm -> m
	dz = spacing[2] / 1000.0;  //unit: mm -> m
}

void SPFD2LinearEquations::SetMagneticVector(float *ax,float *ay,float *az,int dim[3],float fre)
{
	Ax = ax;
	Ay = ay;
	Az = ax;
	frequency = fre;
	if(dims[0] != dim[0] || dims[1] != dim[1] || dims[2] != dim[2])
		cout << "Computation dimension ERROR." <<endl;
}

void SPFD2LinearEquations::Conveter()
{
	float spacing[3] = {dx,dy,dz};
	long N = dims[0]*dims[1]*dims[2];
	Sx = new complex<float>[N];
	memset(Sx,0,N*sizeof(complex<float>));
	Sy = new complex<float>[N];
	memset(Sy,0,N*sizeof(complex<float>));
	Sz = new complex<float>[N];
	memset(Sz,0,N*sizeof(complex<float>));
	SA = new complex<float>[N];
	memset(SA,0,N*sizeof(complex<float>));

	CUDAAnalysisSxyz(cEps,Ax,Ay,Az,frequency,Sx,Sy,Sz,SA,dims,spacing,1);

	cout<<"Constructed SPFD ..."<<endl;
	cout<<"Constructing linear system <Ax=b> ..."<<endl;
	/////////////////////////////////////////////////////////////////////////////
	ColsIndex.clear();
	Aval.clear();
	B.clear();
	int nx = dims[0];
	int ny = dims[1];
	int nz = dims[2];
	int N_layer = nx*ny;
	int LM = (nx - 2)*(ny - 2);
	int matrixM = LM*(nz - 2);
	int matrixN = matrixM;
	long idx_ = 0;
	long idx = 0;
	RowsIndex.clear();
	RowsIndex.push_back(0); // the first element in sparse matrix
	for (int k = 1; k<nz - 1; k++)
	{
		for (int j = 1; j<ny - 1; j++)
		{
			for (int i = 1; i<nx - 1; i++)
			{
					idx = (k - 1)*LM + (j - 1)*(nx - 2) + i - 1;
					idx_ = k*N_layer+j*nx+i;
					int current_row_num = 0;
					if (idx - LM >= 0)
					{
						ColsIndex.push_back(idx - LM);
						Aval.push_back(-Sz[idx_-N_layer]);
						current_row_num++;
					}

					if (idx - (nx - 2) >= 0)
					{
						ColsIndex.push_back(idx - (nx -2));
						Aval.push_back(-Sy[idx_- nx]);
						current_row_num++;
					}

					if (idx - 1 >= 0)
					{
						ColsIndex.push_back(idx - 1);
						Aval.push_back(-Sx[idx_-1]);
						current_row_num++;
					}

					complex<float> Sc = Sx[idx_-1] + Sx[idx_] + Sy[idx_-nx] + Sy[idx_] + Sz[idx_-N_layer] + Sz[idx_];
					ColsIndex.push_back(idx);
					Aval.push_back(Sc);
					current_row_num++;

					if (idx + 1 < matrixM)
					{
						ColsIndex.push_back(idx + 1);
						Aval.push_back(-Sx[idx_]);
						current_row_num++;
					}

					if (idx + (nx - 2) < matrixM)
					{
						ColsIndex.push_back(idx + (nx - 2));
						Aval.push_back(-Sy[idx_]);
						current_row_num++;
					}

					if (idx + LM < matrixM)
					{
						ColsIndex.push_back(idx + LM);
						Aval.push_back(-Sz[idx_]);
						current_row_num++;
					}
					std::vector<int>::iterator end = RowsIndex.end()-1;
					RowsIndex.push_back(*end + current_row_num);

					B.push_back(SA[idx_]);
				}
			}
		}


}

void SPFD2LinearEquations::GetLinearEquations(vector<int> &csr_Row,vector<int> &csr_Col,
		vector<complex<float> > &A,vector<complex<float> > &b)
{
	csr_Row = RowsIndex;
	csr_Col = ColsIndex;
	A = Aval;
	b = B;
}

void SPFD2LinearEquations::SaveSPFD(const char *filename)
{
	size_t Dims[3] = { dims[0],dims[1],dims[2] };
	int N = 0;
	mat_t *matfp = NULL;

	matfp = Mat_CreateVer(filename, NULL, MAT_FT_MAT5);
	matvar_t *parentx = Mat_VarCreate("Sx", MAT_C_SINGLE, MAT_T_SINGLE, 3, Dims, Sx, MAT_F_COMPLEX);
	Mat_VarWrite(matfp, parentx, MAT_COMPRESSION_NONE);
	Mat_VarFree(parentx);

	// field y
	matvar_t *parenty = Mat_VarCreate("Sy", MAT_C_SINGLE, MAT_T_SINGLE, 3,Dims, Sy, MAT_F_COMPLEX);
	Mat_VarWrite(matfp, parenty, MAT_COMPRESSION_NONE);
	Mat_VarFree(parenty);
	// field z
	matvar_t *parentz = Mat_VarCreate("Sz", MAT_C_SINGLE, MAT_T_SINGLE, 3,Dims,Sz, MAT_F_COMPLEX);
	Mat_VarWrite(matfp, parentz, MAT_COMPRESSION_NONE);
	Mat_VarFree(parentz);


	matvar_t *parenta = Mat_VarCreate("SA", MAT_C_SINGLE, MAT_T_SINGLE, 3,Dims,SA, MAT_F_COMPLEX);
	Mat_VarWrite(matfp, parenta, MAT_COMPRESSION_NONE);
	Mat_VarFree(parenta);
	Mat_Close(matfp);
}


Point3 SPFD2LinearEquations::idx2xyz(int i)
{
	Point3 p = {0,0,0};
	int n_layer = dims[0]*dims[1];
	p.z = (int)((float)i / (float)n_layer);
	int res = i - p.z*n_layer;
	p.y = (int)((float)res / (float)dims[0]);
	p.x = res - p.y*dims[0];
	return p;
}

