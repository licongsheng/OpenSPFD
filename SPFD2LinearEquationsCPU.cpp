/*
 * SPFD2LinearEquationsCPU.cpp
 *
 *  Created on: 2018年6月1日
 *      Author: root
 */

#include "SPFD2LinearEquationsCPU.h"
#include <complex>
#include <vector>
#include <iostream>
#include <string.h>
#include "stdlib.h"
#include "matio.h"
#include "common.h"


using namespace std;


SPFD2LinearEquations_CPU::SPFD2LinearEquations_CPU() {
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

SPFD2LinearEquations_CPU::~SPFD2LinearEquations_CPU() {
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


void SPFD2LinearEquations_CPU::SetSphicalVolume(complex<float> *pv,int dim[3],float spacing[3])
{
	cEps = pv;
	solid_dims[0] = dim[0];
	solid_dims[1] = dim[1];
	solid_dims[2] = dim[2];
	dx = spacing[0] / 1000.0;  //unit: mm -> m
	dy = spacing[1] / 1000.0;  //unit: mm -> m
	dz = spacing[2] / 1000.0;  //unit: mm -> m
}

void SPFD2LinearEquations_CPU::SetMagneticVector(float *ax,float *ay,float *az,int dim[3],float fre)
{
	Ax = ax;
	Ay = ay;
	Az = az;
	frequency = fre;
	dims[0] = dim[0];
	dims[1] = dim[1];
	dims[2] = dim[2];
}

void SPFD2LinearEquations_CPU::Conveter()
{
	float xy_z = dx*dy/dz;
	float yz_x = dy*dz/dx;
	float xz_y = dx*dz/dy;
	int x,y,z,i,idx;
	complex<float> Sx_n,Sx_p,Ax_n,Ax_p,Sy_n,Sy_p,Ay_n,Ay_p,Sz_n,Sz_p,Az_n,Az_p;
	complex<float> jw = complex<float>(0,2*M_PI*frequency);
	complex<float> Air = jw*(float)EPSILON;
	complex<float> Zero = complex<float>(0, 0);

	long N = dims[0] * dims[1] * dims[2];
	Sx = new complex<float>[N];
	memset(Sx,0, N *sizeof(complex<float>)); 
	Sy = new complex<float>[N];
	memset(Sy, 0, N * sizeof(complex<float>));
	Sz = new complex<float>[N];
	memset(Sz,0, N *sizeof(complex<float>));
	SA = new complex<float>[N];
	memset(SA,0, N *sizeof(complex<float>));
////////////////////////////////////////////////////////////////////////////////
	cout<<"Constructed SPFD ..."<<endl;

	for (i = 0; i < N; i++)
	{
		Sx[i] = Air*yz_x;
		Sy[i] = Air*xz_y;
		Sz[i] = Air*xy_z;
	}
	int nlayer = dims[0] * dims[1];
	int LM = solid_dims[1] * solid_dims[0];
	for (int k = 0; k < solid_dims[2] - 1; k++)
	{
		for (int j = 0; j < solid_dims[1] - 1; j++)
		{
			for (int i = 0; i < solid_dims[0] - 1; i++)
			{
				long idx = k*nlayer + j*dims[0] + i;
				Sx[idx] =
					(float) 0.25*yz_x*(cEps[k*LM+j*solid_dims[0] + i] + 
						cEps[k*LM + (j + 1)*solid_dims[0] + i] + 
						cEps[(k + 1)*LM + (j + 1)*solid_dims[0] + i] +
						cEps[(k + 1)*LM + j*solid_dims[0] + i]);

				Sy[idx] =
					(float) 0.25*yz_x*(cEps[k*LM + j*solid_dims[0] + i] +
						cEps[k*LM + j*solid_dims[0] + i + 1] +
						cEps[(k + 1)*LM + j*solid_dims[0] + i + 1] +
						cEps[(k + 1)*LM + j*solid_dims[0] + i]);

				Sz[idx] =
					(float) 0.25*xy_z*(cEps[k*LM + j*solid_dims[0] + i] +
						cEps[k*LM + j*solid_dims[0] + i + 1] +
						cEps[k*LM + (j + 1)*solid_dims[0] + i + 1] +
						cEps[k*LM + (j + 1)*solid_dims[0] + i]);
			}
		}
	}

////////////////////////////////////////////////////////////////////////////////

	for (int k = 0; k < dims[2]; k++)
	{
		for (int j = 0; j < dims[1]; j++)
		{
			for (int i = 0; i < dims[0]; i++)
			{
				idx = k*nlayer + j*dims[0] + i;
				if (i == 0)
				{
					Sx_n = Air;
					Sx_p = Sx[idx];
					Ax_n = Zero;
					Ax_p = Ax[idx + 1];
				}
				else if (i == dims[0] - 1)
				{
					Sx_n = Sx[idx - 1];
					Sx_p = Air;
					Ax_n = Ax[idx - 1];
					Ax_p = Zero;
				}
				else
				{
					Sx_n = Sx[idx - 1];
					Sx_p = Sx[idx];
					Ax_n = Ax[idx - 1];
					Ax_p = Ax[idx + 1];
				}

				if (j == 0)
				{
					Sy_n = Air;
					Sy_p = Sy[idx];
					Ay_n = Zero;
					Ay_p = Ay[idx + dims[0]];
				}
				else if (j == dims[1] - 1)
				{
					Sy_n = Sy[idx - dims[0]];
					Sy_p = Air;
					Ay_n = Ay[idx - dims[0]];
					Ay_p = Zero;
				}
				else
				{
					Sy_n = Sy[idx - dims[0]];
					Sy_p = Sy[idx];
					Ay_n = Ay[idx - dims[0]];
					Ay_p = Ay[idx + dims[0]];
				}

				if (k == 0)
				{
					Sz_n = Air;
					Sz_p = Sz[idx];
					Az_n = Zero;
					Az_p = Az[idx + nlayer];
				}
				else if (k == dims[2] - 1)
				{
					Sz_n = Sz[idx];
					Sz_p = Air;
					Az_n = Az[idx - nlayer];
					Az_p = Zero;
				}
				else
				{
					Sz_n = Sz[idx - nlayer];
					Sz_p = Sz[idx];
					Az_n = Az[idx - nlayer];
					Az_p = Az[idx + nlayer];
				}

				SA[idx] = jw*(Ax_p*Sx_p*dx - Ax_n*Sx_n*dx+
					Ay_p*Sy_p*dy - Ay_n*Sy_n*dy+
					Az_p*Sz_p*dz - Az_n*Sz_n*dz);
			}
		}
	}

	cout<<"Constructing linear system <Ax=b> ..."<<endl;
	/////////////////////////////////////////////////////////////////////////////
	ColsIndex.clear();
	Aval.clear();
	B.clear();
	int nx = dims[0];
	int ny = dims[1];
	int nz = dims[2];
	int N_layer = nx*ny;
	LM = (nx - 2)*(ny - 2);
	int matrixM = LM*(nz - 2);
	int matrixN = matrixM;
	long idx_ = 0;
	idx = 0;
	RowsIndex.clear(); 
	RowsCoo.clear();
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
						RowsCoo.push_back(idx);
						Aval.push_back(-Sz[idx_-N_layer]);
						current_row_num++;
					}

					if (idx - (nx - 2) >= 0)
					{
						ColsIndex.push_back(idx - (nx - 2));
						RowsCoo.push_back(idx);
						Aval.push_back(-Sy[idx_- nx]);
						current_row_num++;
					}

					if (idx - 1 >= 0)
					{
						ColsIndex.push_back(idx - 1);
						RowsCoo.push_back(idx);
						Aval.push_back(-Sx[idx_-1]);
						current_row_num++;
					}

					complex<float> Sc = Sx[idx_-1] + Sx[idx_] +
						Sy[idx_-nx] + Sy[idx_] +
						Sz[idx_-N_layer] + Sz[idx_];
					ColsIndex.push_back(idx);
					RowsCoo.push_back(idx);
					Aval.push_back(Sc);
					current_row_num++;

					if (idx + 1 < matrixM)
					{
						ColsIndex.push_back(idx + 1);
						RowsCoo.push_back(idx);
						Aval.push_back(-Sx[idx_]);
						current_row_num++;
					}

					if (idx + (nx - 2) < matrixM)
					{
						ColsIndex.push_back(idx + (nx - 2));
						RowsCoo.push_back(idx);
						Aval.push_back(-Sy[idx_]);
						current_row_num++;
					}

					if (idx + LM < matrixM)
					{
						ColsIndex.push_back(idx + LM);
						RowsCoo.push_back(idx);
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

void SPFD2LinearEquations_CPU::GetLinearEquations(vector<int> &csr_Row,vector<int> &csr_Col,
		vector<complex<float> > &A,vector<complex<float> > &b)
{
	csr_Row = RowsIndex;
	csr_Col = ColsIndex;
	A = Aval;
	b = B;
}

void SPFD2LinearEquations_CPU::GetRowCoo(vector<int> &coo_Row)
{
	coo_Row = RowsCoo;
}

void SPFD2LinearEquations_CPU::SaveSPFD(const char *filename)
{
	mat_complex_split_t Sx_mat = {NULL,NULL};
	mat_complex_split_t Sy_mat = {NULL,NULL};
	mat_complex_split_t Sz_mat = {NULL,NULL};
	mat_complex_split_t SA_mat = {NULL,NULL};
	float *f_real, *f_imag;
	size_t Dims[3] = { dims[0],dims[1],dims[2] };
	mat_t *matfp = NULL;
	long N = dims[0]*dims[1]*dims[2];
	f_real = new float[N];
	f_imag = new float[N];

	matfp = Mat_CreateVer(filename, NULL, MAT_FT_MAT5);

	for(int i=0;i<N;i++)
	{
		f_real[i] = Sx[i].real();
		f_imag[i] = Sx[i].imag();
	}
	Sx_mat.Re = f_real;
	Sx_mat.Im = f_imag;
	matvar_t *parentx = Mat_VarCreate("Sx", MAT_C_SINGLE, MAT_T_SINGLE, 3, Dims, &Sx_mat,MAT_F_COMPLEX);
	Mat_VarWrite(matfp, parentx, MAT_COMPRESSION_NONE);
	Mat_VarFree(parentx);

	// field y
	for(int i=0;i<N;i++)
	{
		f_real[i] = Sy[i].real();
		f_imag[i] = Sy[i].imag();
	}
	Sy_mat.Re = f_real;
	Sy_mat.Im = f_imag;
	matvar_t *parenty = Mat_VarCreate("Sy", MAT_C_SINGLE, MAT_T_SINGLE, 3,Dims, &Sy_mat, MAT_F_COMPLEX);
	Mat_VarWrite(matfp, parenty, MAT_COMPRESSION_NONE);
	Mat_VarFree(parenty);

	// field z
	for(int i=0;i<N;i++)
	{
		f_real[i] = Sz[i].real();
		f_imag[i] = Sz[i].imag();
	}
	Sz_mat.Re = f_real;
	Sz_mat.Im = f_imag;
	matvar_t *parentz = Mat_VarCreate("Sz", MAT_C_SINGLE, MAT_T_SINGLE, 3,Dims,&Sz_mat, MAT_F_COMPLEX);
	Mat_VarWrite(matfp, parentz, MAT_COMPRESSION_NONE);
	Mat_VarFree(parentz);

	// field SA
	for(int i=0;i<N;i++)
	{
		f_real[i] = SA[i].real();
		f_imag[i] = SA[i].imag();
	}
	SA_mat.Re = f_real;
	SA_mat.Im = f_imag;
	matvar_t *parenta = Mat_VarCreate("SA", MAT_C_SINGLE, MAT_T_SINGLE, 3,Dims,&SA_mat, MAT_F_COMPLEX);
	Mat_VarWrite(matfp, parenta, MAT_COMPRESSION_NONE);
	Mat_VarFree(parenta);

	Mat_Close(matfp);
}

Point3 SPFD2LinearEquations_CPU::idx2xyz(int i, int dim[3])
{
	Point3 p = {0,0,0};
	int n_layer = dim[0]* dim[1];
	p.z = (int)((float)i / (float)n_layer);
	int res = i - p.z*n_layer;
	p.y = (int)((float)res / (float)dim[0]);
	p.x = res - p.y*dim[0];
	return p;
}

Point3 SPFD2LinearEquations_CPU::idx2xyz(int i, int nx, int ny, int nz)
{
	Point3 p = { 0,0,0 };
	int n_layer = nx * ny;
	p.z = (int)((float)i / (float)n_layer);
	int res = i - p.z*n_layer;
	p.y = (int)((float)res / (float)nx);
	p.x = res - p.y*nx;
	return p;
}