/*
 * AnalysisMagneticField.cpp
 *
 *  Created on: 2018年5月24日
 *      Author: root
 */

#include "AnalysisMagneticField.h"
#include <iostream>
#include <vector>
#include <string.h>
#include <fstream>
#include <cstring>
#include <vector>
#include "common.h"
#include "math.h"
#include "stdlib.h"
#include "matio.h"
#include <cuda_runtime.h>
using namespace std;


extern "C"
{
cudaError_t CUDAAnalysisMagneticField(float *Rx,float *Ry,float *Rz,float *dLx,float *dLy,float *dLz,int N_seg,
		float *x_axis,int nx,float *y_axis,int ny,float *z_aixs,int nz,float Amp,
		float *Ax,float *Ay,float *Az,float *Bx,float *By,float *Bz,int devid);
}



AnalysisMagneticField::AnalysisMagneticField() {
	// TODO Auto-generated constructor stub
	current_amp = 0.0;
	Ax = NULL;
	Ay = NULL;
	Az = NULL;
	Bx = NULL;
	By = NULL;
	Bz = NULL;
	x_axis = NULL;
	y_axis = NULL;
	z_axis = NULL;
	dims[0] = 0;
	dims[1] = 0;
	dims[2] = 0;
	coil_pts.clear();
}

AnalysisMagneticField::~AnalysisMagneticField() {
	// TODO Auto-generated destructor stub
	if(Ax) delete []Ax;
	if(Ay) delete []Ay;
	if(Az) delete []Az;
	if(Bx) delete []Bx;
	if(By) delete []By;
	if(Bz) delete []Bz;
	coil_pts.clear();
}

void AnalysisMagneticField::SetCoil(vector<Point3> pts,float amp)
{
	coil_pts = pts;
	current_amp = amp;
}

void AnalysisMagneticField::SetComputationVolume(float extent[6],float spacing[3])
{
	dims[0] = (int)ceil((extent[1] - extent[0])/spacing[0]) + 1;
	dims[1] = (int)ceil((extent[3] - extent[2])/spacing[1]) + 1;
	dims[2] = (int)ceil((extent[5] - extent[4])/spacing[2]) + 1;

	if(x_axis) delete []x_axis;x_axis = new float[dims[0]];
	if(y_axis) delete []y_axis;y_axis = new float[dims[1]];
	if(z_axis) delete []z_axis;z_axis = new float[dims[2]];

	for(int i=0;i<dims[0];i++)
		x_axis[i] = (extent[0] + i*spacing[0])/1000.0;

	for(int i=0;i<dims[1];i++)
		y_axis[i] = (extent[2] + i*spacing[1])/1000.0;

	for(int i=0;i<dims[2];i++)
		z_axis[i] = (extent[4] + i*spacing[2])/1000.0;
}

void AnalysisMagneticField::Analysis()
{
	float *Rx, *Ry, *Rz;
	float *dLx, *dLy, *dLz;
	int idx = 0;
	int N_seg = coil_pts.size() - 1;

	Rx = new float[N_seg];
	Ry = new float[N_seg];
	Rz = new float[N_seg];

	dLx = new float[N_seg];
	dLy = new float[N_seg];
	dLz = new float[N_seg];

	for(int i=0;i<N_seg;i++)
	{
		Point3 p1 = coil_pts[i];
		Point3 p2 = coil_pts[i+1];
		Rx[i] = 0.5*(p1.x + p2.x) / 1000.0;
		Ry[i] = 0.5*(p1.y + p2.y) / 1000.0;
		Rz[i] = 0.5*(p1.z + p2.z) / 1000.0;

		dLx[i] = (p2.x - p1.x) / 1000.0;
		dLy[i] = (p2.y - p1.y) / 1000.0;
		dLz[i] = (p2.z - p1.z) / 1000.0;
	}
	
	long N = dims[0]*dims[1]*dims[2];

	Ax = new float[N];
	memset(Ax,0,N*sizeof(float));
	Ay = new float[N];
	memset(Ay,0,N*sizeof(float));
	Az = new float[N];
	memset(Az,0,N*sizeof(float));

	Bx = new float[N];
	memset(Bx,0,N*sizeof(float));
	By = new float[N];
	memset(By,0,N*sizeof(float));
	Bz = new float[N];
	memset(Bz,0,N*sizeof(float));

	CUDAAnalysisMagneticField(Rx,Ry,Rz,dLx,dLy,dLz,N_seg,
			x_axis,dims[0],y_axis,dims[1],z_axis,dims[2],current_amp, Ax, Ay, Az, Bx,By,Bz,1);

	if(x_axis) delete []x_axis;x_axis = NULL;
	if(y_axis) delete []y_axis;y_axis = NULL;
	if(z_axis) delete []z_axis;z_axis = NULL;

	delete []Rx;
	delete []Ry;
	delete []Rz;

	delete []dLx;
	delete []dLy;
	delete []dLz;
}

float *AnalysisMagneticField::GetAxField(int *dim)
{
	dim[0] = dims[0];
	dim[1] = dims[1];
	dim[2] = dims[2];
	return Ax;
}

float *AnalysisMagneticField::GetAyField(int *dim)
{
	dim[0] = dims[0];
	dim[1] = dims[1];
	dim[2] = dims[2];
	return Ay;
}

float *AnalysisMagneticField::GetAzField(int *dim)
{
	dim[0] = dims[0];
	dim[1] = dims[1];
	dim[2] = dims[2];
	return Az;
}

float *AnalysisMagneticField::GetBxField(int *dim)
{
	dim[0] = dims[0];
	dim[1] = dims[1];
	dim[2] = dims[2];
	return Bx;
}

float *AnalysisMagneticField::GetByField(int *dim)
{
	dim[0] = dims[0];
	dim[1] = dims[1];
	dim[2] = dims[2];
	return By;
}

float *AnalysisMagneticField::GetBzField(int *dim)
{
	dim[0] = dims[0];
	dim[1] = dims[1];
	dim[2] = dims[2];
	return Bz;
}

void AnalysisMagneticField::SaveAField(const char *filename)
{
	size_t Dims[3] = { dims[0],dims[1],dims[2] };
	int N = 0;
	mat_t *matfp = NULL;

	matfp = Mat_CreateVer(filename, NULL, MAT_FT_MAT5);
	matvar_t *parentx = Mat_VarCreate("Ax", MAT_C_SINGLE, MAT_T_SINGLE, 3, Dims, Ax, MAT_F_DONT_COPY_DATA);
	Mat_VarWrite(matfp, parentx, MAT_COMPRESSION_NONE);
	Mat_VarFree(parentx);

	// field y
	matvar_t *parenty = Mat_VarCreate("Ay", MAT_C_SINGLE, MAT_T_SINGLE, 3,Dims, Ay, MAT_F_DONT_COPY_DATA);
	Mat_VarWrite(matfp, parenty, MAT_COMPRESSION_NONE);
	Mat_VarFree(parenty);
	// field z
	matvar_t *parentz = Mat_VarCreate("Az", MAT_C_SINGLE, MAT_T_SINGLE, 3,Dims,Az, MAT_F_DONT_COPY_DATA);
	Mat_VarWrite(matfp, parentz, MAT_COMPRESSION_NONE);
	Mat_VarFree(parentz);
	Mat_Close(matfp);
}

void AnalysisMagneticField::SaveBField(const char *filename)
{
	size_t Dims[3] = { dims[0],dims[1],dims[2] };
	int N = 0;
	mat_t *matfp = NULL;

	matfp = Mat_CreateVer(filename, NULL, MAT_FT_MAT5);
	matvar_t *parentx = Mat_VarCreate("Bx", MAT_C_SINGLE, MAT_T_SINGLE, 3, Dims, Bx, MAT_F_DONT_COPY_DATA);
	Mat_VarWrite(matfp, parentx, MAT_COMPRESSION_NONE);
	Mat_VarFree(parentx);

	// field y
	matvar_t *parenty = Mat_VarCreate("By", MAT_C_SINGLE, MAT_T_SINGLE, 3,Dims, By, MAT_F_DONT_COPY_DATA);
	Mat_VarWrite(matfp, parenty, MAT_COMPRESSION_NONE);
	Mat_VarFree(parenty);
	// field z
	matvar_t *parentz = Mat_VarCreate("Bz", MAT_C_SINGLE, MAT_T_SINGLE, 3,Dims,Bz, MAT_F_DONT_COPY_DATA);
	Mat_VarWrite(matfp, parentz, MAT_COMPRESSION_NONE);
	Mat_VarFree(parentz);
	Mat_Close(matfp);
}

float ***AnalysisMagneticField::malloc3Df(int nx,int ny,int nz)
{
	float ***tab = NULL;
	tab = (float***)malloc(nx*sizeof(float**));
	for(int i=0;i<nx;i++)
	{
		tab[i] = (float**)malloc(ny*sizeof(float*));
		for(int j=0;i<ny;j++)
		{
			tab[i][j] = (float*)malloc(nz*sizeof(float));
			for(int k=0;k<nz;k++)
				tab[i][j][k] = 0;
		}
	}
	return tab;
}

float ***AnalysisMagneticField::free3Df(float ***tab,int nx,int ny,int)
{
	for(int i=0;i<nx;i++)
	{
		for(int j=0;i<ny;j++)
			free(tab[i][j]);
		free(tab[i]);
	}
	free(tab);
	return NULL;
}
