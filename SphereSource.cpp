/*
 * SphereSource.cpp
 *
 *  Created on: 2018年5月24日
 *      Author: root
 */

#include "SphereSource.h"
#include "stdlib.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include "matio.h"
#include "math.h"
#include "common.h"

using namespace std;

SphereSource::SphereSource() {
	// TODO Auto-generated constructor stub
	vol = NULL;
	cEps = NULL;
}

SphereSource::~SphereSource() {
	// TODO Auto-generated destructor stub
	if(vol) delete []vol;
	if(cEps) delete []cEps;
}

int *SphereSource::CreateSphere(float r,float spacing,int *dim)
{
	float pading = 10; //mm
	float cx = (r+pading)/spacing;
	float cy = (r+pading)/spacing;
	float cz = (r+pading)/spacing;
	float r2 = r*r;
	dims[0] = (int)ceil(2*(r + pading)/spacing);
	dims[1] = (int)ceil(2*(r + pading)/spacing);
	dims[2] = (int)ceil(2*(r + pading)/spacing);

	long N = dims[0]*dims[1]*dims[2];
	vol = new int[N];

	int i,j,k;
	for(k=0;k<dims[2];k++)
	{

		for(j=0;j<dims[1];j++)
		{
			for(i=0;i<dims[0];i++)
			{
					long idx = k*(dims[1]*dims[0])+j*dims[0]+i;
					float dx = i*spacing - cx;
					float dy = j*spacing - cy;
					float dz = k*spacing - cz;
					if(dx*dx+dy*dy+dz*dz <= r2)
						vol[idx] = 1;
					else
						vol[idx] = 0;
			}
		}
	}

	dim[0] = dims[0];
	dim[1] = dims[1];
	dim[2] = dims[2];
	return vol;
}


complex<float> *SphereSource::CreatePhisicalSphere(float r, float Extent[6],float spacing,float sigma,float epsilon,float frequency,int *dim)
{
	float cx = (Extent[0] + Extent[1])*0.5;
	float cy = (Extent[2] + Extent[3])*0.5;
	float cz = (Extent[4] + Extent[5])*0.5;
	float r2 = r*r;
	dims[0] = (int)ceil((Extent[1] - Extent[0]) / spacing);
	dims[1] = (int)ceil((Extent[3] - Extent[2]) / spacing);
	dims[2] = (int)ceil((Extent[5] - Extent[4]) / spacing);

	long N = dims[0]*dims[1]*dims[2];
	cEps = new complex<float>[N];

	int i,j,k;
	for(k=0;k<dims[2];k++)
	{
		for(j=0;j<dims[1];j++)
		{
			for(i=0;i<dims[0];i++)
			{
					long idx = k*(dims[1]*dims[0])+j*dims[0]+i;
					float dx = i*spacing - cx + Extent[0];
					float dy = j*spacing - cy + Extent[2];
					float dz = k*spacing - cz + Extent[4];
					if(dx*dx+dy*dy+dz*dz <= r2)
						cEps[idx] = complex<float>(sigma, 2*M_PI*frequency*epsilon*EPSILON);
					else
						cEps[idx] = complex<float>(0.0, 2*M_PI*frequency*EPSILON);
			}
		}
	}

	dim[0] = dims[0];
	dim[1] = dims[1];
	dim[2] = dims[2];

	return cEps;
}


void SphereSource::SaveSphere(const char *filename)
{
	long N = dims[0]*dims[1]*dims[2];
	if(vol == NULL)
	{
		cout << "Sphere volume data is empty, please call <CreateSphere> first." <<endl;
		return;
	}

	ofstream outfile(filename, ios::out);
	outfile << dims[0] << endl;
	outfile << dims[1] << endl;
	outfile << dims[2] << endl;
	for(long i=0;i<N;i++)
	{
		outfile << vol[i] << endl;
	}
	outfile.close();
	cout << "Save sphere volume to <" << filename << ">." <<endl;
}

void SphereSource::SavePhicalSphere(const char *filename)
{
	mat_complex_split_t Sx_mat = {NULL,NULL};
	float *f_real, *f_imag;
	size_t Dims[3] = { dims[0],dims[1],dims[2] };
	mat_t *matfp = NULL;
	long N = dims[0]*dims[1]*dims[2];
	f_real = new float[N];
	f_imag = new float[N];

	matfp = Mat_CreateVer(filename, NULL, MAT_FT_MAT5);

	for(int i=0;i<N;i++)
	{
		f_real[i] = cEps[i].real();
		f_imag[i] = cEps[i].imag();
	}
	Sx_mat.Re = f_real;
	Sx_mat.Im = f_imag;
	matvar_t *parentx = Mat_VarCreate("epsilon", MAT_C_SINGLE, MAT_T_SINGLE, 3, Dims, &Sx_mat,MAT_F_COMPLEX);
	Mat_VarWrite(matfp, parentx, MAT_COMPRESSION_NONE);
	Mat_VarFree(parentx);


	Mat_Close(matfp);
}
