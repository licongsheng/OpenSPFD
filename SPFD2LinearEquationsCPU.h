/*
 * SPFD2LinearEquationsCPU.h
 *
 *  Created on: 2018年6月1日
 *      Author: root
 */

#ifndef SPFD2LINEAREQUATIONSCPU_H_
#define SPFD2LINEAREQUATIONSCPU_H_

#include <complex>
#include <vector>
#include "common.h"
using namespace std;


class SPFD2LinearEquations_CPU {
public:
	SPFD2LinearEquations_CPU();
	virtual ~SPFD2LinearEquations_CPU();
	
	void SetSphicalVolume(complex<float> *pv,int dim[3],float spacing[3]);
	void SetMagneticVector(float *Ax,float *Ay,float *Az,int dim[3],float fre);
	void Conveter();
	void GetLinearEquations(vector<int> &csr_Row,vector<int> &csr_Col,vector<complex<float> > &A,vector<complex<float> > &b);
	void GetRowCoo(vector<int> &coo_Row);
	void SaveSPFD(const char *filename);

private:
	float frequency;
	complex<float> *cEps;
	int dims[3];
	int solid_dims[3];
	float dx,dy,dz;
	float *Ax,*Ay,*Az;
	complex<float> *Sx,*Sy,*Sz,*SA;
	vector<int> ColsIndex,RowsIndex, RowsCoo;
	vector<complex<float> > Aval;
	vector<complex<float> > B;

private:
	Point3 idx2xyz(int i, int dim[3]);
	Point3 idx2xyz(int i, int nx,int ny,int nz);

};

#endif /* SPFD2LINEAREQUATIONSCPU_H_ */
