/*
 * SPFD2LinearEquations.h
 *
 *  Created on: 2018年5月28日
 *      Author: root
 */

#ifndef SPFD2LINEAREQUATIONS_H_
#define SPFD2LINEAREQUATIONS_H_

#include <complex>
#include <vector>
#include "common.h"
using namespace std;

class SPFD2LinearEquations {
public:
	SPFD2LinearEquations();
	virtual ~SPFD2LinearEquations();

	void SetSphicalVolume(complex<float> *pv,int dim[3],float spacing[3]);
	void SetMagneticVector(float *Ax,float *Ay,float *Az,int dim[3],float fre);
	void Conveter();
	void GetLinearEquations(vector<int> &csr_Row,vector<int> &csr_Col,vector<complex<float> > &A,vector<complex<float> > &b);

	void SaveSPFD(const char *filename);

private:
	float frequency;
	complex<float> *cEps;
	int dims[3];
	float dx,dy,dz;
	float *Ax,*Ay,*Az;
	complex<float> *Sx,*Sy,*Sz,*SA;
	vector<int> ColsIndex,RowsIndex;
	vector<complex<float> > Aval;
	vector<complex<float> > B;

private:
	Point3 idx2xyz(int i);

};

#endif /* SPFD2LINEAREQUATIONS_H_ */
