/*
 * AnalysisMagneticField.h
 *
 *  Created on: 2018年5月24日
 *      Author: root
 */

#ifndef ANALYSISMAGNETICFIELD_H_
#define ANALYSISMAGNETICFIELD_H_

#include "common.h"
#include <vector>
using namespace std;


class AnalysisMagneticField {
public:
	AnalysisMagneticField();
	virtual ~AnalysisMagneticField();

	void SetCoil(vector<Point3> pts,float amp);
	void SetComputationVolume(float extent[6],float spacing[3]);
	void Analysis();
	float *GetAxField(int *dims);
	float *GetAyField(int *dims);
	float *GetAzField(int *dims);
	float *GetBxField(int *dims);
	float *GetByField(int *dims);
	float *GetBzField(int *dims);

	void SaveAField(const char *filename);
	void SaveBField(const char *filename);

private:
	float current_amp;
	vector<Point3> coil_pts;
	int dims[3];
	float *Ax,*Ay,*Az;
	float *Bx,*By,*Bz;
	float *x_axis,*y_axis,*z_axis;

private:
	float ***malloc3Df(int nx,int ny,int nz);
	float ***free3Df(float ***,int nx,int ny,int);
};

#endif /* ANALYSISMAGNETICFIELD_H_ */
