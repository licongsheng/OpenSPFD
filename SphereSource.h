/*
 * SphereSource.h
 *
 *  Created on: 2018年5月24日
 *      Author: root
 */

#ifndef SPHERESOURCE_H_
#define SPHERESOURCE_H_

#include <complex>
using namespace std;

class SphereSource {
public:
	SphereSource();
	virtual ~SphereSource();
	int *CreateSphere(float r,float spacing,int *dim);
	complex<float> *CreatePhisicalSphere(float r, float Extent[6],float spacing,float sigma,float epsilon,float frequency,int *dim);
	void SaveSphere(const char *filename);
	void SavePhicalSphere(const char *filename);

private:
	complex<float> *cEps;
	int dims[3];
	int *vol;

private:


};

#endif /* SPHERESOURCE_H_ */
