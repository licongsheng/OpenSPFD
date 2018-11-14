/*
 * CoilSource.h
 *
 *  Created on: 2018年5月24日
 *      Author: root
 */

#ifndef COILSOURCE_H_
#define COILSOURCE_H_

#include <vector>
#include "common.h"

using namespace std;

class CoilSource {
public:
	CoilSource();
	virtual ~CoilSource();

	vector<Point3> CircleCoil(float radius,Point3 center);
	vector<Point3> EightCoil(float radius,Point3 center);
	void SaveCoil(const char *filename);

private:
	vector<Point3> coil;
};

#endif /* COILSOURCE_H_ */
