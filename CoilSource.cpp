/*
 * CoilSource.cpp
 *
 *  Created on: 2018年5月24日
 *      Author: root
 */

#include "CoilSource.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <vector>
#include "common.h"
#include "math.h"
using namespace std;


CoilSource::CoilSource() {
	// TODO Auto-generated constructor stub
	coil.clear();
}

CoilSource::~CoilSource() {
	// TODO Auto-generated destructor stub
	coil.clear();
}

vector<Point3> CoilSource::CircleCoil(float radius,Point3 center)
{
	coil.clear();
	float dx = M_PI/64.0;
	for(int i=0;i<128;i++)
	{
		Point3 p;
		p.x = center.x + radius*cos(i*dx);
		p.y = center.y + radius*sin(i*dx);
		p.z = center.z;
		coil.push_back(p);
	}
	Point3 p;
	p.x = center.x + radius*cos(0.0);
	p.y = center.y + radius*sin(0.0);
	p.z = center.z;
	coil.push_back(p);
	return coil;
}


vector<Point3> CoilSource::EightCoil(float radius,Point3 center)
{
	coil.clear();
	float dx = M_PI/64.0;
	for(int i=0;i<128;i++)
	{
		Point3 p;
		p.x = center.x + radius*cos(i*dx) - radius;
		p.y = center.y + radius*sin(i*dx);
		p.z = center.z;
		coil.push_back(p);
	}

	for(float theta = M_PI;theta >0; theta=theta-dx)
	{
		Point3 p;
		p.x = center.x + radius*cos(theta) + radius;
		p.y = center.y + radius*sin(theta);
		p.z = center.z;
		coil.push_back(p);
	}

	for(float theta = 2*M_PI;theta >=M_PI; theta=theta-dx)
	{
		Point3 p;
		p.x = center.x + radius*cos(theta) + radius;
		p.y = center.y + radius*sin(theta);
		p.z = center.z;
		coil.push_back(p);
	}

	Point3 p;
	p.x = center.x + radius*cos(0.0) - radius;
	p.y = center.y + radius*sin(0.0);
	p.z = center.z;
	coil.push_back(p);

	return coil;
}

void CoilSource::SaveCoil(const char *filename)
{
	if(coil.size() < 1)
	{
		cout << "Coil data is empty, please call <CircleCoil or EightCoil> first." <<endl;
		return;
	}

	ofstream outfile(filename, ios::out);
	for(vector<Point3>::iterator it = coil.begin();it!=coil.end();it++)
	{
		Point3 p = *it;
		outfile<<p.x<<" "<<p.y<<" "<<p.z<<endl;
	}
	outfile.close();
	cout << "Save coil data to <" << filename << ">." <<endl;
}
