#pragma once

#include <vector>
#include <complex>
#include "common.h"

using namespace std;


class MeshSolid
{
public:
	MeshSolid();
	~MeshSolid();
	void SetComputeVolume(float extent[6],float spacing[3],float f);
	void SetHeadModel(unsigned char *vol, int dims[3], float spacing[3], vector<Tissue> solids, Point3 c);
	void Mesh();
	complex<float> *GetPhiscalModel(int *dims);
	void SavePhicalModel(const char *filename);
	void SaveModel(const char *filename);

private:
	complex<float> complexPermitivity(unsigned char level);

private:
	float frequency;
	complex<float> *cEps;
	unsigned char *Volume;
	int Dims[3];
	float Extent[6];
	float Spacing[3];
	Point3 center;
	int solid_dims[3];
	float solid_spacing[3];
	unsigned char *Model;
	vector<Tissue> tissues;
};

