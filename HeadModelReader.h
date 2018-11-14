#pragma once

#include <vector>
#include "common.h"

using namespace std;

class HeadModelReader
{
public:
	HeadModelReader();
	virtual ~HeadModelReader();

	void SetFilename(const char *matfile);
	void read();
	unsigned char *GetHeadModel(int *dims,float *spacing, vector<Tissue> &solids);

private:
	char mat_file[500];
	unsigned char *voxel;
	int Dimension[3];
	float Spacing[3];
	vector<Tissue> Solids;

};

