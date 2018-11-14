#include "HeadModelReader.h"
#include <cstring>
#include "stdio.h"
#include "matio.h"
#include "common.h"

using namespace std;

HeadModelReader::HeadModelReader()
{
	memset((char*)mat_file,0,500*sizeof(char));
	voxel = NULL;
	Dimension[0] = 0; Dimension[1] = 0; Dimension[2] = 0;
	Spacing[0] = 0; Spacing[1] = 0; Spacing[2] = 0;
	Solids.clear();
}


HeadModelReader::~HeadModelReader()
{
	if (voxel) delete[]voxel;
	voxel = NULL;
	Dimension[0] = 0; Dimension[1] = 0; Dimension[2] = 0;
	Spacing[0] = 0; Spacing[1] = 0; Spacing[2] = 0;
	Solids.clear();
}

void HeadModelReader::SetFilename(const char *matfile)
{
	strcpy((char*)mat_file,matfile);
}

void HeadModelReader::read()
{
	printf("Read solid from %s\n", (char*)mat_file);
	mat_t *matfp = NULL;
	matfp = Mat_Open((char*)mat_file, MAT_ACC_RDONLY);
	matvar_t *mat_ima = Mat_VarRead(matfp, "ima");
	Dimension[0] = (int)mat_ima->dims[0];
	Dimension[1] = (int)mat_ima->dims[1];
	Dimension[2] = (int)mat_ima->dims[2];
	long N = Dimension[0] * Dimension[1] * Dimension[2];
	voxel = new unsigned char[N];
	memcpy((unsigned char *)voxel, (unsigned char *)mat_ima->data, mat_ima->nbytes);
	
	matvar_t *mat_spacing = Mat_VarRead(matfp, "spacing");
	double delt[3] = { 0 };
	memcpy((double *)delt, (double *)mat_spacing->data, mat_spacing->nbytes);
	Spacing[0] = delt[0];
	Spacing[1] = delt[1];
	Spacing[2] = delt[2];

	matvar_t *mat_solids = Mat_VarRead(matfp, "Tissues");
	for (int i = 0; i < mat_solids->dims[0]; i++)
	{
		Tissue organ;
		strcpy((char *)organ.Name, "\0");
		matvar_t *Name = Mat_VarGetStructFieldByName(mat_solids, "Name", i);
		strcpy((char *)organ.Name, (char*)Name->data);
		matvar_t *Level = Mat_VarGetStructFieldByName(mat_solids, "Level", i);
		double level = 0;
		memcpy((double *)&level, (double*)Level->data, Level->nbytes);
		organ.Level = (unsigned char)level;
		matvar_t *sigma = Mat_VarGetStructFieldByName(mat_solids, "Conductivity", i);
		memcpy((double *)&organ.Conductivity, (double*)sigma->data, sigma->nbytes);
		matvar_t *mu = Mat_VarGetStructFieldByName(mat_solids, "Permitivity",i);
		memcpy((double *)&organ.Permitivity, (double*)mu->data, mu->nbytes);
		matvar_t *rho = Mat_VarGetStructFieldByName(mat_solids, "Density", i);
		memcpy((double *)&organ.Density, (double*)rho->data, rho->nbytes);
		Solids.push_back(organ);
		printf("%40s  %4d     %4.3f     %8.2f      %5.1f\n",
			(char *)organ.Name, (int)organ.Level, (float)organ.Conductivity,
			(float)organ.Permitivity, (float)organ.Density);
	}

	Mat_Close(matfp);
}

unsigned char *HeadModelReader::GetHeadModel(int *dims, float *spacing, vector<Tissue> &solids)
{
	dims[0] = Dimension[0];
	dims[1] = Dimension[1];
	dims[2] = Dimension[2];
	spacing[0] = Spacing[0];
	spacing[1] = Spacing[1];
	spacing[2] = Spacing[2];
	solids = Solids;
	return voxel;
}
