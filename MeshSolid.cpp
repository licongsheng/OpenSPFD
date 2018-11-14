#include "MeshSolid.h"
#include "stdlib.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include "matio.h"
#include "math.h"
#include "common.h"

using namespace std;


MeshSolid::MeshSolid()
{
	Volume = NULL;
	Model = NULL;
	cEps = NULL;
}


MeshSolid::~MeshSolid()
{
	if (Model) delete[]Model;
	if (Volume) delete[]Volume;
	if (cEps) delete[]cEps;
}

void MeshSolid::SetComputeVolume(float extent[6], float spacing[3], float f)
{
	frequency = f;
	for (int i = 0; i < 6; i++)
		Extent[i] = extent[i];
	for (int i = 0; i < 3; i++)
		Spacing[i] = spacing[i];
}

void MeshSolid::SetHeadModel(unsigned char *vol, int dims[3], float spacing[3], vector<Tissue> solids, Point3 c)
{
	Model = vol;
	for (int i = 0; i < 3; i++)
	{
		solid_spacing[i] = spacing[i];
		solid_dims[i] = dims[i];
	}
	tissues = solids;
	center = c;
}

void MeshSolid::Mesh()
{
	complex<float> Air = complex<float>(0.0, 2 * M_PI*frequency*EPSILON);
	Dims[0] = ceil((Extent[1] - Extent[0]) / Spacing[0]);
	Dims[1] = ceil((Extent[3] - Extent[2]) / Spacing[1]);
	Dims[2] = ceil((Extent[5] - Extent[4]) / Spacing[2]);
	long N = Dims[0] * Dims[1] * Dims[2];
	cEps = new complex<float>[N];
	Volume = new unsigned char[N];
	float head_extent[6] =
	{ -(float)solid_dims[0]*solid_spacing[0] / 2.0 + center.x,(float)solid_dims[0] * solid_spacing[0] / 2.0 + center.x,
		-(float)solid_dims[1] * solid_spacing[1] / 2.0 + center.y,(float)solid_dims[1] * solid_spacing[1] / 2.0 + center.y,
		-(float)solid_dims[2] * solid_spacing[2] / 2.0 + center.z,(float)solid_dims[2] * solid_spacing[2] / 2.0 + center.z };

	float offset_x = head_extent[0] - Extent[0];
	float offset_y = head_extent[2] - Extent[2];
	float offset_z = head_extent[4] - Extent[4];

	for (int k = 0; k < Dims[2]; k++)
	{
		printf("Mesh process @ layer %d/%d\n", k, Dims[2]);
		for (int j = 0; j < Dims[1]; j++)
		{
			for (int i = 0; i < Dims[0]; i++)
			{
				long idx = k*Dims[0] * Dims[1] + j*Dims[0] + i;
				float x = Extent[0] + i*Spacing[0];
				float y = Extent[2] + j*Spacing[1];
				float z = Extent[4] + k*Spacing[2];
				if (x >= head_extent[0] && x < head_extent[1] &&
					y >= head_extent[2] && y < head_extent[3] &&
					z >= head_extent[4] && z < head_extent[5])
				{
					int ix = (int)((x - head_extent[0]) / solid_spacing[0]);
					int iy = (int)((y - head_extent[2]) / solid_spacing[1]);
					int iz = (int)((z - head_extent[4]) / solid_spacing[2]);
					unsigned char level = Model[iz*solid_dims[0] * solid_dims[1] + iy*solid_dims[0] + ix];

					if (level != 0)
					{
						Volume[idx] = level;
						cEps[idx] = complexPermitivity(level);
					}
					else
					{
						cEps[idx] = Air;
						Volume[idx] = 0;
					}
				}
				else
				{
					cEps[idx] = Air;
					Volume[idx] = 0;
				}
			}
		}
	}


}

complex<float> *MeshSolid::GetPhiscalModel(int *dims)
{
	for (int i = 0; i < 3; i++)
		 dims[i] = Dims[i];
	return cEps;
}

void MeshSolid::SavePhicalModel(const char *filename)
{
	mat_complex_split_t S_mat = { NULL,NULL };
	float *f_real, *f_imag;
	size_t dims[3] = { Dims[0],Dims[1],Dims[2] };
	mat_t *matfp = NULL;
	long N = Dims[0] * Dims[1] * Dims[2];
	f_real = new float[N];
	f_imag = new float[N];

	matfp = Mat_CreateVer(filename, NULL, MAT_FT_MAT5);

	for (int i = 0; i < N; i++)
	{
		f_real[i] = cEps[i].real();
		f_imag[i] = cEps[i].imag();
	}
	S_mat.Re = f_real;
	S_mat.Im = f_imag;
	matvar_t *parentx = Mat_VarCreate("ceps", MAT_C_SINGLE, MAT_T_SINGLE, 3, dims, &S_mat, MAT_F_COMPLEX);
	Mat_VarWrite(matfp, parentx, MAT_COMPRESSION_NONE);
	Mat_VarFree(parentx);
	Mat_Close(matfp);
}

void MeshSolid::SaveModel(const char *filename)
{
	int i, j, k;
	double *Data = NULL;
	mat_t *matfp = NULL;
	size_t dims[3] = {Dims[0],Dims[1],Dims[2] };
	int N = dims[0] * dims[1] * dims[2];
	matfp = Mat_CreateVer(filename, NULL, MAT_FT_MAT5);

	matvar_t *model = Mat_VarCreate("model", MAT_C_UINT8, MAT_T_UINT8, 3, dims, Volume, MAT_F_DONT_COPY_DATA);
	int flag = Mat_VarWrite(matfp, model, MAT_COMPRESSION_NONE);
	Mat_VarFree(model);

	Mat_Close(matfp);
}


complex<float> MeshSolid::complexPermitivity(unsigned char level)
{
	complex<float> ceps = complex<float>(0.0, 2 * M_PI*frequency*EPSILON);

	if (level == 0)
		return ceps;

	int N = tissues.size();
	for(int i=0;i<N;i++)
		if (tissues[i].Level == level)
		{
			ceps = complex<float>(tissues[i].Conductivity, tissues[i].Permitivity * 2 * M_PI*frequency*EPSILON);
			return ceps;
		}
	return ceps;
}