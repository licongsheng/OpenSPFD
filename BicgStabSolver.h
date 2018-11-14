#pragma once

#include <complex>
#include <vector>
#include "common.h"
#include <Eigen/Eigen>
#include <Eigen/Dense>  
#include <Eigen/IterativeLinearSolvers>

using namespace std;
using namespace Eigen;
typedef Eigen::Triplet<complex<double> > Tri;

class BICGStabSolver
{
public:
	BICGStabSolver();
	~BICGStabSolver();
	void SetParameters(vector<int> coo_Row, vector<int> csr_Col, vector<complex<float> > A, vector<complex<float> > b, int dims[3]);
	void SetMagneticVector(float *Ax, float *Ay, float *Az, complex<float> *pv);
	void Run();
	void SaveElectricField(const char *filename);
	void SaveElectricPotential(const char *filename);

private:
	vector<int> coo_Row;
	vector<int> csr_Col;
	vector<complex<float> > A;
	vector<complex<float> > b;

	std::vector<int> iterations;
	std::vector<double> Residuals;
	int dims[3];
	complex<float> ***Voltage;
	float ***Ex_Field;
	float ***Ey_Field;
	float ***Ez_Field;
	complex<float> *h_b;
	complex<float> *h_x;
	
	int nnz;
	int matrixM;
	int matrixN;
	vector<float> residuals;

	float *Ax;
	float *Ay;
	float *Az;
	complex<float> *Ceps;

private:
	void CalculateEField();
	float ***malloc3F(int nx, int ny, int nz, float val);
	complex<float> ***malloc3C(int nx, int ny, int nz);
	complex<float> ***free3C(complex<float> ***tab, int nx, int ny, int nz);
	float ***free3F(float ***tab, int nx, int ny, int nz);
	void Save3DMatrixestoMat(const char *matfile, const char *fieldx, float ***ex, const char *fieldy, float ***ey, const char *fieldz, float ***ez, int d[3]);
	Point3 idx2xyz(int i);
};

