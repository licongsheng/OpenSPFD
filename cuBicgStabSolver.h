/*
 * cuBicgStabSolver.h
 *
 *  Created on: 2018年6月1日
 *      Author: root
 */

#ifndef CUBICGSTABSOLVER_H_
#define CUBICGSTABSOLVER_H_
#include <complex>
#include <vector>
#include "common.h"
#include "cublas_v2.h"
#include "cusparse_v2.h"
using namespace std;


class cuBicgStabSolver {
public:
	cuBicgStabSolver();
	virtual ~cuBicgStabSolver();
	void SetParameters(vector<int> csr_Row,vector<int> csr_Col,vector<complex<float> > A,vector<complex<float> > b,int dims[3]);
	void SetMagneticVector(float *Ax,float *Ay,float *Az,complex<float> *pv);
	void Run();


	void SaveElectricPotential(const char *filename);

private:
	vector<int> csr_Row;
	vector<int> csr_Col;
	vector<complex<float> > A;
	vector<complex<float> > b;
	int dims[3];
	complex<float> ***Voltage;
	int *h_csrCols;
	int *h_csrRows;
	complex<float> *h_csrA;
	complex<float> *h_b;
	complex<float> *h_x;
	complex<float> *A_ilu0;
	int *d_csrCols;
	int *d_csrRows;
	int *d_csrmCols;
	int *d_csrmRows;
	cublasHandle_t cublasHandle;
	cusparseHandle_t cusparseHandle;
	cusparseMatDescr_t descra;
	cuComplex *d_csrA;
	cusparseMatDescr_t descrm;
	cusparseMatDescr_t descrL;
	cusparseMatDescr_t descrU;
	cuComplex *d_csrM;
	cuComplex *d_b;
	cuComplex *d_x;

	cusparseSolveAnalysisInfo_t info_l;
	cusparseSolveAnalysisInfo_t info_u;
	int nnz;
	int matrixM;
	int matrixN;
	vector<float> residuals;

	float *Ax;
	float *Ay;
	float *Az;
	complex<float> *Ceps;


private:
	bool PrepareCoeffMatrix();
	void CopyDatafromHost2Device();
	void IncompleteLUT();
	void cuBicgStab();
	void CalculateEField();
	float ***malloc3F(int nx, int ny, int nz,float val);
	complex<float> ***malloc3C(int nx, int ny, int nz);
	complex<float> ***free3C(complex<float> ***tab, int nx, int ny, int nz);
	float ***free3F(float ***tab, int nx, int ny, int nz);
	void Save3DMatrixestoMat(const char *matfile, const char *fieldx, float ***ex, const char *fieldy, float ***ey, const char *fieldz, float ***ez, int d[3]);
	Point3 idx2xyz(int i);
};

#endif /* CUBICGSTABSOLVER_H_ */
