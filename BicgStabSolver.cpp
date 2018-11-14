#include "BicgStabSolver.h"
#include <complex>
#include <vector>
#include "common.h"
#include <ctype.h>
#include <iostream>
#include <Eigen/Eigen>
#include <Eigen/Dense>  
#include <Eigen/IterativeLinearSolvers>
#include <time.h>
#include <ctime>
#include <matio.h>

using namespace std;
using namespace Eigen;
typedef Eigen::Triplet<complex<double> > Tri;


BICGStabSolver::BICGStabSolver() {
	// TODO Auto-generated constructor stub
	csr_Col.clear();
	A.clear();
	b.clear();
	h_b = NULL;
	h_x = NULL;
	nnz = 0;
	matrixM = 0;
	matrixN = 0;
	Ex_Field = NULL;
	Ey_Field = NULL;
	Ez_Field = NULL;
}

BICGStabSolver::~BICGStabSolver() {
	// TODO Auto-generated destructor stub
	csr_Col.clear();
	A.clear();
	b.clear();
	h_b = NULL;
	h_x = NULL;
	nnz = 0;
	matrixM = 0;
	matrixN = 0;
	if(Ex_Field) free3F(Ex_Field, dims[0], dims[1], dims[2]);
	if(Ey_Field) free3F(Ey_Field, dims[0], dims[1], dims[2]);
	if(Ez_Field) free3F(Ez_Field, dims[0], dims[1], dims[2]);
}

void BICGStabSolver::SetParameters(vector<int> coo_R, vector<int> csr_C, vector<complex<float> > _A, vector<complex<float> > _b, int _dims[3])
{
	coo_Row = coo_R;
	csr_Col = csr_C;
	A = _A;
	b = _b;
	dims[0] = _dims[0];
	dims[1] = _dims[1];
	dims[2] = _dims[2];
	matrixM = b.size();
	matrixN = matrixM;
	nnz = A.size();
}

void BICGStabSolver::SetMagneticVector(float *ax, float *ay, float *az, complex<float> *pv)
{
	Ax = ax;
	Ay = ay;
	Az = az;
	Ceps = pv;
}

void BICGStabSolver::Run()
{
	SparseMatrix<complex<double> > *M;
	BiCGSTAB<SparseMatrix<complex<double> >, Eigen::IncompleteLUT<complex<double> > > solver;//
	std::vector<Triplet<complex<double> > > triplets;
	M = new SparseMatrix<complex<double> >(matrixM, matrixM);
	VectorXcd x(matrixM), b(matrixM);
	for (int i = 0; i < matrixM; i++)
	{
		b[i] = this->b[i];
		x[i] = 0.0;
	}

	for (long i = 0; i < nnz; i++)
		triplets.push_back(Tri(this->coo_Row[i], this->csr_Col[i], this->A[i]));

	M->setFromTriplets(triplets.begin(), triplets.end());
	solver.setMaxIterations(200);
	solver.setTolerance(1e-8);
	solver.preconditioner().setDroptol(0.001);
	solver.compute(*M);

	iterations.clear();
	Residuals.clear();
	x = solver.solve(b);
	std::cout << "#iterations:     " << solver.iterations() << std::endl;
	std::cout << "estimated error: " << solver.error() << std::endl;

	h_x = new complex<float>[matrixM];
	for (int i = 0; i < matrixM; i++)
		h_x[i] = x[i];
	cout << "Calculate E Field ..." << endl;
	CalculateEField();

	delete[]h_b; h_b = NULL;
	delete(M);
}

void BICGStabSolver::CalculateEField()
{
	float f = 2240; //Hz
	float dx, dy, dz;
	int i, j, k;
	long idx = 0;
	long idx_ = 0;
	int nx, ny, nz;
	nx = dims[0];
	ny = dims[1];
	nz = dims[2];
	long n = (nx - 2)*(ny - 2)*(nz - 2);
	double w = 2 * M_PI*f;
	complex<float> jw(0, w);
	dx = 1.0 / 1000.0;
	dy = 1.0 / 1000.0;
	dz = 1.0 / 1000.0;

	Voltage = malloc3C(nx, ny, nz);
	int LM = (nx - 2)*(ny - 2);

	for (k = 0; k<nz - 2; k++)
	{
		for (j = 0; j<ny - 2; j++)
		{
			for (i = 0; i<nx - 2; i++)
			{
				idx = k*LM + j*(nx - 2) + i;
				Voltage[i + 1][j + 1][k + 1] = h_x[idx];
			}
		}
	}
	delete[]h_x; h_x = NULL;

	Ex_Field = malloc3F(nx, ny, nz, 0);
	Ey_Field = malloc3F(nx, ny, nz, 0);
	Ez_Field = malloc3F(nx, ny, nz, 0);
	complex<float> temp;
	for (k = 1; k < nz - 1; k++)
	{
		for (j = 1; j < ny - 1; j++)
		{
			for (i = 1; i < nx - 1; i++)
			{
				idx = k*nx*ny + j*nx + i;
				idx_ = k*(nx-1)*(ny-1) + j*(nx-1) + i;
				float sigma = Ceps[idx_].real();
				float sigma_i_1 = Ceps[idx_ - 1].real();
				float sigma_j_1 = Ceps[idx_ - nx-1].real();
				float sigma_k_1 = Ceps[idx_ - (nx - 1)*(ny - 1)].real();
				if (sigma > 1e-5 && sigma_i_1 > 1e-5 && sigma_j_1 > 1e-5 && sigma_k_1 > 1e-5)
				{
					float ax = Ax[idx];
					float ay = Ay[idx];
					float az = Az[idx];

					complex<float> Axo(0, w*ax);
					temp = (Voltage[i - 1][j][k] - Voltage[i][j][k]) / dx - Axo;
					Ex_Field[i][j][k] = sqrt(temp.real()*temp.real() + temp.imag()*temp.imag())*0.707;

					complex<float> Ayo(0, w*ay);
					temp = (Voltage[i][j - 1][k] - Voltage[i][j][k]) / dy - Ayo;
					Ey_Field[i][j][k] = sqrt(temp.real()*temp.real() + temp.imag()*temp.imag())*0.707;

					complex<float> Azo(0, w*az);
					temp = (Voltage[i][j][k - 1] - Voltage[i][j][k]) / dz - Azo;
					Ez_Field[i][j][k] = sqrt(temp.real()*temp.real() + temp.imag()*temp.imag())*0.707;
				}
			}
		}
	}

	Voltage = free3C(Voltage, nx, ny, nz);

}

void BICGStabSolver::Save3DMatrixestoMat(const char *matfile, const char *fieldx, float ***ex,
	const char *fieldy, float ***ey, const char *fieldz, float ***ez, int d[3])
{
	int i, j, k;
	double *Data = NULL;
	mat_t *matfp = NULL;
	size_t dims[3] = { d[0],d[1],d[2] };
	int N = dims[0] * dims[1] * dims[2];
	matfp = Mat_CreateVer(matfile, NULL, MAT_FT_MAT5);

	//cout<<"Write field data to disk ..."<< matfp <<endl;
	//cout<<"Ex  ..."<<endl;
	Data = new double[N];
	for (k = 0; k < dims[2]; k++)
		for (j = 0; j < dims[1]; j++)
			for (i = 0; i < dims[0]; i++)
				Data[k*dims[0] * dims[1] + j*dims[0] + i] = ex[i][j][k];
	matvar_t *parentx = Mat_VarCreate(fieldx, MAT_C_DOUBLE, MAT_T_DOUBLE, 3, dims, Data, MAT_F_DONT_COPY_DATA);
	int flag = Mat_VarWrite(matfp, parentx, MAT_COMPRESSION_NONE);
	Mat_VarFree(parentx); delete[]Data;

	//cout<<"Ey  ..."<<endl;
	Data = new double[N];
	for (k = 0; k < dims[2]; k++)
		for (j = 0; j < dims[1]; j++)
			for (i = 0; i < dims[0]; i++)
				Data[k*dims[0] * dims[1] + j*dims[0] + i] = ey[i][j][k];
	matvar_t *parenty = Mat_VarCreate(fieldy, MAT_C_DOUBLE, MAT_T_DOUBLE, 3, dims, Data, MAT_F_DONT_COPY_DATA);
	flag = Mat_VarWrite(matfp, parenty, MAT_COMPRESSION_NONE);
	Mat_VarFree(parenty); delete[]Data;

	//cout<<"Ez  ..."<<endl;
	Data = new double[N];
	for (k = 0; k < dims[2]; k++)
		for (j = 0; j < dims[1]; j++)
			for (i = 0; i < dims[0]; i++)
				Data[k*dims[0] * dims[1] + j*dims[0] + i] = ez[i][j][k];
	matvar_t *parentz = Mat_VarCreate(fieldz, MAT_C_DOUBLE, MAT_T_DOUBLE, 3, dims, Data, MAT_F_DONT_COPY_DATA);
	flag = Mat_VarWrite(matfp, parentz, MAT_COMPRESSION_NONE);
	Mat_VarFree(parentz); delete[]Data;

	Mat_Close(matfp);
}

void BICGStabSolver::SaveElectricField(const char *filename)
{
	Save3DMatrixestoMat("E_loop.mat", "Ex", Ex_Field, "Ey", Ey_Field, "Ez", Ez_Field, dims);
}

void BICGStabSolver::SaveElectricPotential(const char *filename)
{
	mat_complex_split_t Sx_mat = { NULL,NULL };
	mat_complex_split_t Sy_mat = { NULL,NULL };
	mat_complex_split_t Sz_mat = { NULL,NULL };
	mat_complex_split_t SA_mat = { NULL,NULL };
	float *f_real, *f_imag;
	size_t Dims[3] = { dims[0],dims[1],dims[2] };
	size_t Dims_sa[3] = { dims[0] - 2,dims[1] - 2,dims[2] - 2 };
	mat_t *matfp = NULL;
	long N = dims[0] * dims[1] * dims[2];
	f_real = new float[N];
	f_imag = new float[N];

	matfp = Mat_CreateVer(filename, NULL, MAT_FT_MAT5);

	for (int i = 0; i < N; i++)
	{
		Point3 p = idx2xyz(i);
		f_real[i] = Voltage[(int)p.x][(int)p.y][(int)p.z].real();
		f_imag[i] = Voltage[(int)p.x][(int)p.y][(int)p.z].imag();
	}
	Sx_mat.Re = f_real;
	Sx_mat.Im = f_imag;
	matvar_t *parentx = Mat_VarCreate("Voltage", MAT_C_SINGLE, MAT_T_SINGLE, 3, Dims, &Sx_mat, MAT_F_COMPLEX);
	Mat_VarWrite(matfp, parentx, MAT_COMPRESSION_NONE);
	Mat_VarFree(parentx);

	// field SA
	N = (dims[0] - 2) * (dims[1] - 2) * (dims[2] - 2);
	for (int i = 0; i < N; i++)
	{
		f_real[i] = b[i].real();
		f_imag[i] = b[i].imag();
	}
	SA_mat.Re = f_real;
	SA_mat.Im = f_imag;
	matvar_t *parenta = Mat_VarCreate("SA", MAT_C_SINGLE, MAT_T_SINGLE, 3, Dims_sa, &SA_mat, MAT_F_COMPLEX);
	Mat_VarWrite(matfp, parenta, MAT_COMPRESSION_NONE);
	Mat_VarFree(parenta);

	Mat_Close(matfp);
}

float ***BICGStabSolver::malloc3F(int nx, int ny, int nz, float val)
{
	float ***tab = NULL;
	tab = (float***)malloc(nx * sizeof(float **));
	for (int i = 0; i<nx; i++)
	{
		tab[i] = (float**)malloc(ny * sizeof(float *));
		for (int j = 0; j<ny; j++)
		{
			tab[i][j] = (float*)malloc(nz * sizeof(float));
			for (int k = 0; k<nz; k++)
				tab[i][j][k] = val;
		}
	}
	return tab;
}

complex<float> ***BICGStabSolver::malloc3C(int nx, int ny, int nz)
{
	complex<float> ***tab = NULL;
	complex<float> zeros(0.000000, 0);
	tab = (complex<float>***)malloc(nx * sizeof(complex<float> **));
	for (int i = 0; i<nx; i++)
	{
		tab[i] = (complex<float>**)malloc(ny * sizeof(complex<float> *));
		for (int j = 0; j<ny; j++)
		{
			tab[i][j] = (complex<float>*)malloc(nz * sizeof(complex<float>));
			for (int k = 0; k<nz; k++)
				tab[i][j][k] = zeros;
		}
	}
	return tab;
}

complex<float> ***BICGStabSolver::free3C(complex<float> ***tab, int nx, int ny, int nz)
{
	for (int i = 0; i<nx; i++)
	{
		for (int j = 0; j<ny; j++)
		{
			free(tab[i][j]);
		}
		free(tab[i]);
	}
	free(tab);
	return NULL;
}

float ***BICGStabSolver::free3F(float ***tab, int nx, int ny, int nz)
{
	for (int i = 0; i<nx; i++)
	{
		for (int j = 0; j<ny; j++)
		{
			free(tab[i][j]);
		}
		free(tab[i]);
	}
	free(tab);
	return NULL;
}

Point3 BICGStabSolver::idx2xyz(int i)
{
	Point3 p = { 0,0,0 };
	int n_layer = dims[0] * dims[1];
	p.z = (int)((float)i / (float)n_layer);
	int res = i - p.z*n_layer;
	p.y = (int)((float)res / (float)dims[0]);
	p.x = res - p.y*dims[0];
	return p;
}

