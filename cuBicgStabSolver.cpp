/*
 * cuBicgStabSolver.cpp
 *
 *  Created on: 2018年6月1日
 *      Author: root
 */

#include "cuBicgStabSolver.h"
#include <complex>
#include <vector>
#include "common.h"
#include <ctype.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <time.h>
#include <ctime>
#include <matio.h>

using namespace std;

extern "C"
{
	cudaError_t GPUPBICGSTAB(int *RowPtr_host, int *ColIdx_host, complex<float> *A_host, complex<float> *A_ilu_host, complex<float> *b_host, complex<float> *x_host, int N, long nnz, vector<float> residuals, int max_steps, float tol, int devid, const char *logfile);
}

cuBicgStabSolver::cuBicgStabSolver() {
	// TODO Auto-generated constructor stub
	csr_Row.clear();
	csr_Col.clear();
	A.clear();
	b.clear();
	h_csrCols = NULL;
	h_csrRows = NULL;
	h_csrA = NULL;
	h_b = NULL;
	h_x = NULL;
	d_csrCols = NULL;
	d_csrRows = NULL;
	d_csrA = NULL;
	d_b = NULL;
	d_x = NULL;
	cublasHandle = 0;
	cusparseHandle = 0;
	descra = 0;
	d_csrA = NULL;
	descrm = 0;
	d_csrM = NULL;
	d_b = NULL;
	d_x = NULL;
	nnz = 0;
	matrixM = 0;
	matrixN = 0;
	info_l = 0;
	info_u = 0;
	A_ilu0 = NULL;
}

cuBicgStabSolver::~cuBicgStabSolver() {
	// TODO Auto-generated destructor stub
	csr_Row.clear();
	csr_Col.clear();
	A.clear();
	b.clear();
	h_csrCols = NULL;
		h_csrRows = NULL;
		h_csrA = NULL;
		h_b = NULL;
		h_x = NULL;
		d_csrCols = NULL;
		d_csrRows = NULL;
		d_csrA = NULL;
		d_b = NULL;
		d_x = NULL;
		cublasHandle = 0;
		cusparseHandle = 0;
		descra = 0;
		d_csrA = NULL;
		descrm = 0;
		d_csrM = NULL;
		d_b = NULL;
		d_x = NULL;
		nnz = 0;
		matrixM = 0;
		matrixN = 0;
		info_l = 0;
		info_u = 0;
}

void cuBicgStabSolver::SetParameters(vector<int> csr_R,vector<int> csr_C,vector<complex<float> > _A,vector<complex<float> > _b,int _dims[3])
{
	csr_Row = csr_R;
	csr_Col = csr_C;
	A = _A;
	b = _b;
	dims[0] = _dims[0];
	dims[1] = _dims[1];
	dims[2] = _dims[2];
}

void cuBicgStabSolver::SetMagneticVector(float *ax,float *ay,float *az,complex<float> *pv)
{
	Ax = ax;
	Ay = ay;
	Az = az;
	Ceps = pv;
}

void cuBicgStabSolver::Run()
{
	cudaEvent_t start, stop;
	clock_t cup_start, cup_stop;
	cudaStream_t stream = 0;
	cusparseSolveAnalysisInfo_t infoa = 0;
	cudaDeviceProp deviceProp;
	cusparseStatus_t status1, status2, status3;

	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, 0));
	cout<<"GPU Device 0: "<<deviceProp.name<<" with compute capability "<<deviceProp.major<<"."<<deviceProp.minor<<endl;

	/* initialize cusparse*/
	status1 = cusparseCreate(&cusparseHandle);
	if (status1 != CUSPARSE_STATUS_SUCCESS) {
		if (cusparseHandle)   cusparseDestroy(cusparseHandle);
		return;
	}

	// initialize cublas
	if (cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS) {
		if (cublasHandle)     cublasDestroy(cublasHandle);
		return;
	}

	checkCudaErrors(cudaStreamCreate(&stream));
	if (cublasSetStream(cublasHandle, stream) != CUBLAS_STATUS_SUCCESS) {
		if (cublasHandle)     cublasDestroy(cublasHandle);
		if (cusparseHandle)   cusparseDestroy(cusparseHandle);
		return;
	}

	if (PrepareCoeffMatrix())
	{
		cout<<"Copy data from host to device ..."<<endl;
		CopyDatafromHost2Device();

		/* create and setup matrix descriptor */
		status1 = cusparseCreateMatDescr(&descra);
		if (status1 != CUSPARSE_STATUS_SUCCESS) {
			cout<<"Matrix descriptor initialization failed"<<endl;
			return;
		}
		cusparseSetMatType(descra, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(descra, CUSPARSE_INDEX_BASE_ZERO);

		//cup_start = clock();
		IncompleteLUT();
		//cup_stop = clock();
		//float costtime = cup_stop - cup_start;
		//printf("CUDA ilu spend %6.4f s.\n",costtime/1000.0);

		cout<<"BICGStab iterations ..."<<endl;
		cuBicgStab();
/*
		A_ilu0 = new complex<float>[nnz];
		checkCudaErrors(cudaMemcpy(A_ilu0, d_csrM, (size_t)(nnz * sizeof(A_ilu0[0])), cudaMemcpyDeviceToHost));
		GPUPBICGSTAB(h_csrRows, h_csrCols, h_csrA, A_ilu0, h_b, h_x, matrixM, nnz, residuals, MAXIterations, torence, 0, "prebicg.log");
		delete[]A_ilu0;
*/
		/* copy the result into host memory */
		checkCudaErrors(cudaMemcpy(h_x, d_x, (size_t)(matrixM * sizeof(h_x[0])), cudaMemcpyDeviceToHost));

		// free memories ...

		if (cublasHandle)      checkCudaErrors(cublasDestroy(cublasHandle));
		if (cusparseHandle)    checkCudaErrors(cusparseDestroy(cusparseHandle));
		if (d_x)    checkCudaErrors(cudaFree(d_x)); d_x = NULL;
		if (d_csrA)    checkCudaErrors(cudaFree(d_csrA)); d_csrA = NULL;
		if (d_csrM)    checkCudaErrors(cudaFree(d_csrM)); d_csrM = NULL;
		if (d_b)    checkCudaErrors(cudaFree(d_b)); d_b = NULL;
		if (d_csrCols)    checkCudaErrors(cudaFree(d_csrCols)); d_csrCols = NULL;
		if (d_csrRows)    checkCudaErrors(cudaFree(d_csrRows)); d_csrRows = NULL;
		if (info_l)  checkCudaErrors(cusparseDestroySolveAnalysisInfo(info_l)); info_l = 0;
		if (info_u)  checkCudaErrors(cusparseDestroySolveAnalysisInfo(info_u)); info_u = 0;
		cout<<"Calculate E Field ..."<<endl;
		CalculateEField();

		delete[]h_csrA; h_csrA = NULL;
		delete[]h_csrCols; h_csrCols = NULL;
		delete[]h_csrRows; h_csrRows = NULL;
		delete[]h_b; h_b = NULL;
	}
}

bool cuBicgStabSolver::PrepareCoeffMatrix()
{
	matrixM = b.size();
	matrixN = matrixM;
	nnz = A.size();
	h_csrCols = new int[nnz];
	h_csrRows = new int[csr_Row.size()];
	h_csrA = new complex<float>[nnz];
	h_b = new complex<float>[matrixM];
	h_x = new complex<float>[matrixM];

	memset(h_x,0,matrixM*sizeof(complex<float>));

	//cout<<"Copy "<<nnz << " " << matrixM <<" " << csr_Row.size() <<" data from vector to array."<<endl;
	int idx = 0;
	for (vector<int>::iterator it = csr_Row.begin(); it < csr_Row.end(); ++it)
	{
		h_csrRows[idx] = *it;
		idx++;
	}

	idx = 0;
	for (vector<int>::iterator it = csr_Col.begin(); it < csr_Col.end(); ++it)
	{
		h_csrCols[idx] = *it;
		idx++;
	}

	idx = 0;
	for (vector<complex<float> >::iterator it = A.begin(); it < A.end(); ++it)
	{
		h_csrA[idx] = *it;
		idx++;
	}

	idx = 0;
	for (vector<complex<float> >::iterator it = b.begin(); it < b.end(); ++it)
	{
		h_b[idx] = *it;
		idx++;
	}
	return true;
}

void cuBicgStabSolver::CopyDatafromHost2Device()
{
	checkCudaErrors(cudaMalloc((void**)&d_csrRows, sizeof(d_csrRows[0]) * (matrixM + 1)));
	checkCudaErrors(cudaMalloc((void**)&d_csrCols, sizeof(d_csrCols[0]) * nnz));
	checkCudaErrors(cudaMalloc((void**)&d_csrA, sizeof(d_csrA[0]) * nnz));
	checkCudaErrors(cudaMalloc((void**)&d_b, sizeof(d_b[0]) * matrixM));
	checkCudaErrors(cudaMalloc((void**)&d_x, sizeof(d_x[0]) * matrixM));

	/* clean memory */
	checkCudaErrors(cudaMemset((void *)d_x, 0, sizeof(d_x[0]) * matrixM));

	//double start_matrix_copy, stop_matrix_copy, start_preconditioner_copy, stop_preconditioner_copy;
	/* copy the csr matrix and vectors into device memory */
	checkCudaErrors(cudaMemcpy(d_csrA, h_csrA, (size_t)(nnz * sizeof(h_csrA[0])), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_csrCols, h_csrCols, (size_t)(nnz * sizeof(h_csrCols[0])), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_csrRows, h_csrRows, (size_t)((matrixM + 1) * sizeof(h_csrRows[0])), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_b, h_b, (size_t)(matrixM * sizeof(h_b[0])), cudaMemcpyHostToDevice));

}

void cuBicgStabSolver::IncompleteLUT()
{
	cout<<"ILU分解系数矩阵"<<endl;

	cusparseCreateMatDescr(&descrm);
	checkCudaErrors(cusparseSetMatType(descrm, CUSPARSE_MATRIX_TYPE_GENERAL));
	checkCudaErrors(cusparseSetMatIndexBase(descrm, CUSPARSE_INDEX_BASE_ZERO));

	checkCudaErrors(cudaMalloc((void**)&d_csrM, sizeof(d_csrM[0]) * nnz));
	checkCudaErrors(cudaMemset((void *)d_csrM, 0, sizeof(d_csrM[0]) * nnz));
	checkCudaErrors(cudaMemcpy(d_csrM, d_csrA, (size_t)(nnz * sizeof(d_csrM[0])), cudaMemcpyDeviceToDevice));


	checkCudaErrors(cusparseCreateSolveAnalysisInfo(&info_l));
	checkCudaErrors(cusparseCreateSolveAnalysisInfo(&info_u));

	cusparseCreateMatDescr(&descrL);
	checkCudaErrors(cusparseSetMatFillMode(descrL, CUSPARSE_FILL_MODE_LOWER));
	checkCudaErrors(cusparseSetMatDiagType(descrL, CUSPARSE_DIAG_TYPE_UNIT));
	checkCudaErrors(cusparseCcsrsv_analysis(cusparseHandle,
		CUSPARSE_OPERATION_NON_TRANSPOSE, matrixM, nnz, descrL, d_csrA, d_csrRows, d_csrCols, info_l));
	cusparseCreateMatDescr(&descrU);
	checkCudaErrors(cusparseSetMatFillMode(descrU, CUSPARSE_FILL_MODE_UPPER));
	checkCudaErrors(cusparseSetMatDiagType(descrU, CUSPARSE_DIAG_TYPE_NON_UNIT));
	checkCudaErrors(cusparseCcsrsv_analysis(cusparseHandle,
		CUSPARSE_OPERATION_NON_TRANSPOSE, matrixM, nnz, descrU, d_csrA, d_csrRows, d_csrCols, info_u));

	d_csrmCols = d_csrCols;
	d_csrmRows = d_csrRows;

	checkCudaErrors(cusparseCcsrilu0(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
		matrixM, descra, d_csrM, d_csrRows, d_csrCols, info_l));
}

void cuBicgStabSolver::cuBicgStab()
{
	cudaEvent_t start, stop;
	float costtime;
	int n = matrixN;
	cuComplex rho, rho0, beta, alpha, negalpha, omega, omega0, negomega, temp, temp2, tt, ts;
	float nrmr, nrmr0;
	cuComplex zero = make_cuComplex(0.0, 0.0);
	cuComplex one = make_cuComplex(1.0, 0.0);
	cuComplex mone = make_cuComplex(-1.0, 0.0);
	rho = zero;
	int iteration = 0;
	float residual = 1.0;
	cuComplex *r, *rw, *p, *v, *h, *s, *t, *y, *z;

	printf("Memory cost: %6.4f MB.\n",
			(float)ceil((11.0*matrixN*(double)sizeof(cuComplex)+
					nnz*(double)sizeof(cuComplex)+
					(matrixN+1)*(double)sizeof(int)+
					nnz*(double)sizeof(int))/1.0e6));
	cudaSetDevice(0);
	// h
	checkCudaErrors(cudaMalloc((void**)&h, sizeof(h[0]) * matrixN)); //alloc memory
	checkCudaErrors(cudaMemset((void *)h, 0, sizeof(h[0]) * matrixN));
	checkCudaErrors(cudaMemcpy(h, d_x, (size_t)(matrixN * sizeof(h[0])), cudaMemcpyDeviceToDevice));

	//s
	checkCudaErrors(cudaMalloc((void**)&s, sizeof(s[0]) * matrixN)); //alloc memory
	checkCudaErrors(cudaMemset((void *)s, 0, sizeof(s[0]) * matrixN));
	checkCudaErrors(cudaMemcpy(s, d_x, (size_t)(matrixN * sizeof(s[0])), cudaMemcpyDeviceToDevice));

	//t
	checkCudaErrors(cudaMalloc((void**)&t, sizeof(t[0]) * matrixN)); //alloc memory
	checkCudaErrors(cudaMemset((void *)t, 0, sizeof(t[0]) * matrixN));
	checkCudaErrors(cudaMemcpy(t, d_x, (size_t)(matrixN * sizeof(t[0])), cudaMemcpyDeviceToDevice));

	//y
	checkCudaErrors(cudaMalloc((void**)&y, sizeof(y[0]) * matrixN)); //alloc memory
	checkCudaErrors(cudaMemset((void *)y, 0, sizeof(y[0]) * matrixN));
	checkCudaErrors(cudaMemcpy(y, d_x, (size_t)(matrixN * sizeof(y[0])), cudaMemcpyDeviceToDevice));

	//z
	checkCudaErrors(cudaMalloc((void**)&z, sizeof(z[0]) * matrixN)); //alloc memory
	checkCudaErrors(cudaMemset((void *)z, 0, sizeof(z[0]) * matrixN));
	checkCudaErrors(cudaMemcpy(z, d_x, (size_t)(matrixN * sizeof(z[0])), cudaMemcpyDeviceToDevice));

	// 1.0  r0 = b − Ax0, x0 = [0], so r0 = b;
	checkCudaErrors(cudaMalloc((void**)&r, sizeof(r[0]) * matrixN)); //alloc memory
	checkCudaErrors(cudaMemset((void *)r, 0, sizeof(r[0]) * matrixN));
	checkCudaErrors(cudaMemcpy(r, d_b, (size_t)(matrixN * sizeof(r[0])), cudaMemcpyDeviceToDevice));

	// 2.0 Choose an arbitrary vector r̂0 such that (r̂0, r0) ≠ 0, e.g., r̂0 = r0
	checkCudaErrors(cudaMalloc((void**)&rw, sizeof(rw[0]) * matrixN)); //alloc memory
	checkCudaErrors(cudaMemset((void *)rw, 0, sizeof(rw[0]) * matrixN));
	checkCudaErrors(cudaMemcpy(rw, r, (size_t)(matrixN * sizeof(r[0])), cudaMemcpyDeviceToDevice)); //r̂0 = r0

																							  // 3.0 ρ0 = α = ω0 = 1
	rho0 = one;
	rho = one;
	alpha = one;
	omega0 = one;
	omega = one;
	// 4.0 v0 = p0 = 0, x = {0} when initialization
	checkCudaErrors(cudaMalloc((void**)&v, sizeof(v[0]) * matrixN)); //alloc memory
	checkCudaErrors(cudaMemset((void *)v, 0, sizeof(v[0]) * matrixN));
	checkCudaErrors(cudaMemcpy(v, d_x, (size_t)(matrixN * sizeof(v[0])), cudaMemcpyDeviceToDevice)); //r̂0 = r0

	checkCudaErrors(cudaMalloc((void**)&p, sizeof(p[0]) * matrixN)); //alloc memory
	checkCudaErrors(cudaMemset((void *)p, 0, sizeof(p[0]) * matrixN));
	checkCudaErrors(cudaMemcpy(p, d_x, (size_t)(matrixN * sizeof(p[0])), cudaMemcpyDeviceToDevice)); //r̂0 = r0

	checkCudaErrors(cublasScnrm2(cublasHandle, matrixN, r, 1, &nrmr0));
	
	// 5.0 loop ...
	while (iteration < MAXIterations && residual > torence)
	{
		rho0 = rho;
		//5.1 ρi = (r̂0, ri−1)
		checkCudaErrors(cublasCdotc(cublasHandle, n, rw, 1, r, 1, &rho));

		//5.2 β = (ρi/ρi−1)(α/ωi−1)
		beta = cuComplexDoubleToFloat(cuCmul(cuCdiv(cuComplexFloatToDouble(rho), cuComplexFloatToDouble(rho0)),
			cuCdiv(cuComplexFloatToDouble(alpha), cuComplexFloatToDouble(omega0))));

		//5.3 pi = ri−1 + β(pi−1 − ωi−1vi−1)
		// 5.3.1 p = pi−1 − ωi−1*vi−1
		/*
		cublasZaxpy(cublasHandle_t handle, int n, const cuDoubleComplex *alpha,
		const cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy)
		y [j] = α × x [k] + y [j]
		*/
		negomega = make_cuComplex(-omega.x, -omega.y);   ///////////negomega = -omega;
		checkCudaErrors(cublasCaxpy(cublasHandle, n, &negomega, v, 1, p, 1));
		checkCudaErrors(cublasCscal(cublasHandle, n, &beta, p, 1));
		checkCudaErrors(cublasCaxpy(cublasHandle, n, &one, r, 1, p, 1));

		// 5.4 y = K−1pi
		//preconditioning step (lower and upper triangular solve)
		/*
		cusparseZcsrsv_solve(cusparseHandle_t handle, cusparseOperation_t transA,
		int m, const cuDoubleComplex *alpha, const cusparseMatDescr_t descrA,
		const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA,
		const int *csrSortedColIndA, cusparseSolveAnalysisInfo_t info,
		const cuDoubleComplex *f, cuDoubleComplex *x);
		op ( A ) ∗ x = α ∗ f
		*/

		checkCudaErrors(cusparseCcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
			n, &one, descrL, d_csrM, d_csrmRows, d_csrmCols, info_l, p, t));
		checkCudaErrors(cusparseCcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
			n, &one, descrU, d_csrM, d_csrmRows, d_csrmCols, info_u, t, y));
		// 5.5 vi = Ay
		/*
		cusparseZcsrmv(cusparseHandle_t handle, cusparseOperation_t transA,
		int m, int n, int nnz, const cuDoubleComplex *alpha,
		const cusparseMatDescr_t descrA, const cuDoubleComplex *csrValA,
		const int *csrRowPtrA, const int *csrColIndA, const cuDoubleComplex *x,
		const cuDoubleComplex *beta, cuDoubleComplex *y)
		y = α ∗ op ( A ) ∗ x + β ∗ y
		α = 1.0  op = CUSPARSE_OPERATION_NON_TRANSPOSE, β = 0;
		*/
		checkCudaErrors(cusparseCcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
			n, n, nnz, &one, descra, d_csrA, d_csrRows, d_csrCols, y, &zero, v));

		// 5.6 α = ρi/(r̂0, vi)
		checkCudaErrors(cublasCdotc(cublasHandle, n, rw, 1, v, 1, &temp));
		alpha = cuComplexDoubleToFloat(cuCdiv(cuComplexFloatToDouble(rho), cuComplexFloatToDouble(temp)));

		// 5.7 h = xi−1 + αy
		/*
		cublasZaxpy(cublasHandle_t handle, int n, const cuDoubleComplex *alpha,
		const cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy)
		y [j] = α × x [k] + y [j]
		*/
		checkCudaErrors(cudaMemcpy(h, d_x, (size_t)(matrixM * sizeof(h[0])), cudaMemcpyDeviceToDevice));
		checkCudaErrors(cublasCaxpy(cublasHandle, n, &alpha, y, 1, h, 1));

		// 5.9 s = ri−1 − αvi
		negalpha = make_cuComplex(-alpha.x, -alpha.y);
		checkCudaErrors(cudaMemcpy(s, r, (size_t)(matrixM * sizeof(s[0])), cudaMemcpyDeviceToDevice));
		checkCudaErrors(cublasCaxpy(cublasHandle, n, &negalpha, v, 1, s, 1));

		// 5.10 z = K−1s
		checkCudaErrors(cusparseCcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
			n, &one, descrL, d_csrM, d_csrmRows, d_csrmCols, info_l, s, t));
		checkCudaErrors(cusparseCcsrsv_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
			n, &one, descrU, d_csrM, d_csrmRows, d_csrmCols, info_u, t, z));

		// 5.11 t = Az
		/*
		cusparseZcsrmv(cusparseHandle_t handle, cusparseOperation_t transA,
		int m, int n, int nnz, const cuDoubleComplex *alpha,
		const cusparseMatDescr_t descrA, const cuDoubleComplex *csrValA,
		const int *csrRowPtrA, const int *csrColIndA, const cuDoubleComplex *x,
		const cuDoubleComplex *beta, cuDoubleComplex *y)
		y = α ∗ op ( A ) ∗ x + β ∗ y
		α = 1.0  op = CUSPARSE_OPERATION_NON_TRANSPOSE, β = 0;
		*/
		checkCudaErrors(cusparseCcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
			n, n, nnz, &one, descra, d_csrA, d_csrRows, d_csrCols, z, &zero, t));

		// 5.12 ωi = (K−11 t, K−11 s) / (K −11 t, K −11 t)
		checkCudaErrors(cublasCdotc(cublasHandle, n, t, 1, s, 1, &temp));
		checkCudaErrors(cublasCdotc(cublasHandle, n, t, 1, t, 1, &temp2));
		omega = cuComplexDoubleToFloat(cuCdiv(cuComplexFloatToDouble(temp), cuComplexFloatToDouble(temp2)));

		// 5.13 xi = h + ωiz
		/*
		cublasZaxpy(cublasHandle_t handle, int n, const cuDoubleComplex *alpha,
		const cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy)
		y [j] = α × x [k] + y [j]
		*/
		checkCudaErrors(cudaMemcpy(d_x, h, (size_t)(matrixM * sizeof(d_x[0])), cudaMemcpyDeviceToDevice));
		checkCudaErrors(cublasCaxpy(cublasHandle, n, &omega, z, 1, d_x, 1));


		// 5.15 ri = s − ωit
		negomega = make_cuComplex(-omega.x, -omega.y);
		checkCudaErrors(cudaMemcpy(r, s, (size_t)(matrixM * sizeof(r[0])), cudaMemcpyDeviceToDevice));
		checkCudaErrors(cublasCaxpy(cublasHandle, n, &negomega, t, 1, r, 1));
		checkCudaErrors(cublasScnrm2(cublasHandle, n, r, 1, &nrmr));
		residual = nrmr / nrmr0;
		cout<< "PBICGStab @ "<<iteration << ", residual = "<< residual<<endl;

		iteration++;
	}
	//cudaEventDestroy(start);
	//cudaEventDestroy(stop);
	// free memories ...
	if (r) checkCudaErrors(cudaFree(r)); r = NULL;
	if (h) checkCudaErrors(cudaFree(h)); h = NULL;
	if (s) checkCudaErrors(cudaFree(s)); s = NULL;
	if (t) checkCudaErrors(cudaFree(t)); t = NULL;
	if (y) checkCudaErrors(cudaFree(y)); y = NULL;
	if (z) checkCudaErrors(cudaFree(z)); z = NULL;
	if (r) checkCudaErrors(cudaFree(r)); r = NULL;
	if (rw) checkCudaErrors(cudaFree(rw)); rw = NULL;
	if (v) checkCudaErrors(cudaFree(v)); v = NULL;
	if (p) checkCudaErrors(cudaFree(p)); p = NULL;
}

void cuBicgStabSolver::CalculateEField()
{
	float f = 2240; //Hz
	float dx, dy, dz;
	int i, j, k;
	long idx = 0;
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

	float ***Ex_Field = NULL;
	float ***Ey_Field = NULL;
	float ***Ez_Field = NULL;

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

	Ex_Field = malloc3F(nx, ny, nz,0);
	Ey_Field = malloc3F(nx, ny, nz,0);
	Ez_Field = malloc3F(nx, ny, nz,0);
	complex<float> temp;
	for (k = 1; k < nz - 1; k++)
	{
		//cout<<"Process E-field @ "<< k << "/" << nz-1 << "..."<<endl;
		for (j = 1; j < ny - 1; j++)
		{
			for (i = 1; i < nx - 1; i++)
			{
				idx = k*nx*ny + j*nx+ i;
				float sigma = Ceps[idx].real();
				float sigma_i_1 = Ceps[idx-1].real();
				float sigma_j_1 = Ceps[idx-nx].real();
				float sigma_k_1 = Ceps[idx-nx*ny].real();
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
	SaveElectricPotential("voltage.mat");
	Save3DMatrixestoMat("Ebicg2.mat", "Ex", Ex_Field, "Ey", Ey_Field, "Ez", Ez_Field,dims);
	Voltage = free3C(Voltage, nx, ny, nz);
	free3F(Ex_Field, nx, ny, nz);
	free3F(Ey_Field, nx, ny, nz);
	free3F(Ez_Field, nx, ny, nz);
}

void cuBicgStabSolver::Save3DMatrixestoMat(const char *matfile, const char *fieldx, float ***ex,
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


void cuBicgStabSolver::SaveElectricPotential(const char *filename)
{
	mat_complex_split_t Sx_mat = { NULL,NULL };
	mat_complex_split_t Sy_mat = { NULL,NULL };
	mat_complex_split_t Sz_mat = { NULL,NULL };
	mat_complex_split_t SA_mat = { NULL,NULL };
	float *f_real, *f_imag;
	size_t Dims[3] = { dims[0],dims[1],dims[2] };
	size_t Dims_sa[3] = { dims[0]-2,dims[1]-2,dims[2]-2};
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
	N = (dims[0]-2) * (dims[1] - 2) * (dims[2] - 2);
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

float ***cuBicgStabSolver::malloc3F(int nx, int ny, int nz,float val)
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

complex<float> ***cuBicgStabSolver::malloc3C(int nx, int ny, int nz)
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

complex<float> ***cuBicgStabSolver::free3C(complex<float> ***tab, int nx, int ny, int nz)
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

float ***cuBicgStabSolver::free3F(float ***tab, int nx, int ny, int nz)
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


Point3 cuBicgStabSolver::idx2xyz(int i)
{
	Point3 p = { 0,0,0 };
	int n_layer = dims[0] * dims[1];
	p.z = (int)((float)i / (float)n_layer);
	int res = i - p.z*n_layer;
	p.y = (int)((float)res / (float)dims[0]);
	p.x = res - p.y*dims[0];
	return p;
}

