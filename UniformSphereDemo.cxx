/*
 * UniformSphereDemo.cxx
 *
 *  Created on: 2018年11月14日
 *      Author: root
 */


#include <iostream>
#include <vector>
#include <complex>
#include "stdlib.h"
#include "math.h"
#include "SphereSource.h"
#include "CoilSource.h"
#include "HeadModelReader.h"
#include "MeshSolid.h"
#include "AnalysisMagneticField.h"
#include "SPFD2LinearEquationsCPU.h"
#include "cuBicgStabSolver.h"
#include "BicgStabSolver.h"

using namespace std;




int TMS_Sphere()
{
	cout << "This is a open source for ELF MF exposre numerical simulation based on SPFD method."<<endl;

	// class ...
	SphereSource sphere;
	CoilSource coil;
	AnalysisMagneticField analyzer;
	SPFD2LinearEquations_CPU spfder;
	cuBicgStabSolver cu_solver;
	BICGStabSolver solver;

	// variable
	const char *E_fieldname = "./Results/E_loop.mat";  //E-field save to disk ...
	float Sphere_radius = 40; //unit: mm,
	float frequency = 2240; //operating frequency Hz
	float Coil_radius = 35; //mm
	float sigma = 0.33137; //Conductivity 	[S/m]
	float epsilon = 128460; //Relative	permittivity
	float spacing[3] = {1.0,1.0,1.0}; //  mm
	int solid_dims[3] = { 0 };
	float *Ax, *Ay, *Az;
	vector<int> csr_Row;
	vector<int> coo_Row;
	vector<int> csr_Col;
	vector<complex<float> > A;
	vector<complex<float> > b;

	float Ampliter = 1.0;  // Current line amplitude
	Point3 Coil_center = {0.0,0.0,10 + Sphere_radius}; //mm

	int dims[3] = {0};
	complex<float> *pv = NULL;
	vector<Point3> coil_pts;
	float Extent[6] = {-10 - Sphere_radius,10 + Sphere_radius,
		-10 - Sphere_radius,10 + Sphere_radius,
		-10 - Sphere_radius,10 + Sphere_radius }; //mm Calculating volume

	//creat coil
	coil_pts = coil.CircleCoil(Coil_radius,Coil_center);
	//coil_pts = coil.EightCoil(Coil_radius, Coil_center);

	//coil.SaveCoil("Coil.pts");
	cout<<"Analysis Magnetic field induced by Coil"<<endl;
	analyzer.SetCoil(coil_pts, Ampliter);
	analyzer.SetComputationVolume(Extent,spacing);
	analyzer.Analysis();
	//Get Magnetic vector
	Ax = analyzer.GetAxField((int *)dims); //
	Ay = analyzer.GetAyField((int *)dims); //
	Az = analyzer.GetAzField((int *)dims); //

	cout<<"Magnetic field dimension = ("
			     << dims[0]<<","<< dims[1]<<","<< dims[2] <<")" <<endl;

	//mesh ...
	cout<<"Create Sphere"<<endl;
	pv = sphere.CreatePhisicalSphere(Sphere_radius, Extent,1.0,sigma,epsilon,frequency,(int*)solid_dims);
	//sphere.SavePhicalSphere("Phycal_Sphere.mat");
	cout<<"Create Sphere. Dimension = (" << solid_dims[0] <<","<< solid_dims[1] <<","<< solid_dims[2] <<")" <<endl;

	//convert to spfd
	cout<<"Create SPFD formulation ..."<<endl;
	spfder.SetSphicalVolume(pv,solid_dims,spacing);
	spfder.SetMagneticVector(Ax,Ay,Az,dims,frequency);
	spfder.Conveter();
	spfder.GetLinearEquations(csr_Row,csr_Col,A,b);
	spfder.GetRowCoo(coo_Row);
	cout<<"Save SPFD coefficients ..."<<endl;
	//spfder.SaveSPFD("spfd.mat");

	//linear equations solve by iteration method ...
	cout<<"Cuda solver: Ax = b solved by cusparse ..."<<endl;
	solver.SetParameters(coo_Row,csr_Col,A,b,dims);
	solver.SetMagneticVector(Ax,Ay,Az,pv);
	solver.Run();
	solver.SaveElectricField(E_fieldname);

	cout << "All Done."<<endl;

	return 1;
}

int TMS_RealHead()
{
	cout << "This is a open source for ELF MF exposre numerical simulation based on SPFD method." << endl;
	//head model file location ...
	const char* solid_file = "../model/female.mat";
	const char *E_fieldname = "./Results/E_eight.mat";
	//head position ...
	Point3 head_center = { 0,0,0 };
	//head dimension and resolution
	int Soild_Dims[3] = { 0 };
	float Solid_Spacing[3] = { 0 };
	//head model dataset
	unsigned char *Voxel3D = NULL;
	vector<Tissue> Tissues;

	// coil radius, position, frequency, amplitude ...
	float Coil_radius = 35; //mm
	Point3 Coil_center = { 0.0,0.0,0.0}; //mm
	float frequency = 2240; //Hz
	float current = 1.0;

	float Spacing[3] = {1.0,1.0,1.0}; // object size unit in mm ...
	float Extent[6] = { 0 }; // compute volume unit in mm ...
	int Compute_Dims[3] = { 0 };

	complex<float> *CVol = NULL;

	//magnetic vector ...
	float *Ax, *Ay, *Az;
	int dims[3] = { 0 }; //magnetic dimension

	//spfd
	vector<int> csr_Row;
	vector<int> coo_Row;
	vector<int> csr_Col;
	vector<complex<float> > A;
	vector<complex<float> > b;

	// operators ...
	HeadModelReader reader;
	MeshSolid mesher;
	CoilSource coil;
	AnalysisMagneticField analyzer;
	SPFD2LinearEquations_CPU spfder;
	BICGStabSolver solver;

	//////////////////////////////////////////////////////////////////////////////////
	reader.SetFilename(solid_file);
	reader.read();
	Voxel3D = reader.GetHeadModel((int*)Soild_Dims, (float*)Solid_Spacing, Tissues);
	cout << "Magnetic field dimension = ("
		<< Soild_Dims[0] << "," << Soild_Dims[1] << "," << Soild_Dims[2] << ")" << endl;
	cout << "The number of tissues is " << Tissues.size() << endl;

	//set the head to the center of the world
	Extent[0] = -(float)Soild_Dims[0] / 2.0 - 4.0;
	Extent[1] = (float)Soild_Dims[0] / 2.0 + 4.0;
	Extent[2] = -(float)Soild_Dims[1] / 2.0 - 4.0;
	Extent[3] = (float)Soild_Dims[1] / 2.0 + 4.0;
	Extent[4] = -(float)Soild_Dims[2] / 2.0 - 4.0;
	Extent[5] = (float)Soild_Dims[2] / 2.0 + 4.0;

	/////////////////////////////////////////////////////////////////////////////
	//Mesh solid ...
	mesher.SetComputeVolume(Extent, Spacing, frequency);
	mesher.SetHeadModel(Voxel3D, Soild_Dims, Solid_Spacing, Tissues, head_center);
	mesher.Mesh();
	CVol = mesher.GetPhiscalModel((int*)Compute_Dims);

	/////////////////////////////////////////////////////////////////////////////
	//TMS coil design ...
	Coil_center.x = (Extent[0] + Extent[1]) / 2.0;
	Coil_center.y = (Extent[2] + Extent[3]) / 2.0;
	Coil_center.z = Extent[5] + 10.0;
	vector<Point3> coil_pts = coil.EightCoil(Coil_radius, Coil_center);

	/////////////////////////////////////////////////////////////////////////////
	// bio-sarvet
	cout << "Analysis Magnetic field induced by Coil" << endl;
	analyzer.SetCoil(coil_pts, current);
	analyzer.SetComputationVolume(Extent, Spacing);
	analyzer.Analysis();
	Ax = analyzer.GetAxField((int *)dims);
	Ay = analyzer.GetAyField((int *)dims);
	Az = analyzer.GetAzField((int *)dims);

	//convert to spfd
	cout << "Create SPFD formulation ..." << endl;
	spfder.SetSphicalVolume(CVol, Compute_Dims, Spacing);
	spfder.SetMagneticVector(Ax, Ay, Az, dims, frequency);
	spfder.Conveter();
	spfder.GetLinearEquations(csr_Row, csr_Col, A, b);
	spfder.GetRowCoo(coo_Row);
	//cout << "Save SPFD coefficients ..." << endl;
	//spfder.SaveSPFD("spfd.mat");

	//linear equations solve by iteration method ...
	cout << "Cuda solver: Ax = b solved by cusparse ..." << endl;
	solver.SetParameters(coo_Row, csr_Col, A, b, dims);
	solver.SetMagneticVector(Ax, Ay, Az, CVol);
	solver.Run();
	solver.SaveElectricField(E_fieldname);

	//cout << "All Done." << endl;
	return 1;
}



