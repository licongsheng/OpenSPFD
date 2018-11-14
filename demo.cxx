/*
 * main.cpp
 *
 *  Created on: 2018年5月24日
 *      Author: root
 */

#include <iostream>
#include <vector>
#include <complex>
#include "stdlib.h"
#include "math.h"
#include "SphereSource.h"
#include "CoilSource.h"
#include "AnalysisMagneticField.h"
#include "SPFD2LinearEquationsCPU.h"
#include "cuBicgStabSolver.h"
#include "BicgStabSolver.h"

using namespace std;

//Demos
int TMS_Sphere();  //SPFD for uniform sphere exposure to TMS
int TMS_RealHead(); //SPFD for Chinese female head model exposure to TMS

int main()
{
	TMS_Sphere();

	return 1;
}





