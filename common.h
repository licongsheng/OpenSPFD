#pragma once
#pragma warning(disable:4996)
#define _CRT_SECURE_NO_WARNINGS 

#ifndef M_PI
#define M_PI 3.1415926535897
#endif
#ifndef CLOCKS_PER_SEC
#define CLOCKS_PER_SEC ((clock_t)1000)
#endif
#define EPSILON 8.854187817620e-12
#define Mu0 1.2566e-06
#define MAXIterations 1000
#define torence 1e-4

struct Point3 {
	float x;
	float y;
	float z;
};


struct Tissue
{
	char Name[100];
	unsigned char Level;
	double Conductivity;
	double Permitivity;
	double Density;
};


