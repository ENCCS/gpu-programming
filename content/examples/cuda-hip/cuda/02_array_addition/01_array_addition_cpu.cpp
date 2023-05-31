//
// gcc  01_array_addition_cpu.cpp
// nvcc 01_array_addition_cpu.cpp
//
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

const double EPSILON = 1.0E-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;

void array_addition(const double *vecA, const double *vecB, double *vecC, const int NX);
void array_check(const double *vecC, const int NX);


int main(int argc, const char * argv[])
{

	printf("\n--Beginning of the main function.\n");

    const int NX = 1000000;
	int size_array = sizeof(double) * NX;
    double *vecA = (double *)malloc(size_array);
    double *vecB = (double *)malloc(size_array);
    double *vecC = (double *)malloc(size_array);

    for (int i = 0; i < NX; i++)
    {
        vecA[i] = a;
        vecB[i] = b;
    }

    array_addition(vecA, vecB, vecC, NX);
    array_check(vecC, NX);

    free(vecA);
    free(vecB);
    free(vecC);

	printf("\n--Ending of the main function.\n\n");

    return 0;
}


void array_addition(const double *vecA, const double *vecB, double *vecC, const int NX)
{
    for (int i = 0; i < NX; i++)
        vecC[i] = vecA[i] + vecB[i];
}


void array_check(const double *vecC, const int NX)
{
    bool has_error = false;
    for (int i = 0; i < NX; i++)
    {
        if (fabs(vecC[i] - c) > EPSILON)
		{
            has_error = true;
			break;
		}
    }
    printf("\n\tChecking array addition results >>> %s\n", has_error? "|| ERROR ||":"|| NO ERROR ||");
}

