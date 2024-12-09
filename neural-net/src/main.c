#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#define UNUSED_PARAMETER(x) (void)x
#define ARRAY_SIZE(array) (sizeof(array) / sizeof(array[0]))

#include "math.c"
#include "math_matrix.c"
#include "neural_net.c"

int main(int argc, char** argv) {
	UNUSED_PARAMETER(argc);
	UNUSED_PARAMETER(argv);

	srand(time(0));

	/*
	printf("matrix 1\n");
	MAKE_MATRIX(m1, 2, 3, 1,2,3, 4,5,6)
	matrix_print(&m1);

	printf("\nmatrix 2\n");
	MAKE_MATRIX(m2, 3, 2, 7,8,9, 10,11,12);
	matrix_print(&m2);

	printf("\nm1 * m2\n");
	Matrix m_result = {0};
	matrix_multiply(&m1, &m2, &m_result);
	matrix_print(&m_result);
	
	printf("\nmatrix_sigmoid\n");
	MAKE_MATRIX(m3, 3, 1, 0.975, 0.888, 1.254);
	matrix_print(&m3);
	matrix_sigmoid(&m3);
	matrix_print(&m3);
*/
/*
	MAKE_NEURAL_NET(nn1, 3, 3, 3);
	NEURAL_NET_SET_INPUT(nn1, 0.1, 0.2, 0.3);
	NEURAL_NET_SET_WEIGHTS(nn1, 0, 1,2,3,4,5,6,7,8,9);
	neural_net_randomize_weights(&nn1);
	neural_net_print(&nn1);
*/
	printf("matrix multiply test\n");
	MAKE_MATRIX(m1, 3, 1, 1, 2, 3);
	matrix_print(&m1);
	MAKE_MATRIX(m2, 1, 3, 1, 2, 3);
	matrix_print(&m2);
	Matrix m3 = {0};
	matrix_multiply(&m1, &m2, &m3);
	matrix_print(&m3);

	printf("neural net query test\n");
	MAKE_NEURAL_NET(brain, 3, 3, 3);
	NEURAL_NET_SET_WEIGHTS(brain, 0, 
		0.9, 0.3, 0.4,
		0.2, 0.8, 0.2,
		0.1, 0.5, 0.6);
	NEURAL_NET_SET_WEIGHTS(brain, 1,
		0.3, 0.7, 0.5,
		0.6, 0.5, 0.2,
		0.8, 0.1, 0.9);
	MAKE_MATRIX(m_input, 3, 1, 0.9, 0.1, 0.8);
	Matrix m_output = {0};
	matrix_make(&m_output, 3, 1);
	neural_net_query(&brain, &m_input, &m_output);
	printf("neural net state\n");
	neural_net_print(&brain);
	printf("m_output state\n");
	matrix_print(&m_output);
}
