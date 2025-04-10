
#include <stdio.h>
#include <stdint.h>

#define UNUSED(x) (void)(x)

#define u32 uint32_t

struct Matrix {
	u32 Rows;
	u32 Cols;

	Matrix& operator+=(const Matrix& rhs) {

		return *this;
	} 
	friend Matrix operator+(Matrix lhs, const Matrix& rhs) {
		lhs += rhs;
		return lhs;
	}
}


int main(int argc, char **argv) {
	UNUSED(argc);
	UNUSED(argv);

	printf("hello, c++!\n");
}
