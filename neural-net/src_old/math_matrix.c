
typedef struct Vector {
	int size;
	double* data;
} Vector;

void vector_make(Vector* vec, int size) {
	assert(size > 0);
	vec->size = size;
	vec->data = calloc(size, sizeof(double));
}

void vector_destory(Vector* vec) {
	assert(vec->size > 0);
	vec->size = 0;
	free(vec->data);
}

typedef struct Matrix {
	int rows;
	int cols;
	double* data;
} Matrix;

#define MAKE_MATRIX(m_name, rows, cols, ...) Matrix m_name = {0}; \
	matrix_make(&m_name, (rows), (cols)); \
	double m_name##_init_array[] = {__VA_ARGS__}; \
	matrix_setv(&m_name, ARRAY_SIZE(m_name##_init_array), m_name##_init_array);

void matrix_make(Matrix* m, int rows, int cols) {
	assert(rows > 0);
	assert(cols > 0);
	m->rows = rows;
	m->cols = cols;
	m->data = calloc(rows * cols, sizeof(double));
}

void matrix_destroy(Matrix* m) {
	assert(m->rows > 0);
	assert(m->cols > 0);
	free(m->data);
}

void matrix_set(Matrix* m, int row, int col, double value) {
	m->data[col + m->cols * row] = value;
}

void matrix_setv(Matrix* m, int count, double* values) {
	assert((m->rows * m->cols) == count);
	double* data = m->data;
	for(int i = 0; i < count; i++) {
		*data = values[i];
		data++;
	}
}

void matrix_copy(Matrix* src, Matrix* dst) {
	assert(src->rows == dst->rows);
	assert(src->cols == dst->cols);
	matrix_setv(dst, src->rows * src->cols, src->data);
}

double matrix_get(Matrix* m, int row, int col) {
	assert(row < m->rows && row >= 0);
	assert(col < m->cols && col >= 0);
	return m->data[col + m->cols * row];
}

void matrix_multiply(Matrix* m1, Matrix* m2, Matrix* m_result) {
	//assert(m1->rows == m2->cols);
	assert(m1->cols == m2->rows);
	if (m_result->data == 0) {
		matrix_make(m_result, m1->rows, m2->cols);
	}
	assert(m_result->rows == m1->rows);
	assert(m_result->cols == m2->cols);
	for (int row = 0; row < m_result->rows; row++) {
		for (int col = 0; col < m_result->cols; col++) {
			double v = 0;
			for (int i = 0; i < m1->cols; i++) {
				double a = matrix_get(m1, row, i);
				double b = matrix_get(m2, i, col);
				double c = a * b;
				v += c;
			}
			matrix_set(m_result, row, col, v);
		}
	}
}

void matrix_transpose(Matrix* m_in, Matrix* m_result) {
	if (m_result->data == 0) {
		matrix_make(m_result, m_in->cols, m_in->rows);
	}
	assert(m_in->rows == m_result->cols);
	assert(m_in->cols == m_result->rows);
	for (int in_row = 0; in_row < m_in->rows; in_row++) {
		for (int in_col = 0; in_col < m_in->cols; in_col++) {
			int t_row = in_col;
			int t_col = in_row;
			matrix_set(m_result, t_row, t_col, matrix_get(m_in, in_row, in_col));
		}
	}
}

void matrix_sigmoid(Matrix* m) {
	for (int row = 0; row < m->rows; row++) {
		for (int col = 0; col < m->cols; col++) {
			matrix_set(m, row, col, sigmoid(matrix_get(m, row, col)));
		}
	}
}

void matrix_print(Matrix* m) {
	for(int y = 0; y < m->rows; y++) {
		for(int x = 0; x < m->cols; x++) {
			printf("%.3f ", m->data[x + m->cols * y]);
		}
		printf("\n");
	}
}

