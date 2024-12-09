
#define MAKE_NEURAL_NET(nn_name, ...) int nn_name##_init_array[] = {__VA_ARGS__}; \
	NeuralNet nn_name = {0}; \
	neural_net_make(&nn_name, ARRAY_SIZE(nn_name##_init_array), nn_name##_init_array); 

#define NEURAL_NET_SET_WEIGHTS(nn_name, layer, ...) \
	double nn_name##_layer_##layer##_weights_array[] = {__VA_ARGS__}; \
	neural_net_set_weights(&nn_name, layer, \
		ARRAY_SIZE(nn_name##_layer_##layer##_weights_array), \
				   nn_name##_layer_##layer##_weights_array);

#define NEURAL_NET_SET_INPUT(nn_name, ...) double nn_name##_input_array[] = {__VA_ARGS__}; \
	matrix_setv(&nn_name.layers[0], ARRAY_SIZE(nn_name##_input_array), nn_name##_input_array);

typedef struct NeuralNet {
	int layer_count;
	Matrix* layers;
	Matrix* weights;
} NeuralNet;

void neural_net_make(NeuralNet* nn, int layer_count, int* layer_sizes) {
	assert(layer_count > 1);
	nn->layer_count = layer_count;
	nn->layers = calloc(layer_count, sizeof(Matrix));
	int weight_count = layer_count - 1;
	nn->weights = calloc(weight_count, sizeof(Matrix));
	for (int i_layer = 0; i_layer < layer_count; i_layer++) {
		int layer_size = layer_sizes[i_layer];
		matrix_make(&nn->layers[i_layer], layer_size, 1);
		if (i_layer < (weight_count)) {
			int next_layer_size = layer_sizes[i_layer + 1];
			matrix_make(&nn->weights[i_layer], next_layer_size, layer_size);
		}
	}
}

void neural_net_set_weights(NeuralNet* nn, int layer_index, int weight_count, double* weights) {
	assert(layer_index < (nn->layer_count - 1));
	matrix_setv(&nn->weights[layer_index], weight_count, weights);
}

void neural_net_randomize_weights(NeuralNet* nn) {
	int weights_count = nn->layer_count - 1;
	for (int i = 0; i < weights_count; i++) {
		Matrix* w = &nn->weights[i];
		for (int j = 0; j < (w->rows * w->cols); j++) {
			double x = rand();
			w->data[j] = x / RAND_MAX;
		}
	}
}

void neural_net_process(NeuralNet* nn) {
	assert(nn->layer_count > 1);
	int weight_count = nn->layer_count - 1;
	for (int i = 0; i < weight_count; i++) {
		Matrix* m_in = &nn->layers[i];
		Matrix* m_out = &nn->layers[i + 1];
		Matrix* m_weights = &nn->weights[i];
		matrix_multiply(m_weights, m_in, m_out);
		matrix_sigmoid(m_out);
	}
}

void neural_net_load_weights(NeuralNet* nn, int count, Matrix** weights_array) {
	assert((nn->layer_count - 1) == count);
	for (int i = 0; i < count; i++) {
		matrix_copy(weights_array[i], &nn->weights[i]);
	}
}

void neural_net_query(NeuralNet* nn, Matrix* m_input, Matrix* m_output) {
	assert(nn->layer_count > 1);
	matrix_copy(m_input, &nn->layers[0]);
	neural_net_process(nn);
	matrix_copy(&nn->layers[nn->layer_count - 1], m_output);
}

void neural_net_print(NeuralNet* nn) {
	int weight_count = nn->layer_count - 1;
	for (int i_layer = 0; i_layer < nn->layer_count; i_layer++) {
		if (i_layer == 0)
			printf("layer %d (input) and weights\n", i_layer+1);
		else if (i_layer < weight_count)
			printf("\nlayer %d and weights\n", i_layer+1);
		else
			printf("\nlayer %d (output)\n", i_layer+1);

		Matrix* layer = &nn->layers[i_layer];
		for (int i_row = 0; i_row < layer->rows; i_row++) {
			printf("%.3f", matrix_get(layer, i_row, 0));
			if (i_layer < weight_count) {
				Matrix* weight = &nn->weights[i_layer];
				for (int i_col = 0; i_col < weight->cols; i_col++) {
					printf(" %.3f", matrix_get(weight, i_row, i_col));
				}
			}
			printf("\n");
		}
	}
}

