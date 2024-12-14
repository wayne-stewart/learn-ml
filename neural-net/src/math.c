
const double e =  2.718281828459045;
const double pi = 3.141592653589793;

// 1 / (1 + e^(-x))
double sigmoid(double x) { return 1.0 / ( 1.0 + pow(e, -x) ); }

/*  Generate a random number with a normal distribution
 *
 *  THIS METHOD IS NOT THREAD SAFE
 *
 *  Parameters
 *      mu:    the mean of the distribution    
 *      sigma: the standard deviation of the distribution 
 */
double randn(double mu, double sigma) {
	static int m = 1;
	static double u, v, y, z, r;
	m++;
	if (m % 2 == 1) return z;
	do {
		// u cannot be 0 so loop until we get
		// a non-zero result
		u = (double)rand() / (double)RAND_MAX; 
	}
	while( u == 0);
	v = rand() / (double)RAND_MAX;
	r = sigma * sqrt(-2.0 * log(u));
	y = r * cos(2.0 * pi * v) + mu;
	z = r * sin(2.0 * pi * v) + mu;
	return y;
}

