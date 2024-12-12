
const double e = 2.718281828459045;

// 1 / (1 + e^(-x))
double sigmoid(double x) { return 1.0 / ( 1.0 + pow(e, -x) ); }


