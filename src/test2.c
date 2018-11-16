
#include "matrix.h"
#include "matrix.c"
#include <stdio.h>
#include "batch_norm.c"


int main() {
	// image size 2x2x3
	// 10 images
	matrix x = random_matrix(5, 12, 1);
	print_matrix(x);

	// spatial = 2x2 = 4

	// use every group of 4 to calculate
	matrix m = mean(x, 4);
	print_matrix(m);

	matrix v = variance(x, m, 4);
	print_matrix(v);

	matrix n = normalize(x, m, v, 4);
	print_matrix(n);

	// http://www.alcula.com/calculators/statistics/variance/
	// copy the first spatial columns (of all rows)
	// and check that the variance and mean are 1 and 0 respectively
}