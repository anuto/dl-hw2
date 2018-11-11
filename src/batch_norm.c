#include "uwnet.h"

#include <stdlib.h>
#include <assert.h>
#include <math.h>

#define EPSILON .001

matrix mean(matrix x, int spatial)
{
    // x.rows = # of images (in a batch)
    // x.cols = every pixel of every channel in a single image
    // spatial = # of elements in a channel (W x H of an image)

    // m should be
    // 1 by # of channels
    matrix m = make_matrix(1, x.cols/spatial);
    int i, j;
    for(i = 0; i < x.rows; ++i){
        for(j = 0; j < x.cols; ++j){
            m.data[j/spatial] += x.data[i*x.cols + j];
        }
    }
    for(i = 0; i < m.cols; ++i){
        m.data[i] = m.data[i] / x.rows / spatial;
    }
    return m;
}

matrix variance(matrix x, matrix m, int spatial)
{
    matrix v = make_matrix(1, x.cols/spatial);
    // TODO: 7.1 - calculate variance
    int i, j;
    float pre_squared;
    for(i = 0; i < x.rows; ++i){
        for(j = 0; j < x.cols; ++j){
            pre_squared = (x.data[i*x.cols + j] - m.data[j/spatial]);
            v.data[j/spatial] += pre_squared * pre_squared;
        }
    }

    // divide by m which is the # of items in batch
    for(i = 0; i < v.cols; ++i){
        v.data[i] = v.data[i] / x.rows / spatial;
    }
    return v;
}


// normalize(x, l.rolling_mean, l.rolling_variance, spatial);
// rolling mean and rolling variance are 1 by # of filters/channels
// x is 
matrix normalize(matrix x, matrix m, matrix v, int spatial)
{
    matrix norm = make_matrix(x.rows, x.cols);
    // TODO: 7.2 - normalize array, 
    // norm = (x - mean) / sqrt(variance + eps)
    int i, j;
    for(i = 0; i < x.rows; ++i){
        for(j = 0; j < x.cols; ++j){
            norm.data[i*x.cols+j] 
                = x.data[i*x.cols+j] - m.data[j/spatial];
            norm.data[i*x.cols+j] /= sqrt(v.data[j/spatial] + EPSILON);
        }
    }

    return norm;
    
}

matrix batch_normalize_forward(layer l, matrix x)
{
    float s = .1;
    int spatial = x.cols / l.rolling_mean.cols;
    if (x.rows == 1){
        return normalize(x, l.rolling_mean, l.rolling_variance, spatial);
    }
    matrix m = mean(x, spatial);
    matrix v = variance(x, m, spatial);

    matrix x_norm = normalize(x, m, v, spatial);

    scal_matrix(1-s, l.rolling_mean);
    axpy_matrix(s, m, l.rolling_mean);

    scal_matrix(1-s, l.rolling_variance);
    axpy_matrix(s, v, l.rolling_variance);

    free_matrix(m);
    free_matrix(v);

    free_matrix(l.x[0]);
    l.x[0] = x;

    return x_norm;
}


matrix delta_mean(matrix d, matrix variance, int spatial)
{
    matrix dm = make_matrix(1, variance.cols);
    // TODO: 7.3 - calculate dL/dmean
    int i, j;
    // TODO verify dimensions of d
    // 1 x # images in a mini batch?
    for(i = 0; i < d.rows; ++i){
        for (j = 0; j < d.cols; ++j){
            dm.data[j/spatial] += d.data[i*d.cols + j] 
                * (-1 / sqrt(variance.data[j/spatial] + EPSILON));
        }
    }
    return dm;
}

matrix delta_variance(matrix d, matrix x, matrix mean, matrix variance, int spatial)
{
    matrix dv = make_matrix(1, variance.cols);
    // TODO: 7.4 - calculate dL/dvariance
    int i, j;
    for(i = 0; i < d.rows; ++i){
        for (j = 0; j < d.cols; ++j){
            dv.data[j/spatial] += d.data[i*x.cols + j] 
                * (x.data[i*x.cols + j] - mean.data[j/spatial]) 
                * -.5
                * pow((variance.data[j/spatial] + EPSILON), -1.5);
        }
    }

    return dv;
}

matrix delta_batch_norm(matrix d, matrix dm, matrix dv, matrix mean, matrix variance, matrix x, int spatial)
{
    int i, j;
    matrix dx = make_matrix(d.rows, d.cols);
    // TODO: 7.5 - calculate dL/dx

    for(i = 0; i < d.rows; ++i){
        for(j = 0; j < d.cols; ++j){
            dx.data[i*d.cols + j] = d.data[i*d.cols + j] * (1 / sqrt(variance.data[j/spatial] + EPSILON))
                + dv.data[j/spatial] * 2 * (x.data[i*x.cols + j] - mean.data[j/spatial]) / (x.rows / spatial)
                + dm.data[j/spatial] / (x.rows / spatial);
        }
    }
    return dx;
}

matrix batch_normalize_backward(layer l, matrix d)
{
    int spatial = d.cols / l.rolling_mean.cols;
    matrix x = l.x[0];

    matrix m = mean(x, spatial);
    matrix v = variance(x, m, spatial);

    matrix dm = delta_mean(d, v, spatial);
    matrix dv = delta_variance(d, x, m, v, spatial);
    matrix dx = delta_batch_norm(d, dm, dv, m, v, x, spatial);

    free_matrix(m);
    free_matrix(v);
    free_matrix(dm);
    free_matrix(dv);

    return dx;
}
