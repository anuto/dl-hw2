#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "uwnet.h"

/*
Maxpooling is another core building block of convolutional neural networks. 
Implementing maxpooling will be similar to implementing convolutions in some 
ways, you will have to iterate over the image, process a window of pixels with 
me fixed size, and in this case find the maximum value to put into the output.

6.1 forward_maxpool_layer
Write the forward method to find the maximum value in a given window size,
 moving by some strided amount between applications. Note: maxpooling happens
  on each channel independently.

6.2 backward_maxpool_layer
The backward method will be similar to forward. Even though the window size
 may be large, only one element contributed to the error in the prediction 
 so we only backpropagate our deltas to a single element in the input per 
 window. Thus, you'll iterate through again and find the maximum value and 
 then backpropagate error to the appropriate element in prev_delta 
 corresponding to the position of the maximum element.
*/
// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_maxpool_layer(layer l, matrix in)
{
    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    matrix out = make_matrix(in.rows, outw*outh*l.channels);

    // TODO: 6.1 - iterate over the input and fill in the output with max values

    int cols;
    int rows;
    int pool_col;
    int pool_row;
    float max;
    float next;

    int channel;
    int image;

    // The images for a batch are stored in the successive rows of the input 
    // matrix, you should be able to loop over each image, each channel of the 
    // image, and each application of your filter (similar to your im2col function), 
    // find max values in some range, and put them into the output.
    int out_index = 0;
    int offset;
    // printf("size: %d", l.size);

    // for each image (one row = one image)
    for (image = 0; image < in.rows; image++) 
    {
        // for each channel in the image
        for (channel = 0; channel < l.channels; channel++) 
        {
            // for each row in the image channel. We move 'stride' amount
            for (rows = 0; rows < l.height; rows += l.stride)
            {
                // for each column in the row of the image channel. We move 'stride' amount
                for (cols = 0; cols < l.width; cols += l.stride)
                {
                    // calculate one value

                    offset = image * in.cols 
                        + channel * l.width * l.height
                        + rows * l.width 
                        + cols;

                    // printf("    offset: %d", offset);
                    max = in.data[offset];

                    for (pool_row = 0; pool_row < l.size; pool_row++)
                    {
                        for (pool_col = 0; pool_col < l.size; pool_col++)
                        {
                            if(rows + pool_row < l.height && cols + pool_col < l.width) {

                                next = in.data[offset + pool_col + pool_row * l.width];
                                if (next > max) 
                                {
                                    max = next;
                                }
                            }
                        }
                    }
                    out.data[out_index] = max;
                    out_index++;
                }
            }
        }
    }
    // printf("out_index %d\n", out_index);
    // printf("dimension %d\n", out.rows * out.cols);
    // printf("\n");

    l.in[0] = in;
    free_matrix(l.out[0]);
    l.out[0] = out;
    free_matrix(l.delta[0]);
    l.delta[0] = make_matrix(out.rows, out.cols);
    return out;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix prev_delta: error term for the previous layer
void backward_maxpool_layer(layer l, matrix prev_delta)
{
    matrix in    = l.in[0];
    matrix out   = l.out[0];
    matrix delta = l.delta[0];

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;

    // TODO: 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.

    // printf("in %d, %d : delta %d, %d \n", in.rows / l.stride, in.cols / l.stride
    //     , delta.rows, delta.cols);

    int cols;
    int rows;
    int pool_col;
    int pool_row;
    float max;
    float next;
    int max_index;
    // printf("delta.rows: %d\n", delta.rows);
    // printf("delta.columns: %d\n", delta.cols);
    //     printf("in.rows: %d\n", in.rows);
    // printf("in.columns: %d\n", in.cols);
    // printf("prev_delta.rows: %d\n", prev_delta.rows);
    // printf("prev_delta.columns: %d\n", prev_delta.cols);
    //  printf("out.rows: %d\n", out.rows);
    // printf("out.columns: %d\n", out.cols);
    // return;

    int out_index = 0;
    int image;
    int offset;
    int channel;
    for (image = 0; image < in.rows; image++) 
    {
        for (channel = 0; channel < l.channels; channel++) 
        {
            for (rows = 0; rows < l.height; rows += l.stride)
            {
                for (cols = 0; cols < l.width; cols += l.stride)
                {
                    // calculate one value
                    offset = image * in.cols 
                        + channel * l.width * l.height
                        + rows * l.width
                        + cols;

                    max = in.data[offset];
                    max_index = offset;
                    // printf("where\n");
                    for (pool_row = 0; pool_row < l.size; pool_row++)
                    {
                        for (pool_col = 0; pool_col < l.size; pool_col++)
                        {
                                                // printf("is \n");

                            if(rows + pool_row < l.height && cols + pool_col < l.width) {
                                next = in.data[offset + pool_col + pool_row * l.width];
                                                    // printf("the\n");

                                if (next > max) 
                                {
                                    max_index = offset + pool_col + pool_row * l.width;
                                    max = next;
                                }
                            }
                        }
                    }
                    // in.data[max_index] = out.data[out_index];
                    // fill in corresponding delta w delta from output
                    prev_delta.data[max_index] += delta.data[out_index];
                    out_index++;
                                        // printf("issue?\n");

                }
            }
        }
    }
    /*for (rows = 0; rows < delta.rows; rows++) 
    {
        for (cols = 0; cols < delta.cols; cols++)
        { 
            // printf("cur col: %d\n", cols);
            // index of the upper left corner of the pool
            int offset = (cols * l.stride) + (rows * l.stride * l.width);

            max = in.data[offset];
            max_index = offset;
            for (pool_row = 0; pool_row < l.size; pool_row++)
            {
                for (pool_col = 0; pool_col < l.size; pool_col++)
                {
                    next = in.data[offset + pool_col + pool_row * l.width];
                    if (next > max) 
                    {
                        max_index = offset + pool_col + pool_row * l.width;
                        max = next;
                    }
                }
            }
            // fill in corresponding delta w delta from output
            prev_delta.data[max_index] += delta.data[cols + rows * delta.cols];
            
        }
    }*/

}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer l, float rate, float momentum, float decay)
{
}

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(int w, int h, int c, int size, int stride)
{
    layer l = {0};
    l.width = w;
    l.height = h;
    l.channels = c;
    l.size = size;
    l.stride = stride;
    l.in = calloc(1, sizeof(matrix));
    l.out = calloc(1, sizeof(matrix));
    l.delta = calloc(1, sizeof(matrix));
    l.forward  = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update   = update_maxpool_layer;
    return l;
}

