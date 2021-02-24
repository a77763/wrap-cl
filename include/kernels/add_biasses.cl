kernel void add_bias_kernel_32(__global float *output, 
                                       int output_offset, 
                                       __global float *biases, 
                                       int bias_offset, 
                                       int batch, int n, int size)
{
    output = output + output_offset;
    biases = biases + bias_offset;
    int index = get_global_id(1) * get_global_size(0) + get_global_id(0);
    if (index >= n*size*batch) return;
    int i = index % size;
    index /= size;
    int j = index % n;
    index /= n;
    int k = index;

    output[(k*n+j)*size + i] += biases[j];
}