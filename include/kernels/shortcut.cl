kernel void shortcut_kernel_32(
    int size, int minw, int minh, int minc, 
    int stride, int sample, int batch, int w1, 
    int h1, int c1, __global float *add, int add_offset,
    int w2, int h2, int c2, 
    float s1, float s2, __global float *out, int out_offset)
{
    int id = get_global_id(1) * get_global_size(0) + get_global_id(0);
    if (id >= size) return;
    int i = id % minw;
    id /= minw;
    int j = id % minh;
    id /= minh;
    int k = id % minc;
    id /= minc;
    int b = id % batch;

    int out_index = i*sample + w2*(j*sample + h2*(k + c2*b));
    int add_index = i*stride + w1*(j*stride + h1*(k + c1*b));
    out[out_index] = s1*out[out_index] + s2*add[add_index];
}