kernel void upsample_kernel_32(int N, __global float *x, int x_offset, 
                               int w, int h, int c, int batch, int stride, 
                               float scale, __global float *out, int out_offset)
{
    int i = get_global_id(1) * get_global_size(0) + get_global_id(0);
    x = x+x_offset;
    out = out+out_offset;
    if(i >= N) return;
    int out_index = i;
    int out_w = i%(w*stride);
    i = i/(w*stride);
    int out_h = i%(h*stride);
    i = i/(h*stride);
    int out_c = i%c;
    i = i/c;
    int b = i%batch;
    int in_w = out_w / stride;
    int in_h = out_h / stride;
    int in_c = out_c;
    int in_index = b*w*h*c + in_c*w*h + in_h*w + in_w;
    out[out_index] += scale * x[in_index];
}