float get_pixel_kernel_32(global float *image, int w, int h, int x, int y, int c)
		{
			if(x < 0 || x >= w || y < 0 || y >= h) return 0;
			return image[x + w*(y + c*h)];
		}



float bilinear_interpolate_kernel_32(global float *image, int w, int h, float x, float y, int c)
{
    int ix = (int32_t) floor(x);
    int iy = (int32_t) floor(y);

    float dx = x - ix;
    float dy = y - iy;

    float val = (1-dy) * (1-dx) * get_pixel_kernel_32(image, w, h, ix, iy, c) + 
        dy     * (1-dx) * get_pixel_kernel_32(image, w, h, ix, iy+1, c) + 
        (1-dy) *   dx   * get_pixel_kernel_32(image, w, h, ix+1, iy, c) +
        dy     *   dx   * get_pixel_kernel_32(image, w, h, ix+1, iy+1, c);
    return val;
}


kernel void crop_forward_kernel_32( __global float *input_data, int input_offset, int size, int c, int h, int w, int crop_height, int crop_width, __global float *output, int output_offset)
{
	global float *input = input_data + input_offset;
	output = output + output_offset;
	int id = get_global_id(1) * get_global_size(0) + get_global_id(0);
    if(id >= size)
        return;

    int count = id;
    int j = id % crop_width;
    id /= crop_width;
    int i = id % crop_height;
    id /= crop_height;
    int k = id % c;
    id /= c;
    int b = id;

    int dw = (w - crop_width)/2.f;
    int dh = (h - crop_height)/2.f;

    input += w*h*c*b;

    float x = j + dw;    
    float y = i + dh;


    output[count] = bilinear_interpolate_kernel_32(input, w, h, x, y, k);
	printf("%f ", output[count]);
}