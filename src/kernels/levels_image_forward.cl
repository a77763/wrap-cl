kernel void levels_image_forward_kernel_32(__global float *image_data, int input_offset, int batch, int w, int h, float translate, float scale)
{
	global float *image = image_data + input_offset;
    int size = batch * w * h;
	int id = get_global_id(1) * get_global_size(0) + get_global_id(0);	

    if(id >= size)
        return;

    image[id * 3 + 0] = image[id * 3 + 0]*scale + translate;
    image[id * 3 + 1] = image[id * 3 + 1]*scale + translate;
    image[id * 3 + 2] = image[id * 3 + 2]*scale + translate;
}