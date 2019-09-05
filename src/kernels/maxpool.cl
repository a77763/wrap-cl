// CL: grid stride looping
#define CL_KERNEL_LOOP(i, n)                        \
  for (int i = get_group_id(0) * get_local_size(0) + get_local_id(0); \
      i < (n);                                       \
      i += get_local_size(0) * get_num_groups(0))

#define Dtype float

kernel void MaxPoolForward(
    const int nthreads,
    global const Dtype* bottom_data_data, int bottom_data_offset,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    global Dtype* top_data_data, int top_data_offset
  ) {

  global const Dtype *bottom_data = bottom_data_data + bottom_data_offset;
  global Dtype *top_data = top_data_data + top_data_offset;

  CL_KERNEL_LOOP(index, nthreads) {
    if(index < nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height);
    int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    Dtype maxval = -FLT_MAX;
    global const float *bottom_data_img = bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        float val = bottom_data_img[h * width + w];
        if (val > maxval) {
          // int maxidx = h * width + w;
          maxval = val;
        }
      }
    }
    top_data[index] = maxval;
  }
  }
}