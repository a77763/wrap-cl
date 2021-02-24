// CL: grid stride looping
#define CL_KERNEL_LOOP(i, n)                        \
  for (int i = get_group_id(0) * get_local_size(0) + get_local_id(0); \
      i < (n);                                       \
      i += get_local_size(0) * get_num_groups(0))

#define Dtype float
#define {ACTIVATION_TYPE}

kernel void {ACTIVATION_TYPE}Forward(
    const int nthreads,
    global const Dtype* input_data, int input_offset,
    global Dtype* output_data, int output_offset
  ) {

  global const Dtype *input = input_data + input_offset;
  global Dtype *output = output_data + output_offset;

  CL_KERNEL_LOOP(index, nthreads) {
    if(index < nthreads) {
        float inval = input[index];

        #ifdef RELU
        output[index] = inval > 0 ? inval : 0.0f;
        #endif

        #ifdef SIGMOID
        output[index] = 1.0f / (1.0f + exp(- inval) );
        #endif

        #ifdef TANH
        output[index] = tanh(inval);
        #endif
    }
  }
}