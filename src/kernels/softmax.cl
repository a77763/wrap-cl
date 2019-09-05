// CL: grid stride looping
#define CL_KERNEL_LOOP(i, n)                        \
  for (int i = get_group_id(0) * get_local_size(0) + get_local_id(0); \
      i < (n);                                       \
      i += get_local_size(0) * get_num_groups(0))

#define Dtype float

// each thread will be one example (provide result over all c values for that example)
kernel void SoftmaxForward(
    const int N,
    global const Dtype* input_data, int input_offset,
    const int C,
    global Dtype* output_data, int output_offset
  ) {

  global const Dtype *input = input_data + input_offset;
  global Dtype *output = output_data + output_offset;

  CL_KERNEL_LOOP(n, N) {
    if(n < N) {
        // float inval = input[index];
        // output[index] = inval > 0 ? inval : 0.0f;

        const global float *inputCube = &input[n * C];
        global float *outputCube = &output[n * C];

        // first get the max
        float maxValue = inputCube[0];
        for(int c = 1; c < C; c++) {
            maxValue = max(maxValue, inputCube[c]);
        }
        // calculate sum, under this max
        float denominator = 0;
        for(int c = 0; c < C; c++) {
            denominator += exp(inputCube[c] - maxValue);
        }
        // now calc the softmaxes:
        for(int c = 0; c < C; c++) {
            outputCube[c] = exp(inputCube[c] - maxValue) / denominator;
        }
    }
  }
}