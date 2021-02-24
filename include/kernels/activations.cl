typedef enum{
    LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN
} ACTIVATION;
float lhtan_activate_kernel_32(float x)
{
    if(x < 0) return .001f*x;
    if(x > 1) return .001f*(x-1.f) + 1.f;
    return x;
}

float hardtan_activate_kernel_32(float x)
{
    if (x < -1) return -1;
    if (x > 1) return 1;
    return x;
}
float linear_activate_kernel_32(float x){return x;}
float logistic_activate_kernel_32(float x){return 1.f/(1.f + exp(-x));}
float loggy_activate_kernel_32(float x){return 2.f/(1.f + exp(-x)) - 1;}
float relu_activate_kernel_32(float x){return x*(x>0);}
float elu_activate_kernel_32(float x){return (x >= 0)*x + (x < 0)*(exp(x)-1);}
float relie_activate_kernel_32(float x){return (x>0) ? x : .01f*x;}
float ramp_activate_kernel_32(float x){return x*(x>0)+.1f*x;}
float leaky_activate_kernel_32(float x){return (x>0) ? x : .1f*x;}
float tanh_activate_kernel_32(float x){return (2.f/(1 + exp(-2*x)) - 1);}
float plse_activate_kernel_32(float x)
{
    if(x < -4) return .01f * (x + 4);
    if(x > 4)  return .01f * (x - 4) + 1;
    return .125f*x + .5f;
}
float stair_activate_kernel_32(float x)
{
    int n = floor(x);
    if (n%2 == 0) return floor(x/2);
    else return (x - n) + floor(x/2);
}
 

float activate_kernel_32(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return x;
        case LOGISTIC:
            return logistic_activate_kernel_32(x);
        case LOGGY:
            return loggy_activate_kernel_32(x);
        case RELU:
            return relu_activate_kernel_32(x);
        case ELU:
            return elu_activate_kernel_32(x);
        case RELIE:
            return relie_activate_kernel_32(x);
        case RAMP:
            return ramp_activate_kernel_32(x);
        case LEAKY:
            return leaky_activate_kernel_32(x);
        case TANH:
            return tanh_activate_kernel_32(x);
        case PLSE:
            return plse_activate_kernel_32(x);
        case STAIR:
            return stair_activate_kernel_32(x);
        case HARDTAN:
            return hardtan_activate_kernel_32(x);
        case LHTAN:
            return lhtan_activate_kernel_32(x);
    }
    return 0;
}



kernel void activate_array_kernel_32(__global float *x, int input_offset, int n, ACTIVATION a)
{
    x = x+input_offset;
    int i = get_global_id(1) * get_global_size(0) + get_global_id(0);
    if(i < n) x[i] = activate_kernel_32(x[i], a);
}