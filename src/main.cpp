#include <stdio.h>
#include <clcore.hpp>
#include <clmemory.hpp>
#include <clkernel.hpp>
#include <sstream>
#include <iostream>

int * rand_array(int count){
    int * arr = new int[count];
    srand(time(NULL));
    for (size_t i = 0; i < count; i++)
    {
        arr [i] = rand()%(count);
    }
        
    return arr;
}

template<class T>
void print_array(std::string info, T *arr, int count){
    std::stringstream ss;
    ss<<info;
    ss<<" ";
    for(size_t i = 0; i < count; ++i)
    {
        if(i != 0)
        ss << ",";
        ss << arr[i];
    }
    ss <<std::endl;
    std::string s = ss.str();
    std::cout << s;
}

void setVal (float * arr, int count, float val){
    for (size_t i = 0; i < 20; i++)
    {
        for (size_t j = 0; j < 20; j++)
            arr[i*20+j] = j*20 + i;
    }
    
}

int main(int argc, char** argv){
    float * array = (float* ) malloc(sizeof(float)*400);
    float * array2 = (float* ) malloc(sizeof(float)*400);
    float * array3 = (float* ) malloc(sizeof(float)*400);
    setVal(array,400,1);
    setVal(array2,400,2);
    print_array("before:", array2,400);
    void * dev_ptr_a, * dev_ptr_b, * dev_ptr_c;
    clMalloc(&dev_ptr_a, sizeof(float)*400);
    clMalloc(&dev_ptr_b, sizeof(float)*400);
    clMalloc(&dev_ptr_c, sizeof(float)*400);
    clMemcpy(dev_ptr_a, array, sizeof(float)*400, clMemcpyHostToDevice);
    clMemcpy(dev_ptr_b, array2, sizeof(float)*400, clMemcpyHostToDevice);
    clcore::CLKernel * gemm = clcore::createKernel("/Users/trident/Documents/WrapCL/src/transpose.cl", "transpose");
    int K,N;
    K=N=20;
    CLDim3 dim = clcore::createGrid2D((size_t)32,(size_t)32,16,16);
    gemm->run(dim,clcore::getCLQueue(),K,N,dev_ptr_b,dev_ptr_c);
    clMemcpy(array3,dev_ptr_c,400,clMemcpyDeviceToHost);
    print_array("result:", array3, 400);

    clFree(dev_ptr_a);
    clFree(dev_ptr_b);
    clFree(dev_ptr_c);
    free(array);
    free(array2);
    free(array3);
    return 0;
}