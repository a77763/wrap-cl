#include "clmemory.h"
#include <stdexcept>

void clMemcpy(void *dst, const void *src, size_t bytes, clMemcpyKind kind) {
    cl_int err;
    if(kind == clMemcpyDeviceToHost) {
        err = clEnqueueReadBuffer(clcore::getCLQueue(), (cl_mem)src, CL_TRUE, 0,
                                         bytes, dst, 0, NULL, NULL);
        clcore::checkError("Read buffer", err);
    } else if(kind == clMemcpyHostToDevice) {
        err = clEnqueueWriteBuffer(clcore::getCLQueue(), (cl_mem)dst, CL_TRUE, 0,
                                          bytes, src, 0, NULL, NULL);
        clcore::checkError("Write buffer", err);
    } else if(kind == clMemcpyDeviceToDevice) {
        err = clEnqueueCopyBuffer(
            clcore::getCLQueue(),
            (cl_mem)src, (cl_mem)dst,
            0, 0,
            bytes,
            0, 0, 0);
        clcore::checkError("Copy buffer", err);
    } else {
        throw std::runtime_error("unhandled clMemcpyKind");
    }
}
void clMemsetD32(void * ptr, unsigned int value, size_t count){
    cl_int err = clEnqueueFillBuffer(clcore::getCLQueue(), (cl_mem) ptr, &value, sizeof(unsigned int), 0, count * sizeof(unsigned int), 0, 0, 0);
    clcore::checkError("Memset",err);
}	

void clMalloc( void ** ptr, size_t size) {
    CLContext context = clcore::getCLContext();
    cl_int err;
    (*ptr) = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &err);
    clcore::checkError("Malloc", err);
}

void clFree(void * ptr) {
    cl_int err = clReleaseMemObject((cl_mem)ptr);
    clcore::checkError("Free",err);
}
