#pragma once
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#define __CL_ENABLE_EXCEPTIONS
#include <string>
typedef cl_command_queue CLQueue;
typedef cl_context CLContext;
typedef cl_device_id * CLDevice;


namespace clcore{
    
    CLDevice getCLDevice();
    CLQueue getCLQueue();
    CLContext getCLContext();
    void releaseCLQueue();
    void releaseCLContext();
    void checkError(std::string message, cl_int error);
};
