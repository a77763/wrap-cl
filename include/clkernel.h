#ifndef CLKERNEL
#define CLKERNEL
#include "clcore.h"
#include <string>
#include <unordered_map>
#include <iostream>
typedef struct _cl_dim3{
    size_t ndim;
    size_t *global_ws;
    size_t *local_ws;
}CLDim3;
typedef std::unordered_map<std::string,void*> CLCache;


namespace clcore{
    class CLKernel{
        private:
            cl_kernel kernel;
            cl_program program; 
            std::string name;
            template<class T>
                void setArgs(int index, T arg){
                    cl_int error;
                    error = clSetKernelArg(this->kernel, index, sizeof(arg), (void *)&arg);
                    std::cout << index << ": " << sizeof(arg)<< std::endl;
                    clcore::checkError("Set kernel argument", error);
                }
            template<class T, typename... Args>
                void setArgs(int index, T arg, Args... args){
                    cl_int error;
                    error = clSetKernelArg(this->kernel, index, sizeof(arg), (void *)&arg);
                    std::cout << index << ": " << sizeof(arg)<< std::endl;
                    clcore::checkError("Set kernel argument", error);
                    index++;
                    setArgs(index, args...);
                }

        public:
            CLKernel(cl_kernel kernel, cl_program program, std::string name);
            ~CLKernel();
            template<typename... Args>
                void run(CLDim3 workspace, CLQueue queue, Args... args){
                    cl_int error;
                    setArgs(0, args ...);
                    error = clEnqueueNDRangeKernel(queue, this->kernel, workspace.ndim, NULL, 
                            workspace.global_ws, workspace.local_ws, 0, NULL, NULL);
                    clcore::checkError("Enqueue", error);
                    error = clFinish(clcore::getCLQueue());
                    clcore::checkError("Finish", error);
                }
            
    };
    void* getKernel(std::string name);
    void addKernel(std::string name, void * kernel);
    CLDim3 createGrid(size_t elements);
    CLDim3 createGrid2D(size_t gx, size_t gy, size_t lx, size_t ly);
    CLKernel* createKernel(std::string kernel_path, std::string kernel_name);
}
#endif