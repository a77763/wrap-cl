#include <clkernel.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#define BLOCK 128

namespace clcore{
    CLCache cache;

    void* getKernel(std::string name){
        std::unordered_map<std::string,void*>::const_iterator elem = cache.find(name);
        if (elem == cache.end())
        {
            return nullptr;
        }
        return elem->second;
    }

    void addKernel(std::string name, void * kernel){
        cache.insert(std::make_pair(name, kernel));
    }

    CLDim3 createGrid2D(size_t gx, size_t gy, size_t lx, size_t ly){
        CLDim3 grid;
        grid.ndim = 2;
        grid.global_ws = (size_t*) malloc(sizeof(size_t)*2);
        grid.local_ws = (size_t*) malloc(sizeof(size_t)*2);

        grid.global_ws[0] = gx;
        grid.global_ws[1] = gy;
        grid.local_ws[0] = lx;
        grid.local_ws[1] = ly;
        return grid;
    }

    
    CLDim3 createGrid(size_t elements)
    {   
        CLDim3 grid;
        size_t k = (elements-1) / BLOCK + 1;
        size_t x = k;
        size_t y = 1;
        //TODO: Get max grid size
        if(x > 65535){
            x = ceil(sqrt(k));
            y = (elements-1)/(x*BLOCK) + 1;
        }
        x = x * BLOCK;

        grid.ndim = 3;
        grid.global_ws = (size_t*) malloc(sizeof(size_t)*3);
        grid.local_ws = (size_t*) malloc(sizeof(size_t)*3);
        grid.global_ws[0] = x;
        grid.global_ws[1] = y;
        grid.global_ws[2] = 1;
        grid.local_ws[0] = BLOCK;
        grid.local_ws[1] = 1;
        grid.local_ws[2] = 1;
        return grid;
    }


    CLKernel::CLKernel(cl_kernel _kernel, cl_program _program, std::string _name) : 
                            kernel(_kernel), program(_program), name(_name)
    {}

    CLKernel::~CLKernel(){
        cl_int error;
        error = clReleaseKernel(kernel);
        clcore::checkError("Release Kernel",error);
        error = clReleaseProgram(program);
        clcore::checkError("Release Program",error);
    }

    size_t readFile(std::string file_path, char** buffer){
        FILE *f = fopen(file_path.c_str(), "r");
        fseek(f, 0, SEEK_END);
        long fsize = ftell(f);
        fseek(f, 0, SEEK_SET);  /* same as rewind(f); */
        *buffer = (char*) malloc(fsize + 1);
        fread(*buffer, 1, fsize, f);
        fclose(f);
        (*buffer)[fsize] = '\0';
        return fsize;
    }


    CLKernel* createKernel(std::string kernel_path, std::string kernel_name){
        CLKernel * kernel = (CLKernel*) clcore::getKernel(kernel_name);
        if( kernel != nullptr)
            return kernel;
        else{
            cl_int error;
            char* source_str;
            size_t source_size;
            source_size = readFile(kernel_path, &source_str);
            cl_program program = clCreateProgramWithSource(clcore::getCLContext(), 1, 
                    (const char **)&source_str, (const size_t *)&source_size, &error);
            error = clBuildProgram(program, 1, clcore::getCLDevice(), NULL, NULL, NULL);
            clcore::checkError("Building program", error);
            cl_kernel clkernel = clCreateKernel(program, kernel_name.c_str(), &error);
            clcore::checkError("Creating kernel", error);
            kernel = new CLKernel(clkernel, program, kernel_name);
            clcore::addKernel(kernel_name, (void*)kernel);
            return kernel;
        }

    }


    

}