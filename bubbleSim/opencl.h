#pragma once
#pragma comment(lib, "OpenCL.lib")
#include<CL/cl.hpp>
#include "base.h"

cl::Context CreateContext(std::vector<cl::Device>& devices);
cl::Program CreateProgram(cl::Context& context, cl::Device& device, std::string& kernelFile);
cl::Kernel CreateKernel(cl::Program& program, const char* name);
cl::CommandQueue CreateQueue(cl::Context& context, cl::Device& device);