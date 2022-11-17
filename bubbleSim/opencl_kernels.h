#pragma once

#define CL_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#include <CL/cl.hpp>
#pragma comment(lib, "OpenCL.lib")

#include "base.h"
#include "objects.h"

class OpenCLKernelLoader {
  /*
   * Using same kernel for different buffers?
   * https://community.khronos.org/t/calling-the-same-kernel-object-multiple-times/1340
   * "Advised to create multiple kernels"
   *
   * https://stackoverflow.com/questions/57954753/opencl-same-kernel-in-separate-queues
   *
   * Copying buffers to device (when creating CL_MEM_COPY_HOST_PTR)
   * https://stackoverflow.com/questions/50041546/opencl-clsetkernelarg-vs-clsetkernelarg-clenqueuewritebuffer
   * https://registry.khronos.org/OpenCL/sdk/1.2/docs/man/xhtml/clCreateBuffer.html
   *
   */

  // OpenCL stuff
  std::vector<cl::Device> m_devices;
  cl::Platform m_platform;

  cl::Device m_deviceUsed;
  cl::Program m_program;

 public:
  cl::Context m_context;
  cl::Kernel m_kernel;
  cl::CommandQueue m_queue;

  OpenCLKernelLoader() {}
  OpenCLKernelLoader(std::string fileName, std::string kernelName);

  cl::CommandQueue& getCommandQueue() { return m_queue; }
  cl::Context& getContext() { return m_context; }
  cl::Kernel& getKernel() { return m_kernel; }
  void createContext(std::vector<cl::Device>& devices);
  void createProgram(cl::Context& context, cl::Device& device,
                     std::string& kernelFile);
  void createKernel(cl::Program& program, const char* name);
  void createQueue(cl::Context& context, cl::Device& device);
};