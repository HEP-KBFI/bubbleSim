#pragma once

#include <CL/cl.hpp>
#pragma comment(lib, "OpenCL.lib")

#include "base.h"
#include "objects.h"
#include "kernels.h"

class OpenCLLoader {
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
  cl::Context m_context;

  cl::Kernel m_labelKernel;

  // Cell assignment kernel
  // Lable if in bubble or not kernel
  // Collision kernel

  cl::CommandQueue m_queue;

 public:
  OpenCLLoader() {}
  OpenCLLoader(std::string kernelsPath);
  OpenCLLoader(std::string kernelPath, std::string kernelName);
  cl::Kernel m_kernel;

  ParticleStepLinearKernel m_particleLinearStepKernel;
  AssignParticleToCollisionCellTwoMassStateKernel m_cellAssignmentKernelTwoMassState;
  AssignParticleToCollisionCellKernel m_cellAssignmentKernel;
  MomentumRotationKernel m_rotationKernel;
  ParticleBoundaryCheckKernel m_particleBoundaryKernel;
  ParticleStepWithBubbleKernel m_particleStepWithBubbleKernel;
  CollisionCellResetKernel m_collisionCellResetKernel;
  CollisionCellGenerationKernel m_collisionCellCalculateGenerationKernel;
  CollisionCellSumParticlesKernel m_collisionCellSumParticlesKernel;
  ParticleLabelByCoordinateKernel m_particleInBubbleKernel;

  // cl::Kernel m_collisionCellCalculateSummationKernel;

  cl::CommandQueue& getCommandQueue() { return m_queue; }
  cl::Context& getContext() { return m_context; }
  cl::Kernel& getKernel() { return m_kernel; }

  void createContext(std::vector<cl::Device>& devices);
  void createProgram(cl::Context& context, cl::Device& device,
                     std::string& kernelFile);
  void createKernel(cl::Program& program, cl::Kernel& kernel, const char* name);
  void createQueue(cl::Context& context, cl::Device& device);
};