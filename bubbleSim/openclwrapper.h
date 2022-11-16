#pragma once

#define CL_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#include <CL/cl.hpp>
#pragma comment(lib, "OpenCL.lib")

#include "base.h"
#include "objects.h"

class OpenCLWrapper {
  // OpenCL stuff
  std::vector<cl::Device> m_devices;
  cl::Platform m_platform;
  cl::Context m_context;
  cl::Device m_deviceUsed;
  cl::Program m_program;
  cl::Kernel m_kernel;
  cl::CommandQueue m_queue;

  // Buffers
  cl::Buffer m_bufferParticle;
  cl::Buffer m_buffer_dP;

  cl::Buffer m_buffer_dt;
  cl::Buffer m_bufferMassIn;
  cl::Buffer m_bufferMassOut;
  cl::Buffer m_bufferMassDelta2;

  cl::Buffer m_bufferBubble;

  cl::Buffer m_bufferInteractedFalse;
  cl::Buffer m_bufferPassedFalse;
  cl::Buffer m_bufferInteractedTrue;

 public:
  OpenCLWrapper() {}
  OpenCLWrapper(std::string fileName, std::string kernelName,
                std::vector<Particle> t_particles, std::vector<numType>& t_dP,
                numType& t_dt, numType& t_massTrue, numType& t_massFalse,
                numType t_massDelta2, Bubble t_bubble,
                std::vector<int8_t>& t_interactedFalse,
                std::vector<int8_t>& t_passedFalse,
                std::vector<int8_t>& t_interactedTrue,
                bool t_isBubbleTrueVacuum);

  cl::CommandQueue& getReferenceQueue() { return m_queue; }
  cl::Buffer& getReferenceInteractedFalse() { return m_bufferInteractedFalse; }
  cl::Buffer& getReferenceInteractedTrue() { return m_bufferInteractedTrue; }
  cl::Buffer& getReferencePassedFalse() { return m_bufferPassedFalse; }
  void createContext(std::vector<cl::Device>& devices);
  void createProgram(cl::Context& context, cl::Device& device,
                     std::string& kernelFile);
  void createKernel(cl::Program& program, const char* name);
  void createQueue(cl::Context& context, cl::Device& device);

  void makeStep1(Bubble& t_bubble);
  void makeStep2(int& particleCount);
  void makeStep3(int& particleCount, std::vector<numType>& t_dP);
  void makeStep4(Bubble& t_bubble);
  void readBufferParticle(std::vector<Particle>& t_vectorParticle);
  void readBuffer_dP(std::vector<numType>& t_data_dP);

  void readBufferInteractedFalse(std::vector<int8_t>& t_dataInteractedFalse);
  void readBufferPassedFalse(std::vector<int8_t>& t_dataPassedFalse);
  void readBufferInteractedTrue(std::vector<int8_t>& t_dataTrue);

  void writeResetInteractedFalseBuffer(std::vector<int8_t>& v_interacted);
  void writeResetPassedFalseBuffer(std::vector<int8_t>& v_interacted);
  void writeResetInteractedTrueBuffer(std::vector<int8_t>& v_interacted);
};