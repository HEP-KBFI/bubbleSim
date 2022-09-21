#pragma once

#include <CL/cl.hpp>

#include "base.h"
#pragma comment(lib, "OpenCL.lib")

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
  cl::Buffer m_bufferX;
  cl::Buffer m_bufferP;
  cl::Buffer m_bufferE;
  cl::Buffer m_bufferM;
  cl::Buffer m_buffer_dP;

  cl::Buffer m_buffer_dt;
  cl::Buffer m_bufferMassIn;
  cl::Buffer m_bufferMassOut;
  cl::Buffer m_bufferMassDelta2;

  cl::Buffer m_bufferBubbleRadius;
  cl::Buffer m_bufferBubbleRadius2;
  cl::Buffer m_bufferBubbleRadiusSpeedDt2;
  cl::Buffer m_bufferBubbleSpeed;
  cl::Buffer m_bufferBubbleGamma;
  cl::Buffer m_bufferBubbleGammaSpeed;

  cl::Buffer m_bufferInteractedFalse;
  cl::Buffer m_bufferPassedFalse;
  cl::Buffer m_bufferInteractedTrue;
  cl::Buffer m_bufferPassedTrue;

 public:
  OpenCLWrapper() {}
  OpenCLWrapper(std::string fileName, std::string kernelName,
                u_int t_particleCount, std::vector<numType>& t_X,
                std::vector<numType>& t_P, std::vector<numType>& t_E,
                std::vector<numType>& t_M, std::vector<numType>& t_dP,
                numType& t_dt, numType& t_massIn, numType& t_massOut,
                numType t_massDelta2, numType& t_bubbleRadius,
                numType& t_bubbleRadius2, numType& t_bubbleRadiusAfterDt2,
                numType& t_bubbleSpeed, numType& t_bubbleGamma,
                numType& t_bubbleGammaSpeed,
                std::vector<int8_t>& t_interactedFalse,
                std::vector<int8_t>& t_passedFalse,
                std::vector<int8_t>& t_interactedTrue,
                bool t_isBubbleTrueVacuum);

  void createContext(std::vector<cl::Device>& devices);
  void createProgram(cl::Context& context, cl::Device& device,
                     std::string& kernelFile);
  void createKernel(cl::Program& program, const char* name);
  void createQueue(cl::Context& context, cl::Device& device);

  void makeStep1(numType& t_radius, numType& t_radius2, numType& t_speed,
                 numType& t_gamma, numType& t_gammaSpeed,
                 numType& radiusAfterDt2);
  void makeStep2(int& particleCount);
  void makeStep3(int& particleCount, std::vector<numType>& t_dP);

  void readBufferX(std::vector<numType>& t_dataX);
  void readBufferP(std::vector<numType>& t_dataP);
  void readBuffer_dP(std::vector<numType>& t_data_dP);
  void readBufferM(std::vector<numType>& t_dataM);
  void readBufferE(std::vector<numType>& t_dataE);
  void readBufferInteractedFalse(std::vector<int8_t>& t_dataInteractedFalse);
  void readBufferPassedFalse(std::vector<int8_t>& t_dataPassedFalse);
  void readBufferInteractedTrue(std::vector<int8_t>& t_dataTrue);
  void readBufferR(numType& t_dataR);
  void readBufferSpeed(numType& t_dataSpeed);
  void readBufferBubble(numType& t_dataR, numType& t_dataSpeed);
};