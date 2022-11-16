#pragma once
#define _USE_MATH_DEFINES

#include <fstream>

#include "base.h"
#include "bubble.h"
#include "opencl_kernels.h"
#include "simulation.h"

class DataStreamer {
  bool m_readBufferParticle;
  bool m_readBufferBubble;
  bool m_readBuffer_dP;

  bool m_readBufferInteractedFalse;
  bool m_readBufferPassedFalse;
  bool m_readBufferInteractedTrue;

  ParticleCollection& m_sim;
  PhaseBubble& m_bubble;
  OpenCLKernelLoader& m_openCLWrapper;

 public:
  DataStreamer(ParticleCollection& t_sim, PhaseBubble& t_bubble,
               OpenCLKernelLoader& t_openCLWrapper);

  void reset();

  void streamBaseData(std::fstream& t_stream, bool t_isBubbleTrueVacuum);

  int countMassRadiusDifference(bool t_isBubbleTrueVacuum);

  void streamParticleInfo();

  void streamParticleInfo(std::fstream& t_stream);

  void streamProfiles(std::fstream& t_nStream, std::fstream& t_rhoStream,
                      std::fstream& t_pInStream, std::fstream& t_pOutStream,
                      int t_densityCountBins, int t_pCountBins,
                      numType t_radiusMax, numType t_pMax,
                      numType t_energyDensityNormalizer);
};
