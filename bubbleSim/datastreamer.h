#pragma once
#define _USE_MATH_DEFINES

#include <fstream>

#include "base.h"
#include "bubble.h"
#include "openclwrapper.h"
#include "simulation.h"

class DataStreamer {
  bool m_readBufferParticle;
  bool m_readBufferBubble;
  bool m_readBuffer_dP;

  bool m_readBufferInteractedFalse;
  bool m_readBufferPassedFalse;
  bool m_readBufferInteractedTrue;

  Simulation& m_sim;
  PhaseBubble& m_bubble;
  OpenCLWrapper& m_openCLWrapper;

 public:
  DataStreamer(Simulation& t_sim, PhaseBubble& t_bubble,
               OpenCLWrapper& t_openCLWrapper);

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
