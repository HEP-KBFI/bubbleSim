#pragma once
#include <fstream>

#include "base.h"
#include "bubble.h"
#include "openclwrapper.h"
#include "simulation.h"

class DataStreamer {
  bool m_readBufferX;
  bool m_readBufferP;
  bool m_readBuffer_dP;
  bool m_readBufferM;
  bool m_readBufferE;

  bool m_readBufferInteractedFalse;
  bool m_readBufferPassedFalse;
  bool m_readBufferInteractedTrue;

  bool m_readBufferR;
  bool m_readBufferSpeed;

  Simulation& m_sim;
  Bubble& m_bubble;
  OpenCLWrapper& m_openCLWrapper;

 public:
  DataStreamer(Simulation& t_sim, Bubble& t_bubble,
               OpenCLWrapper& t_openCLWrapper);

  void reset();

  void streamBaseData(std::fstream& t_stream, bool t_isBubbleTrueVacuum);
  /*
  Streams for:

  ==	time, dP, Radius, Speed, bubbleEnergy, bubbleEnergy - bubbleEnergyOld,
          particleEnergy, particleEnergy - particleEnergyOld, particleEnergyIn,
          (bubbleEnergy + particleEnergy) / sim.m_energyTotalInitial,
          countParticleIn, countParticleInteractedFalse,
  countParticleInteractedPassedFalse, countParticleInteratedTrue

  == number density profile

  == energy density profile

  == momentum profile

  */

  void streamMassRadiusDifference(bool t_isBubbleTrueVacuum);

  void streamParticleInfo();

  void streamParticleInfo(std::fstream& t_stream);
};

