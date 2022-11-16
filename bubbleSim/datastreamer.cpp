#include "datastreamer.h"

DataStreamer::DataStreamer(Simulation& t_sim, PhaseBubble& t_bubble,
                           OpenCLWrapper& t_openCLWrapper)
    : m_sim(t_sim), m_bubble(t_bubble), m_openCLWrapper(t_openCLWrapper) {
  m_readBufferParticle = true;
  m_readBuffer_dP = true;

  m_readBufferInteractedFalse = true;
  m_readBufferPassedFalse = true;
  m_readBufferInteractedTrue = true;

  m_readBufferBubble = true;
}

void DataStreamer::reset() {
  m_readBufferParticle = true;
  m_readBuffer_dP = true;

  m_readBufferInteractedFalse = true;
  m_readBufferPassedFalse = true;
  m_readBufferInteractedTrue = true;

  m_readBufferBubble = true;
}

void DataStreamer::streamBaseData(std::fstream& t_stream,
                                  bool t_isBubbleTrueVacuum) {
  int countParticleFalse = 0, countParticleInteractedFalse = 0,
      countParticlePassedFalse = 0, countParticleInteratedTrue = 0;
  numType particleEnergy = 0., particleEnergyFalse = 0.;
  numType changeInPressure = 0.;
  numType particlesEnergy = 0.;

  if (m_readBufferParticle) {
    m_openCLWrapper.readBufferParticle(m_sim.getRef_Particles());
    m_readBufferParticle = false;
  }

  if (m_readBufferInteractedFalse) {
    m_openCLWrapper.readBufferInteractedFalse(m_sim.getRef_InteractedFalse());
    m_readBufferInteractedFalse = false;
  }

  if (m_readBufferInteractedTrue) {
    m_openCLWrapper.readBufferInteractedTrue(m_sim.getRef_InteractedTrue());
    m_readBufferInteractedTrue = false;
  }

  if (m_readBufferPassedFalse) {
    m_openCLWrapper.readBufferPassedFalse(m_sim.getRef_PassedFalse());
    m_readBufferPassedFalse = false;
  }

  // If true vacuum is inside the bubble
  if (t_isBubbleTrueVacuum) {
    for (int i = 0; i < m_sim.getParticleCountTotal(); i++) {
      particlesEnergy += m_sim.getParticleEnergy(i);
      if (m_sim.calculateParticleRadius(i) > m_bubble.getRadius()) {
        countParticleFalse += 1;
        particleEnergyFalse += m_sim.getParticleEnergy(i);
      }
      countParticleInteractedFalse += m_sim.getRef_InteractedFalse()[i];
      countParticlePassedFalse += m_sim.getRef_PassedFalse()[i];
      countParticleInteratedTrue += m_sim.getRef_InteractedTrue()[i];
    }
  }
  // If true vacuum is outside the bubble
  else {
    for (int i = 0; i < m_sim.getParticleCountTotal(); i++) {
      particlesEnergy += m_sim.getParticleEnergy(i);
      if (m_sim.calculateParticleRadius(i) < m_bubble.getRadius()) {
        countParticleFalse += 1;
        particleEnergyFalse += m_sim.getRef_Particles()[i].E;
      }
      countParticleInteractedFalse += m_sim.getRef_InteractedFalse()[i];
      countParticlePassedFalse += m_sim.getRef_PassedFalse()[i];
      countParticleInteratedTrue += m_sim.getRef_InteractedTrue()[i];
    }
  }
  t_stream << std::setprecision(10);
  t_stream << m_sim.getTime() << "," << m_sim.getdPressureStep() << ","
           << m_bubble.getRadius() << "," << m_bubble.getSpeed() << ",";
  t_stream << m_bubble.calculateEnergy() << ",";
  t_stream << particlesEnergy << ",";
  t_stream << particleEnergyFalse << ","
           << (particlesEnergy + m_bubble.calculateEnergy()) /
                  m_sim.getInitialTotalEnergy()
           << ",";
  t_stream << countParticleFalse << "," << countParticleInteractedFalse << ","
           << countParticlePassedFalse << ",";
  t_stream << countParticleInteratedTrue << std::endl;

  m_openCLWrapper.writeResetInteractedFalseBuffer(
      m_sim.getRef_InteractedFalse());
  m_openCLWrapper.writeResetPassedFalseBuffer(m_sim.getRef_PassedFalse());
  m_openCLWrapper.writeResetInteractedTrueBuffer(m_sim.getRef_InteractedTrue());
}

int DataStreamer::countMassRadiusDifference(bool t_isBubbleTrueVacuum) {
  int countMassTrue = 0, countMassFalse = 0;
  int countRadiusTrue = 0, countRadiusFalse = 0;

  if (m_readBufferParticle) {
    m_openCLWrapper.readBufferParticle(m_sim.getRef_Particles());
    m_readBufferParticle = false;
  }

  for (int i = 0; i < m_sim.getParticleCountTotal(); i++) {
    if (m_sim.getParticleMass(i) == m_sim.getMassFalse()) {
      countMassFalse += 1;
    } else {
      countMassTrue += 1;
    }
    if (t_isBubbleTrueVacuum) {
      if (m_sim.calculateParticleRadius(i) > m_bubble.getRadius()) {
        countRadiusFalse += 1;
      } else {
        countRadiusTrue += 1;
      }
    } else {
      if (m_sim.calculateParticleRadius(i) > m_bubble.getRadius()) {
        countRadiusTrue += 1;
      } else {
        countRadiusFalse += 1;
      }
    }
  }
  std::printf("\nTotal count (M/R): %.5f\n",
              (double)(countMassTrue + countMassFalse) /
                  (countRadiusTrue + countRadiusFalse));
  std::printf("True/False vacuum difference (M-R): %d / %d\n",
              countMassTrue - countRadiusTrue,
              countMassFalse - countRadiusFalse);
  return countMassFalse - countRadiusFalse;
}

void DataStreamer::streamParticleInfo() {
  if (m_readBufferParticle) {
    m_openCLWrapper.readBufferParticle(m_sim.getRef_Particles());
    m_readBufferParticle = false;
  }
  for (int i = 0; i < m_sim.getParticleCountTotal(); i++) {
    std::cout << m_sim.getRef_Particles()[i].x << ", "
              << m_sim.getRef_Particles()[i].y << ", "
              << m_sim.getRef_Particles()[i].z << ", ";
    std::cout << m_sim.getRef_Particles()[i].p_x << ", "
              << m_sim.getRef_Particles()[i].p_y << ", "
              << m_sim.getRef_Particles()[i].p_z << ", ";
    std::cout << m_sim.getRef_Particles()[i].m << ", "
              << m_sim.getRef_Particles()[i].E << std::endl;
  }
}

void DataStreamer::streamParticleInfo(std::fstream& t_stream) {
  for (int i = 0; i < m_sim.getParticleCountTotal(); i++) {
    t_stream << m_sim.getRef_Particles()[i].x << ","
             << m_sim.getRef_Particles()[i].y << ","
             << m_sim.getRef_Particles()[i].z << ",";
    t_stream << m_sim.getRef_Particles()[i].p_x << ","
             << m_sim.getRef_Particles()[i].p_y << ","
             << m_sim.getRef_Particles()[i].p_z << ",";
    t_stream << m_sim.getRef_Particles()[i].m << ","
             << m_sim.getRef_Particles()[i].E << std::endl;
  }
}

void DataStreamer::streamProfiles(std::fstream& t_nStream,
                                  std::fstream& t_rhoStream,
                                  std::fstream& t_pInStream,
                                  std::fstream& t_pOutStream,
                                  int t_densityCountBins, int t_pCountBins,
                                  numType t_radiusMax, numType t_pMax,
                                  numType t_energyDensityNormalizer) {
  std::vector<int> nBins(t_densityCountBins, 0);
  std::vector<numType> rhoBins(t_densityCountBins, 0);
  std::vector<int> pInBins(t_pCountBins, 0);
  std::vector<int> pOutBins(t_pCountBins, 0);

  numType r;
  numType dr = t_radiusMax / t_densityCountBins;
  numType p;
  numType dp = t_pMax / t_pCountBins;
  int j;

  /*
    Data processing, gathering
  */
  for (int i = 0; i < m_sim.getParticleCountTotal(); i++) {
    r = m_sim.calculateParticleRadius(i);
    p = m_sim.calculateParticleMomentum(i);
    if (r > m_bubble.getRadius()) {
      pOutBins[std::min((int)(p / dp), t_pCountBins - 1)] += 1;
    } else {
      pInBins[std::min((int)(p / dp), t_pCountBins - 1)] += 1;
    }
    nBins[std::min((int)(r / dr), t_densityCountBins - 1)] += 1;
    rhoBins[std::min((int)(r / dr), t_densityCountBins - 1)] +=
        m_sim.getParticleEnergy(i);
  }
  /*
    Streaming data to files
  */

  for (j = 0; j < t_densityCountBins; j++) {
    t_nStream << nBins[j];
    t_rhoStream << rhoBins[j] /
                       (4 * M_PI * std::pow(dr, 3) * (j * j + j + 1.0 / 3.0)) /
                       t_energyDensityNormalizer;
    if (j != t_densityCountBins - 1) {
      t_nStream << ",";
      t_rhoStream << ",";
    } else {
      t_nStream << std::endl;
      t_rhoStream << std::endl;
    }
  }

  for (j = 0; j < t_pCountBins; j++) {
    t_pInStream << pInBins[j];
    t_pOutStream << pOutBins[j];
    if (j != t_pCountBins - 1) {
      t_pInStream << ",";
      t_pOutStream << ",";
    } else {
      t_pInStream << std::endl;
      t_pOutStream << std::endl;
    }
  }
}
