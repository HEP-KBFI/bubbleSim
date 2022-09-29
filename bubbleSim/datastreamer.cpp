#include "datastreamer.h"

DataStreamer::DataStreamer(Simulation& t_sim, Bubble& t_bubble,
                           OpenCLWrapper& t_openCLWrapper)
    : m_sim(t_sim), m_bubble(t_bubble), m_openCLWrapper(t_openCLWrapper) {
  m_readBufferX = true;
  m_readBufferP = true;
  m_readBuffer_dP = true;
  m_readBufferM = true;
  m_readBufferE = true;

  m_readBufferInteractedFalse = true;
  m_readBufferPassedFalse = true;
  m_readBufferInteractedTrue = true;

  m_readBufferR = true;
  m_readBufferSpeed = true;
}

void DataStreamer::reset() {
  m_readBufferX = true;
  m_readBufferP = true;
  m_readBuffer_dP = true;
  m_readBufferM = true;
  m_readBufferE = true;

  m_readBufferInteractedFalse = true;
  m_readBufferPassedFalse = true;
  m_readBufferInteractedTrue = true;

  m_readBufferR = true;
  m_readBufferSpeed = true;
}

void DataStreamer::streamBaseData(std::fstream& t_stream,
                                  bool t_isBubbleTrueVacuum) {
  int countParticleFalse = 0, countParticleInteractedFalse = 0,
      countParticlePassedFalse = 0, countParticleInteratedTrue = 0;
  numType particleEnergy = 0., particleEnergyFalse = 0.;
  numType changeInPressure = 0.;
  numType particlesEnergy = 0.;

  if (m_readBufferX) {
    m_openCLWrapper.readBufferX(m_sim.getReferenceX());
    m_readBufferX = false;
  }
  if (m_readBufferE) {
    m_openCLWrapper.readBufferE(m_sim.getReferenceE());
    m_readBufferE = false;
  }

  if (m_readBufferInteractedFalse) {
    m_openCLWrapper.readBufferInteractedFalse(
        m_sim.getReferenceInteractedFalse());
    m_readBufferInteractedFalse = false;
  }

  if (m_readBufferInteractedTrue) {
    m_openCLWrapper.readBufferInteractedTrue(m_sim.getReferenceInteractedTrue());
    m_readBufferInteractedTrue = false;
  }

  if (m_readBufferPassedFalse) {
    m_openCLWrapper.readBufferPassedFalse(
        m_sim.getReferencePassedFalse());
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
      countParticleInteractedFalse += m_sim.getReferenceInteractedFalse()[i];
      countParticlePassedFalse += m_sim.getReferencePassedFalse()[i];
      countParticleInteratedTrue += m_sim.getReferenceInteractedTrue()[i];
    }
  }
  // If true vacuum is outside the bubble
  else {
    for (int i = 0; i < m_sim.getParticleCountTotal(); i++) {
      particlesEnergy += m_sim.getParticleEnergy(i);
      if (m_sim.calculateParticleRadius(i) < m_bubble.getRadius()) {
        countParticleFalse += 1;
        particleEnergyFalse += m_sim.getReferenceE()[i];
      }
      countParticleInteractedFalse += m_sim.getReferenceInteractedFalse()[i];
      countParticlePassedFalse += m_sim.getReferencePassedFalse()[i];
      countParticleInteratedTrue += m_sim.getReferenceInteractedTrue()[i];
    }
  }
  t_stream << std::setprecision(10);
  t_stream << m_sim.getTime() << "," << m_sim.getdPressureStep() << ","
           << m_bubble.getRadius() << "," << m_bubble.getSpeed() << ",";
  t_stream << m_bubble.calculateEnergy() << ",";
  t_stream << particlesEnergy << ",";
  t_stream << particleEnergyFalse << ","
           << (particlesEnergy + m_bubble.calculateEnergy()) / m_sim.getTotalEnergyInitial() << ",";
  t_stream << countParticleFalse << "," << countParticleInteractedFalse << ","
           << countParticlePassedFalse << ",";
  t_stream << countParticleInteratedTrue << std::endl;

  m_openCLWrapper.writeResetInteractedFalseBuffer(
      m_sim.getReferenceInteractedFalse());
  m_openCLWrapper.writeResetPassedFalseBuffer(
      m_sim.getReferencePassedFalse());
  m_openCLWrapper.writeResetInteractedTrueBuffer(
      m_sim.getReferenceInteractedTrue());
 }

int DataStreamer::streamMassRadiusDifference(bool t_isBubbleTrueVacuum) {
  int countMassTrue = 0, countMassFalse = 0;
  int countRadiusTrue = 0, countRadiusFalse = 0;

  if (m_readBufferX) {
    m_openCLWrapper.readBufferX(m_sim.getReferenceX());
    m_readBufferX = false;
  }
  if (m_readBufferM) {
    m_openCLWrapper.readBufferM(m_sim.getReferenceM());
    m_readBufferM = false;
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
  return (countMassTrue - countRadiusTrue);
}

void DataStreamer::streamParticleInfo() {
  if (m_readBufferX) {
    m_openCLWrapper.readBufferX(m_sim.getReferenceX());
    m_readBufferX = false;
  }
  if (m_readBufferP) {
    m_openCLWrapper.readBufferP(m_sim.getReferenceP());
    m_readBufferP = false;
  }
  if (m_readBufferE) {
    m_openCLWrapper.readBufferE(m_sim.getReferenceE());
    m_readBufferE = false;
  }
  if (m_readBufferM) {
    m_openCLWrapper.readBufferM(m_sim.getReferenceM());
    m_readBufferM = false;
  }
  for (int i = 0; i < m_sim.getParticleCountTotal(); i++) {
    std::cout << m_sim.getReferenceX()[3 * i] << ", "
              << m_sim.getReferenceX()[3 * i + 1] << ", "
              << m_sim.getReferenceX()[3 * i + 2] << ", ";
    std::cout << m_sim.getReferenceP()[3 * i] << ", "
              << m_sim.getReferenceP()[3 * i + 1] << ", "
              << m_sim.getReferenceP()[3 * i + 2] << ", ";
    std::cout << m_sim.getParticleMass(i) << ", " << m_sim.getParticleEnergy(i)
              << std::endl;
  }
}

void DataStreamer::streamParticleInfo(std::fstream& t_stream) {
  for (int i = 0; i < m_sim.getParticleCountTotal(); i++) {
    t_stream << m_sim.getReferenceX()[3 * i] << ", "
             << m_sim.getReferenceX()[3 * i + 1] << ", "
             << m_sim.getReferenceX()[3 * i + 2] << ", ";
    t_stream << m_sim.getReferenceP()[3 * i] << ", "
             << m_sim.getReferenceP()[3 * i + 1] << ", "
             << m_sim.getReferenceP()[3 * i + 2] << ", ";
    t_stream << m_sim.getParticleMass(i) << ", " << m_sim.getParticleEnergy(i)
             << std::endl;
  }
}

void DataStreamer::streamProfiles(std::fstream& t_nStream,
                                  std::fstream& t_rhoStream,
                                  std::fstream& t_pInStream,
                                  std::fstream& t_pOutStream,
                                  u_int t_densityCountBins, u_int t_pCountBins,
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
  u_int j;

  /*
    Data processing, gathering
  */
  for (int i = 0; i < m_sim.getParticleCountTotal(); i++) {
    r = m_sim.calculateParticleRadius(i);
    p = m_sim.calculateParticleMomentum(i);
    if (r > m_bubble.getRadius()) {
      pOutBins[(int)(p / dp)] += 1;
    } else {
      pInBins[(int)(p / dp)] += 1;
    }
    nBins[(int)(r / dr)] += 1;
    rhoBins[(int)(r / dr)] += m_sim.getParticleEnergy(i);
  }
  /*
    Streaming data to files
  */

  for (j = 0; j < t_densityCountBins; j++) {
    t_nStream << nBins[j];
    t_rhoStream << rhoBins[j] /
                       (4 * M_PI * std::pow(dr, 3) * (j * j + j + 1.0 / 3.0)) /
                       t_energyDensityNormalizer;
    if (j != t_densityCountBins) {
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
      t_pInStream << ", ";
      t_pOutStream << ", ";
    } else {
      t_pInStream << std::endl;
      t_pOutStream << std::endl;
    }
  }
}
