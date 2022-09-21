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

  if (m_readBufferX) {
    m_openCLWrapper.readBufferX(m_sim.getReferenceX());
    m_readBufferX = false;
  }

  if (m_readBufferInteractedFalse) {
    m_openCLWrapper.readBufferInteractedFalse(
        m_sim.getReferenceInteractedFalse());
    m_readBufferInteractedFalse = false;
  }

  if (m_readBufferInteractedFalse) {
    m_openCLWrapper.readBufferPassedFalse(m_sim.getReferencePassedFalse());
    m_readBufferInteractedFalse = false;
  }

  if (m_readBufferInteractedFalse) {
    m_openCLWrapper.readBufferInteractedFalse(
        m_sim.getReferenceInteractedFalse());
    m_readBufferInteractedFalse = false;
  }
  // If true vacuum is inside the bubble
  if (t_isBubbleTrueVacuum) {
    for (int i = 0; i < m_sim.getParticleCountTotal(); i++) {
      if (m_sim.calculateParticleRadius(i) > m_bubble.getRadius()) {
        countParticleFalse += 1;
        particleEnergyFalse += m_sim.getReferenceE()[i];
      }
      countParticleInteractedFalse += m_sim.getReferenceInteractedFalse()[i];
      countParticlePassedFalse += m_sim.getReferencePassedFalse()[i];
      countParticleInteratedTrue += m_sim.getReferenceInteractedTrue()[i];
    }
  }
  // If true vacuum is outside the bubble
  else {
    for (int i = 0; i < m_sim.getParticleCountTotal(); i++) {
      if (m_sim.calculateParticleRadius(i) < m_bubble.getRadius()) {
        countParticleFalse += 1;
        particleEnergyFalse += m_sim.getReferenceE()[i];
      }
      countParticleInteractedFalse += m_sim.getReferenceInteractedFalse()[i];
      countParticlePassedFalse += m_sim.getReferencePassedFalse()[i];
      countParticleInteratedTrue += m_sim.getReferenceInteractedTrue()[i];
    }
  }
  t_stream << m_sim.getTime() << "," << m_sim.getdPressureStep() << ","
           << m_bubble.getRadius() << "," << m_bubble.getSpeed() << ",";
  t_stream << m_sim.getBubbleEnergy() << ","
           << m_sim.getBubbleEnergy() - m_sim.getBubbleEnergyLastStep() << ",";
  t_stream << m_sim.getParticlesEnergy() << ","
           << m_sim.getParticlesEnergy() - m_sim.getParticlesEnergyLastStep()
           << ",";
  t_stream << particleEnergyFalse << ","
           << m_sim.getTotalEnergy() / m_sim.getTotalEnergyInitial() << ",";
  t_stream << countParticleFalse << "," << countParticleInteractedFalse << ","
           << countParticlePassedFalse << ",";
  t_stream << countParticleInteratedTrue;

  /*
          t_stream << sim.m_time << "," << dP << "," << bubble.m_radius << ","
     << bubble.m_speed << ","; t_stream << bubbleEnergy << "," << bubbleEnergy -
     bubbleEnergyOld << "," << particleEnergy << "," << particleEnergy -
     particleEnergyOld << ","; t_stream << particleEnergyIn << "," <<
     (bubbleEnergy + particleEnergy) / sim.m_energyTotalInitial << ","; t_stream
     << countParticleIn << "," << countParticleInteractedIn << "," <<
     countParticleInteractedPassedIn << ","; t_stream <<
     countParticleInteratedOut << std::endl;
  */
}

void DataStreamer::streamMassRadiusDifference(bool t_isBubbleTrueVacuum) {
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
}

void DataStreamer::streamParticleInfo() { 
    for (int i = 0; i < m_sim.getParticleCountTotal(); i++) {
        std::cout << m_sim.getReferenceX()[3 * i] << ", " << m_sim.getReferenceX()[3 * i + 1] << ", " << m_sim.getReferenceX()[3 * i + 2] << ", ";
        std::cout << m_sim.getReferenceP()[3 * i] << ", " << m_sim.getReferenceP()[3 * i + 1] << ", " << m_sim.getReferenceP()[3 * i + 2] << ", ";
        std::cout << m_sim.getParticleMass(i) << ", " << m_sim.getParticleEnergy(i) << std::endl;
    }
}

void DataStreamer::streamParticleInfo(std::fstream& t_stream) {
    for (int i = 0; i < m_sim.getParticleCountTotal(); i++) {
        t_stream << m_sim.getReferenceX()[3 * i] << ", " << m_sim.getReferenceX()[3 * i + 1] << ", " << m_sim.getReferenceX()[3 * i + 2] << ", ";
        t_stream << m_sim.getReferenceP()[3 * i] << ", " << m_sim.getReferenceP()[3 * i + 1] << ", " << m_sim.getReferenceP()[3 * i + 2] << ", ";
        t_stream << m_sim.getParticleMass(i) << ", " << m_sim.getParticleEnergy(i) << std::endl;
    }
}