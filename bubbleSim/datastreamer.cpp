#include "datastreamer.h"

DataStreamer::DataStreamer(std::string filePath) { m_filePath = filePath; }

void DataStreamer::initMomentumProfile(size_t t_binsCount,
                                       numType t_maxMomentumValue) {
  m_momentumInitialized = true;
  m_momentumBinsCount = t_binsCount;
  m_maxMomentumValue = t_maxMomentumValue;

  m_fileMomentumIn.open(m_filePath / "pIn.csv", std::ios::out);
  m_fileMomentumOut.open(m_filePath / "pOut.csv", std::ios::out);

  m_fileMomentumIn << m_momentumBinsCount << "," << m_maxMomentumValue
                   << std::endl;
  m_fileMomentumOut << m_momentumBinsCount << "," << m_maxMomentumValue
                    << std::endl;
  numType dp = m_maxMomentumValue / m_momentumBinsCount;
  for (size_t i = 1; i < m_momentumBinsCount; i++) {
    std::cout << std::fixed << std::showpoint << std::setprecision(3);
    m_fileMomentumIn << dp * i << ",";
    m_fileMomentumOut << dp * i << ",";
  }
  m_fileMomentumIn << m_maxMomentumValue << std::endl;
  m_fileMomentumOut << m_maxMomentumValue << std::endl;
}

void DataStreamer::initDensityProfile(size_t t_binsCount,
                                      numType t_maxRadiusValue) {
  m_densityInitialized = true;
  m_densityBinsCount = t_binsCount;
  m_maxRadiusValue = t_maxRadiusValue;
  m_fileDensity.open(m_filePath / "density.csv", std::ios::out);

  m_fileDensity << m_densityBinsCount << "," << m_maxRadiusValue << std::endl;
  numType dr = m_maxRadiusValue / m_densityBinsCount;
  for (size_t i = 1; i < m_densityBinsCount; i++) {
    std::cout << std::fixed << std::showpoint << std::setprecision(3);
    m_fileDensity << dr * i << ",";
  }
  m_fileDensity << m_maxRadiusValue << std::endl;
}

void DataStreamer::initData() {
  m_dataInitialized = true;
  m_fileData.open(m_filePath / "data.csv", std::ios::out);
  /*
   * time
   * dP - Pressure/energy change between particles and bubble
   * R_b - bubble radius
   * V_b - bubble speed
   * E_b - bubble energy
   * E_p - particles' energy
   * E_f - particles' energy which are inside the bubble
   * E - total energy / initial total energy (checking energy conservation)
   * C_f - particles inside the bubble count
   * C_if - particles which interacted with bubble form false vacuum (and did
   not get through) count
   * C_pf - particles which interacted with bubble form
   false vacuum (and got through) count
   * C_it - particles which interacted with
   * bubble from true vacuum count
   */
  m_fileData << "time,dP,R_b,V_b,E_b,E_p,E_f,E,C_f,C_if,C_pf,C_it" << std::endl;
}

void DataStreamer::stream(Simulation& simulation,
                          ParticleCollection& particleCollection,
                          PhaseBubble& bubble, cl::CommandQueue& cl_queue) {
  /*
   * Do only initialized streams.
   * 1) Read in necessary buffers
   * 2) Do for cycle over all particles and count/calculate profiles
   * 3) Stream into files
   */

  particleCollection.readParticlesBuffer(cl_queue);
  /*
   * Don't read bubble buffer as it is last steps one.
   * Bubble object values are always "new"
   */
  if (m_dataInitialized) {
    particleCollection.readInteractedBubbleFalseStateBuffer(cl_queue);
    particleCollection.readPassedBubbleFalseStateBuffer(cl_queue);
    particleCollection.readInteractedBubbleTrueStateBuffer(cl_queue);
  }

  std::vector<u_int> countBinsIn;
  std::vector<u_int> countBinsOut;
  numType dp = m_maxMomentumValue / m_momentumBinsCount;
  std::vector<u_int> countBinsDensity;
  numType dr = m_maxRadiusValue / m_densityBinsCount;

  size_t particleInCount;
  size_t particleInteractedFalseCount;
  size_t particlePassedFalseCount;
  size_t particleInteractedTrueCount;
  numType particleInEnergy;
  numType particleTotalEnergy;
  numType totalEnergy;

  numType particleRadius;
  numType particleMomentum;

  /*
   * If only data is streamed
   */
  if ((m_dataInitialized) && (!m_densityInitialized) &&
      (!m_momentumInitialized)) {
    particleInCount = 0;
    particleInteractedFalseCount = 0;
    particlePassedFalseCount = 0;
    particleInteractedTrueCount = 0;
    particleInEnergy = 0.;
    particleTotalEnergy = 0.;

    for (u_int i = 0; i < particleCollection.getParticleCountTotal(); i++) {
      if (particleCollection.getParticles()[i].m ==
          particleCollection.getMassIn()) {
        particleInCount += 1;
        particleInEnergy += particleCollection.getParticleEnergy(i);
      } else {
        particleTotalEnergy += particleCollection.getParticleEnergy(i);
      }
      particleInteractedFalseCount +=
          particleCollection.getInteractedFalse()[i];
      particlePassedFalseCount += particleCollection.getPassedFalse()[i];
      particleInteractedTrueCount += particleCollection.getInteractedTrue()[i];
    }
    particleTotalEnergy += particleInEnergy;
    totalEnergy = particleTotalEnergy + bubble.calculateEnergy();
    std::cout << std::fixed << std::showpoint << std::setprecision(6);
    m_fileData << simulation.getTime() << "," << simulation.get_dP() << ",";
    m_fileData << bubble.getRadius() << "," << bubble.getSpeed() << ",";
    m_fileData << bubble.calculateEnergy() << "," << particleTotalEnergy << ",";
    m_fileData << particleInEnergy << ","
               << totalEnergy / simulation.getTotalEnergy() << ",";
    m_fileData << particleInCount << "," << particleInteractedFalseCount << ",";
    m_fileData << particlePassedFalseCount << "," << particleInteractedTrueCount
               << std::endl;
    auto it = std::max_element(std::begin(particleCollection.getPassedFalse()),
                               std::end(particleCollection.getPassedFalse()));
    particleCollection.resetAndWriteInteractedBubbleFalseState(cl_queue);
    particleCollection.resetAndWritePassedBubbleFalseState(cl_queue);
    particleCollection.resetAndWriteInteractedBubbleTrueState(cl_queue);
  }
  /*
   * If only number density is streamed
   */
  else if ((!m_dataInitialized) && (m_densityInitialized) &&
           (!m_momentumInitialized)) {
    countBinsDensity.resize(m_densityBinsCount);
    for (u_int i = 0; i < particleCollection.getParticleCountTotal(); i++) {
      particleRadius = particleCollection.calculateParticleRadius(i);
      if (particleRadius < m_maxRadiusValue) {
        countBinsDensity[(int)(particleRadius / dr)] += 1;
      }
    }
    std::cout << std::noshowpoint;
    for (size_t i = 0; i < m_densityBinsCount - 1; i++) {
      m_fileDensity << countBinsDensity[i] << ",";
    }
    m_fileDensity << countBinsDensity[m_densityBinsCount - 1] << std::endl;
  }
  /*
   * If only momentum profiles are streamed
   */
  else if ((!m_dataInitialized) && (!m_densityInitialized) &&
           (m_momentumInitialized)) {
    countBinsIn.resize(m_momentumBinsCount);
    countBinsOut.resize(m_momentumBinsCount);
    for (u_int i = 0; i < particleCollection.getParticleCountTotal(); i++) {
      particleMomentum = particleCollection.calculateParticleMomentum(i);
      if (particleMomentum < m_maxMomentumValue) {
        if (particleCollection.getParticleMass(i) ==
            particleCollection.getMassIn()) {
          countBinsIn[(int)(particleMomentum / dp)] += 1;
        } else {
          countBinsOut[(int)(particleMomentum / dp)] += 1;
        }
      }
    }
    std::cout << std::noshowpoint;
    for (size_t i = 0; i < m_momentumBinsCount - 1; i++) {
      m_fileMomentumIn << countBinsIn[i] << ",";
      m_fileMomentumOut << countBinsOut[i] << ",";
    }
    m_fileMomentumIn << countBinsIn[m_momentumBinsCount - 1] << std::endl;
    m_fileMomentumOut << countBinsOut[m_momentumBinsCount - 1] << std::endl;
  }
  /*
   * If data and number density are streamed
   */
  else if ((m_dataInitialized) && (m_densityInitialized) &&
           (!m_momentumInitialized)) {
    particleInCount = 0;
    particleInteractedFalseCount = 0;
    particlePassedFalseCount = 0;
    particleInteractedTrueCount = 0;
    particleInEnergy = 0.;
    particleTotalEnergy = 0.;
    countBinsDensity.resize(m_densityBinsCount);
    for (u_int i = 0; i < particleCollection.getParticleCountTotal(); i++) {
      particleRadius = particleCollection.calculateParticleRadius(i);
      if (particleRadius < m_maxRadiusValue) {
        countBinsDensity[(int)(particleRadius / dr)] += 1;
      }
      if (particleCollection.getParticles()[i].m ==
          particleCollection.getMassIn()) {
        particleInCount += 1;
        particleInEnergy += particleCollection.getParticles()[i].E;
      } else {
        particleTotalEnergy += particleCollection.getParticles()[i].E;
      }
      particleInteractedFalseCount +=
          particleCollection.getInteractedFalse()[i];
      particlePassedFalseCount += particleCollection.getPassedFalse()[i];
      particleInteractedTrueCount += particleCollection.getInteractedTrue()[i];
    }
    particleTotalEnergy += particleInEnergy;
    totalEnergy = particleTotalEnergy + bubble.calculateEnergy();

    std::cout << std::fixed << std::showpoint << std::setprecision(6);
    m_fileData << simulation.getTime() << "," << simulation.get_dP() << ",";
    m_fileData << bubble.getRadius() << "," << bubble.getSpeed() << ",";
    m_fileData << bubble.calculateEnergy() << "," << particleTotalEnergy << ",";
    m_fileData << particleInEnergy << ","
               << totalEnergy / simulation.getTotalEnergy() << ",";
    m_fileData << particleInCount << "," << particleInteractedFalseCount << ",";
    m_fileData << particlePassedFalseCount << "," << particleInteractedTrueCount
               << std::endl;
    particleCollection.resetAndWriteInteractedBubbleFalseState(cl_queue);
    particleCollection.resetAndWritePassedBubbleFalseState(cl_queue);
    particleCollection.resetAndWriteInteractedBubbleTrueState(cl_queue);
    std::cout << std::noshowpoint;
    for (size_t i = 0; i < m_densityBinsCount - 1; i++) {
      m_fileDensity << countBinsDensity[i] << ",";
    }
    m_fileDensity << countBinsDensity[m_densityBinsCount - 1] << std::endl;
  }
  /*
   * If data and momentum profiles are streamed
   */
  else if ((m_dataInitialized) && (!m_densityInitialized) &&
           (m_momentumInitialized)) {
    countBinsIn.resize(m_momentumBinsCount);
    countBinsOut.resize(m_momentumBinsCount);
    particleInCount = 0;
    particleInteractedFalseCount = 0;
    particlePassedFalseCount = 0;
    particleInteractedTrueCount = 0;
    particleInEnergy = 0.;
    particleTotalEnergy = 0.;
    for (u_int i = 0; i < particleCollection.getParticleCountTotal(); i++) {
      particleMomentum = particleCollection.calculateParticleMomentum(i);
      if (particleCollection.getParticles()[i].m ==
          particleCollection.getMassIn()) {
        particleInCount += 1;
        particleInEnergy += particleCollection.getParticles()[i].E;
        if (particleMomentum < m_maxMomentumValue) {
          countBinsIn[(int)(particleMomentum / dp)] += 1;
        }
      } else {
        particleTotalEnergy += particleCollection.getParticles()[i].E;
        if (particleMomentum < m_maxMomentumValue) {
          countBinsOut[(int)(particleMomentum / dp)] += 1;
        }
      }
      particleInteractedFalseCount +=
          particleCollection.getInteractedFalse()[i];
      particlePassedFalseCount += particleCollection.getPassedFalse()[i];
      particleInteractedTrueCount += particleCollection.getInteractedTrue()[i];
    }
    particleTotalEnergy += particleInEnergy;
    totalEnergy = particleTotalEnergy + bubble.calculateEnergy();

    std::cout << std::fixed << std::showpoint << std::setprecision(6);
    m_fileData << simulation.getTime() << "," << simulation.get_dP() << ",";
    m_fileData << bubble.getRadius() << "," << bubble.getSpeed() << ",";
    m_fileData << bubble.calculateEnergy() << "," << particleTotalEnergy << ",";
    m_fileData << particleInEnergy << ","
               << totalEnergy / simulation.getTotalEnergy() << ",";
    m_fileData << particleInCount << "," << particleInteractedFalseCount << ",";
    m_fileData << particlePassedFalseCount << "," << particleInteractedTrueCount
               << std::endl;
    particleCollection.resetAndWriteInteractedBubbleFalseState(cl_queue);
    particleCollection.resetAndWritePassedBubbleFalseState(cl_queue);
    particleCollection.resetAndWriteInteractedBubbleTrueState(cl_queue);

    std::cout << std::noshowpoint;
    for (size_t i = 0; i < m_momentumBinsCount - 1; i++) {
      m_fileMomentumIn << countBinsIn[i] << ",";
      m_fileMomentumOut << countBinsOut[i] << ",";
    }
    m_fileMomentumIn << countBinsIn[m_momentumBinsCount - 1] << std::endl;
    m_fileMomentumOut << countBinsOut[m_momentumBinsCount - 1] << std::endl;
  }
  /*
   * If number density and momentum profiles are streamed
   */
  else if ((!m_dataInitialized) && (m_densityInitialized) &&
           (m_momentumInitialized)) {
    countBinsDensity.resize(m_densityBinsCount);
    countBinsIn.resize(m_momentumBinsCount);
    countBinsOut.resize(m_momentumBinsCount);
    for (u_int i = 0; i < particleCollection.getParticleCountTotal(); i++) {
      particleRadius = particleCollection.calculateParticleRadius(i);
      particleMomentum = particleCollection.calculateParticleMomentum(i);
      if (particleRadius < m_maxRadiusValue) {
        countBinsDensity[(int)(particleRadius / dr)] += 1;
      }

      if (particleMomentum < m_maxMomentumValue) {
        if (particleCollection.getParticleMass(i) ==
            particleCollection.getMassIn()) {
          countBinsIn[(int)(particleMomentum / dp)] += 1;
        } else {
          countBinsOut[(int)(particleMomentum / dp)] += 1;
        }
      }
    }
    std::cout << std::noshowpoint;
    for (size_t i = 0; i < m_densityBinsCount - 1; i++) {
      m_fileDensity << countBinsDensity[i] << ",";
    }
    m_fileDensity << countBinsDensity[m_densityBinsCount - 1] << std::endl;
    for (size_t i = 0; i < m_momentumBinsCount - 1; i++) {
      m_fileMomentumIn << countBinsIn[i] << ",";
      m_fileMomentumOut << countBinsOut[i] << ",";
    }
    m_fileMomentumIn << countBinsIn[m_momentumBinsCount - 1] << std::endl;
    m_fileMomentumOut << countBinsOut[m_momentumBinsCount - 1] << std::endl;
  }
  /*
   * If data, number density and momentum profiles are streamed
   */
  else if ((m_dataInitialized) && (m_densityInitialized) &&
           (m_momentumInitialized)) {
    numType maxMomentum = -1;
    countBinsDensity.resize(m_densityBinsCount);
    countBinsIn.resize(m_momentumBinsCount);
    countBinsOut.resize(m_momentumBinsCount);

    particleInCount = 0;
    particleInteractedFalseCount = 0;
    particlePassedFalseCount = 0;
    particleInteractedTrueCount = 0;
    particleInEnergy = 0.;
    particleTotalEnergy = 0.;
    for (u_int i = 0; i < particleCollection.getParticleCountTotal(); i++) {
      particleRadius = particleCollection.calculateParticleRadius(i);
      particleMomentum = particleCollection.calculateParticleMomentum(i);
      if (particleMomentum > maxMomentum) {
        maxMomentum = particleMomentum;
      }
      if (particleRadius < m_maxRadiusValue) {
        countBinsDensity[(int)(particleRadius / dr)] += 1;
      }
      if (particleCollection.getParticles()[i].m ==
          particleCollection.getMassIn()) {
        particleInCount += 1;
        particleInEnergy += particleCollection.getParticles()[i].E;
        if (particleMomentum < m_maxMomentumValue) {
          countBinsIn[(int)(particleMomentum / dp)] += 1;
        }

      } else {
        particleTotalEnergy += particleCollection.getParticles()[i].E;
        if (particleMomentum < m_maxMomentumValue) {
          countBinsOut[(int)(particleMomentum / dp)] += 1;
        }
      }
      particleInteractedFalseCount +=
          particleCollection.getInteractedFalse()[i];
      particlePassedFalseCount += particleCollection.getPassedFalse()[i];
      particleInteractedTrueCount += particleCollection.getInteractedTrue()[i];
    }
    particleTotalEnergy += particleInEnergy;
    totalEnergy = particleTotalEnergy + bubble.calculateEnergy();

    // std::cout << "Max momentum: " << maxMomentum << std::endl;

    std::cout << std::fixed << std::showpoint << std::setprecision(6);
    m_fileData << simulation.getTime() << "," << simulation.get_dP() << ",";
    m_fileData << bubble.getRadius() << "," << bubble.getSpeed() << ",";
    m_fileData << bubble.calculateEnergy() << "," << particleTotalEnergy << ",";
    m_fileData << particleInEnergy << ","
               << totalEnergy / simulation.getTotalEnergy() << ",";
    m_fileData << particleInCount << "," << particleInteractedFalseCount << ",";
    m_fileData << particlePassedFalseCount << "," << particleInteractedTrueCount
               << std::endl;
    particleCollection.resetAndWriteInteractedBubbleFalseState(cl_queue);
    particleCollection.resetAndWritePassedBubbleFalseState(cl_queue);
    particleCollection.resetAndWriteInteractedBubbleTrueState(cl_queue);

    std::cout << std::noshowpoint;
    for (size_t i = 0; i < m_densityBinsCount - 1; i++) {
      m_fileDensity << countBinsDensity[i] << ",";
    }
    m_fileDensity << countBinsDensity[m_densityBinsCount - 1] << std::endl;

    for (size_t i = 0; i < m_momentumBinsCount - 1; i++) {
      m_fileMomentumIn << countBinsIn[i] << ",";
      m_fileMomentumOut << countBinsOut[i] << ",";
    }
    m_fileMomentumIn << countBinsIn[m_momentumBinsCount - 1] << std::endl;
    m_fileMomentumOut << countBinsOut[m_momentumBinsCount - 1] << std::endl;
  } else {
    std::cerr << "None of the streams were initilized." << std::endl;
    std::terminate();
  }
}
