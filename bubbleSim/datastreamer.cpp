#include "datastreamer.h"

DataStreamer::DataStreamer(std::string filePath) { m_filePath = filePath; }

void DataStreamer::initStream_Data() {
  m_initialized_Data = true;
  m_stream_Data.open(m_filePath / "data.csv", std::ios::out);
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
   * C_pf - particles which interacted with bubble form false vacuum (and got
   through) count
   * C_it - particles which interacted with bubble from true vacuum count
   * C - compactness
   */
  m_stream_Data << "time,dP,R_b,V_b,E_b,E_p,E_f,E,C_f,C_if,C_pf,C_it,C"
                << std::endl;
}

void DataStreamer::initStream_Momentum(size_t t_binsCount,
                                       numType t_maxMomentumValue) {
  m_initialized_Momentum = true;
  m_binsCount_Momentum = t_binsCount;
  m_maxMomentum_Momentum = t_maxMomentumValue;
  m_dp_Momentum = t_maxMomentumValue / t_binsCount;

  m_stream_MomentumX.open(m_filePath / "pX.csv", std::ios::out);
  m_stream_MomentumY.open(m_filePath / "pY.csv", std::ios::out);
  m_stream_MomentumZ.open(m_filePath / "pZ.csv", std::ios::out);

  m_stream_MomentumX << t_binsCount << "," << t_maxMomentumValue << "\n";
  m_stream_MomentumY << t_binsCount << "," << t_maxMomentumValue << "\n";
  m_stream_MomentumZ << t_binsCount << "," << t_maxMomentumValue << "\n";
  for (size_t i = 1; i <= t_binsCount; i++) {
    if (i == t_binsCount) {
      m_stream_MomentumX << i << std::endl;
      m_stream_MomentumY << i << std::endl;
      m_stream_MomentumZ << i << std::endl;
    } else {
      m_stream_MomentumX << i << ",";
      m_stream_MomentumY << i << ",";
      m_stream_MomentumZ << i << ",";
    }
  }
}

void DataStreamer::initStream_MomentumIn(size_t t_binsCount,
                                         numType t_minMomentumValue,
                                         numType t_maxMomentumValue,
                                         bool t_log_scale_on) {
  if ((t_log_scale_on) && (t_minMomentumValue <= 0)) {
    std::cout << "Log scale is on. Minimum momentum value must be > 0.";
    std::exit(0);
  }

  m_initialized_MomentumIn = true;
  m_binsCount_MomentumIn = t_binsCount;
  m_minMomentum_MomentumIn = t_minMomentumValue;
  m_maxMomentum_MomentumIn = t_maxMomentumValue;
  if (t_log_scale_on) {
    m_dp_MomentumIn =
        (std::log10(t_maxMomentumValue) - std::log10(t_minMomentumValue)) /
        t_binsCount;

  } else {
    m_dp_MomentumIn = t_maxMomentumValue / t_binsCount;
  }

  m_stream_MomentumIn.open(m_filePath / "pIn.csv", std::ios::out);
  m_stream_MomentumIn << t_binsCount << "," << t_minMomentumValue << ","
                      << t_maxMomentumValue << "\n";
  for (size_t i = 1; i <= t_binsCount; i++) {
    if (i == t_binsCount) {
      m_stream_MomentumIn << i << std::endl;
    } else {
      m_stream_MomentumIn << i << ",";
    }
  }
}

void DataStreamer::initStream_MomentumOut(size_t t_binsCount,
                                          numType t_minMomentumValue,
                                          numType t_maxMomentumValue,
                                          bool t_log_scale_on) {
  if ((t_log_scale_on) && (t_minMomentumValue <= 0)) {
    std::cout << "Log scale is on. Minimum momentum value must be > 0.";
    std::exit(0);
  }
  m_initialized_MomentumOut = true;
  m_binsCount_MomentumOut = t_binsCount;
  m_minMomentum_MomentumOut = t_minMomentumValue;
  m_maxMomentum_MomentumOut = t_maxMomentumValue;
  if (t_log_scale_on) {
    m_dp_MomentumOut =
        (std::log10(t_maxMomentumValue) - std::log10(t_minMomentumValue)) /
        t_binsCount;

  } else {
    m_dp_MomentumOut = t_maxMomentumValue / t_binsCount;
  }

  m_stream_MomentumOut.open(m_filePath / "pOut.csv", std::ios::out);
  m_stream_MomentumOut << t_binsCount << "," << t_minMomentumValue << ","
                       << t_maxMomentumValue << "\n";
  for (size_t i = 1; i <= t_binsCount; i++) {
    if (i == t_binsCount) {
      m_stream_MomentumOut << i << std::endl;
    } else {
      m_stream_MomentumOut << i << ",";
    }
  }
}

void DataStreamer::initStream_Density(size_t t_binsCount,
                                      numType t_maxRadiusValue) {
  m_initialized_Density = true;
  m_binsCount_Density = t_binsCount;
  m_maxRadius_Density = t_maxRadiusValue;
  m_dr_Density = t_maxRadiusValue / t_binsCount;

  m_stream_Density.open(m_filePath / "numberDensity.csv", std::ios::out);
  m_stream_Density << t_binsCount << "," << t_maxRadiusValue << "\n";
  for (size_t i = 1; i <= t_binsCount; i++) {
    if (i == t_binsCount) {
      m_stream_Density << i << std::endl;
    } else {
      m_stream_Density << i << ",";
    }
  }
}

void DataStreamer::initStream_EnergyDensity(size_t t_binsCount,
                                            numType t_maxRadiusValue) {
  m_initialized_EnergyDensity = true;
  m_binsCount_EnergyDensity = t_binsCount;
  m_maxRadius_EnergyDensity = t_maxRadiusValue;
  m_dr_EnergyDensity = t_maxRadiusValue / t_binsCount;

  m_stream_EnergyDensity.open(m_filePath / "energyDensity.csv", std::ios::out);
  m_stream_EnergyDensity << t_binsCount << "," << t_maxRadiusValue << "\n";
  for (size_t i = 1; i <= t_binsCount; i++) {
    if (i == t_binsCount) {
      m_stream_EnergyDensity << i << std::endl;
    } else {
      m_stream_EnergyDensity << i << ",";
    }
  }
}

void DataStreamer::initStream_RadialVelocity(size_t t_binsCount,
                                             numType t_maxRadiusValue) {
  m_initialized_RadialVelocity = true;
  m_binsCount_RadialVelocity = t_binsCount;
  m_maxRadius_RadialVelocity = t_maxRadiusValue;
  m_dr_RadialVelocity = t_maxRadiusValue / t_binsCount;

  m_stream_RadialVelocity.open(m_filePath / "radialVelocity.csv",
                               std::ios::out);
  m_stream_RadialVelocity << t_binsCount << "," << t_maxRadiusValue << "\n";
  for (size_t i = 1; i <= t_binsCount; i++) {
    if (i == t_binsCount) {
      m_stream_RadialVelocity << i << std::endl;
    } else {
      m_stream_RadialVelocity << i << ",";
    }
  }
}

void DataStreamer::initStream_TangentialVelocity(size_t t_binsCount,
                                                 numType t_maxRadiusValue) {
  m_initialized_TangentialVelocity = true;
  m_binsCount_TangentialVelocity = t_binsCount;
  m_maxRadius_TangentialVelocity = t_maxRadiusValue;
  m_dr_TangentialVelocity = t_maxRadiusValue / t_binsCount;

  m_stream_TangentialVelocity.open(m_filePath / "tangentialVelocity.csv",
                                   std::ios::out);
  m_stream_TangentialVelocity << t_binsCount << "," << t_maxRadiusValue << "\n";
  for (size_t i = 1; i <= t_binsCount; i++) {
    if (i == t_binsCount) {
      m_stream_TangentialVelocity << i << std::endl;
    } else {
      m_stream_TangentialVelocity << i << ",";
    }
  }
}

void DataStreamer::stream(Simulation& simulation,
                          ParticleCollection& particleCollection,
                          PhaseBubble& bubble, bool t_log_scale_on,
                          cl::CommandQueue& cl_queue) {
  // auto programStartTime = std::chrono::high_resolution_clock::now();
  /*
   * Do only initialized streams.
   * 1) Read in necessary buffers
   * 2) Do for cycle over all particles and count/calculate profiles
   * 3) Stream into files
   */

  std::cout << std::setprecision(8) << std::fixed << std::showpoint;

  /*
   * Read particle buffers to get if particle interacted with the bubble or not
   * Don't read bubble buffer as it is not updated. Use PhaseBubble object.
   */
  if (m_initialized_Data) {
    particleCollection.readInteractedBubbleFalseStateBuffer(cl_queue);
    particleCollection.readPassedBubbleFalseStateBuffer(cl_queue);
    particleCollection.readInteractedBubbleTrueStateBuffer(cl_queue);
  }
  // Save general data
  size_t particleInCount;
  size_t particleInteractedFalseCount;
  size_t particlePassedFalseCount;
  size_t particleInteractedTrueCount;
  numType particleInEnergy;
  numType particleTotalEnergy;
  numType totalEnergy;

  // Save momentum data
  std::vector<u_int> bins_MomentumX;
  std::vector<u_int> bins_MomentumY;
  std::vector<u_int> bins_MomentumZ;
  std::vector<u_int> bins_MomentumIn;
  std::vector<u_int> bins_MomentumOut;

  // Save density data
  std::vector<u_int> bins_Density;
  std::vector<numType> bins_EnergyDensity;

  // Save velocity data
  std::vector<numType> bins_RadialVelocity;
  std::vector<numType> bins_TangentialVelocity;
  std::vector<u_int> bins_RadialVelocityCount;
  std::vector<u_int> bins_TangentialVelocityCount;

  // Read data from buffer:
  particleCollection.readParticleCoordinatesBuffer(cl_queue);
  particleCollection.readParticleMomentumsBuffer(cl_queue);
  particleCollection.readParticleEBuffer(cl_queue);

  // Initialize variables
  if (m_initialized_Momentum) {
    bins_MomentumX.resize(m_binsCount_Momentum, 0);
    bins_MomentumY.resize(m_binsCount_Momentum, 0);
    bins_MomentumZ.resize(m_binsCount_Momentum, 0);
  }
  if (m_initialized_MomentumIn) {
    bins_MomentumIn.resize(m_binsCount_MomentumIn, 0);
  }
  if (m_initialized_MomentumOut) {
    bins_MomentumOut.resize(m_binsCount_MomentumOut, 0);
  }
  if (m_initialized_Density) {
    bins_Density.resize(m_binsCount_Density, 0);
  }
  if (m_initialized_EnergyDensity) {
    bins_EnergyDensity.resize(m_binsCount_EnergyDensity, (numType)0.);
  }
  if (m_initialized_RadialVelocity) {
    bins_RadialVelocity.resize(m_binsCount_RadialVelocity, (numType)0.);
    bins_RadialVelocityCount.resize(m_binsCount_RadialVelocity, 0);
  }
  if (m_initialized_TangentialVelocity) {
    bins_TangentialVelocity.resize(m_binsCount_TangentialVelocity, (numType)0.);
    bins_TangentialVelocityCount.resize(m_binsCount_TangentialVelocity, 0);
  }
  if (m_initialized_Data) {
    particleInCount = 0;
    particleInteractedFalseCount = 0;
    particlePassedFalseCount = 0;
    particleInteractedTrueCount = 0;
    particleInEnergy = 0.;
    particleTotalEnergy = 0.;
  }

  numType particleRadius;
  numType particleMomentum;
  numType particleRadialVelocity;
  numType particleTangentialVelocity;
  for (u_int i = 0; i < particleCollection.getParticleCountTotal(); i++) {
    particleRadius = particleCollection.calculateParticleRadius(i);
    particleMomentum = particleCollection.calculateParticleMomentum(i);

    if (m_initialized_Momentum) {
      bins_MomentumX[std::clamp(
          (int)(abs(particleCollection.returnParticlepX(i) / m_dp_Momentum)), 0,
          (int)m_binsCount_Momentum - 1)] += 1;
      bins_MomentumY[std::clamp(
          (int)(abs(particleCollection.returnParticlepY(i) / m_dp_Momentum)), 0,
          (int)m_binsCount_Momentum - 1)] += 1;
      bins_MomentumZ[std::clamp(
          (int)(abs(particleCollection.returnParticlepZ(i) / m_dp_Momentum)), 0,
          (int)m_binsCount_Momentum - 1)] += 1;
    }
    if (m_initialized_Density && (particleRadius < m_maxRadius_Density)) {
      bins_Density[(int)(particleRadius / m_dr_Density)] += 1;
    }
    if (m_initialized_EnergyDensity &&
        (particleRadius < m_maxRadius_EnergyDensity)) {
      bins_EnergyDensity[(int)(particleRadius / m_dr_EnergyDensity)] +=
          particleCollection.returnParticleE(i);
    }
    if (m_initialized_RadialVelocity &&
        (particleRadius < m_maxRadius_RadialVelocity)) {
      // Average A(N) = [A(N-1) * (N-1) + a(N) ]/N
      particleRadialVelocity =
          (particleCollection.returnParticleX(i) *
               particleCollection.returnParticlepX(i) +
           particleCollection.returnParticleY(i) *
               particleCollection.returnParticlepY(i) +
           particleCollection.returnParticleY(i) *
               particleCollection.returnParticlepY(i)) /
          (particleCollection.returnParticleE(i) * particleRadius);
      bins_RadialVelocity[(int)(particleRadius / m_dr_RadialVelocity)] =
          (bins_RadialVelocity[(int)(particleRadius / m_dr_RadialVelocity)] *
               bins_RadialVelocityCount[(int)(particleRadius /
                                              m_dr_RadialVelocity)] +
           particleRadialVelocity) /
          (bins_RadialVelocityCount[(int)(particleRadius /
                                          m_dr_RadialVelocity)] +
           1);

      bins_RadialVelocityCount[(int)(particleRadius / m_dr_RadialVelocity)] +=
          1;
    }
    if (m_initialized_TangentialVelocity &&
        (particleRadius < m_maxRadius_TangentialVelocity)) {
      if (m_initialized_RadialVelocity) {
        particleTangentialVelocity = std::sqrt(1 - particleRadialVelocity);
      } else {
        particleTangentialVelocity =
            particleCollection.calculateParticleTangentialVelocity(i);
      }
      bins_TangentialVelocity[(int)(particleRadius / m_dr_TangentialVelocity)] =
          (bins_TangentialVelocity[(int)(particleRadius /
                                         m_dr_TangentialVelocity)] *
               bins_TangentialVelocityCount[(int)(particleRadius /
                                                  m_dr_TangentialVelocity)] +
           particleTangentialVelocity) /
          (bins_TangentialVelocityCount[(int)(particleRadius /
                                              m_dr_TangentialVelocity)] +
           1);

      bins_TangentialVelocityCount[(int)(particleRadius /
                                         m_dr_TangentialVelocity)] += 1;
    }
    if (m_initialized_MomentumIn && (particleRadius < bubble.getRadius()) &&
        (particleMomentum < m_maxMomentum_MomentumIn) &&
        (particleMomentum >= m_minMomentum_MomentumIn)) {
      if (t_log_scale_on) {
        bins_MomentumIn[(
            int)(std::log10(particleMomentum / m_minMomentum_MomentumIn) /
                 m_dp_MomentumIn)] += 1;
      } else {
        bins_MomentumIn[(int)(particleMomentum / m_dp_MomentumIn)] += 1;
      }
    }
    if (m_initialized_MomentumOut && (particleRadius > bubble.getRadius()) &&
        (particleMomentum < m_maxMomentum_MomentumOut) &&
        (particleMomentum >= m_minMomentum_MomentumOut)) {
      if (t_log_scale_on) {
        
        bins_MomentumOut[(
            int)(std::log10(particleMomentum / m_minMomentum_MomentumOut) /
                 m_dp_MomentumOut)] += 1;
      } else {
        bins_MomentumOut[(int)(particleMomentum / m_dp_MomentumOut)] += 1;
      }
    }
    if (m_initialized_Data) {
      if (particleRadius <= bubble.getRadius()) {
        particleInCount += 1;
        particleInEnergy += particleCollection.returnParticleE(i);
      }
      particleInteractedFalseCount +=
          particleCollection.getInteractedFalse()[i];
      particlePassedFalseCount += particleCollection.getPassedFalse()[i];
      particleInteractedTrueCount += particleCollection.getInteractedTrue()[i];
      particleTotalEnergy += particleCollection.returnParticleE(i);
    }
  }

  // std::cout << "Data collection: successful!" << std::endl;
  //  std::cout << "Max momentum: " << maxMomentum << std::endl;
  if (m_initialized_Data) {
    totalEnergy = particleTotalEnergy + bubble.calculateEnergy();
    std::cout << std::fixed << std::noshowpoint << std::setprecision(8);
    m_stream_Data << simulation.getTime() << ","
                  << simulation.get_dP() / simulation.get_dt_currentStep()
                  << ",";
    m_stream_Data << bubble.getRadius() << "," << bubble.getSpeed() << ",";
    m_stream_Data << bubble.calculateEnergy() << "," << particleTotalEnergy
                  << ",";
    m_stream_Data << particleInEnergy << ","
                  << totalEnergy / simulation.getInitialTotalEnergy() << ",";
    m_stream_Data << particleInCount << "," << particleInteractedFalseCount
                  << ",";
    m_stream_Data << particlePassedFalseCount << ","
                  << particleInteractedTrueCount << ","
                  << (particleInEnergy + bubble.calculateEnergy()) /
                         bubble.getRadius() / simulation.getInitialCompactnes()
                  << std::endl;

    particleCollection.resetAndWriteInteractedBubbleFalseState(cl_queue);
    particleCollection.resetAndWritePassedBubbleFalseState(cl_queue);
    particleCollection.resetAndWriteInteractedBubbleTrueState(cl_queue);

    std::cout << std::noshowpoint;
  }
  if (m_initialized_Density) {
    for (size_t i = 0; i < m_binsCount_Density - 1; i++) {
      m_stream_Density << bins_Density[i] << ",";
    }
    m_stream_Density << bins_Density[m_binsCount_Density - 1] << "\n";
  }
  if (m_initialized_EnergyDensity) {
    for (size_t i = 0; i < m_binsCount_EnergyDensity - 1; i++) {
      m_stream_EnergyDensity << bins_EnergyDensity[i] << ",";
    }
    m_stream_EnergyDensity << bins_EnergyDensity[m_binsCount_EnergyDensity - 1]
                           << "\n";
  }
  if (m_initialized_MomentumIn) {
    for (size_t i = 0; i < m_binsCount_MomentumIn - 1; i++) {
      m_stream_MomentumIn << bins_MomentumIn[i] << ",";
    }
    m_stream_MomentumIn << bins_MomentumIn[m_binsCount_MomentumIn - 1] << "\n";
  }
  if (m_initialized_MomentumOut) {
    for (size_t i = 0; i < m_binsCount_MomentumOut - 1; i++) {
      m_stream_MomentumOut << bins_MomentumOut[i] << ",";
    }
    m_stream_MomentumOut << bins_MomentumOut[m_binsCount_MomentumOut - 1]
                         << "\n";
  }
  if (m_initialized_RadialVelocity) {
    for (size_t i = 0; i < m_binsCount_RadialVelocity - 1; i++) {
      m_stream_RadialVelocity << bins_RadialVelocity[i] << ",";
    }
    m_stream_RadialVelocity
        << bins_RadialVelocity[m_binsCount_RadialVelocity - 1] << "\n";
  }
  if (m_initialized_TangentialVelocity) {
    for (size_t i = 0; i < m_binsCount_TangentialVelocity - 1; i++) {
      m_stream_TangentialVelocity << bins_TangentialVelocity[i] << ",";
    }
    m_stream_TangentialVelocity
        << bins_TangentialVelocity[m_binsCount_TangentialVelocity - 1] << "\n";
  }
  if (m_initialized_Momentum) {
    for (size_t i = 0; i < m_binsCount_Momentum - 1; i++) {
      m_stream_MomentumX << bins_MomentumX[i] << ",";
      m_stream_MomentumY << bins_MomentumY[i] << ",";
      m_stream_MomentumZ << bins_MomentumZ[i] << ",";
    }
    m_stream_MomentumX << bins_MomentumX[m_binsCount_Momentum - 1] << "\n";
    m_stream_MomentumY << bins_MomentumY[m_binsCount_Momentum - 1] << "\n";
    m_stream_MomentumZ << bins_MomentumZ[m_binsCount_Momentum - 1] << "\n";
  }
  /*auto programEndTime = std::chrono::high_resolution_clock::now();
  std::cout << "Time taken (stream): "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   programEndTime - programStartTime)
                   .count()
            << " ms." << std::endl;
  ;*/
}

void DataStreamer::streamMomentumIn(std::ofstream& t_stream, size_t t_binsCount,
                                    numType t_minMomentumValue,
                                    numType t_maxMomentumValue,
                                    ParticleCollection& particleCollection,
                                    PhaseBubble& bubble,
                                    cl::CommandQueue& cl_queue) {
  std::cout << std::setprecision(8) << std::fixed << std::showpoint;
  numType dp = (t_maxMomentumValue - t_minMomentumValue) / t_binsCount;
  std::vector<u_int> bins(t_binsCount);
  numType particleMomentum;
  numType particleRadius;

  t_stream << t_binsCount << "," << t_minMomentumValue << ","
           << t_maxMomentumValue << "\n";
  for (size_t i = 0; i < t_binsCount - 1; i++) {
    t_stream << (i + 1) * dp << ",";
  }
  t_stream << t_binsCount * dp << "\n";

  for (u_int i = 0; i < particleCollection.getParticleCountTotal(); i++) {
    particleMomentum = particleCollection.calculateParticleMomentum(i);
    particleRadius = particleCollection.calculateParticleRadius(i);
    if (particleRadius < bubble.getRadius()) {
      bins[(int)((particleMomentum - t_minMomentumValue) / dp)] += 1;
    }
  }
  for (size_t i = 0; i < t_binsCount - 1; i++) {
    t_stream << bins[i] << ",";
  }
  t_stream << bins[t_binsCount - 1] << "\n";
}

void DataStreamer::streamMomentumOut(
    std::ofstream& t_stream, size_t t_binsCount, numType t_minMomentumValue,
    numType t_maxMomentumValue, ParticleCollection& particleCollection,
    PhaseBubble& bubble, cl::CommandQueue& cl_queue) {
  std::cout << std::setprecision(8) << std::fixed << std::showpoint;
  numType dp = (t_maxMomentumValue - t_minMomentumValue) / t_binsCount;
  std::vector<u_int> bins(t_binsCount);
  numType particleMomentum;
  numType particleRadius;

  t_stream << t_binsCount << "," << t_minMomentumValue << ","
           << t_maxMomentumValue << "\n";
  for (size_t i = 0; i < t_binsCount - 1; i++) {
    t_stream << (i + 1) * dp << ",";
  }
  t_stream << t_binsCount * dp << "\n";

  for (u_int i = 0; i < particleCollection.getParticleCountTotal(); i++) {
    particleMomentum = particleCollection.calculateParticleMomentum(i);
    particleRadius = particleCollection.calculateParticleRadius(i);
    if (particleRadius > bubble.getRadius()) {
      bins[(int)((particleMomentum - t_minMomentumValue) / dp)] += 1;
    }
  }
  for (size_t i = 0; i < t_binsCount - 1; i++) {
    t_stream << bins[i] << ",";
  }
  t_stream << bins[t_binsCount - 1] << "\n";
}

void DataStreamer::streamNumberDensity(std::ofstream& t_stream,
                                       size_t t_binsCount,
                                       numType t_minRadiusValue,
                                       numType t_maxRadiusValue,
                                       ParticleCollection& particleCollection,
                                       cl::CommandQueue& cl_queue) {
  std::cout << std::setprecision(8) << std::fixed << std::showpoint;
  particleCollection.readParticleCoordinatesBuffer(cl_queue);

  numType dr = (t_maxRadiusValue - t_minRadiusValue) / t_binsCount;
  std::vector<u_int> bins(t_binsCount);
  numType particleRadius;
  t_stream << t_binsCount << "," << t_minRadiusValue << "," << t_maxRadiusValue
           << "\n";
  for (size_t i = 0; i < t_binsCount - 1; i++) {
    t_stream << (i + 1) * dr << ",";
  }
  t_stream << t_binsCount * dr << "\n";

  for (size_t i = 0; i < particleCollection.getParticleCountTotal(); i++) {
    particleRadius = particleCollection.calculateParticleRadius(i);
    if ((particleRadius >= t_minRadiusValue) &&
        (particleRadius < t_maxRadiusValue)) {
      bins[(int)((particleRadius - t_minRadiusValue) / dr)] += 1;
    }
  }
  for (size_t i = 0; i < t_binsCount - 1; i++) {
    t_stream << bins[i] << ",";
  }
  t_stream << bins[t_binsCount - 1] << "\n";
}

void DataStreamer::streamEnergyDensity(std::ofstream& t_stream,
                                       size_t t_binsCount,
                                       numType t_minRadiusValue,
                                       numType t_maxRadiusValue,
                                       ParticleCollection& particleCollection,
                                       cl::CommandQueue& cl_queue) {
  particleCollection.readParticleCoordinatesBuffer(cl_queue);
  particleCollection.readParticleEBuffer(cl_queue);

  std::cout << std::setprecision(8) << std::fixed << std::showpoint;
  numType dr = (t_maxRadiusValue - t_minRadiusValue) / t_binsCount;
  std::vector<numType> bins(t_binsCount, 0.);
  numType particleRadius;
  t_stream << t_binsCount << "," << t_minRadiusValue << "," << t_maxRadiusValue
           << "\n";
  for (size_t i = 0; i < t_binsCount - 1; i++) {
    t_stream << (i + 1) * dr << ",";
  }
  t_stream << t_binsCount * dr << "\n";
  for (size_t i = 0; i < particleCollection.getParticleCountTotal(); i++) {
    particleRadius = particleCollection.calculateParticleRadius(i);
    if ((particleRadius >= t_minRadiusValue) &&
        (particleRadius < t_maxRadiusValue)) {
      bins[(int)((particleRadius - t_minRadiusValue) / dr)] +=
          particleCollection.returnParticleE(i);
    }
  }
  for (size_t i = 0; i < t_binsCount - 1; i++) {
    t_stream << bins[i] << ",";
  }
  t_stream << bins[t_binsCount - 1] << "\n";
}

void DataStreamer::streamRadialVelocity(std::ofstream& t_stream,
                                        size_t t_binsCount,
                                        numType t_minRadiusValue,
                                        numType t_maxRadiusValue,
                                        ParticleCollection& particleCollection,
                                        cl::CommandQueue& cl_queue) {
  std::cout << std::setprecision(8) << std::fixed << std::showpoint;
  particleCollection.readParticleCoordinatesBuffer(cl_queue);
  particleCollection.readParticleMomentumsBuffer(cl_queue);
  particleCollection.readParticleEBuffer(cl_queue);

  numType dr = (t_maxRadiusValue - t_minRadiusValue) / t_binsCount;
  std::vector<u_int> bins(t_binsCount);
  std::vector<numType> average_velocity(t_binsCount);
  numType particleRadius;
  numType particleRadialVelocity;
  t_stream << t_binsCount << "," << t_minRadiusValue << "," << t_maxRadiusValue
           << "\n";
  for (size_t i = 0; i < t_binsCount - 1; i++) {
    t_stream << (i + 1) * dr << ",";
  }
  t_stream << t_binsCount * dr << "\n";

  for (u_int i = 0; i < particleCollection.getParticleCountTotal(); i++) {
    particleRadius = particleCollection.calculateParticleRadius(i);
    particleRadialVelocity =
        particleCollection.calculateParticleRadialVelocity(i);

    if ((particleRadius >= t_minRadiusValue) &&
        (particleRadius < t_maxRadiusValue)) {
      average_velocity[(int)(particleRadius / dr)] =
          (average_velocity[(int)(particleRadius / dr)] *
               bins[(int)(particleRadius / dr)] +
           particleRadialVelocity) /
          (bins[(int)(particleRadius / dr)] + 1);
      bins[(int)((particleRadius - t_minRadiusValue) / dr)] += 1;
    }
  }
  for (size_t i = 0; i < t_binsCount - 1; i++) {
    t_stream << average_velocity[i] << ",";
  }
  t_stream << average_velocity[t_binsCount - 1] << "\n";
}

void DataStreamer::streamTangentialVelocity(
    std::ofstream& t_stream, size_t t_binsCount, numType t_minRadiusValue,
    numType t_maxRadiusValue, ParticleCollection& particleCollection,
    cl::CommandQueue& cl_queue) {
  particleCollection.readParticleCoordinatesBuffer(cl_queue);
  particleCollection.readParticleMomentumsBuffer(cl_queue);
  particleCollection.readParticleEBuffer(cl_queue);
  std::cout << std::setprecision(8) << std::fixed << std::showpoint;
  numType dr = (t_maxRadiusValue - t_minRadiusValue) / t_binsCount;
  std::vector<u_int> bins(t_binsCount);
  std::vector<numType> average_velocity(t_binsCount);
  numType particleRadius;
  numType particleTangentialVelocity;

  t_stream << t_binsCount << "," << t_minRadiusValue << "," << t_maxRadiusValue
           << "\n";

  for (size_t i = 0; i < t_binsCount - 1; i++) {
    t_stream << (i + 1) * dr << ",";
  }
  t_stream << t_binsCount * dr << "\n";

  for (u_int i = 0; i < particleCollection.getParticleCountTotal(); i++) {
    particleRadius = particleCollection.calculateParticleRadius(i);
    particleTangentialVelocity =
        particleCollection.calculateParticleTangentialVelocity(i);

    if ((particleRadius >= t_minRadiusValue) &&
        (particleRadius < t_maxRadiusValue)) {
      average_velocity[(int)(particleRadius / dr)] =
          (average_velocity[(int)(particleRadius / dr)] *
               bins[(int)(particleRadius / dr)] +
           particleTangentialVelocity) /
          (bins[(int)(particleRadius / dr)] + 1);
      bins[(int)((particleRadius - t_minRadiusValue) / dr)] += 1;
    }
  }
  for (size_t i = 0; i < t_binsCount - 1; i++) {
    t_stream << average_velocity[i] << ",";
  }
  t_stream << average_velocity[t_binsCount - 1] << "\n";
}

void DataStreamer::streamRadialMomentumProfile(
    std::ofstream& t_stream, size_t t_binsCountRadius,
    size_t t_binsCountMomentum, numType t_minRadiusValue,
    numType t_maxRadiusValue, numType t_minMomentumValue,
    numType t_maxMomentumValue, ParticleCollection& particleCollection,
    cl::CommandQueue& cl_queue) {
  particleCollection.readParticleCoordinatesBuffer(cl_queue);
  particleCollection.readParticleMomentumsBuffer(cl_queue);
  particleCollection.readParticleEBuffer(cl_queue);
  std::cout << std::setprecision(8) << std::fixed << std::showpoint;
  numType particleRadius;
  numType particleMomentum;
  numType dr = (t_maxRadiusValue - t_minRadiusValue) / t_binsCountRadius;
  numType dp = (t_maxMomentumValue - t_minMomentumValue) / t_binsCountMomentum;

  std::vector<std::vector<u_int>> radialMomentumBins(
      t_binsCountRadius, std::vector<u_int>(t_binsCountMomentum, 0));

  t_stream << t_binsCountRadius << "," << t_minRadiusValue << ","
           << t_maxRadiusValue << "," << t_binsCountMomentum << ","
           << t_minMomentumValue << "," << t_maxMomentumValue << "\n";
  for (size_t i = 0; i < t_binsCountRadius; i++) {
    for (size_t j = 0; j < t_binsCountMomentum - 1; j++) {
      t_stream << "(" << (i + 1) * dr << "; " << (j + 1) * dp << ")"
               << ",";
    }
    if (i == t_binsCountRadius - 1) {
      t_stream << "(" << (i + 1) * dr << "; " << t_binsCountMomentum * dp << ")"
               << "\n";
    } else {
      t_stream << "(" << (i + 1) * dr << "; " << t_binsCountMomentum * dp << ")"
               << ",";
    }
  }

  for (u_int i = 0; i < particleCollection.getParticleCountTotal(); i++) {
    particleRadius = particleCollection.calculateParticleRadius(i);
    particleMomentum = particleCollection.calculateParticleMomentum(i);

    if ((particleRadius >= t_minRadiusValue) &&
        (particleRadius < t_maxRadiusValue) &&
        (particleMomentum >= t_minMomentumValue) &&
        (particleMomentum < t_maxMomentumValue)) {
      radialMomentumBins[(int)((particleRadius - t_minRadiusValue) / dr)]
                        [(int)((particleMomentum - t_minMomentumValue) / dp)] +=
          1;
    }
  }

  for (size_t i = 0; i < t_binsCountRadius; i++) {
    for (size_t j = 0; j < t_binsCountMomentum - 1; j++) {
      t_stream << radialMomentumBins[i][j] << ",";
    }
    if (i == t_binsCountRadius - 1) {
      t_stream << radialMomentumBins[i][t_binsCountMomentum - 1] << std::endl;
    } else {
      t_stream << radialMomentumBins[i][t_binsCountMomentum - 1] << ",";
    }
  }
}