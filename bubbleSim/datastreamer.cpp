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

void DataStreamer::initStream_MomentumIn(size_t t_binsCount,
                                         numType t_maxMomentumValue) {
  m_initialized_MomentumIn = true;
  m_binsCount_MomentumIn = t_binsCount;
  m_maxMomentum_MomentumIn = t_maxMomentumValue;
  m_dp_MomentumIn = t_maxMomentumValue / t_binsCount;

  m_stream_MomentumIn.open(m_filePath / "pIn.csv", std::ios::out);
  m_stream_MomentumIn << t_binsCount << "," << t_maxMomentumValue << "\n";
  for (size_t i = 1; i <= t_binsCount; i++) {
    if (i == t_binsCount) {
      m_stream_MomentumIn << i << std::endl;
    } else {
      m_stream_MomentumIn << i << ",";
    }
  }
}

void DataStreamer::initStream_MomentumOut(size_t t_binsCount,
                                          numType t_maxMomentumValue) {
  m_initialized_MomentumOut = true;
  m_binsCount_MomentumOut = t_binsCount;
  m_maxMomentum_MomentumOut = t_maxMomentumValue;
  m_dp_MomentumOut = t_maxMomentumValue / t_binsCount;

  m_stream_MomentumOut.open(m_filePath / "pOut.csv", std::ios::out);
  m_stream_MomentumOut << t_binsCount << "," << t_maxMomentumValue << "\n";
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

  m_stream_Density.open(m_filePath / "density.csv", std::ios::out);
  m_stream_Density << t_binsCount << "," << t_maxRadiusValue << "\n";
  for (size_t i = 1; i < t_binsCount; i++) {
    if (i == t_binsCount) {
      m_stream_EnergyDensity << i << std::endl;
    } else {
      m_stream_EnergyDensity << i << ",";
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
                          PhaseBubble& bubble, cl::CommandQueue& cl_queue) {
  auto programStartTime = std::chrono::high_resolution_clock::now();
  /*
   * Do only initialized streams.
   * 1) Read in necessary buffers
   * 2) Do for cycle over all particles and count/calculate profiles
   * 3) Stream into files
   */

  particleCollection.readParticlesBuffer(cl_queue);
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

  // Initialize variables
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

    if (m_initialized_Density && (particleRadius < m_maxRadius_Density)) {
      if ((int)(particleRadius / m_dr_Density) >= m_binsCount_Density) {
        std::cout << "Number density problem." << std::endl;
      }
      bins_Density[(int)(particleRadius / m_dr_Density)] += 1;
    }
    if (m_initialized_EnergyDensity &&
        (particleRadius < m_maxRadius_EnergyDensity)) {
      if ((int)(particleRadius / m_dr_EnergyDensity) >=
          m_binsCount_EnergyDensity) {
        std::cout << "Energy density problem." << std::endl;
      }
      bins_EnergyDensity[(int)(particleRadius / m_dr_EnergyDensity)] += 1;
    }
    if (m_initialized_RadialVelocity &&
        (particleRadius < m_maxRadius_RadialVelocity)) {
      if ((int)(particleRadius / m_dr_RadialVelocity) >=
          m_binsCount_RadialVelocity) {
        std::cout << "Radial velocity problem." << std::endl;
      }
      // Average A(N) = [A(N-1) * (N-1) + a(N) ]/N
      particleRadialVelocity =
          particleCollection.calculateParticleRadialVelocity(i);
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
      if ((int)(particleRadius / m_dr_TangentialVelocity) >=
          m_binsCount_TangentialVelocity) {
        std::cout << "Tangential velocity problem." << std::endl;
      }
      particleTangentialVelocity =
          particleCollection.calculateParticleTangentialVelocity(i);
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
        (particleMomentum < m_maxMomentum_MomentumIn)) {
      if ((int)(particleMomentum / m_dp_MomentumIn) >= m_binsCount_MomentumIn) {
        std::cout << "Momentum in problem." << std::endl;
      }
      bins_MomentumIn[(int)(particleMomentum / m_dp_MomentumIn)] += 1;
    }
    if (m_initialized_MomentumOut && (particleRadius > bubble.getRadius()) &&
        (particleMomentum < m_maxMomentum_MomentumIn)) {
      if ((int)(particleMomentum / m_dp_MomentumOut) >=
          m_binsCount_MomentumOut) {
        std::cout << "Momentum out problem." << std::endl;
      }
      bins_MomentumOut[(int)(particleMomentum / m_dp_MomentumOut)] += 1;
    }
    if (m_initialized_Data) {
      if (particleRadius <= bubble.getRadius()) {
        particleInCount += 1;
        particleInEnergy += particleCollection.getParticleEnergy(i);
      }
      particleInteractedFalseCount +=
          particleCollection.getInteractedFalse()[i];
      particlePassedFalseCount += particleCollection.getPassedFalse()[i];
      particleInteractedTrueCount += particleCollection.getInteractedTrue()[i];
      particleTotalEnergy += particleCollection.getParticleEnergy(i);
    }
  }

  // std::cout << "Data collection: successful!" << std::endl;
  //  std::cout << "Max momentum: " << maxMomentum << std::endl;
  if (m_initialized_Data) {
    totalEnergy = particleTotalEnergy + bubble.calculateEnergy();
    std::cout << std::fixed << std::noshowpoint << std::setprecision(8);
    m_stream_Data << simulation.getTime() << "," << simulation.get_dP() << ",";
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
                         bubble.getRadius()
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
  auto programEndTime = std::chrono::high_resolution_clock::now();
  std::cout << "Time taken (stream): "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   programEndTime - programStartTime)
                   .count()
            << " ms." << std::endl;
  ;
}
