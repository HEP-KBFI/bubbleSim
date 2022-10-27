#include "simulation.h"

Simulation::Simulation(int t_seed, numType t_massTrue, numType t_massFalse,
                       numType t_temperatureTrue, numType t_temperatureFalse,
                       unsigned int t_particleCountTrue,
                       unsigned int t_particleCountFalse, numType t_coupling) {
  // Set up random number generator
  m_seed = t_seed;
  if (t_seed == 0) {
    m_generator = std::mt19937_64(m_randDev());
  } else {
    m_generator = std::mt19937_64(t_seed);
  }

  m_distribution = std::uniform_real_distribution<numType>(0, 1);

  // Masses
  m_massTrue = t_massTrue;
  m_massFalse = t_massFalse;
  m_massDelta2 = std::pow(t_massTrue - t_massFalse, 2);

  // Temperatures
  m_temperatureTrue = t_temperatureTrue;
  m_temperatureFalse = t_temperatureFalse;

  // Particle counts
  m_particleCountTrue = t_particleCountTrue;
  m_particleCountFalse = t_particleCountFalse;
  m_particleCountTotal = m_particleCountTrue + t_particleCountFalse;

  if (m_particleCountTotal <= 0) {
    std::cout << "Particle count is < 0.\nExiting program..." << std::endl;
    exit(0);
  }

  m_rhoTrue = 0, m_rhoFalse = 0, m_nTrue = 0, m_nFalse = 0;

  numType dpFalse = 1e-5 * m_temperatureFalse;
  numType pMaxFalse = 30 * t_temperatureFalse;
  numType dpTrue = 1e-5 * m_temperatureTrue;
  numType pMaxTrue = 30 * t_temperatureTrue;
  int vectorSizeFalse = (int)(pMaxFalse / dpFalse);
  int vectorSizeTrue = (int)(pMaxFalse / dpFalse);

  if ((m_particleCountFalse > 0) && (m_temperatureFalse > 0)) {
    m_nFalse = calculateNumberDensity(m_massFalse, m_temperatureFalse, dpFalse,
                                      pMaxFalse);
    m_rhoFalse = calculateEnergyDensity(m_massFalse, m_temperatureFalse,
                                        dpFalse, pMaxFalse);
    calculateCPD(m_massFalse, m_temperatureFalse, dpFalse, pMaxFalse,
                 vectorSizeFalse, m_cpdFalse, m_pFalse);
  }
  if ((m_particleCountTrue > 0) && (m_temperatureTrue > 0)) {
    m_rhoTrue =
        calculateNumberDensity(m_massTrue, m_temperatureTrue, dpTrue, pMaxTrue);
    m_nTrue =
        calculateEnergyDensity(m_massTrue, m_temperatureTrue, dpTrue, pMaxTrue);
    calculateCPD(m_massTrue, m_temperatureTrue, dpTrue, pMaxTrue,
                 vectorSizeTrue, m_cpdTrue, m_pTrue);
  }

  m_coupling = t_coupling;

  // Time
  m_dt = 0.;
  m_time = 0.;
  m_dPressureStep = 0.;

  m_dP = std::vector<double>(m_particleCountTotal, 0.);

  // Data reserve
  m_interactedFalse = std::vector<int8_t>(m_particleCountTotal, 0);
  m_passedFalse = std::vector<int8_t>(m_particleCountTotal, 0);
  m_interactedTrue = std::vector<int8_t>(m_particleCountTotal, 0);

  // Other sim parameters
  m_rhoTrueSim = 0, m_rhoFalseSim = 0, m_nTrueSim = 0, m_nFalseSim = 0;
  m_nTrueSimInitial = 0, m_rhoTrueSimInitial = 0, m_nFalseSimInitial = 0,
  m_rhoFalseSimInitial = 0;
  m_energyTotalInitial = 0, m_energyTotal = 0;
  m_energyParticlesInitial = 0, m_energyParticles = 0;
  m_energyBubble = 0, m_energyBubbleInitial = 0;

  m_energyBubbleLastStep = 0., m_energyParticlesLastStep = 0.;
}

void Simulation::set_dt(numType t_dt) {
  if (t_dt <= 0) {
    std::cout << "dt is <= 0. (" << t_dt << ")" << std::endl;
    exit(0);
  }
  m_dt = t_dt;
}

// Particle functions
numType Simulation::calculateParticleRadius(u_int i) {
  // dot(X, X)
  return std::sqrt(std::fma(m_particles[i].x, m_particles[i].x,
                            std::fma(m_particles[i].y, m_particles[i].y,
                                     m_particles[i].z * m_particles[i].z)));
}

numType Simulation::calculateParticleMomentum(u_int i) {
  // return std::sqrt(m_P[3 * i] * m_P[3 * i] + m_P[3 * i + 1] * m_P[3 * i + 1]
  // + m_P[3 * i + 2] * m_P[3 * i + 2]);

  return std::sqrt(std::fma(m_particles[i].p_x, m_particles[i].p_x,
                            std::fma(m_particles[i].p_y, m_particles[i].p_y,
                                     m_particles[i].p_z * m_particles[i].p_z)));
}

numType Simulation::calculateParticleEnergy(u_int i) {
  return std::sqrt(
      std::fma(m_particles[i].p_x, m_particles[i].p_x,
               std::fma(m_particles[i].p_y, m_particles[i].p_y,
                        std::fma(m_particles[i].p_z, m_particles[i].p_z,
                                 m_particles[i].m * m_particles[i].m))));
}

// Calculate distributions
void Simulation::calculateCPD(numType t_mass, numType t_temperature,
                              numType t_dp, numType t_pMax, int t_vectorSize,
                              std::vector<numType>& t_cpd,
                              std::vector<numType>& t_p) {
  t_p.reserve(t_vectorSize);
  t_cpd.reserve(t_vectorSize);
  numType m2 = std::pow(t_mass, 2);
  numType last_cpdValue = 0;
  numType last_pValue = 0;

  t_cpd.push_back(last_cpdValue);
  t_p.push_back(last_pValue);

  for (int i = 1; i < t_vectorSize; i++) {
    t_p.push_back(last_pValue + t_dp);
    t_cpd.push_back(
        last_cpdValue +
        t_dp * std::pow(last_pValue, 2) *
            std::exp(-std::sqrt(std::fma(last_pValue, last_pValue, m2)) /
                     t_temperature));
    last_cpdValue = t_cpd[i];
    last_pValue = t_p[i];
  }

  for (int i = 0; i < t_vectorSize; i++) {
    t_cpd[i] = t_cpd[i] / last_cpdValue;
  }
}

numType Simulation::calculateNumberDensity(numType t_mass,
                                           numType t_temperature, numType t_dp,
                                           numType t_pMax) {
  numType n = 0;
  numType p = 0;
  numType m2 = std::pow(t_mass, 2);

  for (; p <= t_pMax; p += t_dp) {
    n += t_dp * std::pow(p, 2) *
         std::exp(-std::sqrt(std::fma(p, p, m2)) / t_temperature);
  }
  n = n / (numType)(2. * std::pow(M_PI, 2));
  return n;
}

numType Simulation::calculateEnergyDensity(numType t_mass,
                                           numType t_temperature, numType t_dp,
                                           numType t_pMax) {
  numType rho = 0;
  numType p = 0;
  numType m2 = std::pow(t_mass, 2);

  numType sqrt_p2_m2;

  for (; p <= t_pMax; p += t_dp) {
    sqrt_p2_m2 = std::sqrt(std::fma(p, p, m2));
    rho += t_dp * std::pow(p, 2) * sqrt_p2_m2 *
           std::exp(-sqrt_p2_m2 / t_temperature);
  }
  rho = rho / (numType)(2 * std::pow(M_PI, 2));
  return rho;
}

// Sampling and generating
numType Simulation::interp(numType t_value, std::vector<numType>& t_x,
                           std::vector<numType>& t_y) {
  if (t_value < 0) {
    return t_y[0];
  } else if (t_value > t_x.back()) {
    return t_y.back();
  } else {
    unsigned int k1 = 0;
    unsigned int k2 = static_cast<unsigned int>(t_x.size() - 1);
    unsigned int k = (k2 + k1) / 2;
    for (; k2 - k1 > 1;) {
      if (t_x[k] > t_value) {
        k2 = k;
      } else {
        k1 = k;
      }
      k = (k2 + k1) / 2;
    }
    return t_y[k1] +
           (t_value - t_x[k1]) * (t_y[k2] - t_y[k1]) / (t_x[k2] - t_x[k1]);
  }
}

void Simulation::generateRandomDirection(numType& x, numType& y, numType& z,
                                         numType t_radius) {
  numType phi = std::acos(1 - 2 * m_distribution(m_generator));  // inclination
  numType theta = 2 * M_PI * m_distribution(m_generator);
  x = t_radius * std::sin(phi) * std::cos(theta);  // x
  y = t_radius * std::sin(phi) * std::sin(theta);  // y
  z = t_radius * std::cos(phi);                    // z
}

void Simulation::generateParticleMomentum(numType& p_x, numType& p_y,
                                          numType& p_z,
                                          std::vector<numType>& t_cpd,
                                          std::vector<numType>& t_p,
                                          numType& t_pResult) {
  t_pResult = interp(m_distribution(m_generator), t_cpd, t_p);
  generateRandomDirection(p_x, p_y, p_z, t_pResult);
}

void Simulation::generatePointInBox(numType& x, numType& y, numType& z,
                                    numType& t_SideHalf) {
  x = t_SideHalf - 2 * t_SideHalf * m_distribution(m_generator);
  y = t_SideHalf - 2 * t_SideHalf * m_distribution(m_generator);
  z = t_SideHalf - 2 * t_SideHalf * m_distribution(m_generator);
}

void Simulation::generatePointInBox(numType& x, numType& y, numType& z,
                                    numType& t_xSideHalf, numType& t_ySideHalf,
                                    numType& t_zSideHalf) {
  x = t_xSideHalf - 2 * t_xSideHalf * m_distribution(m_generator);
  y = t_ySideHalf - 2 * t_ySideHalf * m_distribution(m_generator);
  z = t_zSideHalf - 2 * t_zSideHalf * m_distribution(m_generator);
}

void Simulation::generateNParticlesInBox(numType t_mass, numType& t_sideHalf,
                                         u_int t_N, std::vector<numType>& t_cpd,
                                         std::vector<numType>& t_p) {
  numType x, y, z;
  numType p_x, p_y, p_z;
  numType E;
  numType m2 = std::pow(t_mass, 2);
  numType pValue;

  for (u_int i = 0; i < t_N; i++) {
    // Generates 3D space coordinates and pushes to m_X vector
    generatePointInBox(x, y, z, t_sideHalf);

    // Generates 3D space coordinates and pushes to m_P vector
    generateParticleMomentum(p_x, p_y, p_z, t_cpd, t_p, pValue);
    E = std::sqrt(m2 + pow(pValue, 2));
    m_particles.push_back(Particle{x, y, z, p_x, p_y, p_z, E, t_mass});
  }
}

void Simulation::generateNParticlesInBox(numType t_mass, numType& t_xSideHalf,
                                         numType& t_ySideHalf,
                                         numType& t_zSideHalf, u_int t_N,
                                         std::vector<numType>& t_cpd,
                                         std::vector<numType>& t_p) {
  numType x, y, z;
  numType p_x, p_y, p_z;
  numType E;
  numType m2 = std::pow(t_mass, 2);
  numType pValue;

  for (u_int i = 0; i < t_N; i++) {
    // Generates 3D space coordinates and pushes to m_X vector
    generatePointInBox(x, y, z, t_xSideHalf, t_ySideHalf, t_zSideHalf);
    // Generates 3D space coordinates and pushes to m_P vector
    generateParticleMomentum(p_x, p_y, p_z, t_cpd, t_p, pValue);

    E = std::sqrt(m2 + pow(pValue, 2));
    m_particles.push_back(Particle{x, y, z, p_x, p_y, p_z, E, t_mass});
  }
}

void Simulation::generateNParticlesInBox(numType t_mass, numType& t_radiusIn,
                                         numType& t_sideHalf, u_int t_N,
                                         std::vector<numType>& t_cpd,
                                         std::vector<numType>& t_p) {
  numType x, y, z;
  numType p_x, p_y, p_z;
  numType E;
  numType m2 = std::pow(t_mass, 2);
  numType pValue;
  numType radius;
  for (u_int i = 0; i < t_N; i++) {
    // Generates 3D space coordinates and pushes to m_X vector
    do {
      generatePointInBox(x, y, z, t_sideHalf);
      radius = std::sqrt(std::fma(x, x, fma(y, y, z * z)));
    } while (radius < t_radiusIn);
    // Generates 3D space coordinates and pushes to m_P vector
    generateParticleMomentum(p_x, p_y, p_z, t_cpd, t_p, pValue);
    E = std::sqrt(m2 + pow(pValue, 2));
    m_particles.push_back(Particle{x, y, z, p_x, p_y, p_z, E, t_mass});
  }
}

void Simulation::generateNParticlesInBox(numType t_mass, numType& t_radiusIn,
                                         numType& t_xSideHalf,
                                         numType& t_ySideHalf,
                                         numType& t_zSideHalf, u_int t_N,
                                         std::vector<numType>& t_cpd,
                                         std::vector<numType>& t_p) {
  numType x, y, z;
  numType p_x, p_y, p_z;
  numType E;
  numType m2 = std::pow(t_mass, 2);
  numType pValue;
  numType radius;
  for (u_int i = 0; i < t_N; i++) {
    // Generates 3D space coordinates and pushes to m_X vector
    do {
      generatePointInBox(x, y, z, t_xSideHalf, t_ySideHalf, t_zSideHalf);
      radius = std::sqrt(std::fma(x, x, fma(y, y, z * z)));
    } while (radius < t_radiusIn);
    // Generates 3D space coordinates and pushes to m_P vector
    generateParticleMomentum(p_x, p_y, p_z, t_cpd, t_p, pValue);
    E = std::sqrt(m2 + pow(pValue, 2));
    m_particles.push_back(Particle{x, y, z, p_x, p_y, p_z, E, t_mass});
  }
}

void Simulation::generateNParticlesInSphere(numType t_mass, numType& t_radius1,
                                            u_int t_N,
                                            std::vector<numType>& t_cpd,
                                            std::vector<numType>& t_p) {
  numType x, y, z;
  numType p_x, p_y, p_z;
  numType E;
  numType m2 = std::pow(t_mass, 2);
  numType pValue;
  numType radius;

  for (u_int i = 0; i < t_N; i++) {
    // Generates 3D space coordinates and pushes to m_X vector
    do {
      generatePointInBox(x, y, z, t_radius1);
      radius = std::sqrt(std::fma(x, x, fma(y, y, z * z)));
    } while (radius > t_radius1);

    // Generates 3D space coordinates and pushes to m_P vector
    generateParticleMomentum(p_x, p_y, p_z, t_cpd, t_p, pValue);
    E = std::sqrt(m2 + pow(pValue, 2));
    m_particles.push_back(Particle{x, y, z, p_x, p_y, p_z, E, t_mass});
  }
}

void Simulation::generateNParticlesInSphere(numType t_mass, numType& t_radius1,
                                            numType t_radius2, u_int t_N,
                                            std::vector<numType>& t_cpd,
                                            std::vector<numType>& t_p) {
  numType x, y, z;
  numType p_x, p_y, p_z;
  numType E;
  numType m2 = std::pow(t_mass, 2);
  numType pValue;
  numType radius;

  for (u_int i = 0; i < t_N; i++) {
    // Generates 3D space coordinates and pushes to m_X vector
    do {
      generatePointInBox(x, y, z, t_radius1);
      radius = std::sqrt(std::fma(x, x, fma(y, y, z * z)));
    } while ((t_radius1 > radius) || (radius > t_radius2));

    // Generates 3D space coordinates and pushes to m_P vector
    generateParticleMomentum(p_x, p_y, p_z, t_cpd, t_p, pValue);
    E = std::sqrt(m2 + pow(pValue, 2));
    m_particles.push_back(Particle{x, y, z, p_x, p_y, p_z, E, t_mass});
  }
}

// Get values from the simulation

numType Simulation::countParticleNumberDensity(numType t_radius1) {
  u_int counter = 0;
  numType volume = 4 * (pow(t_radius1, 3) * M_PI) / 3;
  for (int i = 0; i < m_particleCountTotal; i++) {
    if (calculateParticleRadius(i) < t_radius1) {
      counter += 1;
    }
  }
  return (numType)counter / volume;
}

numType Simulation::countParticleNumberDensity(numType t_radius1,
                                               numType t_radius2) {
  u_int counter = 0;
  numType volume = 4 * ((pow(t_radius2, 3) - pow(t_radius1, 3)) * M_PI) / 3;
  for (int i = 0; i < m_particleCountTotal; i++) {
    if ((calculateParticleRadius(i) > t_radius1) &&
        (calculateParticleRadius(i) < t_radius2)) {
      counter += 1;
    }
  }
  return (numType)counter / volume;
}

numType Simulation::countParticleEnergyDensity(numType t_radius1) {
  numType energy = countParticlesEnergy(t_radius1);
  std::cout << energy << std::endl;
  numType volume = 4 * (pow(t_radius1, 3) * M_PI) / 3;
  return energy / volume;
}

numType Simulation::countParticleEnergyDensity(numType t_radius1,
                                               numType t_radius2) {
  numType energy = countParticlesEnergy(t_radius1, t_radius2);
  numType volume = 4 * ((pow(t_radius2, 3) - pow(t_radius1, 3)) * M_PI) / 3;
  return (numType)energy / volume;
}

numType Simulation::countParticlesEnergy() {
  numType energy = 0.;
  for (int i = 0; i < m_particleCountTotal; i++) {
    energy += m_particles[i].E;
  }
  return energy;
}

numType Simulation::countParticlesEnergy(numType t_radius1) {
  numType energy = 0.;
  for (int i = 0; i < m_particleCountTotal; i++) {
    if (calculateParticleRadius(i) < t_radius1) {
      energy += m_particles[i].E;
    }
  }
  return energy;
}

numType Simulation::countParticlesEnergy(numType t_radius1, numType t_radius2) {
  numType energy = 0.;
  numType radius;
  for (int i = 0; i < m_particleCountTotal; i++) {
    radius = calculateParticleRadius(i);
    if ((radius > t_radius1) && (radius < t_radius2)) {
      energy += m_particles[i].E;
    }
  }
  return energy;
}

void Simulation::step(PhaseBubble& bubble, numType t_dP) {
  m_time += m_dt;
  bubble.evolveWall(m_dt, m_dPressureStep);
}

void Simulation::step(PhaseBubble& phaseBubble, OpenCLWrapper& openCLWrapper) {
  if (m_dt <= 0) {
    std::cout << "Error: dt <= 0.\nExiting program." << std::endl;
    exit(1);
  }
  m_time += m_dt;
  // Write bubble parameters to GPU
  phaseBubble.calculateRadiusAfterStep2(m_dt);
  openCLWrapper.makeStep1(phaseBubble.getRef_Bubble());
  // Run one step on device
  openCLWrapper.makeStep2(m_particleCountTotal);
  // Read dP vector. dP is "Energy" change for particles -> PhaseBubble energy
  // change is -dP
  openCLWrapper.makeStep3(m_particleCountTotal, m_dP);
  m_dPressureStep = 0;
  for (int i = 0; i < m_particleCountTotal; i++) {
    m_dPressureStep += m_dP[i];
  }
  m_dPressureStep /= -phaseBubble.calculateArea();

  phaseBubble.evolveWall(m_dt, m_dPressureStep);
  openCLWrapper.makeStep4(phaseBubble.getRef_Bubble());
}