#include "simulation.h"

Simulation::Simulation(int t_seed, numType t_massTrue, numType t_massFalse,
                       numType t_temperatureTrue, numType t_temperatureFalse,
                       unsigned int t_particleCountTrue,
                       unsigned int t_particleCountFalse, numType t_coupling) {
  // Set up random number generator
  m_seed = t_seed;

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

  // Initial simulation values
  m_initialTotalEnergy = 0.;
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