#include "particle.h"

ParticleGenerator::ParticleGenerator(numType t_mass, numType t_temperature,
                                     numType t_maxMomentumValue, numType t_dp) {
  m_mass = t_mass;
  calculateCPD(t_temperature, t_maxMomentumValue, t_dp);
}

ParticleGenerator::ParticleGenerator(numType t_mass, numType t_momentum) {
  m_mass = t_mass;
  calculateCPD(t_momentum);
}

void ParticleGenerator::calculateCPD(numType t_temperature, numType t_pMax,
                                     numType t_dp) {
  size_t vectorSize = (size_t)(t_pMax / t_dp);

  m_cumulativeProbabilityFunction[0].clear();
  m_cumulativeProbabilityFunction[1].clear();

  m_cumulativeProbabilityFunction[0].reserve(vectorSize);
  m_cumulativeProbabilityFunction[1].reserve(vectorSize);

  numType m2 = std::pow(m_mass, (numType)2.);
  numType lastCPFValue = 0.;
  numType lastMomentumValue = 0.;

  m_cumulativeProbabilityFunction[0].push_back(lastCPFValue);
  m_cumulativeProbabilityFunction[1].push_back(lastMomentumValue);

  for (size_t i = 1; i < vectorSize; i++) {
    // Integral of: dp * p^2 * exp(-sqrt(p^2+m^2)/T)
    m_cumulativeProbabilityFunction[0].push_back(
        lastCPFValue +
        t_dp * std::pow(lastMomentumValue, (numType)2.) *
            std::exp(
                -std::sqrt(std::fma(lastMomentumValue, lastMomentumValue, m2)) /
                t_temperature));
    m_cumulativeProbabilityFunction[1].push_back(lastMomentumValue + t_dp);
    lastCPFValue = m_cumulativeProbabilityFunction[0][i];
    lastMomentumValue = m_cumulativeProbabilityFunction[1][i];
  }

  for (size_t i = 0; i < vectorSize; i++) {
    m_cumulativeProbabilityFunction[0][i] =
        m_cumulativeProbabilityFunction[0][i] / lastCPFValue;
  }
}

void ParticleGenerator::calculateCPD(numType t_momentumValue) {
  m_cumulativeProbabilityFunction[0].clear();
  m_cumulativeProbabilityFunction[0].reserve(2);
  m_cumulativeProbabilityFunction[0].push_back(0);
  m_cumulativeProbabilityFunction[0].push_back(1);

  m_cumulativeProbabilityFunction[1].clear();
  m_cumulativeProbabilityFunction[1].reserve(2);
  m_cumulativeProbabilityFunction[1].push_back(t_momentumValue);
  m_cumulativeProbabilityFunction[1].push_back(t_momentumValue);
}

numType ParticleGenerator::interp(numType t_xValue,
                                  std::vector<numType>& t_xArray,
                                  std::vector<numType>& t_yArray) {
  if (t_xValue < 0) {
    return t_yArray[0];
  } else if (t_xValue > t_xArray.back()) {
    return t_yArray.back();
  } else {
    unsigned int k1 = 0;
    unsigned int k2 = static_cast<unsigned int>(t_xArray.size() - 1);
    unsigned int k = (k2 + k1) / 2;
    for (; k2 - k1 > 1;) {
      if (t_xArray[k] > t_xValue) {
        k2 = k;
      } else {
        k1 = k;
      }
      k = (k2 + k1) / 2;
    }
    return t_yArray[k1] + (t_xValue - t_xArray[k1]) *
                              (t_yArray[k2] - t_yArray[k1]) /
                              (t_xArray[k2] - t_xArray[k1]);
  }
}

void ParticleGenerator::generateRandomDirection(
    numType& x, numType& y, numType& z, numType t_radius,
    RandomNumberGenerator& t_generator) {
  numType phi =
      std::acos((numType)1. - (numType)2. * t_generator.generate_number());  // inclination
  numType theta = (numType)2. * (numType)M_PI * t_generator.generate_number();
  x = t_radius * std::sin(phi) * std::cos(theta);  // x
  y = t_radius * std::sin(phi) * std::sin(theta);  // y
  z = t_radius * std::cos(phi);                    // z
}

void ParticleGenerator::generateParticleMomentum(
    numType& p_x, numType& p_y, numType& p_z, numType& t_pResult,
    RandomNumberGenerator& t_generator) {
  t_pResult =
      interp(t_generator.generate_number(), m_cumulativeProbabilityFunction[0],
             m_cumulativeProbabilityFunction[1]);

  generateRandomDirection(p_x, p_y, p_z, t_pResult, t_generator);
}

void ParticleGenerator::generatePointInBox(numType& x, numType& y, numType& z,
                                           numType& t_SideHalf,
                                           RandomNumberGenerator& t_generator) {
  x = t_SideHalf - 2 * t_SideHalf * t_generator.generate_number();
  y = t_SideHalf - 2 * t_SideHalf * t_generator.generate_number();
  z = t_SideHalf - 2 * t_SideHalf * t_generator.generate_number();
}

void ParticleGenerator::generatePointInBox(numType& x, numType& y, numType& z,
                                           numType& t_xSideHalf,
                                           numType& t_ySideHalf,
                                           numType& t_zSideHalf,
                                           RandomNumberGenerator& t_generator) {
  x = t_xSideHalf - 2 * t_xSideHalf * t_generator.generate_number();
  y = t_ySideHalf - 2 * t_ySideHalf * t_generator.generate_number();
  z = t_zSideHalf - 2 * t_zSideHalf * t_generator.generate_number();
}

void ParticleGenerator::generatePointInSphere(
    numType& x, numType& y, numType& z, numType t_maxRadius,
    RandomNumberGenerator& t_generator) {
  numType phi =
      std::acos((numType)1. - (numType)2. * t_generator.generate_number());  // inclination
  numType theta = (numType)2. * (numType)M_PI * t_generator.generate_number();
  numType radius = std::cbrt(t_generator.generate_number()) * t_maxRadius;
  x = radius * std::sin(phi) * std::cos(theta);  // x
  y = radius * std::sin(phi) * std::sin(theta);  // y
  z = radius * std::cos(phi);                    // z
}

numType ParticleGenerator::generateNParticlesInBox(
    numType t_sideHalf, u_int t_N, RandomNumberGenerator& t_generator,
    std::vector<Particle>& t_particles) {
  numType totalEnergy = 0.;

  numType x, y, z;
  numType p_x, p_y, p_z;
  numType E;
  numType m2 = std::pow(m_mass, (numType)2.);
  numType pValue;

  for (u_int i = 0; i < t_N; i++) {
    // Generates 3D space coordinates and pushes to m_X vector
    generatePointInBox(x, y, z, t_sideHalf, t_generator);

    // Generates 3D space coordinates and pushes to m_P vector
    generateParticleMomentum(p_x, p_y, p_z, pValue, t_generator);
    E = std::sqrt(m2 + pow(pValue, (numType)2.));
    totalEnergy += E;
    t_particles.push_back(Particle{x, y, z, p_x, p_y, p_z, E, m_mass});
  }
  return totalEnergy;
}

numType ParticleGenerator::generateNParticlesInBox(
    numType t_xSideHalf, numType t_ySideHalf, numType t_zSideHalf, u_int t_N,
    RandomNumberGenerator& t_generator, std::vector<Particle>& t_particles) {
  numType totalEnergy = 0.;

  numType x, y, z;
  numType p_x, p_y, p_z;
  numType E;
  numType m2 = std::pow(m_mass, (numType)2.);
  numType pValue;

  for (u_int i = 0; i < t_N; i++) {
    // Generates 3D space coordinates and pushes to m_X vector
    generatePointInBox(x, y, z, t_xSideHalf, t_ySideHalf, t_zSideHalf,
                       t_generator);
    // Generates 3D space coordinates and pushes to m_P vector
    generateParticleMomentum(p_x, p_y, p_z, pValue, t_generator);

    E = std::sqrt(m2 + pow(pValue, (numType)2.));
    totalEnergy += E;
    t_particles.push_back(Particle{x, y, z, p_x, p_y, p_z, E, m_mass});
  }
  return totalEnergy;
}

numType ParticleGenerator::generateNParticlesInBox(
    numType t_radiusIn, numType t_sideHalf, u_int t_N,
    RandomNumberGenerator& t_generator, std::vector<Particle>& t_particles) {
  numType totalEnergy = 0.;

  numType x, y, z;
  numType p_x, p_y, p_z;
  numType E;
  numType m2 = std::pow(m_mass, (numType)2.);
  numType pValue;
  numType radius;
  for (u_int i = 0; i < t_N; i++) {
    // Generates 3D space coordinates and pushes to m_X vector
    do {
      generatePointInBox(x, y, z, t_sideHalf, t_generator);
      radius = std::sqrt(std::fma(x, x, fma(y, y, z * z)));
    } while (radius < t_radiusIn);
    // Generates 3D space coordinates and pushes to m_P vector
    generateParticleMomentum(p_x, p_y, p_z, pValue, t_generator);
    E = std::sqrt(m2 + pow(pValue, (numType)2.));
    totalEnergy += E;
    t_particles.push_back(Particle{x, y, z, p_x, p_y, p_z, E, m_mass});
  }
  return totalEnergy;
}

numType ParticleGenerator::generateNParticlesInBox(
    numType t_radiusIn, numType t_xSideHalf, numType t_ySideHalf,
    numType t_zSideHalf, u_int t_N, RandomNumberGenerator& t_generator,
    std::vector<Particle>& t_particles) {
  numType totalEnergy = 0.;

  numType x, y, z;
  numType p_x, p_y, p_z;
  numType E;
  numType m2 = std::pow(m_mass, (numType)2.);
  numType pValue;
  numType radius;
  for (u_int i = 0; i < t_N; i++) {
    // Generates 3D space coordinates and pushes to m_X vector
    do {
      generatePointInBox(x, y, z, t_xSideHalf, t_ySideHalf, t_zSideHalf,
                         t_generator);
      radius = std::sqrt(std::fma(x, x, fma(y, y, z * z)));
    } while (radius < t_radiusIn);
    // Generates 3D space coordinates and pushes to m_P vector
    generateParticleMomentum(p_x, p_y, p_z, pValue, t_generator);
    E = std::sqrt(m2 + pow(pValue, (numType)2.));
    totalEnergy += E;
    t_particles.push_back(Particle{x, y, z, p_x, p_y, p_z, E, m_mass});
  }
  return totalEnergy;
}

numType ParticleGenerator::generateNParticlesInSphere(
    numType t_radiusMax, u_int t_N, RandomNumberGenerator& t_generator,
    std::vector<Particle>& t_particles) {
  numType totalEnergy = 0.;
  numType x, y, z;
  numType p_x, p_y, p_z;
  numType E;
  numType m2 = std::pow(m_mass, (numType)2.);
  numType pValue;
  numType radius;

  for (u_int i = 0; i < t_N; i++) {
    // Generates 3D space coordinates and pushes to m_X vector
    do {
      generatePointInSphere(x, y, z, t_radiusMax, t_generator);
      radius = std::sqrt(std::fma(x, x, fma(y, y, z * z)));
    } while (radius > t_radiusMax);

    // Generates 3D space coordinates and pushes to m_P vector
    generateParticleMomentum(p_x, p_y, p_z, pValue, t_generator);
    E = std::sqrt(m2 + pow(pValue, (numType)2.));
    totalEnergy += E;
    t_particles.push_back(Particle{x, y, z, p_x, p_y, p_z, E, m_mass});
  }
  return totalEnergy;
}

numType ParticleGenerator::generateNParticlesInSphere(
    numType t_radiusMin, numType t_radiusMax, u_int t_N,
    RandomNumberGenerator& t_generator, std::vector<Particle>& t_particles) {
  numType totalEnergy = 0.;
  numType x, y, z;
  numType p_x, p_y, p_z;
  numType E;
  numType m2 = std::pow(m_mass, (numType)2.);
  numType pValue;
  numType radius;

  for (u_int i = 0; i < t_N; i++) {
    // Generates 3D space coordinates and pushes to m_X vector
    do {
      generatePointInSphere(x, y, z, t_radiusMax, t_generator);
      radius = std::sqrt(std::fma(x, x, fma(y, y, z * z)));
    } while ((t_radiusMin > radius) || (radius > t_radiusMax));

    // Generates 3D space coordinates and pushes to m_P vector
    generateParticleMomentum(p_x, p_y, p_z, pValue, t_generator);
    E = std::sqrt(m2 + pow(pValue, (numType)2.));
    totalEnergy += E;
    t_particles.push_back(Particle{x, y, z, p_x, p_y, p_z, E, m_mass});
  }
  return totalEnergy;
}

ParticleCollection::ParticleCollection(
    numType t_massTrue, numType t_massFalse, numType t_temperatureTrue,
    numType t_temperatureFalse, unsigned int t_particleCountTrue,
    unsigned int t_particleCountFalse, numType t_coupling,
    bool t_bubbleIsTrueVacuum, cl::Context& cl_context) {
  // Set up random number generator
  int openCLerrNum;
  // Masses
  if (t_bubbleIsTrueVacuum) {
    m_massIn = t_massTrue;
    m_massOut = t_massFalse;
    m_temperatureIn = t_temperatureTrue;
    m_temperatureOut = t_temperatureFalse;
    m_particleCountIn = t_particleCountTrue;
    m_particleCountOut = t_particleCountFalse;
  } else {
    m_massIn = t_massFalse;
    m_massOut = t_massTrue;
    m_temperatureIn = t_temperatureFalse;
    m_temperatureOut = t_temperatureTrue;
    m_particleCountIn = t_particleCountFalse;
    m_particleCountOut = t_particleCountTrue;
  }
  m_massDelta2 = std::abs(std::pow(t_massTrue, (numType)2.) - std::pow(t_massFalse, (numType)2.));

  m_massInBuffer =
      cl::Buffer(cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 sizeof(numType), &m_massIn, &openCLerrNum);

  m_massOutBuffer =
      cl::Buffer(cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 sizeof(numType), &m_massOut, &openCLerrNum);
  m_massDelta2Buffer =
      cl::Buffer(cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 sizeof(numType), &m_massDelta2, &openCLerrNum);

  // Temperatures
  m_massTrue = t_massTrue;
  m_massFalse = t_massFalse;
  m_temperatureTrue = t_temperatureTrue;
  m_temperatureFalse = t_temperatureFalse;
  m_particleCountTrue = t_particleCountTrue;
  m_particleCountFalse = t_particleCountFalse;

  // Particle counts
  m_particleCountTotal = m_particleCountIn + m_particleCountOut;

  if (m_particleCountTotal <= 0) {
    std::cout << "Particle count is < 0.\nExiting program..." << std::endl;
    exit(0);
  }

  m_coupling = t_coupling;

  m_particles.reserve(m_particleCountTotal);
  m_particlesBuffer =
      cl::Buffer(cl_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                 m_particleCountTotal * sizeof(Particle), m_particles.data(),
                 &openCLerrNum);

  m_dP = std::vector<numType>(m_particleCountTotal, 0.);
  m_dPBuffer = cl::Buffer(cl_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                          m_particleCountTotal * sizeof(numType), m_dP.data(),
                          &openCLerrNum);
  // Data reserve
  m_interactedBubbleFalseState = std::vector<int8_t>(m_particleCountTotal, 0);
  m_interactedBubbleFalseStateBuffer =
      cl::Buffer(cl_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                 m_particleCountTotal * sizeof(int8_t),
                 m_interactedBubbleFalseState.data(), &openCLerrNum);

  m_passedBubbleFalseState = std::vector<int8_t>(m_particleCountTotal, 0);
  m_passedBubbleFalseStateBuffer =
      cl::Buffer(cl_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                 m_particleCountTotal * sizeof(int8_t),
                 m_passedBubbleFalseState.data(), &openCLerrNum);
  m_interactedBubbleTrueState = std::vector<int8_t>(m_particleCountTotal, 0);
  m_interactedBubbleTrueStateBuffer =
      cl::Buffer(cl_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                 m_particleCountTotal * sizeof(int8_t),
                 m_interactedBubbleTrueState.data(), &openCLerrNum);
  // Initial simulation values
  m_initialTotalEnergy = 0.;
}

/*
 *
 * ParticleCollection functions
 *
 */
numType ParticleCollection::calculateParticleRadius(u_int i) {
  // dot(X, X)
  return std::sqrt(std::fma(m_particles[i].x, m_particles[i].x,
                            std::fma(m_particles[i].y, m_particles[i].y,
                                     m_particles[i].z * m_particles[i].z)));
}

numType ParticleCollection::calculateParticleMomentum(u_int i) {
  // return std::sqrt(m_P[3 * i] * m_P[3 * i] + m_P[3 * i + 1] * m_P[3 * i + 1]
  // + m_P[3 * i + 2] * m_P[3 * i + 2]);

  return std::sqrt(std::fma(m_particles[i].p_x, m_particles[i].p_x,
                            std::fma(m_particles[i].p_y, m_particles[i].p_y,
                                     m_particles[i].p_z * m_particles[i].p_z)));
}

numType ParticleCollection::calculateParticleEnergy(u_int i) {
  return std::sqrt(
      std::fma(m_particles[i].p_x, m_particles[i].p_x,
               std::fma(m_particles[i].p_y, m_particles[i].p_y,
                        std::fma(m_particles[i].p_z, m_particles[i].p_z,
                                 m_particles[i].m * m_particles[i].m))));
}

numType ParticleCollection::calculateParticleRadialVelocity(u_int i) {
  return std::fma(m_particles[i].p_x, m_particles[i].x,
                  std::fma(m_particles[i].p_y, m_particles[i].y,
                           m_particles[i].p_z * m_particles[i].z)) /
         (m_particles[i].E * calculateParticleRadius(i));
}

numType ParticleCollection::calculateParticleTangentialVelocity(u_int i) {
  return std::sqrt(1 - std::pow(calculateParticleRadialVelocity(i), (numType)2.));
}

numType ParticleCollection::calculateNumberDensity(numType t_mass,
                                                   numType t_temperature,
                                                   numType t_dp,
                                                   numType t_pMax) {
  numType n = 0;
  numType p = 0;
  numType m2 = std::pow(t_mass, (numType)2.);

  for (; p <= t_pMax; p += t_dp) {
    n += t_dp * std::pow(p, (numType)2.) *
         std::exp(-std::sqrt(std::fma(p, p, m2)) / t_temperature);
  }
  n = n / (numType)(2. * std::pow(M_PI, (numType)2.));
  return n;
}

numType ParticleCollection::calculateEnergyDensity(numType t_mass,
                                                   numType t_temperature,
                                                   numType t_dp,
                                                   numType t_pMax) {
  numType rho = 0;
  numType p = 0;
  numType m2 = std::pow(t_mass, (numType)2.);

  numType sqrt_p2_m2;

  for (; p <= t_pMax; p += t_dp) {
    sqrt_p2_m2 = std::sqrt(std::fma(p, p, m2));
    rho += t_dp * std::pow(p, (numType)2.) * sqrt_p2_m2 *
           std::exp(-sqrt_p2_m2 / t_temperature);
  }
  rho = rho / (numType)(2 * std::pow((numType)M_PI, (numType)2.));
  return rho;
}

// Get values from the simulation

numType ParticleCollection::countParticleNumberDensity(numType t_radius1) {
  u_int counter = 0;
  numType volume = (numType)4. * (pow(t_radius1, (numType)3.) * (numType)M_PI) / (numType)3.;
  for (int i = 0; i < m_particleCountTotal; i++) {
    if (calculateParticleRadius(i) < t_radius1) {
      counter += 1;
    }
  }
  return (numType)counter / volume;
}

numType ParticleCollection::countParticleNumberDensity(numType t_radius1,
                                                       numType t_radius2) {
  u_int counter = 0;
  numType volume = (numType)4. * ((pow(t_radius2, (numType)3.) - pow(t_radius1, (numType)3.)) * (numType)M_PI) / (numType)3.;
  for (int i = 0; i < m_particleCountTotal; i++) {
    if ((calculateParticleRadius(i) > t_radius1) &&
        (calculateParticleRadius(i) < t_radius2)) {
      counter += 1;
    }
  }
  return (numType)counter / volume;
}

numType ParticleCollection::countParticleEnergyDensity(numType t_radius1) {
  numType energy = countParticlesEnergy(t_radius1);
  std::cout << energy << std::endl;
  numType volume = (numType)4. * (pow(t_radius1, (numType)3.) * (numType)M_PI) / (numType)3.;
  return energy / volume;
}

numType ParticleCollection::countParticleEnergyDensity(numType t_radius1,
                                                       numType t_radius2) {
  numType energy = countParticlesEnergy(t_radius1, t_radius2);
  numType volume = (numType)4. * ((pow(t_radius2, (numType)3.) - pow(t_radius1, (numType)3.)) * (numType)M_PI) / (numType)3.;
  return (numType)energy / volume;
}

numType ParticleCollection::countParticlesEnergy() {
  numType energy = 0.;
  for (int i = 0; i < m_particleCountTotal; i++) {
    energy += m_particles[i].E;
  }
  return energy;
}

numType ParticleCollection::countParticlesEnergy(numType t_radius1) {
  numType energy = 0.;
  for (int i = 0; i < m_particleCountTotal; i++) {
    if (calculateParticleRadius(i) < t_radius1) {
      energy += m_particles[i].E;
    }
  }
  return energy;
}

numType ParticleCollection::countParticlesEnergy(numType t_radius1,
                                                 numType t_radius2) {
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

void ParticleCollection::print_info(ConfigReader t_config,
                                    PhaseBubble& t_bubble) {
  // exp(mu/T) -> ~ chemical potential ->
  std::cout << std::setprecision(6);
  numType exp_mu_T =
      std::log((m_particleCountIn * std::pow((numType)M_PI, (numType)2.)) /
               (t_bubble.calculateVolume() * std::pow(m_temperatureFalse, (numType)3.)));
  numType nFromParameters = m_particleCountIn / t_bubble.calculateVolume();
  // Assuming m = 0
  numType rhoFromParameters =
      m_particleCountIn * 3 * m_massTrue /
      (t_config.parameterEta * t_bubble.calculateVolume());
  numType energyFromParameters = 3 * m_massTrue / t_config.parameterEta;

  numType particleTotalEnergy = countParticlesEnergy();
  numType nSim = m_particleCountIn / t_bubble.calculateVolume();
  numType rhoSim = particleTotalEnergy / t_bubble.calculateVolume();
  numType energySim = particleTotalEnergy / m_particleCountFalse;
  numType dVrho = abs(t_bubble.getdV()) / rhoSim;

  std::string sublabel_prefix = "==== ";
  std::string sublabel_sufix = " ====";

  std::cout << "=============== Particles ===============" << std::endl;
  std::cout << sublabel_prefix + "exp(mu/T) (m-=0): " << exp_mu_T
            << sublabel_sufix << std::endl;
  std::cout << sublabel_prefix + "dV/rho: " << dVrho << sublabel_sufix
            << std::endl;

  std::cout << sublabel_prefix + "Number density" + sublabel_sufix << std::endl;
  std::cout << "Sim: " << nSim << ", Param: " << nFromParameters
            << ", Ratio (sim/param): " << nSim / nFromParameters << std::endl;
  std::cout << sublabel_prefix + "Energy density" + sublabel_sufix << std::endl;
  std::cout << "Sim: " << rhoSim << ", Param (m=0): " << rhoFromParameters

            << ", Ratio (sim/param): " << rhoSim / rhoFromParameters
            << std::endl;
  std::cout << sublabel_prefix + "<E>" + sublabel_sufix << std::endl;
  std::cout << "Sim: " << energySim << ", Param (m=0): " << energyFromParameters
            << ", Ratio (sim/param): " << energySim / energyFromParameters
            << std::endl;
}
