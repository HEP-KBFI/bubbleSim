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

  numType m2 = std::pow(m_mass, 2);
  numType lastCPFValue = 0.;
  numType lastMomentumValue = 0.;

  m_cumulativeProbabilityFunction[0].push_back(lastCPFValue);
  m_cumulativeProbabilityFunction[1].push_back(lastMomentumValue);

  for (size_t i = 1; i < vectorSize; i++) {
    // Integral of: dp * p^2 * exp(-sqrt(p^2+m^2)/T)
    m_cumulativeProbabilityFunction[0].push_back(
        lastCPFValue +
        t_dp * std::pow(lastMomentumValue, 2) *
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
      std::acos(1 - 2 * t_generator.generate_number());  // inclination
  numType theta = 2 * M_PI * t_generator.generate_number();
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
      std::acos(1 - 2 * t_generator.generate_number());  // inclination
  numType theta = 2 * M_PI * t_generator.generate_number();
  numType radius = std::cbrt(t_generator.generate_number()) * t_maxRadius;
  x = radius * std::sin(phi) * std::cos(theta);  // x
  y = radius * std::sin(phi) * std::sin(theta);  // y
  z = radius * std::cos(phi);                    // z
}

numType ParticleGenerator::generateNParticlesInBox(
    numType t_mass, numType& t_sideHalf, u_int t_N,
    RandomNumberGenerator& t_generator, std::vector<Particle>& t_particles) {
  numType totalEnergy = 0.;

  numType x, y, z;
  numType p_x, p_y, p_z;
  numType E;
  numType m2 = std::pow(t_mass, 2);
  numType pValue;

  for (u_int i = 0; i < t_N; i++) {
    // Generates 3D space coordinates and pushes to m_X vector
    generatePointInBox(x, y, z, t_sideHalf, t_generator);

    // Generates 3D space coordinates and pushes to m_P vector
    generateParticleMomentum(p_x, p_y, p_z, pValue, t_generator);
    E = std::sqrt(m2 + pow(pValue, 2));
    totalEnergy += E;
    t_particles.push_back(Particle{x, y, z, p_x, p_y, p_z, E, t_mass});
  }
  return totalEnergy;
}

numType ParticleGenerator::generateNParticlesInBox(
    numType t_mass, numType& t_xSideHalf, numType& t_ySideHalf,
    numType& t_zSideHalf, u_int t_N, RandomNumberGenerator& t_generator,
    std::vector<Particle>& t_particles) {
  numType totalEnergy = 0.;

  numType x, y, z;
  numType p_x, p_y, p_z;
  numType E;
  numType m2 = std::pow(t_mass, 2);
  numType pValue;

  for (u_int i = 0; i < t_N; i++) {
    // Generates 3D space coordinates and pushes to m_X vector
    generatePointInBox(x, y, z, t_xSideHalf, t_ySideHalf, t_zSideHalf,
                       t_generator);
    // Generates 3D space coordinates and pushes to m_P vector
    generateParticleMomentum(p_x, p_y, p_z, pValue, t_generator);

    E = std::sqrt(m2 + pow(pValue, 2));
    totalEnergy += E;
    t_particles.push_back(Particle{x, y, z, p_x, p_y, p_z, E, t_mass});
  }
  return totalEnergy;
}

numType ParticleGenerator::generateNParticlesInBox(
    numType t_mass, numType& t_radiusIn, numType& t_sideHalf, u_int t_N,
    RandomNumberGenerator& t_generator, std::vector<Particle>& t_particles) {
  numType totalEnergy = 0.;

  numType x, y, z;
  numType p_x, p_y, p_z;
  numType E;
  numType m2 = std::pow(t_mass, 2);
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
    E = std::sqrt(m2 + pow(pValue, 2));
    totalEnergy += E;
    t_particles.push_back(Particle{x, y, z, p_x, p_y, p_z, E, t_mass});
  }
  return totalEnergy;
}

numType ParticleGenerator::generateNParticlesInBox(
    numType t_mass, numType& t_radiusIn, numType& t_xSideHalf,
    numType& t_ySideHalf, numType& t_zSideHalf, u_int t_N,
    RandomNumberGenerator& t_generator, std::vector<Particle>& t_particles) {
  numType totalEnergy = 0.;

  numType x, y, z;
  numType p_x, p_y, p_z;
  numType E;
  numType m2 = std::pow(t_mass, 2);
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
    E = std::sqrt(m2 + pow(pValue, 2));
    totalEnergy += E;
    t_particles.push_back(Particle{x, y, z, p_x, p_y, p_z, E, t_mass});
  }
  return totalEnergy;
}

numType ParticleGenerator::generateNParticlesInSphere(
    numType t_mass, numType& t_radiusMax, u_int t_N,
    RandomNumberGenerator& t_generator, std::vector<Particle>& t_particles) {
  numType totalEnergy = 0.;
  numType x, y, z;
  numType p_x, p_y, p_z;
  numType E;
  numType m2 = std::pow(t_mass, 2);
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
    E = std::sqrt(m2 + pow(pValue, 2));
    totalEnergy += E;
    t_particles.push_back(Particle{x, y, z, p_x, p_y, p_z, E, t_mass});
  }
  return totalEnergy;
}

numType ParticleGenerator::generateNParticlesInSphere(
    numType t_mass, numType& t_radiusMin, numType t_radiusMax, u_int t_N,
    RandomNumberGenerator& t_generator, std::vector<Particle>& t_particles) {
  numType totalEnergy = 0.;
  numType x, y, z;
  numType p_x, p_y, p_z;
  numType E;
  numType m2 = std::pow(t_mass, 2);
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
    E = std::sqrt(m2 + pow(pValue, 2));
    totalEnergy += E;
    t_particles.push_back(Particle{x, y, z, p_x, p_y, p_z, E, t_mass});
  }
  return totalEnergy;
}