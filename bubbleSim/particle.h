#pragma once
#include "base.h"
#include "random_number.hpp"

typedef struct Particle {
  cl_numType x;
  cl_numType y;
  cl_numType z;

  cl_numType p_x;
  cl_numType p_y;
  cl_numType p_z;

  cl_numType E;
  cl_numType m;

} Particle;

class ParticleGenerator {
 public:
  ParticleGenerator() {}
  // Constructur to generate particles according to Boltzmann distribution
  ParticleGenerator(numType t_mass, numType t_temperature,
                    numType t_maxMomentumValue, numType t_dp);
  // Constructur to generate all particles with same momentum value
  ParticleGenerator(numType t_mass, numType t_momentum);

  std::array<std::vector<numType>, 2> m_cumulativeProbabilityFunction;

  numType generateNParticlesInBox(numType t_mass, numType& t_sideHalf,
                                  u_int t_N, RandomNumberGenerator& t_generator,
                                  std::vector<Particle>& t_particles);

  numType generateNParticlesInBox(numType t_mass, numType& t_radiusIn,
                                  numType& t_sideHalf, u_int t_N,
                                  RandomNumberGenerator& t_generator,
                                  std::vector<Particle>& t_particles);

  numType generateNParticlesInBox(numType t_mass, numType& t_xSideHalf,
                                  numType& t_ySideHalf, numType& t_zSideHalf,
                                  u_int t_N, RandomNumberGenerator& t_generator,
                                  std::vector<Particle>& t_particles);

  numType generateNParticlesInBox(numType t_mass, numType& t_radiusIn,
                                  numType& t_xSideHalf, numType& t_ySideHalf,
                                  numType& t_zSideHalf, u_int t_N,
                                  RandomNumberGenerator& t_generator,
                                  std::vector<Particle>& t_particles);

  numType generateNParticlesInSphere(numType t_mass, numType& t_radiusMax,
                                     u_int t_N,
                                     RandomNumberGenerator& t_generator,
                                     std::vector<Particle>& t_particles);

  numType generateNParticlesInSphere(numType t_mass, numType& t_radiusMin,
                                     numType t_radiusMax, u_int t_N,
                                     RandomNumberGenerator& t_generator,
                                     std::vector<Particle>& t_particles);

 private:
  numType m_mass;
  // Index [0] = Probability, Index [1] = Momentum value

  numType interp(numType t_xValue, std::vector<numType>& t_xArray,
                 std::vector<numType>& t_yArray);

  void generateRandomDirection(numType& x, numType& y, numType& z,
                               numType t_radius,
                               RandomNumberGenerator& t_generator);

  void generateParticleMomentum(numType& p_x, numType& p_y, numType& p_z,
                                numType& t_pResult,
                                RandomNumberGenerator& t_generator);

  void calculateCPD(numType t_temperature, numType t_pMax, numType t_dp);
  void calculateCPD(numType t_momentum);
  void generatePointInBox(numType& x, numType& y, numType& z,
                          numType& t_SideHalf,
                          RandomNumberGenerator& t_generator);

  void generatePointInSphere(numType& x, numType& y, numType& z,
                             numType t_maxRadius,
                             RandomNumberGenerator& t_generator);

  void generatePointInBox(numType& x, numType& y, numType& z,
                          numType& t_xSideHalf, numType& t_ySideHalf,
                          numType& t_zSideHalf,
                          RandomNumberGenerator& t_generator);
};
