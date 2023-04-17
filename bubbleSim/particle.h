#pragma once
#include "base.h"
#include "bubble.h"
#include "config_reader.hpp"
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

  cl_char b_collide = (cl_char)1;
  cl_char b_inBubble = (cl_char)0;
  cl_int idxCollisionCell = (cl_int)0;
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

  numType generateNParticlesInBox(numType t_sideHalf, u_int t_N,
                                  RandomNumberGenerator& t_generator,
                                  std::vector<Particle>& t_particles);

  numType generateNParticlesInBox(numType t_radiusIn, numType t_sideHalf,
                                  u_int t_N, RandomNumberGenerator& t_generator,
                                  std::vector<Particle>& t_particles);

  numType generateNParticlesInBox(numType t_xSideHalf, numType t_ySideHalf,
                                  numType t_zSideHalf, u_int t_N,
                                  RandomNumberGenerator& t_generator,
                                  std::vector<Particle>& t_particles);

  numType generateNParticlesInBox(numType t_radiusIn, numType t_xSideHalf,
                                  numType t_ySideHalf, numType t_zSideHalf,
                                  u_int t_N, RandomNumberGenerator& t_generator,
                                  std::vector<Particle>& t_particles);

  numType generateNParticlesInSphere(numType t_radiusMax, u_int t_N,
                                     RandomNumberGenerator& t_generator,
                                     std::vector<Particle>& t_particles);

  numType generateNParticlesInSphere(numType t_radiusMin, numType t_radiusMax,
                                     u_int t_N,
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

class ParticleCollection {
  /*
  False at the end of variable means False vaccum (lower mass)
  True at the end of variable means False vaccum (higher mass)
  */

  // Masses of particles in true and false vacuum
  numType m_massIn, m_massOut, m_massDelta2;
  numType m_massTrue, m_massFalse;
  cl::Buffer m_massInBuffer;
  cl::Buffer m_massOutBuffer;
  cl::Buffer m_massDelta2Buffer;
  numType m_coupling;
  // Temperatures in true and false vacuum
  numType m_temperatureTrue, m_temperatureFalse, m_temperatureIn,
      m_temperatureOut;
  // Particle counts total / true vacuum / false vacuum
  size_t m_particleCountTotal, m_particleCountIn, m_particleCountOut,
      m_particleCountTrue, m_particleCountFalse;

  // Initial simulation values
  numType m_initialTotalEnergy;

  // Data objects for buffers
  std::vector<Particle> m_particles;
  cl::Buffer m_particlesBuffer;
  std::vector<numType> m_dP;
  cl::Buffer m_dPBuffer;
  /*
   * Count particle-bubble interactions when particle started in
   * false vacuum.
   */
  std::vector<int8_t> m_interactedBubbleFalseState;
  cl::Buffer m_interactedBubbleFalseStateBuffer;
  std::vector<int8_t> m_passedBubbleFalseState;
  cl::Buffer m_passedBubbleFalseStateBuffer;
  /*
   * Interaction with bubble when particle started in true vacuum.
   * This also means that particle gets always through the bubble.
   */
  std::vector<int8_t> m_interactedBubbleTrueState;
  cl::Buffer m_interactedBubbleTrueStateBuffer;

 public:
  /*
   * NB!!! When initializing the vectors and buffers it is important
   * that vector sizes stay the same as buffers need reference
   * to vector data. When vector size is changed during the
   * runtime, vector data address also changes and buffer has wrong
   * memory address
   */
  ParticleCollection(numType t_massTrue, numType t_massFalse,
                     numType t_temperatureTrue, numType t_temperatureFalse,
                     unsigned int t_particleCountTrue,
                     unsigned int t_particleCountFalse, numType t_coupling,
                     bool t_bubbleIsTrueVacuum, cl::Context& cl_context);

  ParticleCollection& operator=(const ParticleCollection& t) { return *this; }

  std::vector<Particle>& getParticles() { return m_particles; }

  std::vector<numType>& get_dP() { return m_dP; }

  std::vector<int8_t>& getInteractedFalse() {
    return m_interactedBubbleFalseState;
  }

  std::vector<int8_t>& getPassedFalse() { return m_passedBubbleFalseState; }

  std::vector<int8_t>& getInteractedTrue() {
    return m_interactedBubbleTrueState;
  }

  cl::Buffer& getParticlesBuffer() { return m_particlesBuffer; }

  void writeParticlesBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_particlesBuffer, CL_TRUE, 0,
                                m_particles.size() * sizeof(Particle),
                                m_particles.data());
  }

  void readParticlesBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_particlesBuffer, CL_TRUE, 0,
                               m_particles.size() * sizeof(Particle),
                               m_particles.data());
  }

  cl::Buffer& get_dPBuffer() { return m_dPBuffer; }

  void write_dPBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_dPBuffer, CL_TRUE, 0,
                                m_dP.size() * sizeof(numType), m_dP.data());
  }

  void read_dPBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_dPBuffer, CL_TRUE, 0,
                               m_dP.size() * sizeof(numType), m_dP.data());
  }

  cl::Buffer& getInteractedBubbleFalseStateBuffer() {
    return m_interactedBubbleFalseStateBuffer;
  }

  void resetInteractedBubbleFalseState() {
    std::fill(m_interactedBubbleFalseState.begin(),
              m_interactedBubbleFalseState.end(), 0);
  }

  void writeInteractedBubbleFalseStateBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(
        m_interactedBubbleFalseStateBuffer, CL_TRUE, 0,
        m_interactedBubbleFalseState.size() * sizeof(int8_t),
        m_interactedBubbleFalseState.data());
  }

  void readInteractedBubbleFalseStateBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(
        m_interactedBubbleFalseStateBuffer, CL_TRUE, 0,
        m_interactedBubbleFalseState.size() * sizeof(int8_t),
        m_interactedBubbleFalseState.data());
  }

  void resetAndWriteInteractedBubbleFalseState(cl::CommandQueue& cl_queue) {
    resetInteractedBubbleFalseState();
    writeInteractedBubbleFalseStateBuffer(cl_queue);
  }

  cl::Buffer& getPassedBubbleFalseStateBuffer() {
    return m_passedBubbleFalseStateBuffer;
  }

  void resetPassedBubbleFalseState() {
    std::fill(m_passedBubbleFalseState.begin(), m_passedBubbleFalseState.end(),
              0);
  }

  void writePassedBubbleFalseStateBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(
        m_passedBubbleFalseStateBuffer, CL_TRUE, 0,
        m_passedBubbleFalseState.size() * sizeof(int8_t),
        m_passedBubbleFalseState.data());
  }

  void readPassedBubbleFalseStateBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_passedBubbleFalseStateBuffer, CL_TRUE, 0,
                               m_passedBubbleFalseState.size() * sizeof(int8_t),
                               m_passedBubbleFalseState.data());
  }

  void resetAndWritePassedBubbleFalseState(cl::CommandQueue& cl_queue) {
    resetPassedBubbleFalseState();
    writePassedBubbleFalseStateBuffer(cl_queue);
  }

  cl::Buffer& getInteractedBubbleTrueStateBuffer() {
    return m_interactedBubbleTrueStateBuffer;
  }

  void resetInteractedBubbleTrueState() {
    std::fill(m_interactedBubbleTrueState.begin(),
              m_interactedBubbleTrueState.end(), 0);
  }

  void writeInteractedBubbleTrueStateBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(
        m_interactedBubbleTrueStateBuffer, CL_TRUE, 0,
        m_interactedBubbleTrueState.size() * sizeof(int8_t),
        m_interactedBubbleTrueState.data());
  }

  void readInteractedBubbleTrueStateBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(
        m_interactedBubbleTrueStateBuffer, CL_TRUE, 0,
        m_interactedBubbleTrueState.size() * sizeof(int8_t),
        m_interactedBubbleTrueState.data());
  }

  void resetAndWriteInteractedBubbleTrueState(cl::CommandQueue& cl_queue) {
    resetInteractedBubbleTrueState();
    writeInteractedBubbleTrueStateBuffer(cl_queue);
  }

  cl::Buffer& getMassInBuffer() { return m_massInBuffer; };

  void writeMassInBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_massInBuffer, CL_TRUE, 0, sizeof(numType),
                                &m_massIn);
  }

  void readMassInBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_massInBuffer, CL_TRUE, 0, sizeof(numType),
                               &m_massIn);
  }

  cl::Buffer& getMassOutBuffer() { return m_massOutBuffer; }

  void writeMassOutBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_massOutBuffer, CL_TRUE, 0, sizeof(numType),
                                &m_massOut);
  }

  void readMassOutBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_massOutBuffer, CL_TRUE, 0, sizeof(numType),
                               &m_massOut);
  }

  cl::Buffer& getMassDelta2Buffer() { return m_massDelta2Buffer; };

  void writeMassDelta2Buffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_massDelta2Buffer, CL_TRUE, 0, sizeof(numType),
                                &m_massDelta2);
  }

  void readMassDelta2Buffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_massDelta2Buffer, CL_TRUE, 0, sizeof(numType),
                               &m_massDelta2);
  }

  void writeAllBuffersToKernel(cl::CommandQueue cl_queue) {
    writeParticlesBuffer(cl_queue);
    write_dPBuffer(cl_queue);
    writeInteractedBubbleFalseStateBuffer(cl_queue);
    writePassedBubbleFalseStateBuffer(cl_queue);
    writeInteractedBubbleTrueStateBuffer(cl_queue);
    writeMassInBuffer(cl_queue);
    writeMassOutBuffer(cl_queue);
    writeMassDelta2Buffer(cl_queue);
  }

  numType getMassIn() { return m_massIn; }

  numType getMassOut() { return m_massOut; }

  size_t getParticleCountTotal() { return m_particleCountTotal; }

  size_t getParticleCountIn() { return m_particleCountIn; }

  size_t getParticleCountOut() { return m_particleCountOut; }

  size_t getParticleCountTrue() { return m_particleCountTrue; }

  size_t getParticleCountFalse() { return m_particleCountFalse; }

  // Particle functions
  numType getParticleEnergy(u_int i) { return m_particles[i].E; }

  numType getParticleMomentum(u_int i) {
    return std::sqrt(
        std::fma(m_particles[i].p_x, m_particles[i].p_x,
                 std::fma(m_particles[i].p_y, m_particles[i].p_y,
                          m_particles[i].p_z * m_particles[i].p_z)));
  }

  numType getParticleMass(u_int i) { return m_particles[i].m; }

  numType calculateParticleRadius(u_int i);

  numType calculateParticleMomentum(u_int i);

  numType calculateParticleEnergy(u_int i);

  numType calculateParticleRadialVelocity(u_int i);

  numType calculateParticleTangentialVelocity(u_int i);

  // Calculate distributions
  numType calculateNumberDensity(numType t_mass, numType t_temperature,
                                 numType t_dp, numType t_pMax);

  numType calculateEnergyDensity(numType t_mass, numType t_temperature,
                                 numType t_dp, numType t_pMax);

  void add_to_total_initial_energy(numType energy) {
    m_initialTotalEnergy += energy;
  }
  numType getInitialTotalEnergy() { return m_initialTotalEnergy; }

  // Get values from the simulation
  numType countParticleNumberDensity(numType t_radius1);

  numType countParticleNumberDensity(numType t_radius1, numType t_radius2);

  numType countParticleEnergyDensity(numType t_radius1);

  numType countParticleEnergyDensity(numType t_radius1, numType t_radius2);

  numType countParticlesEnergy();

  numType countParticlesEnergy(numType t_radius1);

  numType countParticlesEnergy(numType t_radius1, numType t_radius2);

  void print_info(ConfigReader t_config, PhaseBubble& t_bubble);

  void print_particle_info(unsigned int i) {
    Particle particle = m_particles[i];
    std::cout << "(" << particle.x << ", " << particle.y << ", " << particle.z
              << ") (" << particle.E << ", " << particle.p_x << ", "
              << particle.p_y << ", " << particle.p_z << ")" << std::endl;
  }
};
