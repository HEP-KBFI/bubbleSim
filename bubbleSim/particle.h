#pragma once
#include "base.h"
#include "bubble.h"
#include "cmath"
#include "config_reader.hpp"
#include "random_number.hpp"

// TODO: Flags for all processes of array initialization..
class ParticleCollection {
  // Particle counts total / true vacuum / false vacuum
  size_t m_particleCountTotal, m_particleCountIn, m_particleCountOut,
      m_particleCountTrue, m_particleCountFalse;

  // Initial simulation values
  numType m_initialTotalEnergy;

  // Data objects for buffers
  std::vector<numType> m_particle_X;
  std::vector<numType> m_particle_X_copy;
  cl::Buffer m_particle_X_buffer;

  std::vector<numType> m_particle_Y;
  std::vector<numType> m_particle_Y_copy;
  cl::Buffer m_particle_Y_buffer;

  std::vector<numType> m_particle_Z;
  std::vector<numType> m_particle_Z_copy;
  cl::Buffer m_particle_Z_buffer;

  std::vector<numType> m_particle_pX;
  std::vector<numType> m_particle_pX_copy;
  cl::Buffer m_particle_pX_buffer;

  std::vector<numType> m_particle_pY;
  std::vector<numType> m_particle_pY_copy;
  cl::Buffer m_particle_pY_buffer;

  std::vector<numType> m_particle_pZ;
  std::vector<numType> m_particle_pZ_copy;
  cl::Buffer m_particle_pZ_buffer;

  std::vector<numType> m_particle_E;
  std::vector<numType> m_particle_E_copy;
  cl::Buffer m_particle_E_buffer;

  std::vector<numType> m_particle_M;
  std::vector<numType> m_particle_M_copy;
  cl::Buffer m_particle_M_buffer;

  std::vector<cl_char> m_particle_bool_collide;
  std::vector<cl_char> m_particle_bool_collide_copy;
  cl::Buffer m_particle_bool_collide_buffer;

  std::vector<cl_char> m_particle_bool_in_bubble;
  std::vector<cl_char> m_particle_bool_in_bubble_copy;
  cl::Buffer m_particle_bool_in_bubble_buffer;

  std::vector<cl_uint> m_particle_collision_cell_index;
  std::vector<cl_uint> m_particle_collision_cell_index_copy;
  cl::Buffer m_particle_collision_cell_index_buffer;

  std::vector<numType> m_dP;
  std::vector<numType> m_dP_copy;
  cl::Buffer m_dP_buffer;

  std::vector<int8_t> m_interacted_bubble_false_state;
  std::vector<int8_t> m_interacted_bubble_false_state_copy;
  cl::Buffer m_interacted_bubble_false_state_buffer;

  std::vector<int8_t> m_passed_bubble_false_state;
  std::vector<int8_t> m_passed_bubble_false_state_copy;
  cl::Buffer m_passed_bubble_false_state_buffer;

  std::vector<int8_t> m_interacted_bubble_true_state;
  std::vector<int8_t> m_interacted_bubble_true_state_copy;
  cl::Buffer m_interacted_bubble_true_state_buffer;

 public:
  /*
   * NB!!! When initializing the vectors and buffers it is important
   * that vector sizes stay the same as buffers need reference
   * to vector data. When vector size is changed during the
   * runtime, vector data address also changes and buffer has wrong
   * memory address
   */
  std::vector<numType> m_mean_free_path;

  ParticleCollection(unsigned int t_particleCountTrue,
                     unsigned int t_particleCountFalse,
                     bool t_bubbleIsTrueVacuum, std::uint64_t& t_buffer_flags,
                     cl::Context& cl_context);

  ParticleCollection& operator=(const ParticleCollection& t) { return *this; }

  u_int returnCollisionCellIndex(size_t particle_i) {
    return (u_int)m_particle_collision_cell_index[particle_i];
  }

  numType returnParticleX(size_t particle_i) {
    return m_particle_X[particle_i];
  }

  numType returnParticleY(size_t particle_i) {
    return m_particle_Y[particle_i];
  }

  numType returnParticleZ(size_t particle_i) {
    return m_particle_Z[particle_i];
  }

  numType returnParticlepX(size_t particle_i) {
    return m_particle_pX[particle_i];
  }

  numType returnParticlepY(size_t particle_i) {
    return m_particle_pY[particle_i];
  }

  numType returnParticlepZ(size_t particle_i) {
    return m_particle_pZ[particle_i];
  }

  numType returnParticleE(size_t particle_i) {
    return m_particle_E[particle_i];
  }

  numType returnParticleM(size_t particle_i) {
    return m_particle_M[particle_i];
  }

  numType returnParticledP(size_t particle_i) { return m_dP[particle_i]; }

  void createMomentumCopy() {
    if (m_particle_pX.size() != m_particle_pX_copy.size() ||
        m_particle_pY.size() != m_particle_pY_copy.size() ||
        m_particle_pZ.size() != m_particle_pZ_copy.size()) {
      std::cout << m_particle_pX.size() << ", " << m_particle_pY.size() << ", "
                << m_particle_pZ.size() << std::endl;
      std::cout << m_particle_pX_copy.size() << ", "
                << m_particle_pY_copy.size() << ", "
                << m_particle_pZ_copy.size() << std::endl;
      std::cout << "Momentum vector and it's copy length(s) are not same."
                << std::endl;
      std::exit(0);
    }
    std::copy(m_particle_pX.cbegin(), m_particle_pX.cend(),
              m_particle_pX_copy.begin());
    std::copy(m_particle_pY.cbegin(), m_particle_pY.cend(),
              m_particle_pY_copy.begin());
    std::copy(m_particle_pZ.cbegin(), m_particle_pZ.cend(),
              m_particle_pZ_copy.begin());
  }

  // TODO: Particle data buffer writing, reading
  // TODO - boolean buffers

  /*
   * ================================================================
   * ================================================================
   *                           Getters
   * ================================================================
   * ================================================================
   */
  size_t getParticleCountTotal() { return m_particleCountTotal; }

  size_t getParticleCountIn() { return m_particleCountIn; }

  size_t getParticleCountOut() { return m_particleCountOut; }

  size_t getParticleCountTrue() { return m_particleCountTrue; }

  size_t getParticleCountFalse() { return m_particleCountFalse; }

  // Particle functions

  std::vector<numType>& getParticleX() { return m_particle_X; }
  std::vector<numType>& getParticleY() { return m_particle_Y; }
  std::vector<numType>& getParticleZ() { return m_particle_Z; }
  std::vector<numType>& getParticlepX() { return m_particle_pX; }
  std::vector<numType>& getParticlepY() { return m_particle_pY; }
  std::vector<numType>& getParticlepZ() { return m_particle_pZ; }
  std::vector<numType>& getParticleE() { return m_particle_E; }
  std::vector<numType>& getParticleM() { return m_particle_M; }
  std::vector<cl_uint>& getParticleCollisionCellIndex() {
    return m_particle_collision_cell_index;
  }
  std::vector<int8_t>& getInteractedFalse() {
    return m_interacted_bubble_false_state;
  }

  std::vector<int8_t>& getPassedFalse() { return m_passed_bubble_false_state; }

  std::vector<int8_t>& getInteractedTrue() {
    return m_interacted_bubble_true_state;
  }

  std::vector<numType>& getParticlepXCopy() { return m_particle_pX_copy; }
  std::vector<numType>& getParticlepYCopy() { return m_particle_pY_copy; }
  std::vector<numType>& getParticlepZCopy() { return m_particle_pZ_copy; }

  cl::Buffer& getParticleXBuffer() { return m_particle_X_buffer; }
  cl::Buffer& getParticleYBuffer() { return m_particle_Y_buffer; }
  cl::Buffer& getParticleZBuffer() { return m_particle_Z_buffer; }
  cl::Buffer& getParticlepXBuffer() { return m_particle_pX_buffer; }
  cl::Buffer& getParticlepYBuffer() { return m_particle_pY_buffer; }
  cl::Buffer& getParticlepZBuffer() { return m_particle_pZ_buffer; }
  cl::Buffer& getParticleEBuffer() { return m_particle_E_buffer; }
  cl::Buffer& getParticleMBuffer() { return m_particle_M_buffer; }
  cl::Buffer& getInteractedBubbleFalseStateBuffer() {
    return m_interacted_bubble_false_state_buffer;
  }
  cl::Buffer& getPassedBubbleFalseStateBuffer() {
    return m_passed_bubble_false_state_buffer;
  }
  cl::Buffer& getInteractedBubbleTrueStateBuffer() {
    return m_interacted_bubble_true_state_buffer;
  }

  cl::Buffer& getdPBuffer() { return m_dP_buffer; }
  std::vector<numType>& getdP() { return m_dP; }

  cl::Buffer& getParticleInBubbleBuffer() {
    return m_particle_bool_in_bubble_buffer;
  }
  cl::Buffer& getParticleCollisionCellIndexBuffer() {
    return m_particle_collision_cell_index_buffer;
  }

  /*
   * ================================================================
   * ================================================================
   *                           Buffer writers
   * ================================================================
   * ================================================================
   */

  void writeParticleXBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_particle_X_buffer, CL_TRUE, 0,
                                m_particle_X.size() * sizeof(numType),
                                m_particle_X.data());
  }

  void writeParticleYBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_particle_Y_buffer, CL_TRUE, 0,
                                m_particle_Y.size() * sizeof(numType),
                                m_particle_Y.data());
  }

  void writeParticleZBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_particle_Z_buffer, CL_TRUE, 0,
                                m_particle_Z.size() * sizeof(numType),
                                m_particle_Z.data());
  }

  void writeParticleCoordinatesBuffer(cl::CommandQueue& cl_queue) {
    writeParticleXBuffer(cl_queue);
    writeParticleYBuffer(cl_queue);
    writeParticleZBuffer(cl_queue);
  }

  void writeParticlepXBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_particle_pX_buffer, CL_TRUE, 0,
                                m_particle_pX.size() * sizeof(numType),
                                m_particle_pX.data());
  }

  void writeParticlepYBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_particle_pY_buffer, CL_TRUE, 0,
                                m_particle_pY.size() * sizeof(numType),
                                m_particle_pY.data());
  }

  void writeParticlepZBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_particle_pZ_buffer, CL_TRUE, 0,
                                m_particle_pZ.size() * sizeof(numType),
                                m_particle_pZ.data());
  }

  void writeParticleMomentumsBuffer(cl::CommandQueue& cl_queue) {
    writeParticlepXBuffer(cl_queue);
    writeParticlepYBuffer(cl_queue);
    writeParticlepZBuffer(cl_queue);
  }

  void writeParticleEBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_particle_E_buffer, CL_TRUE, 0,
                                m_particle_E.size() * sizeof(numType),
                                m_particle_E.data());
  }

  void writeParticleMBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_particle_M_buffer, CL_TRUE, 0,
                                m_particle_M.size() * sizeof(numType),
                                m_particle_M.data());
  }

  void writeParticleInBubbleBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(
        m_particle_bool_in_bubble_buffer, CL_TRUE, 0,
        m_particle_bool_in_bubble.size() * sizeof(cl_char),
        m_particle_bool_in_bubble.data());
  }

  void writeParticleBuffer(cl::CommandQueue& cl_queue) {
    writeParticleCoordinatesBuffer(cl_queue);
    writeParticleMomentumsBuffer(cl_queue);
    writeParticleEBuffer(cl_queue);
    writeParticleMBuffer(cl_queue);
  }

  void writeInteractedBubbleFalseStateBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(
        m_interacted_bubble_false_state_buffer, CL_TRUE, 0,
        m_interacted_bubble_false_state.size() * sizeof(int8_t),
        m_interacted_bubble_false_state.data());
  }
  void writePassedBubbleFalseStateBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(
        m_passed_bubble_false_state_buffer, CL_TRUE, 0,
        m_passed_bubble_false_state.size() * sizeof(int8_t),
        m_passed_bubble_false_state.data());
  }
  void writeInteractedBubbleTrueStateBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(
        m_interacted_bubble_true_state_buffer, CL_TRUE, 0,
        m_interacted_bubble_true_state.size() * sizeof(int8_t),
        m_interacted_bubble_true_state.data());
  }

  void writedPBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_dP_buffer, CL_TRUE, 0,
                                m_dP.size() * sizeof(numType), m_dP.data());
  }

  void writeParticleCollisionCellIndexBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(
        m_particle_collision_cell_index_buffer, CL_TRUE, 0,
        m_particle_collision_cell_index.size() * sizeof(cl_uint),
        m_particle_collision_cell_index.data());
  }

  /*
   * ================================================================
   * ================================================================
   *                           Buffer readers
   * ================================================================
   * ================================================================
   */

  void readParticleXBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_particle_X_buffer, CL_TRUE, 0,
                               m_particle_X.size() * sizeof(numType),
                               m_particle_X.data());
  }

  void readParticleYBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_particle_Y_buffer, CL_TRUE, 0,
                               m_particle_Y.size() * sizeof(numType),
                               m_particle_Y.data());
  }

  void readParticleZBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_particle_Z_buffer, CL_TRUE, 0,
                               m_particle_Z.size() * sizeof(numType),
                               m_particle_Z.data());
  }

  void readParticleCoordinatesBuffer(cl::CommandQueue& cl_queue) {
    readParticleXBuffer(cl_queue);
    readParticleYBuffer(cl_queue);
    readParticleZBuffer(cl_queue);
  }

  void readParticlepXBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_particle_pX_buffer, CL_TRUE, 0,
                               m_particle_pX.size() * sizeof(numType),
                               m_particle_pX.data());
  }

  void readParticlepYBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_particle_pY_buffer, CL_TRUE, 0,
                               m_particle_pY.size() * sizeof(numType),
                               m_particle_pY.data());
  }

  void readParticlepZBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_particle_pZ_buffer, CL_TRUE, 0,
                               m_particle_pZ.size() * sizeof(numType),
                               m_particle_pZ.data());
  }

  void readParticleMomentumsBuffer(cl::CommandQueue& cl_queue) {
    readParticlepXBuffer(cl_queue);
    readParticlepYBuffer(cl_queue);
    readParticlepZBuffer(cl_queue);
  }

  void readParticleEBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_particle_E_buffer, CL_TRUE, 0,
                               m_particle_E.size() * sizeof(numType),
                               m_particle_E.data());
  }

  void readParticleMBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_particle_M_buffer, CL_TRUE, 0,
                               m_particle_M.size() * sizeof(numType),
                               m_particle_M.data());
  }

  void readInteractedBubbleFalseStateBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(
        m_interacted_bubble_false_state_buffer, CL_TRUE, 0,
        m_interacted_bubble_false_state.size() * sizeof(int8_t),
        m_interacted_bubble_false_state.data());
  }

  void readInteractedBubbleTrueStateBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(
        m_interacted_bubble_true_state_buffer, CL_TRUE, 0,
        m_interacted_bubble_true_state.size() * sizeof(int8_t),
        m_interacted_bubble_true_state.data());
  }

  void readPassedBubbleFalseStateBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(
        m_passed_bubble_false_state_buffer, CL_TRUE, 0,
        m_passed_bubble_false_state.size() * sizeof(int8_t),
        m_passed_bubble_false_state.data());
  }

  void readdPBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_dP_buffer, CL_TRUE, 0,
                               m_dP.size() * sizeof(numType), m_dP.data());
  }

  void readParticleInBubbelBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(
        m_particle_bool_in_bubble_buffer, CL_TRUE, 0,
        m_particle_bool_in_bubble.size() * sizeof(int8_t),
        m_particle_bool_in_bubble.data());
  }

  void readParticleCollisionCellIndexBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(
        m_particle_collision_cell_index_buffer, CL_TRUE, 0,
        m_particle_collision_cell_index.size() * sizeof(cl_uint),
        m_particle_collision_cell_index.data());
  }

  /*
   * ================================================================
   * ================================================================
   *                           Reset buffers
   * ================================================================
   * ================================================================
   */

  void resetInteractedBubbleFalseState() {
    std::fill(m_interacted_bubble_false_state.begin(),
              m_interacted_bubble_false_state.end(), 0);
  }

  void resetPassedBubbleFalseState() {
    std::fill(m_passed_bubble_false_state.begin(),
              m_passed_bubble_false_state.end(), 0);
  }

  void resetInteractedBubbleTrueState() {
    std::fill(m_interacted_bubble_true_state.begin(),
              m_interacted_bubble_true_state.end(), 0);
  }

  void resetAndWriteInteractedBubbleFalseState(cl::CommandQueue& cl_queue) {
    resetInteractedBubbleFalseState();
    writeInteractedBubbleFalseStateBuffer(cl_queue);
  }

  void resetAndWritePassedBubbleFalseState(cl::CommandQueue& cl_queue) {
    resetPassedBubbleFalseState();
    writePassedBubbleFalseStateBuffer(cl_queue);
  }

  void resetAndWriteInteractedBubbleTrueState(cl::CommandQueue& cl_queue) {
    resetInteractedBubbleTrueState();
    writeInteractedBubbleTrueStateBuffer(cl_queue);
  }

  void writeAllBuffersToKernel(cl::CommandQueue cl_queue) {
    writeParticleBuffer(cl_queue);
    writedPBuffer(cl_queue);
    writeInteractedBubbleFalseStateBuffer(cl_queue);
    writePassedBubbleFalseStateBuffer(cl_queue);
    writeInteractedBubbleTrueStateBuffer(cl_queue);
  }

  /*
   * ================================================================
   * ================================================================
   *                            Other
   * ================================================================
   * ================================================================
   */

  numType getParticleMass(size_t i) { return m_particle_M[i]; }

  numType calculateParticleRadius(size_t i);

  numType calculateParticleMomentum(size_t i);
  numType calculateParticleMomentumCopy(size_t i);

  numType calculateParticleEnergy(size_t i);

  numType calculateParticleRadialVelocity(size_t i);

  numType calculateParticleTangentialVelocity(size_t i);

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

  /*
   * ================================================================
   * ================================================================
   *                           Step revert
   * ================================================================
   * ================================================================
   */

  void revertParticlesToLastStep(cl::CommandQueue& cl_queue) {
    m_particle_X = m_particle_X_copy;
    m_particle_Y = m_particle_Y_copy;
    m_particle_Z = m_particle_Z_copy;
    m_particle_pX = m_particle_pX_copy;
    m_particle_pY = m_particle_pY_copy;
    m_particle_pZ = m_particle_pZ_copy;
    m_particle_E = m_particle_E_copy;
    m_particle_M = m_particle_M_copy;

    writeParticleBuffer(cl_queue);
  }

  void revertdPToLastStep(cl::CommandQueue& cl_queue) {
    m_dP = m_dP_copy;
    writedPBuffer(cl_queue);
  }

  void revertInteractedBubbleFalseStateToLastStep(cl::CommandQueue& cl_queue) {
    m_interacted_bubble_false_state = m_interacted_bubble_false_state_copy;
    writeInteractedBubbleFalseStateBuffer(cl_queue);
  }

  void revertPassedBubbleFalseStateToLastStep(cl::CommandQueue& cl_queue) {
    m_passed_bubble_false_state = m_passed_bubble_false_state_copy;
    writePassedBubbleFalseStateBuffer(cl_queue);
  }

  void revertInteractedBubbleTrueStateToLastStep(cl::CommandQueue& cl_queue) {
    m_interacted_bubble_true_state = m_interacted_bubble_true_state_copy;
    writeInteractedBubbleTrueStateBuffer(cl_queue);
  }

  void revertToLastStep(cl::CommandQueue& cl_queue) {
    revertParticlesToLastStep(cl_queue);
    revertdPToLastStep(cl_queue);
    revertInteractedBubbleFalseStateToLastStep(cl_queue);
    revertInteractedBubbleTrueStateToLastStep(cl_queue);
    revertPassedBubbleFalseStateToLastStep(cl_queue);
  }

  void makeParticlesCopy() {
    m_particle_X_copy = m_particle_X;
    m_particle_Y_copy = m_particle_Y;
    m_particle_Z_copy = m_particle_Z;
    m_particle_pX_copy = m_particle_pX;
    m_particle_pY_copy = m_particle_pY;
    m_particle_pZ_copy = m_particle_pZ;
    m_particle_E_copy = m_particle_E;
    m_particle_M_copy = m_particle_M;
  }

  void make_dPCopy() { m_dP_copy = m_dP; }

  void makeInteractedBubbleFalseStateCopy() {
    m_interacted_bubble_false_state_copy = m_interacted_bubble_false_state;
  }

  void makePassedBubbleFalseStateCopy() {
    m_passed_bubble_false_state_copy = m_passed_bubble_false_state;
  }

  void makeInteractedBubbleTrueStateCopy() {
    m_interacted_bubble_true_state_copy = m_interacted_bubble_true_state;
  }

  void makeCopy() {
    makeParticlesCopy();
    make_dPCopy();
    makeInteractedBubbleFalseStateCopy();
    makeInteractedBubbleTrueStateCopy();
    makePassedBubbleFalseStateCopy();
  }

  /*
   * ================================================================
   * ================================================================
   *                           Print info
   * ================================================================
   * ================================================================
   */

  void print_particle_info(unsigned int i) {
    std::cout << "(" << m_particle_X[i] << ", " << m_particle_Y[i] << ", "
              << m_particle_Z[i] << ") (" << m_particle_E[i] << ", "
              << m_particle_pX[i] << ", " << m_particle_pY[i] << ", "
              << m_particle_pZ[i] << ")" << std::endl;
  }
};

class ParticleGenerator {
 public:
  ParticleGenerator() {}
  // Constructur to generate particles according to Boltzmann distribution
  ParticleGenerator(numType t_mass);

  std::array<std::vector<numType>, 2> m_cumulativeProbabilityFunction;

  bool checkCPDInitialization() { return m_CPD_initialized; }

  numType generateNParticlesInCube(numType t_sideHalf, u_int t_N,
                                   RandomNumberGeneratorNumType& t_generator,
                                   ParticleCollection& t_particles);

  numType generateNParticlesInCube(numType t_radiusIn, numType t_sideHalf,
                                   u_int t_N,
                                   RandomNumberGeneratorNumType& t_generator,
                                   ParticleCollection& t_particles);

  numType generateNParticlesInCube(numType t_xSideHalf, numType t_ySideHalf,
                                   numType t_zSideHalf, u_int t_N,
                                   RandomNumberGeneratorNumType& t_generator,
                                   ParticleCollection& t_particles);

  numType generateNParticlesInCube(numType t_radiusIn, numType t_xSideHalf,
                                   numType t_ySideHalf, numType t_zSideHalf,
                                   u_int t_N,
                                   RandomNumberGeneratorNumType& t_generator,
                                   ParticleCollection& t_particles);

  numType generateNParticlesInSphere(numType t_radiusMax, u_int t_N,
                                     RandomNumberGeneratorNumType& t_generator,
                                     ParticleCollection& t_particles);

  numType generateNParticlesInSphere(numType t_radiusMin, numType t_radiusMax,
                                     u_int t_N,
                                     RandomNumberGeneratorNumType& t_generator,
                                     ParticleCollection& t_particles);
  void calculateCPDBoltzmann(numType t_temperature, numType t_pMax,
                             numType t_dp);
  void calculateCPDDelta(numType t_momentum);

  void calculateCPDBeta(numType t_leftShift, numType t_pMax, numType t_alpha,
                        numType t_beta, numType t_dp);

  void generateRandomDirection(numType& x, numType& y, numType& z,
                               numType t_radius,
                               RandomNumberGeneratorNumType& t_generator);

  

 private:
  bool m_CPD_initialized = false;
  numType m_mass = 0;
  // Index [0] = Probability, Index [1] = Momentum value

  numType interp(numType t_xValue, std::vector<numType>& t_xArray,
                 std::vector<numType>& t_yArray);

  void generateParticleMomentum(numType& p_x, numType& p_y, numType& p_z,
                                numType& t_pResult,
                                RandomNumberGeneratorNumType& t_generator);

  void generatePointInCube(numType& x, numType& y, numType& z,
                           numType& t_SideHalf,
                           RandomNumberGeneratorNumType& t_generator);

  void generatePointInSphere(numType& x, numType& y, numType& z,
                             numType t_maxRadius,
                             RandomNumberGeneratorNumType& t_generator);

  void generatePointInCube(numType& x, numType& y, numType& z,
                           numType& t_xSideHalf, numType& t_ySideHalf,
                           numType& t_zSideHalf,
                           RandomNumberGeneratorNumType& t_generator);
};
