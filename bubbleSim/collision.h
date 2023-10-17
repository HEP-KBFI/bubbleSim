#pragma once
#include "base.h"
#include "objects.h"

class CollisionCellCollection {
 public:
  CollisionCellCollection(){};
  CollisionCellCollection(numType t_meanFreePath, unsigned int t_cellCount,
                          bool t_two_mass_state_on,
                          std::uint64_t& t_buffer_flags,
                          cl::Context& cl_context);

  void recalculate_cells(ParticleCollection& t_particles, numType t_dt,
                         numType t_tau, RandomNumberGeneratorNumType& t_rng);

  void recalculate_cells2(ParticleCollection& t_particles);

  u_int recalculate_cells3(ParticleCollection& t_particles);

  void generateSeed(RandomNumberGeneratorULong& t_rng);

  void resetShiftVector();

  void generateShiftVector(RandomNumberGeneratorNumType& t_rng);

  void calculate_new_no_collision_probability(double dt, double tau) {
    m_no_collision_probability = std::exp(-dt / tau);
  }

  void generate_new_seed(RandomNumberGeneratorULong t_rng) {
    m_seed_int64 = t_rng.generate_number();
  }

  uint32_t returnParticleCountInCell(u_int i) {
    return m_cell_particle_count[i];
  }
  bool returnCellCollideBoolean(u_int i) { return m_cell_collide_boolean[i]; }
  numType returnCellE(u_int i) { return m_cell_E[i]; }
  numType returnCellLogE(u_int i) { return m_cell_logE[i]; }
  numType returnCellpX(u_int i) { return m_cell_pX[i]; }
  numType returnCellpY(u_int i) { return m_cell_pY[i]; }
  numType returnCellpZ(u_int i) { return m_cell_pZ[i]; }

  /*
   * ================================================================
   * ================================================================
   *                            Setters
   * ================================================================
   * ================================================================
   */

  /*
   * ================================================================
   * ================================================================
   *                            Getters
   * ================================================================
   * ================================================================
   */
  std::array<numType, 3>& getShiftVector() { return m_shiftVector; }

  u_int getCellCount() { return m_cellCount; }

  double getNoCollisionProbability() { return m_no_collision_probability; }

  int64_t getSeed() { return m_seed_int64; }

  bool getTwoMassStateOn() { return m_two_mass_state_on; }

  uint32_t getCollisionCount() { return m_collision_count; }

  cl::Buffer& getCellLengthBuffer() { return m_cellLengthBuffer; }

  cl::Buffer& getCellCountInOneAxisBuffer() {
    return m_cellCountInOneAxisBuffer;
  }

  cl::Buffer& getShiftVectorBuffer() { return m_shiftVectorBuffer; }

  cl::Buffer& getCellCountBuffer() { return m_cellCountBuffer; }

  cl::Buffer& getNoCollisionProbabilityBuffer() {
    return m_no_collision_probability_buffer;
  }

  cl::Buffer& getSeedBuffer() { return m_seed_int64_buffer; }

  cl::Buffer& getCellThetaAxisBuffer() { return m_cell_theta_axis_buffer; }

  cl::Buffer& getCellPhiAxisBuffer() { return m_cell_phi_axis_buffer; }

  cl::Buffer& getCellThetaRotationBuffer() {
    return m_cell_theta_rotation_buffer;
  }

  cl::Buffer& getCellEBuffer() { return m_cell_E_buffer; }

  cl::Buffer& getCellLogEBuffer() { return m_cell_logE_buffer; }

  cl::Buffer& getCellpXBuffer() { return m_cell_pX_buffer; }

  cl::Buffer& getCellpYBuffer() { return m_cell_pY_buffer; }

  cl::Buffer& getCellpZBuffer() { return m_cell_pZ_buffer; }

  cl::Buffer& getCellCollideBooleanBuffer() {
    return m_cell_collide_boolean_buffer;
  }

  cl::Buffer& getCellParticleCountBuffer() {
    return m_cell_particle_count_buffer;
  }

  /*
   * ================================================================
   * ================================================================
   *                        Buffer writers
   * ================================================================
   * ================================================================
   */

  void writeCellCountBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_cellCountBuffer, CL_TRUE, 0, sizeof(u_int),
                                &m_cellCount);
  };

  void writeCellCountInOneAxisBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_cellCountInOneAxisBuffer, CL_TRUE, 0,
                                sizeof(u_int), &m_cellCountInOneAxis);
  };

  void writeCellLengthBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_cellLengthBuffer, CL_TRUE, 0, sizeof(numType),
                                &m_cellLength);
  };

  void writeShiftVectorBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_shiftVectorBuffer, CL_TRUE, 0,
                                3 * sizeof(numType), m_shiftVector.data());
  };

  void writeSeedBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_seed_int64_buffer, CL_TRUE, 0,
                                sizeof(int64_t), &m_seed_int64);
  }

  void writeNoCollisionProbabilityBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_no_collision_probability_buffer, CL_TRUE, 0,
                                sizeof(double), &m_no_collision_probability);
  }

  void writeCellThetaAxisBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_cell_theta_axis_buffer, CL_TRUE, 0,
                                m_cellCount * sizeof(numType),
                                m_cell_theta_axis.data());
  }

  void writeCellPhiAxisBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_cell_phi_axis_buffer, CL_TRUE, 0,
                                m_cellCount * sizeof(numType),
                                m_cell_phi_axis.data());
  }

  void writeCellThetaRotationBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_cell_theta_rotation_buffer, CL_TRUE, 0,
                                m_cellCount * sizeof(numType),
                                m_cell_theta_rotation.data());
  }

  void writeCellEBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_cell_E_buffer, CL_TRUE, 0,
                                m_cellCount * sizeof(numType), m_cell_E.data());
  }

  void writeCellLogEBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_cell_logE_buffer, CL_TRUE, 0,
                                m_cellCount * sizeof(numType),
                                m_cell_logE.data());
  }

  void writeCellpXBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_cell_pX_buffer, CL_TRUE, 0,
                                m_cellCount * sizeof(numType),
                                m_cell_pX.data());
  }

  void writeCellpYBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_cell_pY_buffer, CL_TRUE, 0,
                                m_cellCount * sizeof(numType),
                                m_cell_pY.data());
  }

  void writeCellpZBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_cell_pZ_buffer, CL_TRUE, 0,
                                m_cellCount * sizeof(numType),
                                m_cell_pZ.data());
  }

  void writeCellCollideBooleanBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_cell_collide_boolean_buffer, CL_TRUE, 0,
                                m_cellCount * sizeof(cl_char),
                                m_cell_collide_boolean.data());
  }

  void writeCellParticleCountBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_cell_particle_count_buffer, CL_TRUE, 0,
                                m_cellCount * sizeof(uint32_t),
                                m_cell_particle_count.data());
  }
  /////////////////////////////

  void writeCollisionCellRotationBuffers(cl::CommandQueue& cl_queue) {
    writeCollisionCellMomentumBuffers(cl_queue);
    writeCollisionCellAngleBuffers(cl_queue);
    writeCellCollideBooleanBuffer(cl_queue);
  }

  void writeCollisionCellMomentumBuffers(cl::CommandQueue& cl_queue) {
    writeCellEBuffer(cl_queue);
    writeCellpXBuffer(cl_queue);
    writeCellpYBuffer(cl_queue);
    writeCellpZBuffer(cl_queue);
  }

  void writeCollisionCellAngleBuffers(cl::CommandQueue& cl_queue) {
    writeCellThetaAxisBuffer(cl_queue);
    writeCellPhiAxisBuffer(cl_queue);
    writeCellThetaRotationBuffer(cl_queue);
  }

  void writeCollisionCellBuffers(cl::CommandQueue& cl_queue) {
    writeCollisionCellMomentumBuffers(cl_queue);  // 4
    writeCollisionCellAngleBuffers(cl_queue);     // 3
    writeCellLogEBuffer(cl_queue);
    writeCellParticleCountBuffer(cl_queue);
    writeCellCollideBooleanBuffer(cl_queue);
  }

  void writeAllBuffersToKernel(cl::CommandQueue& cl_queue) {
    // writeCollisionCellBuffer(cl_queue);
    writeCollisionCellBuffers(cl_queue);
    writeCellCountInOneAxisBuffer(cl_queue);
    writeCellLengthBuffer(cl_queue);
    writeShiftVectorBuffer(cl_queue);
  }

  /*
   * ================================================================
   * ================================================================
   *                        Buffer readers
   * ================================================================
   * ================================================================
   */

  void readCellCountInOneAxisBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_cellCountInOneAxisBuffer, CL_TRUE, 0,
                               sizeof(u_int), &m_cellCountInOneAxis);
  };

  void readShiftVectorBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_shiftVectorBuffer, CL_TRUE, 0,
                               3 * sizeof(numType), m_shiftVector.data());
  };

  void readCellLengthBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_cellLengthBuffer, CL_TRUE, 0, sizeof(numType),
                               &m_cellLength);
  };

  void readCellCountBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_cellCountBuffer, CL_TRUE, 0,
                               sizeof(unsigned int), &m_cellCount);
  };

  void readCellThetaAxisBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_cell_theta_axis_buffer, CL_TRUE, 0,
                               m_cellCount * sizeof(numType),
                               m_cell_theta_axis.data());
  }

  void readCellPhiAxisBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_cell_phi_axis_buffer, CL_TRUE, 0,
                               m_cellCount * sizeof(numType),
                               m_cell_phi_axis.data());
  }

  void readCellThetaRotationBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_cell_theta_rotation_buffer, CL_TRUE, 0,
                               m_cellCount * sizeof(numType),
                               m_cell_theta_rotation.data());
  }

  void readCellEBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_cell_E_buffer, CL_TRUE, 0,
                               m_cellCount * sizeof(numType), m_cell_E.data());
  }

  void readCellLogEBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_cell_logE_buffer, CL_TRUE, 0,
                               m_cellCount * sizeof(numType),
                               m_cell_logE.data());
  }

  void readCellpXBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_cell_pX_buffer, CL_TRUE, 0,
                               m_cellCount * sizeof(numType), m_cell_pX.data());
  }

  void readCellpYBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_cell_pY_buffer, CL_TRUE, 0,
                               m_cellCount * sizeof(numType), m_cell_pY.data());
  }

  void readCellpZBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_cell_pZ_buffer, CL_TRUE, 0,
                               m_cellCount * sizeof(numType), m_cell_pZ.data());
  }

  void readCellCollideBooleanBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_cell_collide_boolean_buffer, CL_TRUE, 0,
                               m_cellCount * sizeof(cl_char),
                               m_cell_collide_boolean.data());
  }

  void readCellParticleCountBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_cell_particle_count_buffer, CL_TRUE, 0,
                               m_cellCount * sizeof(uint32_t),
                               m_cell_particle_count.data());
  }

  void readCollisionCellMomentumBuffers(cl::CommandQueue& cl_queue) {
    readCellEBuffer(cl_queue);
    readCellpXBuffer(cl_queue);
    readCellpYBuffer(cl_queue);
    readCellpZBuffer(cl_queue);
  }

  void readCollisionCellAngleBuffers(cl::CommandQueue& cl_queue) {
    readCellThetaAxisBuffer(cl_queue);
    readCellPhiAxisBuffer(cl_queue);
    readCellThetaRotationBuffer(cl_queue);
  }

  void readCollisionCellBuffers(cl::CommandQueue& cl_queue) {
    readCollisionCellMomentumBuffers(cl_queue);  // 4
    readCollisionCellAngleBuffers(cl_queue);     // 3
    readCellLogEBuffer(cl_queue);
    readCellParticleCountBuffer(cl_queue);
    readCellCollideBooleanBuffer(cl_queue);
  }

 private:
  std::vector<numType> m_cell_theta_axis;
  cl::Buffer m_cell_theta_axis_buffer;
  std::vector<numType> m_cell_phi_axis;
  cl::Buffer m_cell_phi_axis_buffer;
  std::vector<numType> m_cell_theta_rotation;
  cl::Buffer m_cell_theta_rotation_buffer;
  std::vector<numType> m_cell_E;
  cl::Buffer m_cell_E_buffer;
  std::vector<numType> m_cell_logE;
  cl::Buffer m_cell_logE_buffer;
  std::vector<numType> m_cell_pX;
  cl::Buffer m_cell_pX_buffer;
  std::vector<numType> m_cell_pY;
  cl::Buffer m_cell_pY_buffer;
  std::vector<numType> m_cell_pZ;
  cl::Buffer m_cell_pZ_buffer;
  std::vector<cl_char> m_cell_collide_boolean;
  cl::Buffer m_cell_collide_boolean_buffer;
  std::vector<uint32_t> m_cell_particle_count;
  cl::Buffer m_cell_particle_count_buffer;

  numType m_cellLength;
  cl::Buffer m_cellLengthBuffer;

  u_int m_cellCountInOneAxis;
  cl::Buffer m_cellCountInOneAxisBuffer;

  u_int m_cellCount;
  cl::Buffer m_cellCountBuffer;

  std::array<numType, 3> m_shiftVector;
  cl::Buffer m_shiftVectorBuffer;

  int64_t m_seed_int64;
  cl::Buffer m_seed_int64_buffer;

  double m_no_collision_probability;
  cl::Buffer m_no_collision_probability_buffer;
  uint32_t m_collision_count;
  /*
   * Cell counts should be odd to have nice symmetry.
   * (x/2 + 1, y/2 + 1, z/2 + 1) CollisionCell should be centered
   * such that coordinate (0,0,0) is in the center of that cell
   */

  bool m_two_mass_state_on;
};