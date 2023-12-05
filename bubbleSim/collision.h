#pragma once
#include "base.h"
#include "objects.h"

class CollisionCellCollection {
 public:
  CollisionCellCollection(){};
  CollisionCellCollection(numType t_meanFreePath,
                          unsigned int t_cellCountInOneAxis,
                          bool t_two_mass_state_on,
                          u_int collision_cell_duplication,
                          double number_density_equilibrium,
                          std::uint64_t& t_buffer_flags,
                          cl::Context& cl_context);

  void recalculate_cells(ParticleCollection& t_particles, numType t_dt,
                         numType t_tau, RandomNumberGeneratorNumType& t_rng);

  u_int recalculate_cells3(ParticleCollection& t_particles);

  void resetShiftVector();

  void generateShiftVector(RandomNumberGeneratorNumType& t_rng);

  void calculate_new_no_collision_probability(double dt, double tau) {
    if (tau > 0) {
      m_no_collision_probability = std::exp(-dt / tau);
    } else if (tau == 0) {
      m_no_collision_probability = 0.;
    } else {
      std::cerr << "tau < 0" << std::endl;
      std::terminate();
    }
    }
   

  void generate_collision_seeds(RandomNumberGeneratorULong& t_rng);

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

  u_int getCellCount() { return m_cell_count; }

  u_int getCellCountInOneAxis() { return m_cellCountInOneAxis; }

  double getNoCollisionProbability() { return m_no_collision_probability; }

  bool getTwoMassStateOn() { return m_two_mass_state_on; }

  uint32_t getCollisionCount() { return m_collision_count; }

  uint32_t getCellDuplication() { return m_cell_duplication; }

  cl::Buffer& getCellLengthBuffer() { return m_cellLengthBuffer; }

  cl::Buffer& getCellCountInOneAxisBuffer() {
    return m_cellCountInOneAxisBuffer;
  }

  cl::Buffer& getShiftVectorBuffer() { return m_shiftVectorBuffer; }

  cl::Buffer& getCellCountBuffer() { return m_cell_count_buffer; }

  cl::Buffer& getNoCollisionProbabilityBuffer() {
    return m_no_collision_probability_buffer;
  }

  cl::Buffer& getCollisionSeedsBuffer() { return m_seeds_uint64_buffer; }

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

  cl::Buffer& getCellDuplicationBuffer() { return m_cell_duplication_buffer; }

  cl::Buffer& getCellParticleCountBuffer() {
    return m_cell_particle_count_buffer;
  }

  cl::Buffer& getTwoMassStateOnBuffer() { return m_two_mass_state_on_buffer; }

  cl::Buffer& getNequilibriumBuffer() { return m_N_equilibrium_buffer; }

  /*
   * ================================================================
   * ================================================================
   *                        Buffer writers
   * ================================================================
   * ================================================================
   */

  void writeCellCountBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_cell_count_buffer, CL_TRUE, 0, sizeof(u_int),
                                &m_cell_count);
  };

  void writeCellCountInOneAxisBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_cellCountInOneAxisBuffer, CL_TRUE, 0,
                                sizeof(uint32_t), &m_cellCountInOneAxis);
  };

  void writeCellLengthBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_cellLengthBuffer, CL_TRUE, 0, sizeof(numType),
                                &m_cellLength);
  };

  void writeShiftVectorBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_shiftVectorBuffer, CL_TRUE, 0,
                                3 * sizeof(numType), m_shiftVector.data());
  };

  void writeCollisionSeedsBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_seeds_uint64_buffer, CL_TRUE, 0,
                                m_cell_count * sizeof(uint64_t),
                                m_seeds_uint64.data());
  }

  void writeNoCollisionProbabilityBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_no_collision_probability_buffer, CL_TRUE, 0,
                                sizeof(double), &m_no_collision_probability);
  }

  void writeCellThetaAxisBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_cell_theta_axis_buffer, CL_TRUE, 0,
                                m_cell_count * sizeof(numType),
                                m_cell_theta_axis.data());
  }

  void writeCellPhiAxisBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_cell_phi_axis_buffer, CL_TRUE, 0,
                                m_cell_count * sizeof(numType),
                                m_cell_phi_axis.data());
  }

  void writeCellThetaRotationBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_cell_theta_rotation_buffer, CL_TRUE, 0,
                                m_cell_count * sizeof(numType),
                                m_cell_theta_rotation.data());
  }

  void writeCellEBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_cell_E_buffer, CL_TRUE, 0,
                                m_cell_count * sizeof(numType),
                                m_cell_E.data());
  }

  void writeCellLogEBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_cell_logE_buffer, CL_TRUE, 0,
                                m_cell_count * sizeof(numType),
                                m_cell_logE.data());
  }

  void writeCellpXBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_cell_pX_buffer, CL_TRUE, 0,
                                m_cell_count * sizeof(numType),
                                m_cell_pX.data());
  }

  void writeCellpYBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_cell_pY_buffer, CL_TRUE, 0,
                                m_cell_count * sizeof(numType),
                                m_cell_pY.data());
  }

  void writeCellpZBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_cell_pZ_buffer, CL_TRUE, 0,
                                m_cell_count * sizeof(numType),
                                m_cell_pZ.data());
  }

  void writeCellCollideBooleanBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_cell_collide_boolean_buffer, CL_TRUE, 0,
                                m_cell_count * sizeof(cl_char),
                                m_cell_collide_boolean.data());
  }

  void writeCellDuplicationBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_cell_duplication_buffer, CL_TRUE, 0,
                                sizeof(uint32_t), &m_cell_duplication);
  }

  void writeCellParticleCountBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_cell_particle_count_buffer, CL_TRUE, 0,
                                m_cell_count * sizeof(uint32_t),
                                m_cell_particle_count.data());
  }

  void writeTwoMassStateOnBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_two_mass_state_on_buffer, CL_TRUE, 0,
                                sizeof(cl_char), &m_two_mass_state_on);
  }

  void writeNEquilibriumBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_N_equilibrium_buffer, CL_TRUE, 0,
                                sizeof(numType), &m_N_equilibrium);
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
                               sizeof(uint32_t), &m_cellCountInOneAxis);
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
    cl_queue.enqueueReadBuffer(m_cell_count_buffer, CL_TRUE, 0,
                               sizeof(unsigned int), &m_cell_count);
  };

  void readCellThetaAxisBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_cell_theta_axis_buffer, CL_TRUE, 0,
                               m_cell_count * sizeof(numType),
                               m_cell_theta_axis.data());
  }

  void readCellPhiAxisBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_cell_phi_axis_buffer, CL_TRUE, 0,
                               m_cell_count * sizeof(numType),
                               m_cell_phi_axis.data());
  }

  void readCellThetaRotationBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_cell_theta_rotation_buffer, CL_TRUE, 0,
                               m_cell_count * sizeof(numType),
                               m_cell_theta_rotation.data());
  }

  void readCellEBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_cell_E_buffer, CL_TRUE, 0,
                               m_cell_count * sizeof(numType), m_cell_E.data());
  }

  void readCellLogEBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_cell_logE_buffer, CL_TRUE, 0,
                               m_cell_count * sizeof(numType),
                               m_cell_logE.data());
  }

  void readCellpXBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_cell_pX_buffer, CL_TRUE, 0,
                               m_cell_count * sizeof(numType),
                               m_cell_pX.data());
  }

  void readCellpYBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_cell_pY_buffer, CL_TRUE, 0,
                               m_cell_count * sizeof(numType),
                               m_cell_pY.data());
  }

  void readCellpZBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_cell_pZ_buffer, CL_TRUE, 0,
                               m_cell_count * sizeof(numType),
                               m_cell_pZ.data());
  }

  void readCellCollideBooleanBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_cell_collide_boolean_buffer, CL_TRUE, 0,
                               m_cell_count * sizeof(cl_char),
                               m_cell_collide_boolean.data());
  }

  void readCellParticleCountBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_cell_particle_count_buffer, CL_TRUE, 0,
                               m_cell_count * sizeof(uint32_t),
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

  uint32_t m_cellCountInOneAxis;
  cl::Buffer m_cellCountInOneAxisBuffer;

  u_int m_cell_count;
  cl::Buffer m_cell_count_buffer;

  std::array<numType, 3> m_shiftVector;
  cl::Buffer m_shiftVectorBuffer;

  std::vector<uint64_t> m_seeds_uint64;
  cl::Buffer m_seeds_uint64_buffer;

  uint32_t m_cell_duplication;
  cl::Buffer m_cell_duplication_buffer;

  double m_no_collision_probability;
  cl::Buffer m_no_collision_probability_buffer;
  uint32_t m_collision_count;
  /*
   * Cell counts should be odd to have nice symmetry.
   * (x/2 + 1, y/2 + 1, z/2 + 1) CollisionCell should be centered
   * such that coordinate (0,0,0) is in the center of that cell
   */

  cl_char m_two_mass_state_on;
  cl::Buffer m_two_mass_state_on_buffer;

  numType m_N_equilibrium;
  cl::Buffer m_N_equilibrium_buffer;
};