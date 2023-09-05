#pragma once
#include "base.h"
#include "objects.h"

typedef struct CollisionCell {
  // Lorentz transformation
  cl_numType vX;
  cl_numType vY;
  cl_numType vZ;

  // Rotation matrix
  cl_numType x;
  cl_numType y;
  cl_numType z;
  cl_numType theta;

  // Average 4-momentum values
  cl_numType pE;
  cl_numType pX;
  cl_numType pY;
  cl_numType pZ;

  cl_char b_collide;
  cl_numType total_mass;
  cl_uint particle_count;
} CollisionCell;

class CollisionCellCollection {
 public:
  CollisionCellCollection(numType t_meanFreePath, unsigned int t_cellCount,
                          bool t_doubleCellCount, cl::Context& cl_context);

  void recalculate_cells(ParticleCollection& t_particles, numType t_dt,
                         numType t_tau, RandomNumberGeneratorNumType& t_rng);

  void recalculate_cells2(ParticleCollection& t_particles);

  u_int recalculate_cells3(ParticleCollection& t_particles);

  void generateSeed(RandomNumberGeneratorULong& t_rng);

  void generateShiftVector(RandomNumberGeneratorNumType& t_rng);

  void calculate_new_no_collision_probability(double dt, double tau) {
    m_no_collision_probability = std::exp(-dt / tau);
  }

  void generate_new_seed(RandomNumberGeneratorULong t_rng) {
    m_seed_int64 = t_rng.generate_number();
  }

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

  std::vector<CollisionCell>& getCollisionCells() { return m_collisionCells; }

  cl::Buffer& getCellBuffer() { return m_collisionCellsBuffer; }

  cl::Buffer& getCellLengthBuffer() { return m_cellLengthBuffer; }

  cl::Buffer& getCellCountInOneAxisBuffer() {
    return m_cellCountInOneAxisBuffer;
  }

  cl::Buffer& getShiftVectorBuffer() { return m_shiftVectorBuffer; }

  cl::Buffer& getCellCountBuffer() { return m_cellCountBuffer; }

  cl::Buffer& getStructureRadiusBuffer() { return m_structureRadiusBuffer; }

  cl::Buffer& getNoCollisionProbabilityBuffer() {
    return m_no_collision_probability_buffer;
  }

  cl::Buffer& getSeedBuffer() { return m_seed_int64_buffer; }
  /*
   * ================================================================
   * ================================================================
   *                        Buffer writers
   * ================================================================
   * ================================================================
   */

  void writeCollisionCellBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_collisionCellsBuffer, CL_TRUE, 0,
                                m_collisionCells.size() * sizeof(CollisionCell),
                                m_collisionCells.data());
  }

  void writeCollisionCellBuffer(cl::CommandQueue& cl_queue, size_t t_N, size_t t_offsetN=0) {
    assert(t_offsetN + t_N <= m_collisionCells.size());
    cl_queue.enqueueWriteBuffer(
        m_collisionCellsBuffer, CL_TRUE, t_offsetN * sizeof(CollisionCell),
                                t_N * sizeof(CollisionCell),
                                m_collisionCells.data());
  }

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

  void writeStructureRadiusBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_structureRadiusBuffer, CL_TRUE, 0,
                                sizeof(numType), &m_structureRadius);
  };

  void writeSeedBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_seed_int64_buffer, CL_TRUE, 0,
                                sizeof(int64_t), &m_seed_int64);
  }

  void writeNoCollisionProbabilityBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_no_collision_probability_buffer, CL_TRUE, 0,
                                sizeof(double), &m_no_collision_probability);
  }

  void writeAllBuffersToKernel(cl::CommandQueue& cl_queue) {
    writeCollisionCellBuffer(cl_queue);
    writeCellCountBuffer(cl_queue);
    writeCellCountInOneAxisBuffer(cl_queue);
    writeCellLengthBuffer(cl_queue);
    writeShiftVectorBuffer(cl_queue);
    writeStructureRadiusBuffer(cl_queue);
  }

  /*
   * ================================================================
   * ================================================================
   *                        Buffer readers
   * ================================================================
   * ================================================================
   */

  void readCollisionCellBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_collisionCellsBuffer, CL_TRUE, 0,
                               m_collisionCells.size() * sizeof(CollisionCell),
                               m_collisionCells.data());
  }

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

  void readStructureRadiusBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_structureRadiusBuffer, CL_TRUE, 0,
                               sizeof(numType), &m_structureRadius);
  };

 private:
  std::vector<CollisionCell> m_collisionCells;
  cl::Buffer m_collisionCellsBuffer;
  numType m_meanFreePath;

  numType m_cellLength;
  cl::Buffer m_cellLengthBuffer;

  u_int m_cellCountInOneAxis;
  cl::Buffer m_cellCountInOneAxisBuffer;

  u_int m_cellCount;
  cl::Buffer m_cellCountBuffer;

  std::array<numType, 3> m_shiftVector;
  cl::Buffer m_shiftVectorBuffer;

  numType m_structureRadius;
  cl::Buffer m_structureRadiusBuffer;

  int64_t m_seed_int64;
  cl::Buffer m_seed_int64_buffer;

  double m_no_collision_probability;
  cl::Buffer m_no_collision_probability_buffer;

  /*
   * Cell counts should be odd to have nice symmetry.
   * (x/2 + 1, y/2 + 1, z/2 + 1) CollisionCell should be centered
   * such that coordinate (0,0,0) is in the center of that cell
   */

  bool m_doubleCellCount;
};