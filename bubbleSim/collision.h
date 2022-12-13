#pragma once
#include "base.h"
#include "objects.h"
typedef struct CollisionCell {
  cl_numType v_x;
  cl_numType v_y;
  cl_numType v_z;

  cl_numType x;
  cl_numType y;
  cl_numType z;

  cl_numType theta;

  cl_numType p_E;
  cl_numType p_x;
  cl_numType p_y;
  cl_numType p_z;

  cl_numType v2;
  cl_numType gamma;
  cl_numType total_mass;
  cl_uint particle_count;
} CollisionCell;

class CollisionCellCollection {
 public:
  CollisionCellCollection(numType t_meanFreePath, unsigned int t_cellCount,
                          bool t_doubleCellCount, cl::Context& cl_context);

  std::array<numType, 3>& getShiftVector() { return m_shiftVector; }

  void recalculate_cells(std::vector<Particle>& t_particles,
                         RandomNumberGenerator& t_rng);

  void generateShiftVector(RandomNumberGenerator& t_rng);

  cl::Buffer& getCellBuffer() { return m_collisionCellsBuffer; }
  void writeCollisionCellBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_collisionCellsBuffer, CL_TRUE, 0,
                                m_collisionCells.size() * sizeof(CollisionCell),
                                m_collisionCells.data());
  }

  void readCollisionCellBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_collisionCellsBuffer, CL_TRUE, 0,
                               m_collisionCells.size() * sizeof(CollisionCell),
                               m_collisionCells.data());
  }

  cl::Buffer& getCellLengthBuffer() { return m_cellLengthBuffer; };

  void writeCellLengthBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_cellLengthBuffer, CL_TRUE, 0, sizeof(numType),
                                &m_cellLength);
  };

  void readCellLengthBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_cellLengthBuffer, CL_TRUE, 0, sizeof(numType),
                               &m_cellLength);
  };

  cl::Buffer& getCellCountInOneAxisBuffer() {
    return m_cellCountInOneAxisBuffer;
  };

  void writeCellCountInOneAxisBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_cellCountInOneAxisBuffer, CL_TRUE, 0,
                                sizeof(unsigned int), &m_cellCountInOneAxis);
  };

  void readCellCountInOneAxisBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_cellCountInOneAxisBuffer, CL_TRUE, 0,
                               sizeof(unsigned int), &m_cellCountInOneAxis);
  };

  cl::Buffer& getShiftVectorBuffer() { return m_shiftVectorBuffer; };

  void writeShiftVectorBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_shiftVectorBuffer, CL_TRUE, 0,
                                3 * sizeof(numType), m_shiftVector.data());
  };

  void readShiftVectorBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_shiftVectorBuffer, CL_TRUE, 0,
                               3 * sizeof(numType), m_shiftVector.data());
  };

  cl::Buffer& getCellCountBuffer() { return m_cellCountBuffer; }

  void writeCellCountBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_cellCountBuffer, CL_TRUE, 0,
                                sizeof(unsigned int), &m_cellCount);
  };

  void readCellCountBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_cellCountBuffer, CL_TRUE, 0,
                               sizeof(unsigned int), &m_cellCount);
  };

  cl::Buffer& getStructureRadiusBuffer() { return m_structureRadiusBuffer; }

  void writeStructureRadiusBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueWriteBuffer(m_structureRadiusBuffer, CL_TRUE, 0,
                                sizeof(numType), &m_structureRadius);
  };

  void readStructureRadiusBuffer(cl::CommandQueue& cl_queue) {
    cl_queue.enqueueReadBuffer(m_structureRadiusBuffer, CL_TRUE, 0,
                               sizeof(numType), &m_structureRadius);
  };

  void writeAllBuffersToKernel(cl::CommandQueue& cl_queue) {
    writeCollisionCellBuffer(cl_queue);
    writeCellCountBuffer(cl_queue);
    writeCellCountInOneAxisBuffer(cl_queue);
    writeCellLengthBuffer(cl_queue);
    writeShiftVectorBuffer(cl_queue);
    writeStructureRadiusBuffer(cl_queue);
  }

  std::vector<CollisionCell>& getCollisionCells() { return m_collisionCells; }

 private:
  std::vector<CollisionCell> m_collisionCells;
  cl::Buffer m_collisionCellsBuffer;
  numType m_meanFreePath;

  numType m_cellLength;
  cl::Buffer m_cellLengthBuffer;

  unsigned int m_cellCountInOneAxis;
  cl::Buffer m_cellCountInOneAxisBuffer;

  unsigned int m_cellCount;
  cl::Buffer m_cellCountBuffer;

  std::array<numType, 3> m_shiftVector;
  cl::Buffer m_shiftVectorBuffer;

  numType m_structureRadius;
  cl::Buffer m_structureRadiusBuffer;

  /*
   * Cell counts should be odd to have nice symmetry.
   * (x/2 + 1, y/2 + 1, z/2 + 1) CollisionCell should be centered
   * such that coordinate (0,0,0) is in the center of that cell
   */

  bool m_doubleCellCount;
};