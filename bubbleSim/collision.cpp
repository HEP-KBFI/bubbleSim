#include "collision.h"

CollisionCellCollection::CollisionCellCollection(
    numType t_meanFreePath, unsigned int t_cellCountInOneAxis,
    bool t_doubleCellCount, cl::Context& cl_context) {
  /*
  TODO:
    Differentiate if particle is inside or outside the bubble. (Different collision cells)
    Change name of t_doubleCellCount to more accurate name.

  */
  int openCLerrNum;

  if (t_cellCountInOneAxis % 2 != 1) {
    std::cerr << "Cell count in one axis must be odd. (" << t_cellCountInOneAxis
              << ")" << std::endl;
    std::terminate();
  }

  if (t_doubleCellCount) {
    m_cellCount = 2 * (unsigned int)std::pow(t_cellCountInOneAxis, 3) + 1;
  } else {
    m_cellCount = (unsigned int)std::pow(t_cellCountInOneAxis, 3) + 1;
  }
  m_cellCountBuffer =
      cl::Buffer(cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 sizeof(unsigned int), &m_cellCount, &openCLerrNum);

  m_doubleCellCount = t_doubleCellCount;
  m_meanFreePath = t_meanFreePath;

  m_cellCountInOneAxis = t_cellCountInOneAxis;
  m_cellCountInOneAxisBuffer =
      cl::Buffer(cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 sizeof(unsigned int), &m_cellCountInOneAxis, &openCLerrNum);

  m_collisionCells.resize(m_cellCount);
  m_collisionCellsBuffer =
      cl::Buffer(cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 m_cellCount * sizeof(CollisionCell), m_collisionCells.data(),
                 &openCLerrNum);

  m_cellLength = t_meanFreePath;
  m_cellLengthBuffer =
      cl::Buffer(cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 sizeof(numType), &m_cellLength, &openCLerrNum);

  m_structureRadius = m_cellLength * m_cellCountInOneAxis / 2;
  m_structureRadiusBuffer =
      cl::Buffer(cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 sizeof(numType), &m_structureRadius, &openCLerrNum);

  m_shiftVector = {0., 0., 0.};
  m_shiftVectorBuffer =
      cl::Buffer(cl_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                 3 * sizeof(numType), m_shiftVector.data(), &openCLerrNum);
}

void CollisionCellCollection::generateShiftVector(
    RandomNumberGenerator& t_rng) {
  m_shiftVector = {m_cellLength / (numType)2. - t_rng.generate_number() * m_cellLength,
      m_cellLength / (numType)2. - t_rng.generate_number() * m_cellLength,
      m_cellLength / (numType)2. - t_rng.generate_number() * m_cellLength};
}

void CollisionCellCollection::recalculate_cells(
    std::vector<Particle>& t_particles, RandomNumberGenerator& t_rng) {
  // 0 index cell is for particles outside the collision cell grid

  numType phi, theta;

  std::vector<std::array<cl_numType, 6>> frames(m_cellCount, {(cl_numType)0.,(cl_numType)0.,(cl_numType)0.,(cl_numType)0.,(cl_numType)0.,(cl_numType)1.});

  /*
   * Sum up all momentums, energies and masses for each cell
   */
  for (Particle& particle : t_particles) {
    if (particle.idxCollisionCell == 0) continue;
    frames[particle.idxCollisionCell][0] += particle.pX;
    frames[particle.idxCollisionCell][1] += particle.pY;
    frames[particle.idxCollisionCell][2] += particle.pZ;
    frames[particle.idxCollisionCell][3] += particle.E;
    frames[particle.idxCollisionCell][4] += 1;
    frames[particle.idxCollisionCell][5] *= particle.E;
  }
  /*
   * Calculate velocities for each collision cell
   */
  for (size_t i = 1; i < m_cellCount; i++) {
    m_collisionCells[i].particle_count = (int)frames[i][4];
    if (frames[i][4] != 2) {
      m_collisionCells[i].particle_count = (int)0;
      continue;
    }
    //if (frames[i][4] > 1) {
    else{
    /* 
      * If T ^ N / (9 * prod(E_i)) <= RNG then skip rotation
      * T - temperature, prod(E_i) is product of particles' energies in a cell, N is number of particles in a cell
      */ 
      if (t_rng.generate_number() >=
          std::pow(0.01, 1) / (3 * std::sqrt(frames[i][5]))) {
        m_collisionCells[i].particle_count = (int)0;
        continue;
      }
      m_collisionCells[i].vX = frames[i][0] / frames[i][3];
      m_collisionCells[i].vY = frames[i][1] / frames[i][3];
      m_collisionCells[i].vZ = frames[i][2] / frames[i][3];
      m_collisionCells[i].total_mass =
          std::sqrt(std::pow(frames[i][3], 2) - std::pow(frames[i][0], 2) -
                    std::pow(frames[i][1], 2) - std::pow(frames[i][2], 2));
      m_collisionCells[i].pX = frames[i][0] / frames[i][4];
      m_collisionCells[i].pY = frames[i][1] / frames[i][4];
      m_collisionCells[i].pZ = frames[i][2] / frames[i][4];
      m_collisionCells[i].pE = frames[i][3] / frames[i][4];

      /*
       * beta = sqrt(v_x^2 + v_y^2 + v_z^2);
       */
      m_collisionCells[i].v2 =
          std::fma(m_collisionCells[i].vX, m_collisionCells[i].vX,
                   std::fma(m_collisionCells[i].vY, m_collisionCells[i].vY,
                            m_collisionCells[i].vZ * m_collisionCells[i].vZ));
      /*
       * gamma = 1/sqrt(1-beta^2)
       */
      m_collisionCells[i].gamma = 1 / std::sqrt(1 - m_collisionCells[i].v2);
      phi = std::acos((numType)1. - (numType)2. * t_rng.generate_number());
      theta = (numType)2 * (numType)M_PI * t_rng.generate_number();
      /*
       * Generate rotation axis and angle for CollisionCell
       */
      m_collisionCells[i].theta =
          (numType)2. * (numType)M_PI * t_rng.generate_number();
      m_collisionCells[i].x = std::sin(phi) * std::cos(theta);
      m_collisionCells[i].y = std::sin(phi) * std::sin(theta);
      m_collisionCells[i].z = std::cos(phi);
    }
  }
}