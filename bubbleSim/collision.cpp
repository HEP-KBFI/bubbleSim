#include "collision.h"

CollisionCellCollection::CollisionCellCollection(
    numType t_meanFreePath, unsigned int t_cellCountInOneAxis,
    bool t_doubleCellCount, cl::Context& cl_context) {
  /*
  TODO:
    Differentiate if particle is inside or outside the bubble. (Different
  collision cells) Change name of t_doubleCellCount to more accurate name.

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
                 sizeof(u_int), &m_cellCount, &openCLerrNum);

  m_doubleCellCount = t_doubleCellCount;
  m_meanFreePath = t_meanFreePath;

  m_cellCountInOneAxis = t_cellCountInOneAxis;
  m_cellCountInOneAxisBuffer =
      cl::Buffer(cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 sizeof(u_int), &m_cellCountInOneAxis, &openCLerrNum);

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
  m_shiftVector = {
      m_cellLength / (numType)2. - t_rng.generate_number() * m_cellLength,
      m_cellLength / (numType)2. - t_rng.generate_number() * m_cellLength,
      m_cellLength / (numType)2. - t_rng.generate_number() * m_cellLength};
}

void CollisionCellCollection::recalculate_cells(
    ParticleCollection& t_particles, numType t_dt, numType t_tau, RandomNumberGenerator& t_rng) {
  numType phi, theta;

  /* Initialize count vector
   * 0 index: Sum pX
   * 1 index: Sum pY
   * 2 index: Sum pZ
   * 3 index: Sum E
   * 4 index: Particle count in a cell
   * 5 index: Prod(E_i) in a cell
   */
  std::vector<std::array<cl_numType, 7>> cell_values(m_cellCount,
                                                     {
                                                         (cl_numType)0.,
                                                         (cl_numType)0.,
                                                         (cl_numType)0.,
                                                         (cl_numType)0.,
                                                         (cl_numType)0.,
                                                         (cl_numType)1.,
                                                         (cl_numType)0.,
                                                     });
  // std::array<u_int, 30> cell_counter = {0};
  /*
   * Sum up all momentums, energies and masses for each cell
   */
  int collision_cell_index;
  // m_particle_collision_cell_index[gid]

  for (size_t i = 0; i < t_particles.getParticleCountTotal(); i++) {
    collision_cell_index = t_particles.returnCollisionCellIndex(i);

    if (collision_cell_index == 0) {
      continue;
    }
    cell_values[collision_cell_index][0] += t_particles.returnParticlepX(i);
    cell_values[collision_cell_index][1] += t_particles.returnParticlepY(i);
    cell_values[collision_cell_index][2] += t_particles.returnParticlepZ(i);
    cell_values[collision_cell_index][3] += t_particles.returnParticleE(i);
    cell_values[collision_cell_index][4] += 1;
    cell_values[collision_cell_index][5] *= t_particles.returnParticleE(i);
    cell_values[collision_cell_index][6] +=
        std::log(t_particles.returnParticleE(i));
  }

  /*
   * Calculate velocities for each collision cell
   */

  double no_collision_probability = 0.;
  double generated_probability = 0;
  numType no_collision_probabilit_thermalization = std::exp(-t_dt / t_tau);

  for (size_t i = 1; i < m_cellCount; i++) {
    m_collisionCells[i].particle_count = (cl_uint)cell_values[i][4];
    /*if (m_collisionCells[i].particle_count < 30) {
      cell_counter[m_collisionCells[i].particle_count] += 1;
    }*/

    if ((cl_uint)cell_values[i][4] < 2) {
      m_collisionCells[i].b_collide = 0;
      continue;
    }
    generated_probability = t_rng.generate_number();
    if (generated_probability <= no_collision_probabilit_thermalization) {
      m_collisionCells[i].b_collide = 0;
      continue;
    }

    /*
     * If T ^ N / prod_N(E_i) <= RNG then skip rotation
     * T - temperature, prod(E_i) is product of particles' energies in a cell,
     * N is number of particles in a cell
     */

    no_collision_probability = std::exp(
        -std::exp(cell_values[i][4] *
                      std::log(cell_values[i][3] / (3 * cell_values[i][4])) -
                  cell_values[i][6]));

    generated_probability = t_rng.generate_number();
    if (generated_probability <= no_collision_probability) {
      m_collisionCells[i].b_collide = 0;
      continue;
    }
    m_collisionCells[i].b_collide = 1;
    m_collisionCells[i].particle_count = (int)cell_values[i][4];
    m_collisionCells[i].vX = cell_values[i][0] / cell_values[i][3];
    m_collisionCells[i].vY = cell_values[i][1] / cell_values[i][3];
    m_collisionCells[i].vZ = cell_values[i][2] / cell_values[i][3];
    m_collisionCells[i].total_mass = std::sqrt(
        std::pow(cell_values[i][3], 2) - std::pow(cell_values[i][0], 2) -
        std::pow(cell_values[i][1], 2) - std::pow(cell_values[i][2], 2));
    m_collisionCells[i].pX = cell_values[i][0] / cell_values[i][4];
    m_collisionCells[i].pY = cell_values[i][1] / cell_values[i][4];
    m_collisionCells[i].pZ = cell_values[i][2] / cell_values[i][4];
    m_collisionCells[i].pE = cell_values[i][3] / cell_values[i][4];

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

  /*for (u_int count : cell_counter) {
    std::cout << count << ", ";
  }
  std::cout << std::endl;*/
  // exit(0);
}