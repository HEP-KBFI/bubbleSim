#include "collision.h"

#include <algorithm>

CollisionCellCollection::CollisionCellCollection(
    numType t_cellLength, unsigned int t_cellCountInOneAxis,
    bool t_two_mass_state_on, std::uint64_t& t_buffer_flags,
    cl::Context& cl_context) {
  /*
  TODO:
    Differentiate if particle is inside or outside the bubble. (Different
  collision cells) Change name of t_two_mass_state_on to more accurate name.

  */
  int openCLerrNum;

  if (t_cellCountInOneAxis % 2 != 1) {
    std::cerr << "Cell count in one axis must be odd. (" << t_cellCountInOneAxis
              << ")" << std::endl;
    std::terminate();
  }
  m_collision_count = 0;

  m_two_mass_state_on = t_two_mass_state_on;
  if (t_two_mass_state_on) {
    m_cellCount = 2 * (unsigned int)std::pow(t_cellCountInOneAxis, 3) + 1;
  } else {
    m_cellCount = (unsigned int)std::pow(t_cellCountInOneAxis, 3) + 1;
  }

  // Collision cell buffers
  m_cell_theta_axis.resize(m_cellCount, 0.);
  m_cell_theta_axis_buffer = cl::Buffer(
      cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      m_cellCount * sizeof(numType), m_cell_theta_axis.data(), &openCLerrNum);
  t_buffer_flags |= CELL_THETA_AXIS_BUFFER;
  m_cell_phi_axis.resize(m_cellCount, 0.);
  m_cell_phi_axis_buffer = cl::Buffer(
      cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      m_cellCount * sizeof(numType), m_cell_phi_axis.data(), &openCLerrNum);
  t_buffer_flags |= CELL_PHI_AXIS_BUFFER;
  m_cell_theta_rotation.resize(m_cellCount, 0.);
  m_cell_theta_rotation_buffer =
      cl::Buffer(cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 m_cellCount * sizeof(numType), m_cell_theta_rotation.data(),
                 &openCLerrNum);
  t_buffer_flags |= CELL_THETA_ROTATION_BUFFER;
  m_cell_E.resize(m_cellCount, 0.);
  m_cell_E_buffer =
      cl::Buffer(cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 m_cellCount * sizeof(numType), m_cell_E.data(), &openCLerrNum);
  t_buffer_flags |= CELL_E_BUFFER;
  m_cell_logE.resize(m_cellCount, 0.);
  m_cell_logE_buffer = cl::Buffer(
      cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      m_cellCount * sizeof(numType), m_cell_logE.data(), &openCLerrNum);
  t_buffer_flags |= CELL_LOGE_BUFFER;
  m_cell_pX.resize(m_cellCount, 0.);
  m_cell_pX_buffer = cl::Buffer(
      cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      m_cellCount * sizeof(numType), m_cell_pX.data(), &openCLerrNum);
  t_buffer_flags |= CELL_PX_BUFFER;
  m_cell_pY.resize(m_cellCount, 0.);
  m_cell_pY_buffer = cl::Buffer(
      cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      m_cellCount * sizeof(numType), m_cell_pY.data(), &openCLerrNum);
  t_buffer_flags |= CELL_PY_BUFFER;
  m_cell_pZ.resize(m_cellCount, 0.);
  m_cell_pZ_buffer = cl::Buffer(
      cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      m_cellCount * sizeof(numType), m_cell_pZ.data(), &openCLerrNum);
  t_buffer_flags |= CELL_PZ_BUFFER;
  m_cell_collide_boolean.resize(m_cellCount, (cl_char)0);
  m_cell_collide_boolean_buffer =
      cl::Buffer(cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 m_cellCount * sizeof(cl_char), m_cell_collide_boolean.data(),
                 &openCLerrNum);
  t_buffer_flags |= CELL_COLLIDE_BUFFER;
  m_cell_particle_count.resize(m_cellCount, (uint64_t)0);
  m_cell_particle_count_buffer =
      cl::Buffer(cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 m_cellCount * sizeof(uint64_t), m_cell_particle_count.data(),
                 &openCLerrNum);
  t_buffer_flags |= CELL_PARTICLE_COUNT_BUFFER;


  m_cellCountInOneAxis = t_cellCountInOneAxis;
  m_cellCountInOneAxisBuffer =
      cl::Buffer(cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 sizeof(u_int), &m_cellCountInOneAxis, &openCLerrNum);
  t_buffer_flags |= CELL_COUNT_IN_ONE_AXIS_BUFFER;

  m_cellLength = t_cellLength;
  m_cellLengthBuffer =
      cl::Buffer(cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 sizeof(numType), &m_cellLength, &openCLerrNum);
  t_buffer_flags |= CELL_LENGTH_BUFFER;
  m_shiftVector = {0., 0., 0.};
  m_shiftVectorBuffer =
      cl::Buffer(cl_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                 3 * sizeof(numType), m_shiftVector.data(), &openCLerrNum);
  t_buffer_flags |= CELL_SHIFT_VECTOR_BUFFER;
  m_seed_int64 = (int64_t)0;
  m_seed_int64_buffer =
      cl::Buffer(cl_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                 sizeof(int64_t), &m_seed_int64, &openCLerrNum);
  t_buffer_flags |= CELL_SEED_INT64_BUFFER;
  m_no_collision_probability = 0.;
  m_no_collision_probability_buffer =
      cl::Buffer(cl_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                 sizeof(double), &m_no_collision_probability, &openCLerrNum);
  t_buffer_flags |= CELL_NO_COLLISION_PROBABILITY_BUFFER;
}

void CollisionCellCollection::generateSeed(RandomNumberGeneratorULong& t_rng) {
  m_seed_int64 = t_rng.generate_number();
}

void CollisionCellCollection::generateShiftVector(
    RandomNumberGeneratorNumType& t_rng) {
  m_shiftVector = {
      m_cellLength / (numType)2. - t_rng.generate_number() * m_cellLength,
      m_cellLength / (numType)2. - t_rng.generate_number() * m_cellLength,
      m_cellLength / (numType)2. - t_rng.generate_number() * m_cellLength};
}

void CollisionCellCollection::recalculate_cells(
    ParticleCollection& t_particles, numType t_dt, numType t_tau,
    RandomNumberGeneratorNumType& t_rng) {
  /* Initialize count vector
   * 0 index: Sum pX
   * 1 index: Sum pY
   * 2 index: Sum pZ
   * 3 index: Sum E
   * 4 index: Particle count in a cell
   * 5 index: Prod(E_i) in a cell
   */
  m_collision_count = (uint64_t)0;
  std::vector<std::array<cl_numType, 6>> cell_values(m_cellCount,
                                                     {
                                                         (cl_numType)0.,
                                                         (cl_numType)0.,
                                                         (cl_numType)0.,
                                                         (cl_numType)0.,
                                                         (cl_numType)0.,
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
    if (collision_cell_index >
        t_particles.getParticleCollisionCellIndex().size()) {
      std::cerr << "In recalculate_cell algorithm occured collision cell index "
                   "then maximum number of cells. (Particle id: "
                << i << ", cell id: " << collision_cell_index << ")"
                << std::endl;
      std::exit(0);
    }
    
    
    cell_values[collision_cell_index][0] +=
        t_particles.returnParticlepX(i);
    cell_values[collision_cell_index][1] += t_particles.returnParticlepY(i);
    cell_values[collision_cell_index][2] += t_particles.returnParticlepZ(i);
    cell_values[collision_cell_index][3] += t_particles.returnParticleE(i);
    cell_values[collision_cell_index][4] += 1;
    cell_values[collision_cell_index][5] +=
        std::log(t_particles.returnParticleE(i));
  }
  /*
   * Calculate velocities for each collision cell
   */

  double no_collision_probability_cell = 0.;
  double generated_probability = 0;
  numType no_collision_probability_thermalization = std::exp(-t_dt / t_tau);

  for (size_t i = 1; i < m_cellCount; i++) {
    m_cell_particle_count[i] = (cl_uint)cell_values[i][4];
    /*if (m_collisionCells[i].particle_count < 30) {
      cell_counter[m_collisionCells[i].particle_count] += 1;
    }*/

    if ((cl_uint)cell_values[i][4] < 2) {
      m_cell_collide_boolean[i] = (cl_char)0;
      continue;
    }
    generated_probability = t_rng.generate_number();
    if (generated_probability <= no_collision_probability_thermalization) {
      m_cell_collide_boolean[i] = (cl_char)0;
      continue;
    }

    /*
     * If T ^ N / a*prod_N(E_i) <= RNG then skip rotation
     * T - temperature, prod(E_i) is product of particles' energies in a cell,
     * N is number of particles in a cell
     */

    double fix_constant = 1;
    // This requires that generated probaility < no_collision_probability_cell

    no_collision_probability_cell = std::exp(
        -std::exp(cell_values[i][4] * (std::log(cell_values[i][3]) -
                                       std::log(3 * cell_values[i][4])) -
                  cell_values[i][5] - std::log(fix_constant)));
    /*std::cout << "Particle count: " << cell_values[i][4]
              << ", No collision: " << no_collision_probability_cell
              << ", E: " << cell_values[i][3]
              << ", log(E): " << std::exp(cell_values[i][5]) << std::endl;*/
    // This requires that generated probaility > no_collision_probability_cell
    /*no_collision_probability_cell =
        std::exp(cell_values[i][4] * (std::log(cell_values[i][3]) -
                                      std::log(3 * cell_values[i][4])) -
                 cell_values[i][5] - std::log(1));*/
    generated_probability = t_rng.generate_number();
    if (generated_probability <= no_collision_probability_cell) {
      m_cell_collide_boolean[i] = (cl_char)0;
      continue;
    }

    m_collision_count += m_cell_particle_count[i];
    m_cell_collide_boolean[i] = (cl_char)1;
    m_cell_particle_count[i] = (uint64_t)cell_values[i][4];

    m_cell_pX[i] = cell_values[i][0];
    m_cell_pY[i] = cell_values[i][1];
    m_cell_pZ[i] = cell_values[i][2];
    m_cell_E[i] = cell_values[i][3];

    m_cell_phi_axis[i] =
        std::acos((numType)2. * t_rng.generate_number() - (numType)1.);
    m_cell_theta_axis[i] = (numType)2 * (numType)M_PI * t_rng.generate_number();
    /*
     * Generate rotation axis and angle for CollisionCell
     */
    m_cell_theta_rotation[i] =
        (numType)2. * (numType)M_PI * t_rng.generate_number();
  }
  /*for (u_int count : cell_counter) {
    std::cout << count << ", ";
  }
  std::cout << std::endl;*/
  // exit(0);
}

// void CollisionCellCollection::recalculate_cells2(
//     ParticleCollection& t_particles) {
//   int collision_cell_index;
//
//   for (size_t i = 0; i < t_particles.getParticleCountTotal(); i++) {
//     collision_cell_index = t_particles.returnCollisionCellIndex(i);
//     if (collision_cell_index == 0) {
//       continue;
//     }
//     m_collisionCells[collision_cell_index].pX +=
//         t_particles.returnParticlepX(i);
//     m_collisionCells[collision_cell_index].pY +=
//         t_particles.returnParticlepY(i);
//     m_collisionCells[collision_cell_index].pZ +=
//         t_particles.returnParticlepZ(i);
//     m_collisionCells[collision_cell_index].pE +=
//     t_particles.returnParticleE(i);
//     m_collisionCells[collision_cell_index].particle_count += (cl_uint)1;
//     m_collisionCells[collision_cell_index].total_mass +=
//         std::log(t_particles.returnParticleE(i));
//   }
// }

u_int CollisionCellCollection::recalculate_cells3(
    ParticleCollection& t_particles) {
  auto start = std::chrono::high_resolution_clock::now();

  u_int cell_index = 1;

  ankerl::unordered_dense::map<u_int, std::array<u_int, 2>> cell_map;
  cell_map.reserve(t_particles.getParticleCountTotal());
  for (u_int i = 0; i < t_particles.getParticleCountTotal(); i++) {
    auto it = cell_map.find(t_particles.returnCollisionCellIndex(i));
    if (it == cell_map.end()) {
      cell_map[t_particles.returnCollisionCellIndex(i)] = {cell_index, 1};
      cell_index += 1;
    } else {
      (it->second)[1] += 1;
    }
  }

  u_int cell_index_new = 1;
  for (auto it = cell_map.begin(); it != cell_map.end();) {
    if (it->second[1] < 2) {
      it = cell_map.erase(it);
    } else {
      (it->second)[1] = cell_index_new;
      cell_index_new += 1;
      ++it;
    }
  }

  u_int cell_index_particle;
  for (u_int i = 0; i < t_particles.getParticleCountTotal(); i++) {
    auto it = cell_map.find(t_particles.returnCollisionCellIndex(i));
    if (it == cell_map.end()) {
      t_particles.getParticleCollisionCellIndex()[i] = 0;
    } else {
      cell_index_particle = (it->second)[0];
      t_particles.getParticleCollisionCellIndex()[i] = cell_index_particle;
      m_cell_E[cell_index_particle] += t_particles.returnParticleE(i);
      m_cell_pX[cell_index_particle] += t_particles.returnParticlepX(i);
      m_cell_pY[cell_index_particle] += t_particles.returnParticlepY(i);
      m_cell_pZ[cell_index_particle] += t_particles.returnParticlepZ(i);
      m_cell_logE[cell_index_particle] +=
          std::log(t_particles.returnParticleE(i));
      m_cell_particle_count[cell_index_particle] += 1;
    }
  }
  return cell_index_new;
}