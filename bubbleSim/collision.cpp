#include "collision.h"

#include <algorithm>

CollisionCellCollection::CollisionCellCollection(
    numType t_cellLength, unsigned int t_cellCountInOneAxis,
    bool t_two_mass_state_on, u_int t_collision_cell_duplication, double number_density_equilibrium,
    std::uint64_t& t_buffer_flags, cl::Context& cl_context) {
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
  m_cell_duplication = (uint32_t)t_collision_cell_duplication;
  m_cell_duplication_buffer =
      cl::Buffer(cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 sizeof(uint32_t), &m_cell_duplication, &openCLerrNum);

  m_two_mass_state_on = (cl_char)t_two_mass_state_on;
  m_two_mass_state_on_buffer =
      cl::Buffer(cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 sizeof(cl_char), &m_two_mass_state_on, &openCLerrNum);

  if (t_two_mass_state_on) {
    // CollisionHack
    m_cell_count = m_cell_duplication * 2 *
                       (unsigned int)std::pow(t_cellCountInOneAxis, 3) +
                   1;
    // m_cell_count = 2 * (unsigned int)std::pow(t_cellCountInOneAxis, 3) + 1;
  } else {
    // CollisionHack
    m_cell_count =
        m_cell_duplication * (unsigned int)std::pow(t_cellCountInOneAxis, 3) +
        1;
    // m_cell_count = (unsigned int)std::pow(t_cellCountInOneAxis, 3) + 1;
  }

  // Collision cell buffers
  m_cell_theta_axis.resize(m_cell_count, 0.);
  m_cell_theta_axis_buffer = cl::Buffer(
      cl_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      m_cell_count * sizeof(numType), m_cell_theta_axis.data(), &openCLerrNum);
  t_buffer_flags |= CELL_THETA_AXIS_BUFFER;
  m_cell_phi_axis.resize(m_cell_count, 0.);
  m_cell_phi_axis_buffer = cl::Buffer(
      cl_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      m_cell_count * sizeof(numType), m_cell_phi_axis.data(), &openCLerrNum);
  t_buffer_flags |= CELL_PHI_AXIS_BUFFER;
  m_cell_theta_rotation.resize(m_cell_count, 0.);
  m_cell_theta_rotation_buffer =
      cl::Buffer(cl_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                 m_cell_count * sizeof(numType), m_cell_theta_rotation.data(),
                 &openCLerrNum);
  t_buffer_flags |= CELL_THETA_ROTATION_BUFFER;
  m_cell_E.resize(m_cell_count, 0.);
  m_cell_E_buffer = cl::Buffer(
      cl_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      m_cell_count * sizeof(numType), m_cell_E.data(), &openCLerrNum);
  t_buffer_flags |= CELL_E_BUFFER;
  m_cell_logE.resize(m_cell_count, 0.);
  m_cell_logE_buffer = cl::Buffer(
      cl_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      m_cell_count * sizeof(numType), m_cell_logE.data(), &openCLerrNum);
  t_buffer_flags |= CELL_LOGE_BUFFER;
  m_cell_pX.resize(m_cell_count, 0.);
  m_cell_pX_buffer = cl::Buffer(
      cl_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      m_cell_count * sizeof(numType), m_cell_pX.data(), &openCLerrNum);
  t_buffer_flags |= CELL_PX_BUFFER;
  m_cell_pY.resize(m_cell_count, 0.);
  m_cell_pY_buffer = cl::Buffer(
      cl_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      m_cell_count * sizeof(numType), m_cell_pY.data(), &openCLerrNum);
  t_buffer_flags |= CELL_PY_BUFFER;
  m_cell_pZ.resize(m_cell_count, 0.);
  m_cell_pZ_buffer = cl::Buffer(
      cl_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      m_cell_count * sizeof(numType), m_cell_pZ.data(), &openCLerrNum);
  t_buffer_flags |= CELL_PZ_BUFFER;
  m_cell_collide_boolean.resize(m_cell_count, (cl_char)0);
  m_cell_collide_boolean_buffer =
      cl::Buffer(cl_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                 m_cell_count * sizeof(cl_char), m_cell_collide_boolean.data(),
                 &openCLerrNum);
  t_buffer_flags |= CELL_COLLIDE_BUFFER;
  m_cell_particle_count.resize(m_cell_count, (uint32_t)0);
  m_cell_particle_count_buffer =
      cl::Buffer(cl_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                 m_cell_count * sizeof(uint32_t), m_cell_particle_count.data(),
                 &openCLerrNum);
  t_buffer_flags |= CELL_PARTICLE_COUNT_BUFFER;

  m_cellCountInOneAxis = t_cellCountInOneAxis;
  m_cellCountInOneAxisBuffer =
      cl::Buffer(cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 sizeof(uint32_t), &m_cellCountInOneAxis, &openCLerrNum);
  t_buffer_flags |= CELL_COUNT_IN_ONE_AXIS_BUFFER;

  m_cellLength = t_cellLength;
  m_cellLengthBuffer =
      cl::Buffer(cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 sizeof(numType), &m_cellLength, &openCLerrNum);
  t_buffer_flags |= CELL_LENGTH_BUFFER;
  m_shiftVector = {0., 0., 0.};
  m_shiftVectorBuffer =
      cl::Buffer(cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 3 * sizeof(numType), m_shiftVector.data(), &openCLerrNum);
  t_buffer_flags |= CELL_SHIFT_VECTOR_BUFFER;

  m_seeds_uint64.reserve(m_cell_count);
  m_seeds_uint64_buffer = cl::Buffer(
      cl_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      m_cell_count * sizeof(uint64_t), m_seeds_uint64.data(), &openCLerrNum);

  t_buffer_flags |= CELL_SEED_INT64_BUFFER;
  m_no_collision_probability = 0.;
  m_no_collision_probability_buffer =
      cl::Buffer(cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 sizeof(double), &m_no_collision_probability, &openCLerrNum);
  t_buffer_flags |= CELL_NO_COLLISION_PROBABILITY_BUFFER;


  m_N_equilibrium = number_density_equilibrium * std::pow(t_cellLength, 3.);
  std::cout << "N equilibrium: "<<  m_N_equilibrium << std::endl;
  m_N_equilibrium_buffer =
      cl::Buffer(cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 sizeof(numType), &t_cellLength, &openCLerrNum);

}

void CollisionCellCollection::generate_collision_seeds(
    RandomNumberGeneratorULong& t_rng) {
  if (m_seeds_uint64.size() > 0) {
    std::cerr << "Sice of seed vector is already generated (size > 0)"
              << std::endl;
    std::exit(0);
  }
  uint64_t seed;
  for (u_int i = 0; i < m_cell_count; i++) {
    do {
      seed = t_rng.generate_number();
    } while (seed == 0);
    m_seeds_uint64.push_back(seed);
  }
}

void CollisionCellCollection::resetShiftVector() {
  m_shiftVector = {0., 0., 0.};
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
  m_collision_count = (uint32_t)0;
  std::vector<std::array<cl_numType, 6>> cell_values(m_cell_count,
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

    cell_values[collision_cell_index][0] += t_particles.returnParticlepX(i);
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
  double collision_probability_cell = 0.;
  double no_collision_probability_cell = 0.;
  double generated_probability = 0;
  numType no_collision_probability_thermalization;
  if (t_tau > 0) {
    no_collision_probability_thermalization = std::exp(-t_dt / t_tau);
  }
  else if (t_tau == 0) { no_collision_probability_thermalization = 0;
  }
  else {
    std::cerr << "tau < 0" << std::endl;
    std::terminate();
  }
  
  for (size_t i = 1; i < m_cell_count; i++) {
    m_cell_particle_count[i] = (cl_uint)cell_values[i][4];
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

    // This requires that generated probaility < no_collision_probability_cell

    no_collision_probability_cell =
        std::exp(-std::exp(cell_values[i][4] * (std::log(cell_values[i][3]) -
                                                std::log(cell_values[i][4])) -
                           cell_values[i][5]));

    collision_probability_cell =
        std::exp(cell_values[i][4] * (std::log(cell_values[i][3]) -
                                      std::log(cell_values[i][4])) -
                 cell_values[i][5]) *
        cell_values[i][4] / 30.;

    generated_probability = t_rng.generate_number();

    // if (generated_probability <= no_collision_probability_cell) {
    if (generated_probability >= collision_probability_cell) {
      m_cell_collide_boolean[i] = (cl_char)0;
      continue;
    }

    m_collision_count += m_cell_particle_count[i];
    m_cell_collide_boolean[i] = (cl_char)1;
    m_cell_particle_count[i] = (uint32_t)cell_values[i][4];

    m_cell_pX[i] = cell_values[i][0];
    m_cell_pY[i] = cell_values[i][1];
    m_cell_pZ[i] = cell_values[i][2];
    m_cell_E[i] = cell_values[i][3];
    // m_cell_logE[i] = cell_values[i][5];
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
