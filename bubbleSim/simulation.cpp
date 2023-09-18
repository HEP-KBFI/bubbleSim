#include "simulation.h"

#ifdef TIME_COLLIDE_DEBUG
template <typename Duration>

std::chrono::microseconds clock_micros(Duration const& duration) {
  return std::chrono::duration_cast<std::chrono::microseconds>(duration);
}

void print_timer_info(const char* text, long long time) {
  std::cout << text << time << std::endl;
}
#endif

Simulation::Simulation(int t_seed, numType t_max_dt,
                       SimulationParameters& t_simulation_parameters,
                       cl::Context& cl_context) {
  m_seed = t_seed;
  m_dt = t_max_dt;
  m_dt_current = t_max_dt;
  m_dt_max = t_max_dt;
  m_parameters = t_simulation_parameters;
  m_dP = 0.;
  }

/*
 * ================================================================
 * ================================================================
 *                        Simulation step
 * ================================================================
 * ================================================================
 */

// Step with bubble kernel
/*
 * Simple bubble step algorithm with bubble
 *
 * // TODO: Energy summation on the GPU
 * // TODO: dP summation on the GPU
 * // TODO: Simulation revert option
 */
void Simulation::stepParticleBubble(ParticleCollection& particles,
                                    PhaseBubble& bubble,
                                    OpenCLLoader& t_kernels) {
  numType dE;
  // m_dt_current = m_parameters.getTimestepAdapter().getTimestep();
  m_parameters.writeDtBuffer(t_kernels.getCommandQueue());
  t_kernels.getCommandQueue().enqueueNDRangeKernel(
      t_kernels.m_particleStepWithBubbleKernel.getKernel(), cl::NullRange,
                                cl::NDRange(particles.getParticleCountTotal()));
  particles.readParticleEBuffer(t_kernels.getCommandQueue());
  particles.readdPBuffer(t_kernels.getCommandQueue());

  m_dP = 0.;
  numType currentTotalEnergy = 0.;
  for (size_t i = 0; i < particles.getParticleCountTotal(); i++) {
    m_dP += particles.returnParticledP(i);
    currentTotalEnergy += particles.returnParticleE(i);
  }
  dE = -m_dP * bubble.getSpeed();
  /*if (m_dP != 0) {
    std::cout << "dP: " << m_dP << std::endl;

    PhaseBubble testBubble(55.456826, 0.23236236, 2.83371e-05, 0.000345447);
    std::cout << "Energy change (bubble): " << -m_dP * testBubble.getSpeed()
              << std::endl;
    testBubble.evolveWall(0.975251, m_dP / testBubble.calculateArea());
    std::cout << testBubble.getRadius() << ", "
              << testBubble.getSpeed() * 0.975251
              << std::endl;


    exit(0);
  }*/
  
  m_dP = m_dP / bubble.calculateArea();
  bubble.evolveWall(m_dt_current, m_dP);
  currentTotalEnergy += bubble.calculateEnergy();
  // bubble.evolveWall2(m_dt_current, dE);
  // currentTotalEnergy += bubble.getEnergy();

  // std::cout << "= = = = = = = = = =" << std::endl;
  // std::cout << "Eb: " << bubble.calculateEnergy() << ", dt: " << m_dt_current
  // << std::endl; std::cout << "Velocity: "
  //           << std::sqrt(1 - std::pow(4 * M_PI * bubble.getSigma() *
  //                             std::pow(bubble.getRadius(), 2.) /
  //                             (bubble.calculateEnergy() +
  //                              4 * M_PI / 3 *
  //                              std::pow(bubble.getRadius(), 3.) *
  //                                  bubble.getdV()), 2.))
  //           << std::endl;
  // std::cout << "dP: " << m_dP << std::endl;
  // std::cout << "dV - dP: " << bubble.getdV() - m_dP / m_dt_current
  //           << std::endl;
  // std::cout << "V_b (before): " << bubble.getSpeed() << std::endl;
  // std::cout << "V_b (after): " << bubble.getSpeed() << std::endl;
  // std::cout << "= = = = = = = = = =" << std::endl;

  bubble.writeBubbleBuffer(t_kernels.getCommandQueue());

  // currentTotalEnergy += bubble.calculateEnergy();

  /* In development: adaptive timestep ; step revert
  numType bubbleStepFinalSpeed = bubble.getSpeed();
  numType bubbleSpeedChange = std::abs(bubbleStepFinalSpeed - bubbleStartSpeed);
  if ((bubbleStartSpeed > 0) && (bubbleStartSpeed - bubbleStepFinalSpeed > 0)) {
    if (bubbleSpeedChange > 0.00005) {
      particles.revertToLastStep(cl_queue);
      bubble.revertBubbleToLastStep(cl_queue);
      m_timestepAdapter.calculateNewTimeStep();
      step(particles, bubble, t_bubbleInteractionKernel, cl_queue);
    }
  } else if (bubbleSpeedChange > 0.005) {
    particles.revertToLastStep(cl_queue);
    bubble.revertBubbleToLastStep(cl_queue);
    m_timestepAdapter.calculateNewTimeStep();
    step(particles, bubble, t_bubbleInteractionKernel, cl_queue);
  } else if (std::abs(bubbleStepFinalSpeed) > 1) {
    particles.revertToLastStep(cl_queue);
    bubble.revertBubbleToLastStep(cl_queue);
    m_timestepAdapter.calculateNewTimeStep();
    step(particles, bubble, t_bubbleInteractionKernel, cl_queue);
  } else {
    m_timestepAdapter.calculateNewTimeStep(bubble);
  }
  */

  m_totalEnergy = currentTotalEnergy;
  m_time += m_dt_current;
  m_step_count += 1;
  // In development: step revert
  // particles.makeCopy();
  // bubble.makeBubbleCopy();
}

// Step with bubble kernel and boundary condition
void Simulation::stepParticleBubbleBoundary(
    ParticleCollection& particles, PhaseBubble& bubble,
                                            OpenCLLoader& t_kernels) {
  Simulation::stepParticleBubble(particles, bubble, t_kernels);

  t_kernels.getCommandQueue().enqueueNDRangeKernel(
      t_kernels.m_particleBoundaryKernel.getKernel(), cl::NullRange,
                                cl::NDRange(particles.getParticleCountTotal()));
}

// Step for only calculating bubble
void Simulation::step(PhaseBubble& bubble, numType t_dP) {
  m_time += m_dt;
  bubble.evolveWall(m_dt, t_dP);
}

void Simulation::collide(ParticleCollection& particles,
                         CollisionCellCollection& cells, numType t_dt,
                         numType t_tau, RandomNumberGeneratorNumType& t_rng,
                         OpenCLLoader& t_kernels) {
#ifdef TIME_COLLIDE_DEBUG
  auto start = my_clock::now();
#endif
  cells.generateShiftVector(t_rng);
  cells.writeShiftVectorBuffer(t_kernels.getCommandQueue());
#ifdef TIME_COLLIDE_DEBUG
  auto stop = my_clock::now();
  std::cout << "= Shift vector generation and write time: "
            << clock_micros(stop - start).count() << std::endl;
  // Assign particles to collision cells
  start = my_clock::now();
#endif
  t_kernels.getCommandQueue().enqueueNDRangeKernel(
      t_kernels.m_cellAssignmentKernel.getKernel(),
                                cl::NullRange,
                                cl::NDRange(particles.getParticleCountTotal()));
#ifdef TIME_COLLIDE_DEBUG
  stop = my_clock::now();
  std::cout << "= Collision cell assignation time: "
            << clock_micros(stop - start).count() << std::endl;
  // Update particle data on CPU
  start = my_clock::now();
#endif
  particles.readParticleCoordinatesBuffer(t_kernels.getCommandQueue());
  particles.readParticleMomentumsBuffer(t_kernels.getCommandQueue());
  particles.readParticleEBuffer(t_kernels.getCommandQueue());
  particles.readParticleCollisionCellIndexBuffer(t_kernels.getCommandQueue());
  // Calculate COM and genrate rotation matrix for each cell
#ifdef TIME_COLLIDE_DEBUG
  stop = my_clock::now();

  std::cout << "= Read particle data to CPU time: "
            << clock_micros(stop - start).count() << std::endl;
  start = my_clock::now();
#endif
  cells.recalculate_cells(particles, t_dt, t_tau, t_rng);
#ifdef TIME_COLLIDE_DEBUG
  stop = my_clock::now();
  std::cout << "= Recalculate cell time: " << clock_micros(stop - start).count()
            << std::endl;

  start = my_clock::now();
#endif
  particles.writeParticleCollisionCellIndexBuffer(t_kernels.getCommandQueue());
  // cells.writeCollisionCellBuffer(cl_queue);
  cells.writeCollisionCellRotationBuffers(t_kernels.getCommandQueue());
#ifdef TIME_COLLIDE_DEBUG
  stop = my_clock::now();
  std::cout << "= Write data to GPU time: "
            << clock_micros(stop - start).count() << std::endl;
  // Rotate momentum
  start = my_clock::now();
#endif
  t_kernels.getCommandQueue().enqueueNDRangeKernel(
      t_kernels.m_rotationKernel.getKernel(), cl::NullRange,
                                cl::NDRange(particles.getParticleCountTotal()));
#ifdef TIME_COLLIDE_DEBUG
  stop = my_clock::now();
  std::cout << "= Rotation time: " << clock_micros(stop - start).count()
            << std::endl;
#endif
}

// void Simulation::collide2(
//     ParticleCollection& particles, CollisionCellCollection& cells, numType
//     t_dt, numType t_tau, RandomNumberGeneratorNumType& t_rng_numtype,
//     RandomNumberGeneratorULong& t_rng_int,
//     cl::Kernel& t_assign_particle_to_collision_cell_kernel,
//     cl::Kernel& t_rotate_momentum_kernel,
//     cl::Kernel& t_reset_collision_cell_kernel,
//     cl::Kernel& t_generate_collision_cell_kernel, cl::CommandQueue& cl_queue)
//     {
//   cells.generateShiftVector(t_rng_numtype);
//   cells.writeShiftVectorBuffer(cl_queue);
//   cl_queue.enqueueNDRangeKernel(t_assign_particle_to_collision_cell_kernel,
//                                 cl::NullRange,
//                                 cl::NDRange(particles.getParticleCountTotal()));
//   particles.readParticleCollisionCellIndexBuffer(cl_queue);
//
//   cells.calculate_new_no_collision_probability(t_dt, t_tau);
//   cells.writeNoCollisionProbabilityBuffer(cl_queue);
//   while (cells.getSeed() == 0) {
//     cells.generateSeed(t_rng_int);
//   }
//   cells.writeSeedBuffer(cl_queue);
//
//   cl_queue.enqueueNDRangeKernel(t_reset_collision_cell_kernel, cl::NullRange,
//                                 cl::NDRange(cells.getCellCount()));
//   cells.readCollisionCellBuffer(cl_queue);
//   auto start = my_clock::now();
//
//   cells.recalculate_cells2(particles);
//   auto stop = my_clock::now();
//   auto duration =
//       std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
//   std::cout << "= Recalcualte cell time: " << clock_micros(stop -
//   start).count()
//             << std::endl;
//
//   cells.writeCollisionCellBuffer(cl_queue);
//
//   /*cl_queue.enqueueNDRangeKernel(t_sum_collision_cell_kernel, cl::NullRange,
//                                 cl::NDRange(particles.getParticleCountTotal()));*/
//   cl_queue.enqueueNDRangeKernel(t_generate_collision_cell_kernel,
//   cl::NullRange,
//                                 cl::NDRange(cells.getCellCount()));
//   cl_queue.enqueueNDRangeKernel(t_rotate_momentum_kernel, cl::NullRange,
//                                 cl::NDRange(particles.getParticleCountTotal()));
// }

void Simulation::collide3(
    ParticleCollection& particles, CollisionCellCollection& cells, numType t_dt,
    numType t_tau, RandomNumberGeneratorNumType& t_rng_numtype,
    RandomNumberGeneratorULong& t_rng_int,
                          OpenCLLoader& t_kernels) {
#ifdef TIME_COLLIDE_DEBUG
  auto start = my_clock::now();
  auto stop = my_clock::now();
#endif
  u_int cell_count;
  // Generate shift vector
  cells.generateShiftVector(t_rng_numtype);
  cells.writeShiftVectorBuffer(t_kernels.getCommandQueue());
#ifdef TIME_COLLIDE_DEBUG
  start = my_clock::now();
#endif
  t_kernels.getCommandQueue().enqueueNDRangeKernel(
      t_kernels.m_cellAssignmentKernel.getKernel(),
                                cl::NullRange,
                                cl::NDRange(particles.getParticleCountTotal()));
  particles.readParticleCollisionCellIndexBuffer(t_kernels.getCommandQueue());
#ifdef TIME_COLLIDE_DEBUG
  stop = my_clock::now();
  std::cout << "= Cell assignment to particle and read: "
            << clock_micros(stop - start).count() << std::endl;
  // Generate probabilty that collision does not happen
  start = my_clock::now();
#endif
  cells.calculate_new_no_collision_probability(t_dt, t_tau);
  cells.writeNoCollisionProbabilityBuffer(t_kernels.getCommandQueue());
#ifdef TIME_COLLIDE_DEBUG
  stop = my_clock::now();
  std::cout << "= No collision probability and write: "
            << clock_micros(stop - start).count() << std::endl;
  // Generate new seed (seed!=0 as algo does not work in that case)
  start = my_clock::now();
#endif
  while (cells.getSeed() == 0) {
    cells.generateSeed(t_rng_int);
  }
  cells.writeSeedBuffer(t_kernels.getCommandQueue());
#ifdef TIME_COLLIDE_DEBUG
  stop = my_clock::now();

  std::cout << "= Seed and write: " << clock_micros(stop - start).count()
            << std::endl;
  start = my_clock::now();
#endif
  // Reset collision cells
  t_kernels.getCommandQueue().enqueueNDRangeKernel(
      t_kernels.m_collisionCellResetKernel.getKernel(), cl::NullRange,
                                cl::NDRange(cells.getCellCount()));
#ifdef TIME_COLLIDE_DEBUG
  stop = my_clock::now();

  std::cout << "= Collision cell reset: " << clock_micros(stop - start).count()
            << std::endl;
  start = my_clock::now();
#endif
  cells.readCollisionCellBuffers(t_kernels.getCommandQueue());
#ifdef TIME_COLLIDE_DEBUG
  auto start2 = my_clock::now();
#endif
  cell_count = cells.recalculate_cells3(particles);
#ifdef TIME_COLLIDE_DEBUG
  auto stop2 = my_clock::now();
  std::cout << "== Recalculate collision cells: "
            << clock_micros(stop2 - start2).count() << std::endl;
#endif
  // Possible addition. Only write cell_count of values.
  cells.writeCollisionCellBuffers(t_kernels.getCommandQueue());
#ifdef TIME_COLLIDE_DEBUG
  stop = my_clock::now();
  std::cout << "= Recalculate collision cells, read and write: "
            << clock_micros(stop - start).count() << std::endl;
  start = my_clock::now();
#endif
  t_kernels.getCommandQueue().enqueueNDRangeKernel(
      t_kernels.m_collisionCellCalculateGenerationKernel.getKernel(),
      cl::NullRange,
                                cl::NDRange(cells.getCellCount()));
#ifdef TIME_COLLIDE_DEBUG
  stop = my_clock::now();

  std::cout << "= Generate collision cells: "
            << clock_micros(stop - start).count() << std::endl;
  start = my_clock::now();
#endif
  t_kernels.getCommandQueue().enqueueNDRangeKernel(
      t_kernels.m_rotationKernel.getKernel(), cl::NullRange,
                                cl::NDRange(particles.getParticleCountTotal()));
#ifdef TIME_COLLIDE_DEBUG
  stop = my_clock::now();
  std::cout << "= Rotate particles' momentum: "
            << clock_micros(stop - start).count() << std::endl;
#endif
}

void Simulation::stepParticleCollisionBoundary(
    ParticleCollection& particles, CollisionCellCollection& cells,
    RandomNumberGeneratorNumType& t_rng_numtype,
    RandomNumberGeneratorULong& t_rng_int,
    OpenCLLoader& t_kernels) {
  // m_timestepAdapter.calculateNewTimeStep(bubble);
  m_dt_current = m_parameters.getTimestepAdapter().getTimestep();
  std::cout << m_dt_current << std::endl;
  m_parameters.writeDtBuffer(t_kernels.getCommandQueue());
#ifdef TIME_COLLIDE_DEBUG
  auto start = my_clock::now();
#endif
  t_kernels.getCommandQueue().enqueueNDRangeKernel(
      t_kernels.m_particleLinearStepKernel.getKernel(), cl::NullRange,
                                cl::NDRange(particles.getParticleCountTotal()));
#ifdef TIME_COLLIDE_DEBUG
  auto stop = my_clock::now();
  print_timer_info("Step time: ", clock_micros(stop - start).count());
  start = my_clock::now();
#endif
  t_kernels.getCommandQueue().enqueueNDRangeKernel(
      t_kernels.m_particleBoundaryKernel.getKernel(), cl::NullRange,
                                cl::NDRange(particles.getParticleCountTotal()));
#ifdef TIME_COLLIDE_DEBUG
  stop = my_clock::now();
  print_timer_info("Boundary check time: ", clock_micros(stop - start).count());

  start = my_clock::now();
#endif
  collide(particles, cells, m_dt_current, getTau(), t_rng_numtype,
          t_kernels);
  /*collide2(particles, cells, m_dt_current, getTau(), t_rng_numtype, t_rng_int,
           t_assign_particle_to_collision_cell_kernel, t_rotate_momentum_kernel,
           t_reset_collision_cell_kernel, t_generate_collision_cell_kernel,
           cl_queue);*/
  /*collide3(particles, cells, m_dt_current, getTau(), t_rng_numtype, t_rng_int,
           t_assign_particle_to_collision_cell_kernel, t_rotate_momentum_kernel,
           t_reset_collision_cell_kernel, t_generate_collision_cell_kernel,
           cl_queue);*/
#ifdef TIME_COLLIDE_DEBUG
  stop = my_clock::now();
  print_timer_info("Collision time: ", clock_micros(stop - start).count());

  start = my_clock::now();
#endif
  particles.readParticleEBuffer(t_kernels.getCommandQueue());
#ifdef TIME_COLLIDE_DEBUG
  stop = my_clock::now();
  print_timer_info("Particle Energy read time: ",
                   clock_micros(stop - start).count());

  start = my_clock::now();
#endif
  numType currentTotalEnergy = 0.;
  for (size_t i = 0; i < particles.getParticleCountTotal(); i++) {
    currentTotalEnergy += particles.returnParticleE(i);
  }
#ifdef TIME_COLLIDE_DEBUG
  stop = my_clock::now();
  print_timer_info("Particle Energy count time: ",
                   clock_micros(stop - start).count());

  std::cout << std::endl;
#endif
  m_totalEnergy = currentTotalEnergy;
  m_time += m_dt_current;
  m_step_count += 1;
}

void Simulation::stepParticleBubbleCollisionBoundary(
    ParticleCollection& particles, PhaseBubble& bubble,
    CollisionCellCollection& cells, RandomNumberGeneratorNumType& t_rng_numtype,
    RandomNumberGeneratorULong& t_rng_int, OpenCLLoader& t_kernels) {
  numType dE;
  // m_timestepAdapter.calculateNewTimeStep(bubble);
  m_dt_current = m_parameters.getTimestepAdapter().getTimestep();
  m_parameters.writeDtBuffer(t_kernels.getCommandQueue());

  t_kernels.getCommandQueue().enqueueNDRangeKernel(
      t_kernels.m_particleStepWithBubbleKernel.getKernel(), cl::NullRange,
                                cl::NDRange(particles.getParticleCountTotal()));
  t_kernels.getCommandQueue().enqueueNDRangeKernel(
      t_kernels.m_particleBoundaryKernel.getKernel(), cl::NullRange,
                                cl::NDRange(particles.getParticleCountTotal()));
#ifdef TIME_COLLIDE_DEBUG
  auto start = my_clock::now();
#endif
  collide(particles, cells, m_dt_current, getTau(), t_rng_numtype,
          t_kernels);

  particles.readParticleEBuffer(t_kernels.getCommandQueue());
  particles.readdPBuffer(t_kernels.getCommandQueue());
#ifdef TIME_COLLIDE_DEBUG
  auto stop = my_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << clock_micros(stop - start).count() << std::endl;
#endif
  m_dP = 0.;
  numType currentTotalEnergy = 0.;
  for (size_t i = 0; i < particles.getParticleCountTotal(); i++) {
    m_dP += particles.returnParticledP(i);
    currentTotalEnergy += particles.returnParticleE(i);
  }
  // Calculate energy change for bubble (m_dP is ~ energy change for particles)
  dE = -m_dP * bubble.getSpeed();
  m_dP = m_dP / bubble.calculateArea();

  bubble.evolveWall(m_dt_current, m_dP);
  currentTotalEnergy += bubble.calculateEnergy();

  // bubble.evolveWall2(m_dt_current, dE);
  // currentTotalEnergy += bubble.getEnergy();

  bubble.writeBubbleBuffer(t_kernels.getCommandQueue());

  m_totalEnergy = currentTotalEnergy;
  m_time += m_dt_current;
  m_step_count += 1;
}