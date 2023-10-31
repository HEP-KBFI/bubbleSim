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

// Utils

void Simulation::calculate_average_particle_count_in_filled_cells(
    ParticleCollection& t_particles, CollisionCellCollection t_cells,
    OpenCLLoader& t_kernels) {
  t_cells.resetShiftVector();
  t_cells.writeShiftVectorBuffer(t_kernels.getCommandQueue());
  if (t_cells.getTwoMassStateOn()) {
    t_kernels.getCommandQueue().enqueueNDRangeKernel(
        t_kernels.m_particleInBubbleKernel.getKernel(), cl::NullRange,
        cl::NDRange(t_particles.getParticleCountTotal()));
    t_kernels.getCommandQueue().enqueueNDRangeKernel(
        t_kernels.m_cellAssignmentKernelTwoMassState.getKernel(), cl::NullRange,
        cl::NDRange(t_particles.getParticleCountTotal()));
  } else {
    t_kernels.getCommandQueue().enqueueNDRangeKernel(
        t_kernels.m_cellAssignmentKernel.getKernel(), cl::NullRange,
        cl::NDRange(t_particles.getParticleCountTotal()));
  }
  t_particles.readParticleCollisionCellIndexBuffer(t_kernels.getCommandQueue());
  std::vector<cl_uint> vect2(t_particles.getParticleCollisionCellIndex());

  std::sort(vect2.begin(), vect2.end());
  u_int particle_count = 0;
  int res = 0;
  u_int test_count = 0;
  for (int i = 0; i < vect2.size(); i++) {
    if (vect2[i] != 0) {
      test_count += 1;
    }
  }

  for (int i = 0; i < vect2.size(); i++) {
    while (i < vect2.size() - 1 && vect2[i] == vect2[i + 1]) {
      i++;
      if (vect2[i] != 0) {
        particle_count += 1;
      }
    }
    if (vect2[i] != 0) {
      particle_count += 1;
    }
    if (vect2[i] == 0) {
      continue;
    }
    res++;
  }
  std::cout << "Unique cell count: " << res << std::endl;
  std::cout << "Particle count: " << particle_count << std::endl;
  std::cout << "Average particle count in filled cells: "
            << (double)particle_count / res << std::endl;
}

void Simulation::count_collision_cells(CollisionCellCollection& cells,
                                       OpenCLLoader& t_kernels) {
  cells.readCellCollideBooleanBuffer(t_kernels.getCommandQueue());
  cells.readCellParticleCountBuffer(t_kernels.getCommandQueue());

  m_active_colliding_particle_count = (uint32_t)0;
  m_active_collision_cell_count = (uint32_t)0;
  for (u_int i = 0; i < cells.getCellCount(); i++) {
    m_active_collision_cell_count +=
        (uint32_t)cells.returnCellCollideBoolean(i);
    m_active_colliding_particle_count +=
        (uint32_t)(cells.returnParticleCountInCell(i) *
                   cells.returnCellCollideBoolean(i));
  }
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

  numType initial_particle_energy = 0.;
  for (size_t i = 0; i < particles.getParticleCountTotal(); i++) {
    initial_particle_energy += particles.returnParticleE(i);
  }
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
  m_dP = m_dP / bubble.calculateArea();
  numType initial_bubble_energy = bubble.calculateEnergy();
  /*if (m_dP != 0) {
    std::cout << "dP: " << m_dP
              << ", dP/dV: " << m_dP / (m_dt_current * bubble.getdV())
              << ", dP/dV: "
              << dE /
                     (bubble.calculateArea() *
  bubble.getSpeed()*m_dt_current*bubble.getdV())
              << std::endl;
  }*/
  bubble.evolveWall(m_dt_current, m_dP);
  /*if (m_dP != 0) {
    std::cout << "dE: " << dE << ", Bubble energy change: "
              << bubble.calculateEnergy() - initial_bubble_energy
              << ", Particle energy change: "
              << currentTotalEnergy - initial_particle_energy << std::endl;
  }*/
  currentTotalEnergy += bubble.calculateEnergy();
  // bubble.evolveWall2(m_dt_current, dE);
  // currentTotalEnergy += bubble.getEnergy();

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
void Simulation::stepParticleBubbleBoundary(ParticleCollection& particles,
                                            PhaseBubble& bubble,
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
  if (cells.getTwoMassStateOn()) {
    t_kernels.getCommandQueue().enqueueNDRangeKernel(
        t_kernels.m_particleInBubbleKernel.getKernel(), cl::NullRange,
        cl::NDRange(particles.getParticleCountTotal()));
    t_kernels.getCommandQueue().enqueueNDRangeKernel(
        t_kernels.m_cellAssignmentKernelTwoMassState.getKernel(), cl::NullRange,
        cl::NDRange(particles.getParticleCountTotal()));
  } else {
    t_kernels.getCommandQueue().enqueueNDRangeKernel(
        t_kernels.m_cellAssignmentKernel.getKernel(), cl::NullRange,
        cl::NDRange(particles.getParticleCountTotal()));
  }
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

void Simulation::collide2(ParticleCollection& particles,
                          CollisionCellCollection& cells, numType t_dt,
                          numType t_tau,
                          RandomNumberGeneratorNumType& t_rng_numtype,
                          RandomNumberGeneratorULong& t_rng_int,
                          OpenCLLoader& t_kernels) {
#ifdef DEBUG_OPENCL_KERNEL_RUNTIME_PROFILE
  cl::Event event;

  cl_ulong time_end, time_start, time_total;
#endif

  cells.generateShiftVector(t_rng_numtype);
  cells.writeShiftVectorBuffer(t_kernels.getCommandQueue());

  // CollisionHack
#ifdef DEBUG_OPENCL_KERNEL_RUNTIME_PROFILE
  t_kernels.getCommandQueue().enqueueNDRangeKernel(
      t_kernels.m_collisionCellResetKernel.getKernel(), cl::NullRange,
      cl::NDRange(cells.getCellCount()), cl::NullRange, NULL, &event);
#endif
#ifndef DEBUG_OPENCL_KERNEL_RUNTIME_PROFILE
  t_kernels.getCommandQueue().enqueueNDRangeKernel(
      t_kernels.m_collisionCellResetKernel.getKernel(), cl::NullRange,
      cl::NDRange(cells.getCellCount()));
#endif

#ifdef DEBUG_OPENCL_KERNEL_RUNTIME_PROFILE
  t_kernels.getCommandQueue().finish();
  time_start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
  time_end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
  time_total = (time_end - time_start) / 1000;
  std::cout << "= Cell reset kernel (us.): " << time_total << std::endl;

  if (cells.getTwoMassStateOn()) {
    t_kernels.getCommandQueue().enqueueNDRangeKernel(
        t_kernels.m_particleInBubbleKernel.getKernel(), cl::NullRange,
        cl::NDRange(particles.getParticleCountTotal()));
    t_kernels.getCommandQueue().enqueueNDRangeKernel(
        t_kernels.m_cellAssignmentKernelTwoMassState.getKernel(), cl::NullRange,
        cl::NDRange(particles.getParticleCountTotal()), cl::NullRange, NULL,
        &event);
  } else {
    t_kernels.getCommandQueue().enqueueNDRangeKernel(
        t_kernels.m_cellAssignmentKernel.getKernel(), cl::NullRange,
        cl::NDRange(particles.getParticleCountTotal()), cl::NullRange, NULL,
        &event);
  }
#endif
#ifndef DEBUG_OPENCL_KERNEL_RUNTIME_PROFILE
  if (cells.getTwoMassStateOn()) {
    t_kernels.getCommandQueue().enqueueNDRangeKernel(
        t_kernels.m_particleInBubbleKernel.getKernel(), cl::NullRange,
        cl::NDRange(particles.getParticleCountTotal()));
    t_kernels.getCommandQueue().enqueueNDRangeKernel(
        t_kernels.m_cellAssignmentKernelTwoMassState.getKernel(), cl::NullRange,
        cl::NDRange(particles.getParticleCountTotal()), cl::NullRange);
  } else {
    t_kernels.getCommandQueue().enqueueNDRangeKernel(
        t_kernels.m_cellAssignmentKernel.getKernel(), cl::NullRange,
        cl::NDRange(particles.getParticleCountTotal()), cl::NullRange);
  }
#endif

#ifdef DEBUG_OPENCL_KERNEL_RUNTIME_PROFILE
  t_kernels.getCommandQueue().finish();
  time_start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
  time_end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
  time_total = (time_end - time_start) / 1000;
  std::cout << "= Cell assignment kernel (us.): " << time_total << std::endl;
#endif

  cells.calculate_new_no_collision_probability(t_dt, t_tau);
  cells.writeNoCollisionProbabilityBuffer(t_kernels.getCommandQueue());

 #ifdef DEBUG_OPENCL_KERNEL_RUNTIME_PROFILE
  t_kernels.getCommandQueue().enqueueNDRangeKernel(
      t_kernels.m_collisionCellResetKernel.getKernel(), cl::NullRange,
      cl::NDRange(cells.getCellCount()), cl::NullRange, NULL, &event);
#endif
#ifndef DEBUG_OPENCL_KERNEL_RUNTIME_PROFILE
  t_kernels.getCommandQueue().enqueueNDRangeKernel(
      t_kernels.m_collisionCellResetKernel.getKernel(), cl::NullRange,
      cl::NDRange(cells.getCellCount()));
#endif
#ifdef DEBUG_OPENCL_KERNEL_RUNTIME_PROFILE
  t_kernels.getCommandQueue().finish();
  time_start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
  time_end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
  time_total = (time_end - time_start) / 1000;
  std::cout << "= Cell reset kernel (us.): " << time_total << std::endl;
  t_kernels.getCommandQueue().enqueueNDRangeKernel(
      t_kernels.m_collisionCellSumParticlesKernel.getKernel(), cl::NullRange,
      cl::NDRange(particles.getParticleCountTotal()), cl::NullRange, NULL,
      &event);
#endif
#ifndef DEBUG_OPENCL_KERNEL_RUNTIME_PROFILE
  t_kernels.getCommandQueue().enqueueNDRangeKernel(
      t_kernels.m_collisionCellSumParticlesKernel.getKernel(), cl::NullRange,
      cl::NDRange(particles.getParticleCountTotal()));
#endif
#ifdef DEBUG_OPENCL_KERNEL_RUNTIME_PROFILE
  t_kernels.getCommandQueue().finish();
  time_start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
  time_end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
  time_total = (time_end - time_start) / 1000;
  std::cout << "= Cell summation kernel (us.): " << time_total << std::endl;
  t_kernels.getCommandQueue().enqueueNDRangeKernel(
      t_kernels.m_collisionCellCalculateGenerationKernel.getKernel(),
      cl::NullRange, cl::NDRange(cells.getCellCount()), cl::NullRange, NULL, &event);
#endif
#ifndef DEBUG_OPENCL_KERNEL_RUNTIME_PROFILE
  t_kernels.getCommandQueue().enqueueNDRangeKernel(
      t_kernels.m_collisionCellCalculateGenerationKernel.getKernel(),
      cl::NullRange, cl::NDRange(cells.getCellCount()));
#endif
#ifdef DEBUG_OPENCL_KERNEL_RUNTIME_PROFILE
  t_kernels.getCommandQueue().finish();
  time_start =
      event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
  time_end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
  time_total = (time_end - time_start) / 1000;
  std::cout << "= Cell generation kernel (us.): " << time_total << std::endl;
  t_kernels.getCommandQueue().enqueueNDRangeKernel(
      t_kernels.m_rotationKernel.getKernel(), cl::NullRange,
      cl::NDRange(particles.getParticleCountTotal()), cl::NullRange, NULL, &event);
#endif
#ifndef DEBUG_OPENCL_KERNEL_RUNTIME_PROFILE
  t_kernels.getCommandQueue().enqueueNDRangeKernel(
      t_kernels.m_rotationKernel.getKernel(), cl::NullRange,
      cl::NDRange(particles.getParticleCountTotal()));
#endif
#ifdef DEBUG_OPENCL_KERNEL_RUNTIME_PROFILE
  t_kernels.getCommandQueue().finish();
  time_start =
      event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
  time_end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
  time_total = (time_end - time_start)/1000;
  std::cout << "= Cell rotation kernel (us.): " << time_total << std::endl;
#endif
}

void Simulation::collide3(ParticleCollection& particles,
                          CollisionCellCollection& cells, numType t_dt,
                          numType t_tau,
                          RandomNumberGeneratorNumType& t_rng_numtype,
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
      t_kernels.m_cellAssignmentKernel.getKernel(), cl::NullRange,
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
      cl::NullRange, cl::NDRange(cells.getCellCount()));
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
    RandomNumberGeneratorULong& t_rng_int, OpenCLLoader& t_kernels) {
  // m_timestepAdapter.calculateNewTimeStep(bubble);
  m_dt_current = m_parameters.getTimestepAdapter().getTimestep();
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

  // Full CPU algorithm
  /* collide(particles, cells, m_dt_current, getTau(), t_rng_numtype,
    t_kernels);*/

  // Full GPU algorithm
  collide2(particles, cells, m_dt_current, getTau(), t_rng_numtype, t_rng_int,
           t_kernels);

  // CollisionHack
  /*cells.readCellParticleCountBuffer(t_kernels.getCommandQueue());
  u_int max_count = 0;
  for (u_int i = 1; i < cells.getCellCount(); i++) {
    if (cells.returnParticleCountInCell(i) > max_count) {
      max_count = cells.returnParticleCountInCell(i);
    }
  }
  std::cout << "Max count: " << max_count << std::endl;
  std::cout << "Index 0: " << cells.returnParticleCountInCell(0) << std::endl;
  exit(0);*/

  /* Not optimized and not working.
  collide3(particles, cells, m_dt_current, getTau(), t_rng_numtype, t_rng_int,
           t_kernels);
  */
  /*u_int collision_cell_count = 0;
  cells.readCellCollideBooleanBuffer(t_kernels.getCommandQueue());
  for (int i = 0; i < cells.getCellCount(); i++) {
    collision_cell_count += cells.returnCellCollideBoolean(i);
  }
  std::cout << "Collision cell count: " << collision_cell_count << std::endl;*/

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
  // collide(particles, cells, m_dt_current, getTau(), t_rng_numtype,
  // t_kernels);
  collide2(particles, cells, m_dt_current, getTau(), t_rng_numtype, t_rng_int,
           t_kernels);
#ifdef TIME_COLLIDE_DEBUG
  auto stop = my_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "Collision time: " << clock_micros(stop - start).count()
            << std::endl;
  start = my_clock::now();
#endif
  particles.readParticleEBuffer(t_kernels.getCommandQueue());
  particles.readdPBuffer(t_kernels.getCommandQueue());
#ifdef TIME_COLLIDE_DEBUG
  stop = my_clock::now();
  duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "particle dP and E read: " << clock_micros(stop - start).count()
            << std::endl;
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