#define _CRT_SECURE_NO_WARNINGS
#include "source.h"

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::seconds;

u_int DEFAULT_COUT_PRECISION = (u_int)std::cout.precision();

// Collision is in development

// numType scale_dV(numType)

numType calculate_boltzmann_number_density(numType T, numType mass) {
  numType n = 0.0;
  numType dp = 1e-4 * T;
  numType p = 0.;
  while (p < 30.0 * T) {
    n += std::pow(p, 2.) * std::exp(-std::sqrt(p * p + mass * mass) / T) * dp;
    p += dp;
  }
  return n / (2. * M_PI * M_PI);
}

numType calculate_boltzmann_energy_density(numType T, numType mass) {
  numType n = 0.0;
  numType dp = 1e-4 * T;
  numType p = 0.;
  while (p < 30.0 * T) {
    n += std::sqrt(p * p + mass * mass) * std::pow(p, 2.) *
         std::exp(-std::sqrt(p * p + mass * mass) / T) * dp;
    p += dp;
  }
  return n / (2. * M_PI * M_PI);
}

void print_simulation_state(ParticleCollection& particles, PhaseBubble& bubble,
                            CollisionCellCollection& cells,
                            Simulation& simulation, ConfigReader& config) {
  std::cout << "Step: " << std::setw(8) << simulation.getStep();
  std::cout.precision(2);
  std::cout << ", Sim. time: " << std::setw(8) << simulation.getTime();
  if (config.SIMULATION_SETTINGS.isFlagSet(BUBBLE_ON)) {
    std::cout.precision(4);
    std::cout << ", R: " << std::setw(8) << bubble.getRadius();
    std::cout << ", R/Rc: " << bubble.getRadius() / bubble.getCriticalRadius();
    std::cout << ", V: " << std::setw(7) << bubble.getSpeed();
  }
  if (config.SIMULATION_SETTINGS.isFlagSet(COLLISION_ON)) {
    std::cout.precision(5);
    std::cout << ", N_col(cell)/N_tot: "
              << (numType)simulation.getActiveCollisionCellCount() /
                     cells.getCellCount()
              << ", N_col(part.)/N_tot: "
              << (numType)simulation.getActiveCollidingParticleCount() /
                     particles.getParticleCountTotal();
  }
  std::cout.precision(6);
  std::cout << ", E: "
            << simulation.getTotalEnergy() / simulation.getInitialTotalEnergy();
  std::cout.precision(DEFAULT_COUT_PRECISION);
}

void scale_R_dV_sigma(numType& bubble_radius, numType& boundary_radius,
                      numType& dV, numType& sigma, numType scale) {
  numType constant1 = sigma / (dV * bubble_radius);

  numType V0 = 8. * std::pow(boundary_radius, 3.) -
               4. * M_PI * std::pow(bubble_radius, 3.) / 3.;
  boundary_radius *= scale;
  numType V1 = 8. * std::pow(boundary_radius, 3.) -
               4. * M_PI * std::pow(bubble_radius, 3.) / 3.;
  dV = V0 / V1 * dV;

  sigma = constant1 * dV * bubble_radius;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: bubbleSim.exe config.json kernel.cl" << std::endl;
    exit(0);
  }

  auto program_start_time = my_clock::now();
  auto program_end_time = my_clock::now();

  std::string s_configPath = argv[1];  // "config.json"
  std::string s_kernelPath = argv[2];  // "kernel.cl";
  std::uint64_t buffer_flags = 0;
  std::cout << "Config path: " << s_configPath << std::endl;
  std::cout << "Kernel path: " << s_kernelPath << std::endl;
  ConfigReader config(s_configPath);

  RandomNumberGeneratorNumType rn_generator(config.m_seed);
  RandomNumberGeneratorULong rn_generator_64uint(config.m_seed);

  /* TODO:
    Tau is defined in config. In current setup dt depends on tau: dt= tau/N.
    Tau should be representing thermalization time in some sence.
    In the future dt and tau are separate but probability to collide still
    depends on exp(-dt/tau).
  */
  numType dV =
      config.lambda * std::pow(config.v, 4.0) * (1 - 2 * config.etaV) / 12.;
  numType sigma = config.sigma;
  // Particles
  numType particle_mass_in_false_vacuum = config.particleMassFalse;
  numType particle_mass_in_true_vacuum = config.v * config.y;
  numType particle_deltaM =
      std::sqrt(std::pow(particle_mass_in_true_vacuum, 2.) -
                std::pow(particle_mass_in_false_vacuum, 2.));
  numType particle_temperature_in_false_vacuum = config.Tn;
  numType n_false_boltzmann = calculate_boltzmann_number_density(
      particle_temperature_in_false_vacuum, config.particleMassFalse);
  numType rho_false_boltzmann = calculate_boltzmann_energy_density(
      particle_temperature_in_false_vacuum, config.particleMassFalse);
  numType particle_eta = particle_deltaM / particle_temperature_in_false_vacuum;

  // Bubble
  numType bubble_critical_radius = 2 * sigma / dV;
  numType bubble_initial_radius =
      config.upsilon * bubble_critical_radius *
      config.SIMULATION_SETTINGS.isFlagSet(BUBBLE_ON);
  numType simulation_boundary_radius =
      std::cbrt(config.particleCountFalse / n_false_boltzmann +
                4. * M_PI * std::pow(bubble_initial_radius, 3.) / 3.) /
      2.0;

  // std::cout << dV << std::endl;
  scale_R_dV_sigma(bubble_initial_radius, simulation_boundary_radius, dV, sigma,
                   config.scale);
  // std::cout << dV << std::endl;
  if (simulation_boundary_radius <= bubble_initial_radius) {
    std::cerr << "Boundary radius <= Initial bubble radius" << std::endl;
    std::exit(0);
  }

  numType dt = simulation_boundary_radius / config.timestep_resolution;
  numType tau = config.tau * dt;

  /*
    =============== Initialization ===============
  */

  std::cout << std::endl
            << "=============== OpenCL initialization ==============="
            << std::endl;
  OpenCLLoader kernels(s_kernelPath);
  std::cout << std::endl
            << "=============== Simulation initialization ==============="
            << std::endl;
  std::cout << std::setprecision(6) << std::fixed << std::showpoint;

  ParticleGenerator particleGenerator1;
  // ParticleGenerator particleGenerator2;

  particleGenerator1 = ParticleGenerator(particle_mass_in_false_vacuum);
  // particleGenerator2 = ParticleGenerator(config.particleMassFalse);

  particleGenerator1.calculateCPDBoltzmann(
      particle_temperature_in_false_vacuum,
      30 * particle_temperature_in_false_vacuum,
      1e-5 * particle_temperature_in_false_vacuum);

  /*particleGenerator1.calculateCPDDelta(3 *
                                       particle_temperature_in_false_vacuum);*/

  // particleGenerator2.calculateCPDBeta(2.5, 1., 2., 2., 0.00001);

  ParticleCollection particles(
      config.particleCountTrue, config.particleCountFalse,
      config.SIMULATION_SETTINGS.isFlagSet(TRUE_VACUUM_BUBBLE_ON), buffer_flags,
      kernels.getContext());

  numType generatedParticleEnergy;
  generatedParticleEnergy = particleGenerator1.generateNParticlesInCube(
      bubble_initial_radius, simulation_boundary_radius,
      (u_int)(particles.getParticleCountTotal()), rn_generator, particles);

  /*generatedParticleEnergy += particleGenerator2.generateNParticlesInCube(
      config.bubbleInitialRadius, config.cyclicBoundaryRadius,
      (u_int)(particles.getParticleCountTotal() -
     particles.getParticleX().size()), rn_generator, particles);*/
  particles.add_to_total_initial_energy(generatedParticleEnergy);

  numType initial_false_vacuum_volume =
      8 * std::pow(simulation_boundary_radius, 3.) -
      4. * M_PI * std::pow(bubble_initial_radius, 3.) / 3.;

  numType generated_plasma_rho =
      generatedParticleEnergy / initial_false_vacuum_volume;
  numType generated_plasma_n =
      config.particleCountFalse / initial_false_vacuum_volume;

  numType dperl = 0.;

  numType mu = initial_false_vacuum_volume *
               std::pow(particle_temperature_in_false_vacuum, 3.) /
               (config.particleCountFalse * M_PI * M_PI);

  std::cout << "m-: " << particle_mass_in_false_vacuum
            << ", m+: " << particle_mass_in_true_vacuum << std::endl;
  std::cout << "T-: " << particle_temperature_in_false_vacuum
            << ", eta: " << particle_eta << std::endl;
  std::cout << "n-: " << n_false_boltzmann << ", rho-: " << rho_false_boltzmann
            << std::endl;
  std::cout << "R_c: " << bubble_critical_radius
            << ", R_b(0): " << bubble_initial_radius << " ("
            << bubble_initial_radius / bubble_critical_radius << "R_c)"
            << ", R_bd: " << simulation_boundary_radius << " ("
            << simulation_boundary_radius / bubble_critical_radius << "R_c)"
            << std::endl;

  std::cout << "Energy density scale factor (m-=0): " << mu << std::endl;
  std::cout << "rho- : " << rho_false_boltzmann
            << ", rho(false): " << generated_plasma_rho
            << ", rho(sim)/rho(anal.): "
            << generated_plasma_rho / rho_false_boltzmann << std::endl;
  std::cout << "n-: " << n_false_boltzmann
            << ", n(false): " << generated_plasma_n
            << ", n(sim)/n(anal.): " << generated_plasma_n / n_false_boltzmann
            << std::endl;

  /* TODO:
  In development: step revert. Add methdos which save state state after each
  step and if necessary then take revert step. Currently most of the necessary
  methods are ready but implementation is poor. Also add option to config to
  turn it on/off.
  */
  // particles.makeCopy();

  std::cout << std::endl
            << "=============== Dimensionless parameters ==============="
            << std::endl;

  std::cout << "dV/rho-: " << dV / generated_plasma_rho
            << ", sigma/(dV*R0): " << sigma / (dV * bubble_initial_radius)
            << std::endl;

  // If bubble is true vacuum then dV is + sign. Otherwise dV sign is -.
  if (!config.SIMULATION_SETTINGS.isFlagSet(TRUE_VACUUM_BUBBLE_ON)) {
    dV = -dV;
  }

  PhaseBubble bubble;
  if (config.SIMULATION_SETTINGS.isFlagSet(BUBBLE_ON)) {
    bubble =
        PhaseBubble(bubble_initial_radius, config.bubbleInitialSpeed, dV, sigma,
                    bubble_critical_radius, buffer_flags, kernels.getContext());
  }

  SimulationParameters simulation_parameters;
  Simulation simulation;

  std::cout << std::endl;
  bubble.print_info(config);
  std::cout << std::endl;

  numType particle_mass_in =
      (config.SIMULATION_SETTINGS.isFlagSet(TRUE_VACUUM_BUBBLE_ON))
          ? particle_mass_in_true_vacuum
          : particle_mass_in_false_vacuum;
  numType particle_mass_out =
      (config.SIMULATION_SETTINGS.isFlagSet(TRUE_VACUUM_BUBBLE_ON))
          ? particle_mass_in_false_vacuum
          : particle_mass_in_true_vacuum;

  if (config.SIMULATION_SETTINGS.isFlagSet(SIMULATION_BOUNDARY_ON)) {
    simulation_parameters = SimulationParameters(
        dt, particle_mass_in, particle_mass_out, simulation_boundary_radius,
        buffer_flags, kernels.getContext());
  } else if (config.SIMULATION_SETTINGS.isFlagSet(BUBBLE_ON)) {
    simulation_parameters =
        SimulationParameters(dt, particle_mass_in, particle_mass_out,
                             buffer_flags, kernels.getContext());
  } else {
    simulation_parameters =
        SimulationParameters(dt, buffer_flags, kernels.getContext());
  }

  simulation = Simulation(config.m_seed, dt, simulation_parameters,
                          kernels.getContext());

  simulation.setTau(tau);

  // Add all energy together in simulation for later use
  simulation.addInitialTotalEnergy(particles.getInitialTotalEnergy());
  simulation.addInitialTotalEnergy(bubble.calculateEnergy());
  simulation.setInitialCompactness(simulation.getInitialTotalEnergy() /
                                   bubble.getInitialRadius());

  CollisionCellCollection cells;
  /*
    Collision cell length is defined by simulation size. Simulation space with =
    2 * boundary_radius. Cell length =  2 * boundary_radius /
    N_collision_cell_count
  */
  numType collision_cell_length =
      2.0 * simulation_boundary_radius / config.collision_cell_count;

  if (config.SIMULATION_SETTINGS.isFlagSet(COLLISION_ON)) {
    cells = CollisionCellCollection(
        collision_cell_length, config.collision_cell_count,
        config.SIMULATION_SETTINGS.isFlagSet(COLLISION_MASS_STATE_ON),
        config.collision_cell_duplication, generated_plasma_n, buffer_flags,
        kernels.getContext());
    cells.generate_collision_seeds(rn_generator_64uint);
  }
  std::cout << "R_cell: " << collision_cell_length << std::endl;
  std::cout << "Cell count: " << cells.getCellCount() << std::endl;
  // Setup buffers for OpenCL
  if (config.SIMULATION_SETTINGS.isFlagSet(BUBBLE_INTERACTION_ON | BUBBLE_ON)) {
    kernels.m_particleStepWithBubbleKernel.setBuffers(
        simulation.getSimulationParameters(), particles, bubble, buffer_flags);
    bubble.writeAllBuffersToKernel(kernels.getCommandQueue());

  } else {
    kernels.m_particleLinearStepKernel.setBuffers(
        simulation.getSimulationParameters(), particles, buffer_flags);
  }
  if (config.SIMULATION_SETTINGS.isFlagSet(SIMULATION_BOUNDARY_ON)) {
    kernels.m_particleBoundaryKernel.setBuffers(
        simulation.getSimulationParameters(), particles, buffer_flags);
  }

  if (config.SIMULATION_SETTINGS.isFlagSet(COLLISION_ON)) {
    if (config.SIMULATION_SETTINGS.isFlagSet(COLLISION_MASS_STATE_ON)) {
      kernels.m_cellAssignmentKernelTwoMassState.setBuffers(particles, cells,
                                                            buffer_flags);
      particles.writeParticleInBubbleBuffer(kernels.getCommandQueue());
      if (config.SIMULATION_SETTINGS.isFlagSet(BUBBLE_ON)) {
        kernels.m_particleInBubbleKernel.setBuffers(particles, bubble);
      }
    } else {
      kernels.m_cellAssignmentKernel.setBuffers(particles, cells, buffer_flags);
    }
    kernels.m_collisionCellSumParticlesKernel.setBuffers(particles, cells,
                                                         buffer_flags);
    kernels.m_rotationKernel.setBuffers(particles, cells, buffer_flags);
    kernels.m_collisionCellResetKernel.setBuffers(cells, buffer_flags);
    kernels.m_collisionCellCalculateGenerationKernel.setBuffers(cells,
                                                                buffer_flags);
    cells.writeAllBuffersToKernel(kernels.getCommandQueue());
    cells.writeNoCollisionProbabilityBuffer(kernels.getCommandQueue());
    cells.writeCollisionSeedsBuffer(kernels.getCommandQueue());
    // cells.writeCellDuplicationBuffer(kernels.getCommandQueue());
    cells.writeTwoMassStateOnBuffer(kernels.getCommandQueue());
    cells.writeNEquilibriumBuffer(kernels.getCommandQueue());
  }

  // Move data to the GPU
  particles.writeParticleCoordinatesBuffer(kernels.getCommandQueue());
  particles.writeParticleEBuffer(kernels.getCommandQueue());
  particles.writeParticleMomentumsBuffer(kernels.getCommandQueue());
  particles.writeParticleMBuffer(kernels.getCommandQueue());
  particles.writedPBuffer(kernels.getCommandQueue());
  simulation.getSimulationParameters().writeDtBuffer(kernels.getCommandQueue());
  simulation.getSimulationParameters().writeMassInBuffer(
      kernels.getCommandQueue());
  simulation.getSimulationParameters().writeMassOutBuffer(
      kernels.getCommandQueue());

  /* TODO
    Modify streaming process. Make it easier to read. Refactor.
  */
  std::string dataFolderName = createFileNameFromCurrentDate();
  std::filesystem::path filePath =
      createSimulationFilePath(config.m_dataSavePath, dataFolderName);

  DataStreamerBinary streamer2(filePath.string());
  streamer2.initStream_Data();

  streamer2.initialize_profile_streaming(
      200 * config.SIMULATION_SETTINGS.isFlagSet(BUBBLE_ON), 200,
      config.SIMULATION_SETTINGS);
  streamer2.initialize_momentum_streaming(
      200 * config.SIMULATION_SETTINGS.isFlagSet(BUBBLE_ON), 200,
      particle_temperature_in_false_vacuum);
  streamer2.initialize_momentum_radial_profile_streaming(
      200, 20 * config.SIMULATION_SETTINGS.isFlagSet(BUBBLE_ON), 20,
      particle_temperature_in_false_vacuum);
  streamer2.stream(simulation, particles, bubble, cells,
                   config.SIMULATION_SETTINGS, kernels.getCommandQueue());

  // std::cout << std::endl;
  // config.print_info();

  std::cout.precision(DEFAULT_COUT_PRECISION);

  std::cout.precision(8);

  simulation.calculate_average_particle_count_in_filled_cells(
      particles, cells, bubble_initial_radius, simulation_boundary_radius,
      collision_cell_length, kernels);
  // Create simulation info file which includes description of the simulation
  std::ofstream infoStream(filePath / "info.txt",
                           std::ios::out | std::ios::trunc);
  createSimulationInfoFile(infoStream, filePath, config, dV,
                           bubble_critical_radius, bubble_initial_radius,
                           simulation_boundary_radius);

  /*
    =============== Run simulation ===============
  */

  numType simTimeSinceLastStream = 0.;
  int stepsSinceLastStream = 0;

  std::cout << "Max steps: " << config.m_max_steps << std::endl;

  std::cout << "=============== Simulation ===============" << std::endl;
  program_end_time = my_clock::now();
  print_simulation_state(particles, bubble, cells, simulation, config);
  std::cout << ", time: "
            << convertTimeToHMS(program_end_time - program_start_time)
            << std::endl;

  auto start_time = my_clock::now();
  auto end_time = my_clock::now();

  kernels.getCommandQueue().enqueueNDRangeKernel(
      kernels.m_particleLinearStepKernel.getKernel(), cl::NullRange,
      cl::NDRange(particles.getParticleCountTotal()));

  for (u_int i = 1; i <= config.m_max_steps; i++) {
    if ((i + 1) % config.stream_step == 0) {
      particles.createMomentumCopy();
    }

    if (config.SIMULATION_SETTINGS.isFlagSet(BUBBLE_ON | BUBBLE_INTERACTION_ON |
                                             COLLISION_ON |
                                             SIMULATION_BOUNDARY_ON)) {
      simulation.stepParticleBubbleCollisionBoundary(
          particles, bubble, cells, rn_generator, rn_generator_64uint, kernels);
    } else if (config.SIMULATION_SETTINGS.isFlagSet(BUBBLE_ON |
                                                    BUBBLE_INTERACTION_ON |
                                                    SIMULATION_BOUNDARY_ON)) {
      simulation.stepParticleBubbleBoundary(particles, bubble, kernels);
    } else if (config.SIMULATION_SETTINGS.isFlagSet(BUBBLE_ON |
                                                    BUBBLE_INTERACTION_ON)) {
      simulation.stepParticleBubble(particles, bubble, kernels);
    } else if (!config.SIMULATION_SETTINGS.isFlagSet(BUBBLE_ON) &&
               config.SIMULATION_SETTINGS.isFlagSet(COLLISION_ON)) {
      simulation.stepParticleCollisionBoundary(particles, cells, rn_generator,
                                               rn_generator_64uint, kernels);
    } else {
      simulation.step(bubble, 0);
    }
    if (i % config.stream_step == 0) {
      simulation.count_collision_cells(cells, kernels);
      streamer2.stream(simulation, particles, bubble, cells,
                       config.SIMULATION_SETTINGS, kernels.getCommandQueue());
      program_end_time = my_clock::now();

      print_simulation_state(particles, bubble, cells, simulation, config);
      /*dperl =
          std::pow(generated_plasma_n, -1.0 / 3.0) *
          simulation.getActiveCollidingParticleCount() /
          (simulation.get_dt_currentStep() * particles.getParticleCountTotal());
      std::cout << ", d/l: " << dperl;*/
      std::cout << ", time: "
                << convertTimeToHMS(program_end_time - program_start_time)
                << std::endl;
    }

   

    if (config.SIMULATION_SETTINGS.isFlagSet(BUBBLE_ON)) {
      if (std::isnan(bubble.getRadius()) || bubble.getRadius() <= 0) {
        std::cerr << "Ending simulaton. Radius is not a number or <= 0. (R_b="
                  << bubble.getRadius() << ")" << std::endl;
        break;
      }
      if (std::isnan(bubble.getSpeed()) || std::abs(bubble.getSpeed()) >= 1) {
        std::cerr << "Ending simulaton. Bubble speed not a number or > 1. (V_b="
                  << bubble.getSpeed() << ")" << std::endl;
        break;
      }
      if ((bubble.getRadius() >= simulation_boundary_radius) &&
          (config.cyclicBoundaryOn)) {
        std::cerr
            << "Ending simulation. Bubble radius >= simulation boundary radius."
            << std::endl;
        break;
      }
      if (simulation_parameters.getBoundaryRadius() <= simulation.getTime()) {
        std::cerr
            << "Ending simulation. Simulation boundary <= simulation time."
            << std::endl;
        break;
      }
    }
  }

  // Stream last state
  // streamer.stream(simulation, particles, bubble, momentum_log_scale_on,
  //                kernels.getCommandQueue());
  streamer2.stream(simulation, particles, bubble, cells,
                   config.SIMULATION_SETTINGS, kernels.getCommandQueue());

  // Measure runtime
  program_end_time = my_clock::now();
  auto ms_int = std::chrono::duration_cast<std::chrono::seconds>(
      program_end_time - program_start_time);

  appendSimulationInfoFile(infoStream, (int)ms_int.count());
}
