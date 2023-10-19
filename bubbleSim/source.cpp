#define _CRT_SECURE_NO_WARNINGS
#include "source.h"

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::seconds;

int DEFAULT_COUT_PRECISION = std::cout.precision();

// Collision is in development

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

  /* TODO:
    Tau is defined in config. In current setup dt depends on tau: dt= tau/N.
    Tau should be representing thermalization time in some sence.
    In the future dt and tau are separate but probability to collide still
    depends on exp(-dt/tau).
  */
  numType tau = config.parameterTau;
  u_int N_steps_tau = 10;
  numType dt = tau / N_steps_tau;
  u_int sim_length_in_tau = config.m_maxSteps;
  /*
    =============== Initialization ===============
  */
  RandomNumberGeneratorNumType rn_generator(config.m_seed);
  RandomNumberGeneratorULong rn_generator_64int(config.m_seed);

  std::cout << std::endl
            << "=============== OpenCL initialization ==============="
            << std::endl;
  OpenCLLoader kernels(s_kernelPath);

  std::cout << std::endl
            << "=============== Simulation initialization ==============="
            << std::endl;
  std::cout << std::setprecision(6) << std::fixed << std::showpoint;

  numType no_particle_radius = (config.SIMULATION_SETTINGS.isFlagSet(BUBBLE_ON))
                                   ? config.bubbleInitialRadius
                                   : 0.;
  numType deltaM = std::sqrt(std::pow(config.particleMassTrue, 2.) -
                             std::pow(config.particleMassFalse, 2.));
  numType particle_temperature_in_false_vacuum =
      std::sqrt(deltaM) / config.parameterEta;

  numType n_boltzmann = calculate_boltzmann_number_density(
      particle_temperature_in_false_vacuum, config.particleMassFalse);
  numType rho_boltzmann = calculate_boltzmann_energy_density(
      particle_temperature_in_false_vacuum, config.particleMassFalse);
  numType boundaryRadius =
      std::cbrt(config.particleCountFalse / n_boltzmann +
                4. * M_PI * std::pow(no_particle_radius, 3.) / 3.) /
      2.0;

  std::cout << "T-/ m + : "
            << particle_temperature_in_false_vacuum / config.particleMassTrue
            << ", R_b(t=0): " << no_particle_radius
            << ", R(boundary): " << boundaryRadius << std::endl;

  ParticleGenerator particleGenerator1;
  // ParticleGenerator particleGenerator2;

  particleGenerator1 = ParticleGenerator(config.particleMassFalse);
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
      no_particle_radius, boundaryRadius,
      (u_int)(particles.getParticleCountTotal()), rn_generator, particles);
  /*generatedParticleEnergy += particleGenerator2.generateNParticlesInCube(
      config.bubbleInitialRadius, config.cyclicBoundaryRadius,
      (u_int)(particles.getParticleCountTotal() -
     particles.getParticleX().size()), rn_generator, particles);*/
  particles.add_to_total_initial_energy(generatedParticleEnergy);

  numType initial_false_vacuum_volume =
      8 * std::pow(boundaryRadius, 3.) -
      4. * M_PI * std::pow(no_particle_radius, 3.) / 3.;

  numType plasma_energy_density =
      generatedParticleEnergy / initial_false_vacuum_volume;
  numType plasma_number_density =
      config.particleCountFalse / initial_false_vacuum_volume;

  numType dperl = 0.;

  numType mu = initial_false_vacuum_volume *
               std::pow(particle_temperature_in_false_vacuum, 3.) /
               (config.particleCountFalse * M_PI * M_PI);

  std::cout << "Energy density scale factor (m-=0): " << mu << std::endl;
  std::cout << "rho- : " << rho_boltzmann
            << ", rho(false): " << plasma_energy_density
            << ", rho(sim)/rho(anal.): "
            << plasma_energy_density / rho_boltzmann << std::endl;
  std::cout << "n-: " << n_boltzmann << ", n(false): " << plasma_number_density
            << ", n(sim)/n(anal.): " << plasma_number_density / n_boltzmann
            << std::endl;

  numType alpha = config.parameterAlpha;
  numType dV = (alpha * plasma_energy_density +
                particle_temperature_in_false_vacuum * plasma_number_density);

  /* TODO:
  In development: step revert. Add methdos which save state state after each
  step and if necessary then take revert step. Currently most of the necessary
  methods are ready but implementation is poor. Also add option to config to
  turn it on/off.
  */
  // particles.makeCopy();
  numType critical_radius =
      config.bubbleInitialRadius / config.parameterUpsilon;

  numType sigma =
      (alpha * plasma_energy_density +
       plasma_number_density * particle_temperature_in_false_vacuum) *
      critical_radius / 2.0;
  std::cout << "dV/rho-: " << dV / plasma_energy_density
            << ", sigma/(dV*R0): " << sigma / (dV * config.bubbleInitialRadius)
            << std::endl;

  // If bubble is true vacuum then dV is + sign. Otherwise dV sign is -.
  if (!config.SIMULATION_SETTINGS.isFlagSet(TRUE_VACUUM_BUBBLE_ON)) {
    dV = -dV;
  }

  PhaseBubble bubble;
  if (config.SIMULATION_SETTINGS.isFlagSet(BUBBLE_ON)) {
    bubble =
        PhaseBubble(config.bubbleInitialRadius, config.bubbleInitialSpeed, dV,
                    sigma, critical_radius, buffer_flags, kernels.getContext());
  }

  SimulationParameters simulation_parameters;
  Simulation simulation;

  numType particle_mass_in =
      (config.SIMULATION_SETTINGS.isFlagSet(TRUE_VACUUM_BUBBLE_ON))
          ? config.particleMassTrue
          : config.particleMassFalse;
  numType particle_mass_out =
      (config.SIMULATION_SETTINGS.isFlagSet(TRUE_VACUUM_BUBBLE_ON))
          ? config.particleMassFalse
          : config.particleMassTrue;

  if (config.SIMULATION_SETTINGS.isFlagSet(SIMULATION_BOUNDARY_ON)) {
    simulation_parameters = SimulationParameters(
        dt, particle_mass_in, particle_mass_out, boundaryRadius, buffer_flags,
        kernels.getContext());
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

  simulation.setTau(config.parameterTau);

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
      boundaryRadius / config.collision_cell_count;
  std::cout << "Collision cell length: " << collision_cell_length << std::endl;
  if (config.SIMULATION_SETTINGS.isFlagSet(COLLISION_ON)) {
    cells = CollisionCellCollection(
        collision_cell_length, config.collision_cell_count,
        config.SIMULATION_SETTINGS.isFlagSet(COLLISION_MASS_STATE_ON),
        buffer_flags, kernels.getContext());
  }

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
      kernels.m_particleInBubbleKernel.setBuffers(particles, bubble);
      particles.writeParticleInBubbleBuffer(kernels.getCommandQueue());
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
    cells.writeSeedBuffer(kernels.getCommandQueue());
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
  // DataStreamer streamer(filePath.string());

  DataStreamerBinary streamer2(filePath.string());
  streamer2.initStream_Data();

  streamer2.initialize_profile_streaming(
      200 * config.SIMULATION_SETTINGS.isFlagSet(BUBBLE_ON), 200);
  streamer2.initialize_momentum_streaming(
      200 * config.SIMULATION_SETTINGS.isFlagSet(BUBBLE_ON), 200,
      particle_temperature_in_false_vacuum);
  streamer2.initialize_momentum_radial_profile_streaming(
      200, 20 * config.SIMULATION_SETTINGS.isFlagSet(BUBBLE_ON), 20,
      particle_temperature_in_false_vacuum);
  /*streamer.stream(simulation, particles, bubble, momentum_log_scale_on,
                  kernels.getCommandQueue());
  */
  streamer2.stream(simulation, particles, bubble, config.SIMULATION_SETTINGS,
                   kernels.getCommandQueue());

  std::cout << std::endl;
  config.print_info();
  std::cout << std::endl;
  bubble.print_info(config);
  std::cout << std::endl;
  std::cout.precision(DEFAULT_COUT_PRECISION);

  std::cout.precision(8);

  simulation.calculate_average_particle_count_in_filled_cells(particles, cells,
                                                              kernels);
  // Create simulation info file which includes description of the simulation
  std::ofstream infoStream(filePath / "info.txt",
                           std::ios::out | std::ios::trunc);
  createSimulationInfoFile(infoStream, filePath, config, dV, boundaryRadius);

  /*
    =============== Run simulation ===============
  */

  numType simTimeSinceLastStream = 0.;
  int stepsSinceLastStream = 0;
  std::cout << "=============== Simulation ===============" << std::endl;
  program_end_time = my_clock::now();
  print_simulation_state(particles, bubble, cells, simulation, config);
  std::cout << ", time: "
            << convertTimeToHMS(program_end_time - program_start_time)
            << std::endl;

  auto start_time = my_clock::now();
  auto end_time = my_clock::now();

  for (u_int i = 1; i <= N_steps_tau * sim_length_in_tau; i++) {
    if (config.SIMULATION_SETTINGS.isFlagSet(BUBBLE_ON | BUBBLE_INTERACTION_ON |
                                             COLLISION_ON |
                                             SIMULATION_BOUNDARY_ON)) {
      simulation.stepParticleBubbleCollisionBoundary(
          particles, bubble, cells, rn_generator, rn_generator_64int, kernels);
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
                                               rn_generator_64int, kernels);
    } else {
      simulation.step(bubble, 0);
    }
    if (i % config.streamStep == 0) {
      simulation.count_collision_cells(cells, kernels);
      streamer2.stream(simulation, particles, bubble,
                       config.SIMULATION_SETTINGS, kernels.getCommandQueue());
      program_end_time = my_clock::now();

      print_simulation_state(particles, bubble, cells, simulation, config);
      dperl =
          std::pow(plasma_number_density, -1.0 / 3.0) *
          simulation.getActiveCollidingParticleCount() /
          (simulation.get_dt_currentStep() * particles.getParticleCountTotal());
      std::cout << ", d/l: " << dperl;
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
      if ((bubble.getRadius() >= boundaryRadius) && (config.cyclicBoundaryOn)) {
        std::cerr
            << "Ending simulation. Bubble radius >= simulation boundary radius."
            << std::endl;
        break;
      }
    }
  }

  // Stream last state
  // streamer.stream(simulation, particles, bubble, momentum_log_scale_on,
  //                kernels.getCommandQueue());
  streamer2.stream(simulation, particles, bubble, config.SIMULATION_SETTINGS,
                   kernels.getCommandQueue());

  // Measure runtime
  program_end_time = my_clock::now();
  auto ms_int = std::chrono::duration_cast<std::chrono::seconds>(
      program_end_time - program_start_time);

  appendSimulationInfoFile(infoStream, (int)ms_int.count());
}
