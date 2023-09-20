#define _CRT_SECURE_NO_WARNINGS
#include "source.h"

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::seconds;

// Collision is in development

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: bubbleSim.exe config.json kernel.cl" << std::endl;
    exit(0);
  }
  auto program_start_time = my_clock::now();
  auto program_end_time = my_clock::now();

  // Read configs and kernel file
  std::string s_configPath = argv[1];  // "config.json"
  std::string s_kernelPath = argv[2];  // "kernel.cl";

  std::uint32_t buffer_flags = 0;

  std::cout << "Config path: " << s_configPath << std::endl;
  std::cout << "Kernel path: " << s_kernelPath << std::endl;
  ConfigReader config(s_configPath);

  numType tau = config.parameterTau;
  u_int N_steps_tau = 10;
  numType dt = tau / N_steps_tau;
  u_int sim_length_in_tau = config.m_maxSteps;
  /*
          ===============  ===============
  */
  // If seed = 0 then it generates random seed.

  /*
    =============== Initialization ===============
 */

  // 1) Initialize random number generators
  RandomNumberGeneratorNumType rn_generator(config.m_seed);
  RandomNumberGeneratorULong rn_generator_64int(config.m_seed);

  // 2) Initialize openCL (kernels, commandQueues)
  OpenCLLoader kernels(s_kernelPath, config.kernelName);

  // 3) Generate particles

  numType deltaM = std::sqrt(std::pow(config.particleMassTrue, 2.) -
                             std::pow(config.particleMassFalse, 2.));
  numType particle_temperature_in_false_vacuum =
      std::sqrt(deltaM) / config.parameterEta;
  ParticleGenerator particleGenerator1;
  ParticleGenerator particleGenerator2;
  std::cout << "T-: " << particle_temperature_in_false_vacuum << std::endl;
  // 3.1) Create generator, calculates distribution(s)
  particleGenerator1 = ParticleGenerator(config.particleMassFalse);
  // particleGenerator2 = ParticleGenerator(config.particleMassFalse);

  /*particleGenerator1.calculateCPDDelta(3 *
                                       particle_temperature_in_false_vacuum);*/
  particleGenerator1.calculateCPDBoltzmann(
      config.particleTemperatureFalse, 30 * config.particleTemperatureFalse,
      1e-5 * config.particleTemperatureFalse);
  // particleGenerator2.calculateCPDBeta(2.5, 1., 2., 2., 0.00001);

  // 3.2) Create particle collection

  ParticleCollection particles(
      config.particleCountTrue, config.particleCountFalse,
      config.SIMULATION_SETTINGS.isFlagSet(TRUE_VACUUM_BUBBLE_ON), buffer_flags,
      kernels.getContext());

  // 3.3) create particles
  numType genreatedParticleEnergy;

  numType initial_false_vacuum_volume =
      (24. * std::pow(config.cyclicBoundaryRadius, 3.) -
       4. * M_PI * std::pow(config.bubbleInitialRadius, 3.)) /
      3.;
  numType energy_density = 3 * particle_temperature_in_false_vacuum *
                           config.particleCountFalse /
                           initial_false_vacuum_volume;
  numType dV = config.parameterAlpha * energy_density;

  genreatedParticleEnergy = particleGenerator1.generateNParticlesInCube(
      config.bubbleInitialRadius, config.cyclicBoundaryRadius,
      (u_int)(particles.getParticleCountTotal()), rn_generator, particles);
  /*genreatedParticleEnergy += particleGenerator2.generateNParticlesInCube(
      config.bubbleInitialRadius, config.cyclicBoundaryRadius,
      (u_int)(particles.getParticleCountTotal() -
     particles.getParticleX().size()), rn_generator, particles);*/

  std::cout << std::setprecision(10)
            << "Particle total energy: " << genreatedParticleEnergy << ", "
            << "Particle count: " << particles.getParticleX().size() << ", T: "
            << genreatedParticleEnergy / particles.getParticleX().size() / 3.
            << std::endl;
  particles.add_to_total_initial_energy(genreatedParticleEnergy);

  // In development: step revert
  // particles.makeCopy();

  // 4) Create bubble
  // numType dV = config.parameter_dV;
  numType alpha = config.parameterAlpha;
  numType sigma = config.parameterUpsilon * dV * config.bubbleInitialRadius;
  numType critical_radius =
      2 * config.parameterUpsilon * config.bubbleInitialRadius;
  if (!config.SIMULATION_SETTINGS.isFlagSet(TRUE_VACUUM_BUBBLE_ON)) {
    // If bubble is true vacuum then dV is correct sign. Otherwise dV sign must
    // be changed as change of direction changes.
    dV = -dV;
  }
  PhaseBubble bubble;
  if (config.SIMULATION_SETTINGS.isFlagSet(BUBBLE_ON)) {
    bubble = PhaseBubble(config.bubbleInitialRadius, config.bubbleInitialSpeed,
                         dV, sigma, buffer_flags, kernels.getContext());
  }

  // 5) Initialize Simulation
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
        dt, particle_mass_in, particle_mass_out, config.cyclicBoundaryRadius,
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

  simulation.setTau(config.parameterTau);

  // Add all energy together in simulation for later use
  simulation.addInitialTotalEnergy(particles.getInitialTotalEnergy());
  simulation.addInitialTotalEnergy(bubble.calculateEnergy());
  simulation.setInitialCompactness(simulation.getInitialTotalEnergy() /
                                   bubble.getInitialRadius());

  // 6) Create collision cells
  CollisionCellCollection cells(config.collision_cell_length,
                                config.collision_cell_count, false,
                                buffer_flags, kernels.getContext());

  // 7) Create buffers and copy data on the GPU
  // NB! Different kernels might need different input

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
    kernels.m_cellAssignmentKernel.setBuffers(particles, cells, buffer_flags);
    kernels.m_rotationKernel.setBuffers(particles, cells, buffer_flags);
    kernels.m_collisionCellResetKernel.setBuffers(cells, buffer_flags);
    kernels.m_collisionCellCalculateGenerationKernel.setBuffers(cells,
                                                                buffer_flags);

    cells.writeAllBuffersToKernel(kernels.getCommandQueue());
    cells.writeNoCollisionProbabilityBuffer(kernels.getCommandQueue());
    cells.writeSeedBuffer(kernels.getCommandQueue());
  }

  // Copy data to the GPU
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

  // 8) Streaming initialization
  std::string dataFolderName = createFileNameFromCurrentDate();
  std::filesystem::path filePath =
      createSimulationFilePath(config.m_dataSavePath, dataFolderName);
  DataStreamer streamer(filePath.string());

  // Momentum streaming is in log scale
  bool log_scale_on = true;

  if (config.STREAM_SETTINGS.isFlagSet(STREAM_DATA)) {
    streamer.initStream_Data();
  }
  if (config.STREAM_SETTINGS.isFlagSet(STREAM_RADIAL_VELOCITY)) {
    streamer.initStream_RadialVelocity(config.binsCountRadialVelocity,
                                       config.maxValueRadialVelocity);
  }
  if (config.STREAM_SETTINGS.isFlagSet(STREAM_TANGENTIAL_VELOCITY)) {
    streamer.initStream_TangentialVelocity(config.binsCountTangentialVelocity,
                                           config.maxValueTangentialVelocity);
  }
  if (config.STREAM_SETTINGS.isFlagSet(STREAM_NUMBER_DENSITY)) {
    streamer.initStream_Density(config.binsCountDensity,
                                config.maxValueDensity);
  }
  if (config.STREAM_SETTINGS.isFlagSet(STREAM_ENERGY_DENSITY)) {
    streamer.initStream_EnergyDensity(config.binsCountEnergy,
                                      config.maxValueEnergy);
  }
  if (config.STREAM_SETTINGS.isFlagSet(STREAM_MOMENTUM)) {
    if (config.STREAM_SETTINGS.isFlagSet(STREAM_MOMENTUM_IN)) {
      streamer.initStream_MomentumIn(config.binsCountMomentumIn,
                                     config.minValueMomentumIn,
                                     config.maxValueMomentumIn, log_scale_on);
    }
    if (config.STREAM_SETTINGS.isFlagSet(STREAM_MOMENTUM_OUT)) {
      streamer.initStream_MomentumOut(config.binsCountMomentumOut,
                                      config.minValueMomentumOut,
                                      config.maxValueMomentumOut, log_scale_on);
    }
  }
  streamer.stream(simulation, particles, bubble, log_scale_on,
                  kernels.getCommandQueue());
  /*
   * =============== Display text ===============
   */
  std::cout << std::endl;
  config.print_info();
  std::cout << std::endl;
  bubble.print_info(config);
  std::cout << std::endl;

  // 9) Streams
  std::cout << std::setprecision(8) << std::endl;
  // Create simulation info file which includes description of the simulation
  std::ofstream infoStream(filePath / "info.txt",
                           std::ios::out | std::ios::trunc);
  /*std::ofstream mfp_stream(filePath / "mfp.csv",
                           std::ios::out | std::ios::trunc);
  mfp_stream << "step,"
             << "mfp" << std::endl;
  */
  createSimulationInfoFile(infoStream, filePath, config, dV);

  /*
    =============== Run simulation ===============
  */


  /*numType mean_free_path = 0.;
  u_int mfp_count = 0;*/


  numType simTimeSinceLastStream = 0.;
  int stepsSinceLastStream = 0;
  std::cout << "=============== Simulation ===============" << std::endl;
  std::cout << std::setprecision(6) << std::fixed << std::showpoint;
  program_end_time = my_clock::now();
  std::cout << "Step: " << simulation.getStep()
            << ", Time: " << simulation.getTime()
            << ", R: " << bubble.getRadius()
            << ", R/Rc: " << bubble.getRadius() / critical_radius
            << ", V: " << bubble.getSpeed()
            << ", dP: " << simulation.get_dP() / simulation.get_dt_currentStep()
            << ", E: "
            << simulation.getTotalEnergy() / simulation.getInitialTotalEnergy()
            << ", time: "
            << convertTimeToHMS(program_end_time - program_start_time)
            << std::endl;
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

    //particles.readParticleCollisionCellIndexBuffer(kernels.getCommandQueue());
    //for (size_t i = 0; i < particles.getParticleCountTotal(); i++) {
    //  if (particles.returnCollisionCellIndex(i) != 0) {
    //    
    //    mean_free_path +=
    //        (particles.m_mean_free_path[i] - mean_free_path) / (numType)(mfp_count + 1);
    //    particles.m_mean_free_path[i] = 0.;
    //    mfp_count++;
    //  } else {
    //    particles.m_mean_free_path[i] += simulation.get_dt_currentStep();
    //  }
    //}
    //mfp_stream << i << ","<<mean_free_path << std::endl;

    // if (i % N_steps_tau >= 0) {
    if (i % 100 == 0) {
      streamer.stream(simulation, particles, bubble, log_scale_on,
                      kernels.getCommandQueue());
      program_end_time = my_clock::now();
      std::cout << "Step: " << simulation.getStep()
                << ", Time: " << simulation.getTime()
                << ", R: " << bubble.getRadius()
                << ", R/Rc: " << bubble.getRadius() / critical_radius
                << ", V: " << bubble.getSpeed() << ", dP: "
                << simulation.get_dP() / simulation.get_dt_currentStep()
                << ", E: "
                << simulation.getTotalEnergy() /
                       simulation.getInitialTotalEnergy()
                << ", time: "
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
      if ((bubble.getRadius() >= config.cyclicBoundaryRadius) &&
          (config.cyclicBoundaryOn)) {
        std::cerr
            << "Ending simulation. Bubble radius >= simulation boundary radius."
            << std::endl;
        break;
      }
    }
  }

  /*
    =============== End simulation ===============
  */

  // Stream last state
  streamer.stream(simulation, particles, bubble, log_scale_on,
                  kernels.getCommandQueue());

  // Measure runtime
  program_end_time = my_clock::now();
  auto ms_int = std::chrono::duration_cast<std::chrono::seconds>(
      program_end_time - program_start_time);

  appendSimulationInfoFile(infoStream, 0, (int)ms_int.count());
}

// std::cout << std::resetiosflags( std::cout.flags() );