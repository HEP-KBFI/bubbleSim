#define _CRT_SECURE_NO_WARNINGS
#include "source.h"

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

// Collision is in development
std::string createFileNameFromCurrentDate() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_int_distribution<> dis(0, 15);
  static std::uniform_int_distribution<> dis2(8, 11);

  char c_name[18];

  int i;
  std::stringstream ss;
  std::string result;

  std::time_t t_cur = time(NULL);
  const auto* t_local = localtime(&t_cur);
  std::strftime(c_name, 18, "%j_%Y_%H_%M_%S", t_local);

  ss << std::hex;
  for (i = 0; i < 8; i++) {
    ss << dis(gen);
  }
  result = std::string(c_name) + "_" + ss.str();
  return result;
}

std::filesystem::path createSimulationFilePath(std::string& t_dataPath,
                                               std::string& t_fileName) {
  std::filesystem::path dataPath(t_dataPath);
  std::filesystem::path filePath = dataPath / t_fileName;
  if (!std::filesystem::is_directory(dataPath) ||
      !std::filesystem::exists(dataPath)) {       // Check if src folder exists
    std::filesystem::create_directory(dataPath);  // create src folder
  }
  if (!std::filesystem::is_directory(filePath) ||
      !std::filesystem::exists(filePath)) {       // Check if src folder exists
    std::filesystem::create_directory(filePath);  // create src folder
  }
  return filePath;
}

void createSimulationInfoFile(std::ofstream& infoStream,
                              std::filesystem::path& filePath,
                              ConfigReader& t_config, numType t_dV, numType t_n,
                              numType t_rho) {
  infoStream << "file_name,seed,alpha,eta,upsilon,coupling,radius,speed,m-,m+,"
                "T-,T+,N-,N+,dt,interactionBubbleOn,cyclicBoundaryOn,"
                "dV,n,rho,deltaN,runtime"
             << std::endl;
  infoStream << filePath.filename() << "," << t_config.m_seed << ","
             << t_config.parameterAlpha << "," << t_config.parameterEta << ",";
  infoStream << t_config.parameterUpsilon << "," << t_config.parameterCoupling
             << ",";
  infoStream << t_config.bubbleInitialRadius << ","
             << t_config.bubbleInitialSpeed << ",";
  infoStream << t_config.particleMassFalse << "," << t_config.particleMassTrue
             << ",";
  infoStream << t_config.particleTemperatureFalse << ","
             << t_config.particleTemperatureTrue << ",";
  infoStream << t_config.particleCountFalse << "," << t_config.particleCountTrue
             << ",";
  infoStream << t_config.dt << "," << t_config.bubbleInteractionsOn << ","
             << t_config.cyclicBoundaryOn << ",";
  infoStream << t_dV << "," << t_n << "," << t_rho << ",";
}
void appendSimulationInfoFile(std::ofstream& infoStream,
                              int t_postionDifference, int t_programRuntime) {
  infoStream << t_postionDifference << "," << t_programRuntime << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: bubbleSim.exe config.json kernel.cl" << std::endl;
    exit(0);
  }
  auto program_start_time = high_resolution_clock::now();
  // auto program_second_time = high_resolution_clock::now();

  // Read configs and kernel file
  std::string s_configPath = argv[1];  // "config.json"
  std::string s_kernelPath = argv[2];  // "kernel.cl";

  std::cout << "Config path: " << s_configPath << std::endl;
  std::cout << "Kernel path: " << s_kernelPath << std::endl;

  ConfigReader config(s_configPath);

  numType tau = config.parameterTau;
  u_int N_steps_tau = 10;
  numType dt = tau / N_steps_tau;
  u_int sim_length_in_tau = 10;
  /*
          ===============  ===============
  */
  // If seed = 0 then it generates random seed.

  /*
    =============== Initialization ===============
 */

  // 1) Initialize random number generators
  RandomNumberGenerator rn_generator(config.m_seed);

  // 2) Initialize openCL (kernels, commandQueues)
  OpenCLLoader kernels(s_kernelPath, config.kernelName);

  // 4) Generate particles
  ParticleGenerator particleGenerator1;
  // 4.1) Create generator (calculates distribution)

  particleGenerator1 = ParticleGenerator(config.particleMassFalse);
  // particleGenerator1.calculateCPDDelta(config.particleTemperatureFalse);
  particleGenerator1.calculateCPDBoltzmann(
      config.particleTemperatureFalse, 30 * config.particleTemperatureFalse,
      1e-5 * config.particleTemperatureFalse);

  // 4.2) Create arrays for particles which hold the data of the particles
  ParticleCollection particles(
      config.particleMassTrue, config.particleMassFalse,
      config.particleTemperatureTrue, config.particleTemperatureFalse,
      config.particleCountTrue, config.particleCountFalse,
      config.parameterCoupling, config.bubbleIsTrueVacuum,
      kernels.getContext());

  // 4.3) Generate particles
  numType genreatedParticleEnergy;

  /* genreatedParticleEnergy = particleGenerator1.generateNParticlesInSphere(
       1., (u_int)(particles.getParticleCountTotal() * 1), rn_generator,
       particles);*/

  genreatedParticleEnergy = particleGenerator1.generateNParticlesInSphere(
      1., 2., (u_int)(particles.getParticleCountTotal()), rn_generator,
      particles);

 
  particles.add_to_total_initial_energy(genreatedParticleEnergy);
  // In development: step revert
  // particles.makeCopy();

  // 5) Initialize simulation object: controls one simulation step
  Simulation simulation;
  if (!config.cyclicBoundaryOn) {
    simulation = Simulation(config.m_seed, dt, kernels.getContext());
  } else {
    simulation = Simulation(config.m_seed, dt, config.cyclicBoundaryRadius,
                            kernels.getContext());
  }
  simulation.setTau(config.parameterTau);

  cl::Kernel* stepKernel;
  // In development: set kernel name in config file.
  // NB! Different kernels might need different input

  stepKernel = &kernels.m_kernel;

  // 7) Initialize bubble object
  numType bubbleVolume = 4 * M_PI / 3 * std::pow(config.bubbleInitialRadius, 3);
  numType initialEnergyDensityFalse =
      particles.countParticlesEnergy() / bubbleVolume;
  numType initialNumberDensityFalse =
      particles.getParticleCountFalse() / bubbleVolume;

  numType dV = config.parameter_dV;
  numType alpha = config.parameterAlpha;

  numType sigma = config.parameterUpsilon * dV * config.bubbleInitialRadius;
  if (!config.bubbleIsTrueVacuum) {
    // If bubble is true vacuum then dV is correct sign. Otherwise dV sign must
    // be changed as change of direction changes.
    dV = -dV;
  }
  PhaseBubble bubble(config.bubbleInitialRadius, config.bubbleInitialSpeed, dV,
                     sigma, kernels.getContext());

  // Add all energy together in simulation for later use
  simulation.addInitialTotalEnergy(particles.getInitialTotalEnergy());
  simulation.addInitialTotalEnergy(bubble.calculateEnergy());
  simulation.setInitialCompactness(simulation.getInitialTotalEnergy() /
                                   bubble.getInitialRadius());

  // In development: collision
  CollisionCellCollection cells(config.collision_cell_length,
                                config.collision_cell_count, false,
                                kernels.getContext());

  if (config.bubbleInteractionsOn) {
    simulation.setBuffersParticleStepWithBubble(particles, bubble, *stepKernel);

  } else {
    simulation.setBuffersParticleStepLinear(particles, *stepKernel);
  }
  if (config.collision_on) {
    simulation.setBuffersParticleBoundaryCheck(particles,
                                               kernels.m_particleBounceKernel);
    simulation.setBuffersAssignParticleToCollisionCell(
        particles, cells, kernels.m_cellAssignmentKernel);
    simulation.setBuffersRotateMomentum(particles, cells,
                                        kernels.m_rotationKernel);
    cells.writeAllBuffersToKernel(kernels.getCommandQueue());
  }

  // Copy data to the GPU
  particles.writeParticleCoordinatesBuffer(kernels.getCommandQueue());
  particles.writeParticleEBuffer(kernels.getCommandQueue());
  particles.writeParticleMomentumsBuffer(kernels.getCommandQueue());
  particles.writeParticleMBuffer(kernels.getCommandQueue());
  simulation.writeAllBuffersToKernel(kernels.getCommandQueue());
  bubble.writeAllBuffersToKernel(kernels.getCommandQueue());

  // 8) Streaming initialization
  std::string dataFolderName = createFileNameFromCurrentDate();

  std::filesystem::path filePath =
      createSimulationFilePath(config.m_dataSavePath, dataFolderName);
  DataStreamer streamer(filePath.string());

  bool log_scale_on = true;

  if (config.streamDataOn) {
    streamer.initStream_Data();
  }
  if (config.streamRadialVelocityOn) {
    streamer.initStream_RadialVelocity(config.binsCountRadialVelocity,
                                       config.maxValueRadialVelocity);
  }
  if (config.streamTangentialVelocityOn) {
    streamer.initStream_TangentialVelocity(config.binsCountTangentialVelocity,
                                           config.maxValueTangentialVelocity);
  }
  if (config.streamDensityOn) {
    streamer.initStream_Density(config.binsCountDensity,
                                config.maxValueDensity);
  }
  if (config.streamEnergyOn) {
    streamer.initStream_EnergyDensity(config.binsCountEnergy,
                                      config.maxValueEnergy);
  }
  if (config.streamMomentumInOn) {
    streamer.initStream_MomentumIn(config.binsCountMomentumIn,
                                   config.minValueMomentumIn,
                                   config.maxValueMomentumIn, log_scale_on);
  }
  if (config.streamMomentumOutOn) {
    streamer.initStream_MomentumOut(config.binsCountMomentumOut,
                                    config.minValueMomentumOut,
                                    config.maxValueMomentumOut, log_scale_on);
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

  createSimulationInfoFile(infoStream, filePath, config, dV,
                           initialNumberDensityFalse,
                           initialEnergyDensityFalse);

  /*
    =============== Run simulation ===============
  */

  numType simTimeSinceLastStream = 0.;
  int stepsSinceLastStream = 0;
  std::cout << "=============== Simulation ===============" << std::endl;
  std::cout << std::setprecision(6) << std::fixed << std::showpoint;
  auto program_second_time = high_resolution_clock::now();
  std::cout << "Step: " << simulation.getStep()
            << ", Time: " << simulation.getTime()
            << ", R: " << bubble.getRadius() << ", V: " << bubble.getSpeed()
            << ", C/C0: "
            << (simulation.getTotalEnergy() / bubble.getRadius()) /
                   simulation.getInitialCompactnes()
            << ", dP: " << simulation.get_dP() / simulation.get_dt_currentStep()
            << ", E: "
            << simulation.getTotalEnergy() / simulation.getInitialTotalEnergy()
            << std::endl;

  // auto streamEndTime = high_resolution_clock::now();
  // auto streamStartTime = high_resolution_clock::now();
  /*for (int i = 1;
       (simulation.getTime() <= config.maxTime) &&
       (config.m_maxSteps > 0 && simulation.getStep() < config.m_maxSteps);
       i++)
  */
  for (u_int i = 1; i <= N_steps_tau * sim_length_in_tau; i++) {
    if (config.bubbleInteractionsOn) {
      if (config.collision_on) {
        simulation.stepParticleBubbleCollisionBoundary(
            particles, bubble, cells, rn_generator, *stepKernel,
            kernels.m_particleBounceKernel, kernels.m_cellAssignmentKernel,
            kernels.m_rotationKernel, kernels.getCommandQueue());
      } else {
        if (config.cyclicBoundaryOn) {
          simulation.stepParticleBubbleBoundary(particles, bubble, *stepKernel,
                                                kernels.m_particleBounceKernel,
                                                kernels.getCommandQueue());
        } else {
          simulation.stepParticleBubble(particles, bubble, *stepKernel,
                                        kernels.getCommandQueue());
        }
      }
    } else {
      if (config.collision_on) {
        simulation.stepParticleCollisionBoundary(
            particles, cells, rn_generator, *stepKernel,
            kernels.m_particleBounceKernel, kernels.m_cellAssignmentKernel,
            kernels.m_rotationKernel, kernels.getCommandQueue());
      } else {
        simulation.step(bubble, 0);
      }
    }

    /*simTimeSinceLastStream += simulation.get_dt_currentStep();
    stepsSinceLastStream += 1;*/

    /*if (config.streamStep > 0) {
      if (stepsSinceLastStream == config.streamStep) {
        if (config.streamOn) {
          streamer.stream(simulation, particles, bubble,
                          kernels.getCommandQueue());
          stepsSinceLastStream = 0;
        }
      }
    }*/
    if (i % N_steps_tau == 0) {
      streamer.stream(simulation, particles, bubble, log_scale_on,
                      kernels.getCommandQueue());
        std::cout << "Step: " << simulation.getStep()
                  << ", Time: " << simulation.getTime()
                  << ", R: " << bubble.getRadius() << ", V: " <<
                  bubble.getSpeed()
                  << ", C/C0: "
                  << (simulation.getTotalEnergy() / bubble.getRadius()) /
                         simulation.getInitialCompactnes()
                  << ", dP: "
                  << simulation.get_dP() / simulation.get_dt_currentStep()
                  << ", E: "
                  << simulation.getTotalEnergy() /
                         simulation.getInitialTotalEnergy()
                  << std::endl;

    }

    // if (simTimeSinceLastStream >= config.streamTime) {
    //   // streamEndTime = high_resolution_clock::now();

    //  std::cout << std::setprecision(6) << std::fixed << std::showpoint;
    //  program_second_time = high_resolution_clock::now();

    //  std::cout << "Step: " << simulation.getStep()
    //            << ", Time: " << simulation.getTime()
    //            << ", R: " << bubble.getRadius() << ", V: " <<
    //            bubble.getSpeed()
    //            << ", C/C0: "
    //            << (simulation.getTotalEnergy() / bubble.getRadius()) /
    //                   simulation.getInitialCompactnes()
    //            << ", dP: "
    //            << simulation.get_dP() / simulation.get_dt_currentStep()
    //            << ", E: "
    //            << simulation.getTotalEnergy() /
    //                   simulation.getInitialTotalEnergy()
    //            << std::endl;

    //  /*std::cout << "Time taken (steps): "
    //            << std::chrono::duration_cast<std::chrono::milliseconds>(
    //                   streamEndTime - streamStartTime)
    //                   .count()
    //            << " ms." << std::endl;*/

    //  if (config.streamOn) {
    //    streamer.stream(simulation, particles, bubble,
    //                    kernels.getCommandQueue());
    //  }
    //  simTimeSinceLastStream = 0.;
    //  stepsSinceLastStream = 0;
    //  // streamStartTime = high_resolution_clock::now();
    //}

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

  /*
    =============== End simulation ===============
  */

  // Stream last state
  streamer.stream(simulation, particles, bubble, log_scale_on,
                  kernels.getCommandQueue());

  // Measure runtime
  auto programEndTime = high_resolution_clock::now();
  auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(
      programEndTime - program_start_time);

  appendSimulationInfoFile(infoStream, 0, (int)ms_int.count());
}

// std::cout << std::resetiosflags( std::cout.flags() );