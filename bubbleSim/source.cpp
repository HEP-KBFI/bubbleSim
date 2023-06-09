#define _CRT_SECURE_NO_WARNINGS
#include "source.h"

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

  // Collision is in development
  bool b_collisionDevelopment = true;

  // Read configs and kernel file
  std::string s_configPath = argv[1];  // "config.json"
  std::string s_kernelPath = argv[2];  // "kernel.cl";

  s_configPath = "D:\\dev\\bubbleSim\\configs\\test_collision.json";

  std::cout << "Config path: " << s_configPath << std::endl;
  std::cout << "Kernel path: " << s_kernelPath << std::endl;

  using std::chrono::duration;
  using std::chrono::duration_cast;
  using std::chrono::high_resolution_clock;
  using std::chrono::milliseconds;
  auto programStartTime = high_resolution_clock::now();
  ConfigReader config(s_configPath);
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

  // 3) Define required physical parameters
  numType temperatureFalse =
      std::sqrt(std::abs(std::pow(config.particleMassTrue, 2) -
                         std::pow(config.particleMassFalse, 2))) /
      config.parameterEta;

  numType temperatureTrue = 0;  // -> No particles generated in true vacuum,
                                // thus set temperature in true vacuum 0

  // 4) Generate particles
  ParticleGenerator particleGenerator1;
  // 4.1) Create generator (calculates distribution)
  if (!b_collisionDevelopment) {
    particleGenerator1 = ParticleGenerator(config.particleMassFalse,
                                           config.particleTemperatureFalse);
  } else {
    particleGenerator1 =
        ParticleGenerator(config.particleMassFalse, config.particleTemperatureFalse,
                          30 * temperatureFalse, 1e-5 * temperatureFalse);
  }
  // 4.2) Create arrays for particles which hold the data of the particles
  ParticleCollection particles(
      config.particleMassTrue, config.particleMassFalse, temperatureTrue,
      temperatureFalse, config.particleCountTrue, config.particleCountFalse,
      config.parameterCoupling, config.bubbleIsTrueVacuum,
      kernels.getContext());

  // 4.3) Generate particles
  numType genreatedParticleEnergy;
  if (b_collisionDevelopment) {
    genreatedParticleEnergy = particleGenerator1.generateNParticlesInBox(
        std::sqrt(3) * config.bubbleInitialRadius, (u_int)particles.getParticleCountTotal(),
        rn_generator, particles.getParticles());
  } else {
    genreatedParticleEnergy = particleGenerator1.generateNParticlesInSphere(
        config.bubbleInitialRadius, (u_int)particles.getParticleCountTotal(),
        rn_generator, particles.getParticles());
  }

  numType total_energy = 0;
  for (unsigned int i = 0; i < config.particleCountFalse; i++) {
    total_energy += particles.getParticleEnergy(i);
  }

  particles.add_to_total_initial_energy(genreatedParticleEnergy);

  // In development: step revert
  // particles.makeCopy();

  // 5) Initialize simulation object: controls one simulation step
  Simulation simulation;
  if (!config.cyclicBoundaryOn) {
    simulation = Simulation(config.m_seed, config.dt, kernels.getContext());
  } else {
    simulation = Simulation(config.m_seed, config.dt,
                            config.cyclicBoundaryRadius, kernels.getContext());
  }

  // 6) Define which openCL kernel to use. Kernel defines calculation process on
  // the GPU.
  //
  cl::Kernel* stepKernel;
  // In development: set kernel name in config file.
  // NB! Different kernels might need different input

  /*if (!config.bubbleInteractionsOn) {
    stepKernel = &kernels.m_particleStepKernel;
  } else if ((config.bubbleInteractionsOn) && (!config.cyclicBoundaryOn)) {
    stepKernel = &kernels.m_particleBubbleStepKernel;
  } else if ((config.bubbleInteractionsOn) && (config.cyclicBoundaryOn)) {
    stepKernel = &kernels.m_particleBubbleBoundaryStepKernel;
  } else {
    std::cerr << "Kernel for current configuration is not available"
              << std::endl;
    std::terminate();
  }*/
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
  CollisionCellCollection cells(config.collisionCellLength,
                                config.collisionCellCount, false,
                                kernels.getContext());
  if (b_collisionDevelopment) {
    simulation.set_particle_interaction_buffers(particles, cells,
                                                kernels.m_cellAssignmentKernel,
                                                kernels.m_rotationKernel);

    simulation.set_particle_step_buffers(particles, cells, *stepKernel);
    simulation.set_particle_bounce_buffers(particles, cells,
                                           kernels.m_particleBounceKernel);

  } else if (config.bubbleInteractionsOn) {  // Set up buffers for GPU
    simulation.set_particle_step_buffers(particles, bubble, *stepKernel);
  }

  // Copy data to the GPU
  particles.writeAllBuffersToKernel(kernels.getCommandQueue());
  simulation.writeAllBuffersToKernel(kernels.getCommandQueue());
  bubble.writeAllBuffersToKernel(kernels.getCommandQueue());
  if (b_collisionDevelopment) {
    cells.writeAllBuffersToKernel(kernels.getCommandQueue());
  }

  // 8) Streaming initialization
  std::string dataFolderName = createFileNameFromCurrentDate();

  std::filesystem::path filePath =
      createSimulationFilePath(config.m_dataSavePath, dataFolderName);
  
  DataStreamer streamer(filePath.string());
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
                                   config.maxValueMomentumIn);
  }
  if (config.streamMomentumOutOn) {
    streamer.initStream_MomentumOut(config.binsCountMomentumOut,
                                    config.maxValueMomentumOut);
  }
  streamer.initStream_Momentum(config.binsCountMomentumIn,
      config.maxValueMomentumIn);

  streamer.stream(simulation, particles, bubble, kernels.getCommandQueue());



  /*
   * =============== Display text ===============
   */
  std::cout << std::endl;
  config.print_info();
  std::cout << std::endl;
  particles.print_info(config, bubble);
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


  std::ofstream pLocStream("cellIdx.csv");
  std::vector<unsigned int> pLocArray;
  pLocArray.resize(config.collisionCellCount* config.collisionCellCount* config.collisionCellCount + 1, 0);
  /*
    =============== Run simulation ===============
  */

  std::array<int, 5> particleIdx = { 84522, 355324 , 429220, 538040, 704867 };
  for (int i : particleIdx) {
      particles.printParticleInfo(i);
  }


  numType simTimeSinceLastStream = 0.;
  int stepsSinceLastStream = 0;
  std::cout << "=============== Simulation ===============" << std::endl;
  std::cout << std::setprecision(6) << std::fixed << std::showpoint;
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
  double px = 0;
  double py = 0;
  double pz = 0;
  // auto streamEndTime = high_resolution_clock::now();
  // auto streamStartTime = high_resolution_clock::now();
  for (int i = 1; (simulation.getTime() <= config.maxTime); i++) {
    if (config.m_maxSteps > 0 && simulation.getStep() > config.m_maxSteps) {
      break;
    }
    if (b_collisionDevelopment) {
      simulation.step(particles, cells, rn_generator, i, *stepKernel,
                      kernels.m_cellAssignmentKernel, kernels.m_rotationKernel,
                      kernels.m_particleBounceKernel,
                      kernels.getCommandQueue());

      /*for (Particle& p : particles.getParticles()) {
          pLocArray[p.idxCollisionCell] += 1;
      }
      for (unsigned int loc : pLocArray) {
          pLocStream << loc << ",";
      }
      pLocStream << "\n";
      std::fill(pLocArray.begin(), pLocArray.end(), 0);*/
    } else {
      if (config.bubbleInteractionsOn) {
        simulation.step(particles, bubble, *stepKernel,
                        kernels.getCommandQueue());
      } else {
        simulation.step(bubble, 0);
      }
    }
    simTimeSinceLastStream += simulation.get_dt_currentStep();
    stepsSinceLastStream += 1;

    if (config.streamStep > 0) {
      if (stepsSinceLastStream == config.streamStep) {
        if (config.streamOn) {
          streamer.stream(simulation, particles, bubble,
                          kernels.getCommandQueue());
          stepsSinceLastStream = 0;
        }
      }
    }

    if (simTimeSinceLastStream >= config.streamTime) {
      // streamEndTime = high_resolution_clock::now();

      std::cout << std::setprecision(6) << std::fixed << std::showpoint;

      std::cout << "Step: " << simulation.getStep()
                << ", Time: " << simulation.getTime()
                << ", R: " << bubble.getRadius() << ", V: " << bubble.getSpeed()
                << ", C/C0: "
                << (simulation.getTotalEnergy() / bubble.getRadius()) /
                       simulation.getInitialCompactnes()
                << ", dP: "
                << simulation.get_dP() / simulation.get_dt_currentStep()
                << ", E: "
                << simulation.getTotalEnergy() /
                       simulation.getInitialTotalEnergy()
                << std::endl;

      /*std::cout << "Time taken (steps): "
                << std::chrono::duration_cast<std::chrono::milliseconds>(
                       streamEndTime - streamStartTime)
                       .count()
                << " ms." << std::endl;*/

      if (config.streamOn) {
        streamer.stream(simulation, particles, bubble,
                        kernels.getCommandQueue());
      }
      simTimeSinceLastStream = 0.;
      stepsSinceLastStream = 0;
      // streamStartTime = high_resolution_clock::now();
    }

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
  }

  /*
    =============== End simulation ===============
  */

  // Stream last state
  streamer.stream(simulation, particles, bubble, kernels.getCommandQueue());

  // Measure runtime
  auto programEndTime = high_resolution_clock::now();
  auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(
      programEndTime - programStartTime);
  int seconds = (int)(ms_int.count() / 1000) % 60;
  int minutes = ((int)(ms_int.count() / (1000 * 60)) % 60);
  int hours = ((int)(ms_int.count() / (1000 * 60 * 60)) % 24);

  appendSimulationInfoFile(infoStream, 0, (int)ms_int.count());

  std::cout << std::endl
            << "Program run: " << hours << "h " << minutes << "m " << seconds
            << "s " << std::endl;
}
