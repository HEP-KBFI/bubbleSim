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

void createSimulationInfoFile(std::fstream& infoStream,
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
void appendSimulationInfoFile(std::fstream& infoStream, int t_postionDifference,
                              int t_programRuntime) {
  infoStream << t_postionDifference << "," << t_programRuntime << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: bubbleSim.exe config.json kernel.cl" << std::endl;
    exit(0);
  }

  bool b_collisionDevelopment = false;

  std::string s_configPath = argv[1];  // "config.json"
  std::string s_kernelPath = argv[2];  // "kernel.cl";

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
    === TODO ===
 */

  // 1) Initialize random number generators
  RandomNumberGenerator rn_generator(config.m_seed);

  // 2) Initialize openCL
  OpenCLLoader kernels(s_kernelPath);
  // 3) Define required physical parameters
  /*
  alpha = dV/rho, eta = sqrt(M+^2 - m-^2)/T, Upsilon = sigma/(dV * R_0)

  Define M+ -> Get T -> Get rho -> get dV -> get sigma
  M+ is defined in a config
  */
  numType temperatureFalse = std::sqrt(std::pow(config.particleMassTrue, 2) -
                                       std::pow(config.particleMassFalse, 2)) /
                             config.parameterEta;
  numType temperatureTrue = 0;  // -> No particles generated in true vacuum

  // 4) Generate particles

  ParticleGenerator particleGenerator1;
  if (b_collisionDevelopment) {
    particleGenerator1 = ParticleGenerator(config.particleMassFalse, 3.);
  } else {
    particleGenerator1 =
        ParticleGenerator(config.particleMassFalse, temperatureFalse,
                          30 * temperatureFalse, 1e-5 * temperatureFalse);
  }

  ParticleCollection particles(
      config.particleMassTrue, config.particleMassFalse, temperatureTrue,
      temperatureFalse, config.particleCountTrue, config.particleCountFalse,
      config.parameterCoupling, config.bubbleIsTrueVacuum,
      kernels.getContext());

  numType genreatedParticleEnergy;
  if (b_collisionDevelopment) {
    genreatedParticleEnergy = particleGenerator1.generateNParticlesInBox(
        config.bubbleInitialRadius, (u_int)particles.getParticleCountTotal(),
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

  // 5) Initialize simulation object

  Simulation simulation;
  numType cycleBoundaryRadius;
  if (!config.cyclicBoundaryOn) {
    simulation = Simulation(config.m_seed, config.dt, kernels.getContext());
  } else {
    cycleBoundaryRadius = 2 * config.bubbleInitialRadius;
    simulation = Simulation(config.m_seed, config.dt, cycleBoundaryRadius,
                            kernels.getContext());
  }

  cl::Kernel* stepKernel;
  if (!config.bubbleInteractionsOn) {
    stepKernel = &kernels.m_particleStepKernel;
  } else if ((config.bubbleInteractionsOn) && (!config.cyclicBoundaryOn)) {
    stepKernel = &kernels.m_particleBubbleStepKernel;
  } else if ((config.bubbleInteractionsOn) && (config.cyclicBoundaryOn)) {
    stepKernel = &kernels.m_particleBubbleBoundaryStepKernel;
  } else {
    std::cerr << "Kernel for current configuration is not available"
              << std::endl;
    std::terminate();
  }

  // 6) Initialize bubble object

  numType bubbleVolume = 4 * M_PI / 3 * std::pow(config.bubbleInitialRadius, 3);
  numType initialEnergyDensityFalse =
      particles.countParticlesEnergy() / bubbleVolume;
  numType initialNumberDensityFalse =
      particles.getParticleCountFalse() / bubbleVolume;
  /*
  numType Tn = initialNumberDensityFalse * temperatureFalse;            // > 0
  numType dV = config.parameterAlpha * initialEnergyDensityFalse + Tn;  // > 0
  */

  numType dV = config.parameter_dV;
  numType alpha = config.parameterAlpha;

  numType sigma = config.parameterUpsilon * dV * config.bubbleInitialRadius;

  if (!config.bubbleIsTrueVacuum) {
    // If bubble is true vacuum then dV is correct. Otherwise dV sign must be
    // channged as change of direction changes
    dV = -dV;
  }
  PhaseBubble bubble(config.bubbleInitialRadius, config.bubbleInitialSpeed, dV,
                     sigma, kernels.getContext());

  CollisionCellCollection cells(config.collisionCellLength,
                                config.collisionCellCount, false,
                                kernels.getContext());

  simulation.set_particle_interaction_buffers(particles, cells,
                                              kernels.m_cellAssignmentKernel,
                                              kernels.m_rotationKernel);
  if (b_collisionDevelopment) {
    simulation.set_particle_step_buffers(particles, cells,
                                         kernels.m_particleStepKernel);
  } else if (config.bubbleInteractionsOn) {
    simulation.set_particle_step_buffers(particles, bubble, *stepKernel);
  }
  simulation.set_particle_bounce_buffers(particles, cells,
                                         kernels.m_particleBounceKernel);

  // Add all energy together

  simulation.addTotalEnergy(particles.getInitialTotalEnergy());
  simulation.addTotalEnergy(bubble.calculateEnergy());

  // 8) Streaming object

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

  particles.writeAllBuffersToKernel(kernels.getCommandQueue());
  simulation.writeAllBuffersToKernel(kernels.getCommandQueue());
  bubble.writeAllBuffersToKernel(kernels.getCommandQueue());
  cells.writeAllBuffersToKernel(kernels.getCommandQueue());

  std::string dataFolderName = createFileNameFromCurrentDate();

  std::filesystem::path filePath =
      createSimulationFilePath(config.m_dataSavePath, dataFolderName);
  DataStreamer streamer(filePath.string());

  if (config.streamDataOn) {
    streamer.initData();
  }
  if (config.streamDensityOn) {
    streamer.initDensityProfile(config.streamDensityBinsCount,
                                config.bubbleInitialRadius * 2 * std::sqrt(3));
    // If cyclic condition is set to 2*R_b then max distance is sqrt(3) 2 * R_b
  }
  if (config.streamMomentumOn) {
    streamer.initMomentumProfile(config.streamMomentumBinsCount, 30);
  }
  streamer.stream(simulation, particles, bubble, kernels.getCommandQueue());

  std::fstream infoStream(filePath / "info.txt",
                          std::ios::out | std::ios::trunc);
  std::cout << std::setprecision(8) << std::endl;
  createSimulationInfoFile(infoStream, filePath, config, dV,
                           initialNumberDensityFalse,
                           initialEnergyDensityFalse);

  // Energy, p_x, p_y, p_z, p^2
  std::array<double, 5> previous = {0., 0., 0., 0., 0.};
  std::array<double, 5> current = {0., 0., 0., 0., 0.};
  if (b_collisionDevelopment) {
    for (Particle p : particles.getParticles()) {
      current[0] += p.E;
      current[1] += p.p_x;  // std::abs(p.p_x);
      current[2] += p.p_y;  // std::abs(p.p_y);
      current[3] += p.p_z;  // std::abs(p.p_z);
      current[4] += std::sqrt(
          std::fma(p.p_x, p.p_x, std::fma(p.p_y, p.p_y, p.p_z * p.p_z)));
    }
    previous[0] = current[0];
    previous[1] = current[1];
    previous[2] = current[2];
    previous[3] = current[3];
    previous[4] = current[4];

    std::cout << std::setprecision(10) << "Current energy: " << current[0]
              << ", delta: " << current[0] - previous[0] << std::endl;
    std::cout << "Current p1: " << current[1]
              << ", delta: " << current[1] - previous[1] << std::endl;
    std::cout << "Current p2: " << current[2]
              << ", delta: " << current[2] - previous[2] << std::endl;
    std::cout << "Current p3: " << current[3]
              << ", delta: " << current[3] - previous[3] << std::endl;
    std::cout << "Current p: " << current[4]
              << ", delta: " << current[4] - previous[4] << std::endl
              << std::endl;
  }
  std::cout << "Time: " << simulation.getTime() << ", R: " << bubble.getRadius()
            << ", V: " << bubble.getSpeed() << ", dP: " << simulation.get_dP()
            << std::endl;

  for (int i = 1; i <= config.m_maxSteps; i++) {
    /*
      simulation.step(bubble, 0);
    }*/
    if (b_collisionDevelopment) {
      simulation.step(particles, cells, rn_generator, i, *stepKernel,
                      kernels.m_cellAssignmentKernel, kernels.m_rotationKernel,
                      kernels.m_particleBounceKernel,
                      kernels.getCommandQueue());

      particles.readParticlesBuffer(kernels.getCommandQueue());
      current[0] = 0.;
      current[1] = 0.;
      current[2] = 0.;
      current[3] = 0.;
      current[4] = 0.;
      for (Particle p : particles.getParticles()) {
        current[0] += p.E;
        current[1] += p.p_x;  // std::abs(p.p_x);
        current[2] += p.p_y;  // std::abs(p.p_y);
        current[3] += p.p_z;  // std::abs(p.p_z);
        current[4] += std::sqrt(
            std::fma(p.p_x, p.p_x, std::fma(p.p_y, p.p_y, p.p_z * p.p_z)));
      }
      std::cout << std::setprecision(10) << "Current energy: " << current[0]
                << ", delta: " << current[0] - previous[0] << std::endl;
      std::cout << "Current p1: " << current[1]
                << ", delta: " << current[1] - previous[1] << std::endl;
      std::cout << "Current p2: " << current[2]
                << ", delta: " << current[2] - previous[2] << std::endl;
      std::cout << "Current p3: " << current[3]
                << ", delta: " << current[3] - previous[3] << std::endl;
      std::cout << "Current p: " << current[4]
                << ", delta: " << current[4] - previous[4] << std::endl
                << std::endl;
      previous[0] = current[0];
      previous[1] = current[1];
      previous[2] = current[2];
      previous[3] = current[3];
      previous[4] = current[4];
    } else {
      if (config.bubbleInteractionsOn) {
        simulation.step(particles, bubble, *stepKernel,
                        kernels.getCommandQueue());
      } else {
        simulation.step(bubble, 0);
      }
    }

    if (i % config.streamFreq == 0) {
      std::cout << std::setprecision(10) << std::fixed << std::showpoint;
      std::cout << "Time: " << simulation.getTime()
                << ", R: " << bubble.getRadius() << ", V: " << bubble.getSpeed()
                << ", dP: " << simulation.get_dP() << std::endl;
      if (config.streamOn) {
        streamer.stream(simulation, particles, bubble,
                        kernels.getCommandQueue());
      }
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
