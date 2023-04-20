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
  OpenCLLoader kernels(s_kernelPath, config.kernelName);
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
  particles.makeCopy();
  // 5) Initialize simulation object
  Simulation simulation;
  if (!config.cyclicBoundaryOn) {
    simulation = Simulation(config.m_seed, config.dt, kernels.getContext());
  } else {
    simulation = Simulation(config.m_seed, config.dt,
                            config.cyclicBoundaryRadius, kernels.getContext());
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
  stepKernel = &kernels.m_kernel;

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

  simulation.addInitialTotalEnergy(particles.getInitialTotalEnergy());
  simulation.addInitialTotalEnergy(bubble.calculateEnergy());

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

  streamer.stream(simulation, particles, bubble, kernels.getCommandQueue());

  std::ofstream infoStream(filePath / "info.txt",
                           std::ios::out | std::ios::trunc);
  std::cout << std::setprecision(8) << std::endl;
  createSimulationInfoFile(infoStream, filePath, config, dV,
                           initialNumberDensityFalse,
                           initialEnergyDensityFalse);

  std::ofstream radialMomentumStream(filePath / "radialMomentum.csv",
                                     std::ios::out | std::ios::trunc);

  // Energy, p_x, p_y, p_z, p^2
  std::array<double, 5> previous = {0., 0., 0., 0., 0.};
  std::array<double, 5> current = {0., 0., 0., 0., 0.};

  numType timeSinceLastStream = 0.;

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

  size_t counter = 0;
  std::vector<numType> fullStreamTime = {0.52396,
                                         1.04792,
                                         2.22159,
                                         3.39526,
                                         3.975625,
                                         4.55599,
                                         5.126620000000001,
                                         5.69725,
                                         6.27995,
                                         6.86265,
                                         7.435350000000001,
                                         8.00805,
                                         8.603480000000001,
                                         9.19891,
                                         9.784604999999999,
                                         10.3703,
                                         10.95995,
                                         11.5496,
                                         12.136700000000001,
                                         12.7238,
                                         13.3084,
                                         13.893,
                                         14.459900000000001,
                                         15.0268,
                                         15.6161,
                                         16.2054,
                                         16.78845,
                                         17.3715,
                                         17.95845,
                                         18.5454,
                                         19.130000000000003,
                                         19.7146,
                                         20.30705,
                                         20.8995,
                                         21.4962,
                                         22.0929,
                                         22.6672,
                                         23.2415,
                                         23.8065,
                                         24.3715,
                                         24.96735,
                                         25.5632,
                                         26.148249999999997,
                                         26.7333,
                                         27.3245,
                                         27.9157,
                                         28.50335,
                                         29.091,
                                         29.677950000000003,
                                         30.2649,
                                         30.85695,
                                         31.449,
                                         32.01965,
                                         32.5903,
                                         33.175399999999996,
                                         33.7605,
                                         34.35395,
                                         34.9474,
                                         35.5373,
                                         36.1272,
                                         36.71565,
                                         37.3041,
                                         37.89875,
                                         38.4934,
                                         39.06755,
                                         39.6417,
                                         40.21695,
                                         40.7922,
                                         41.39105,
                                         41.9899,
                                         42.576750000000004,
                                         43.1636,
                                         43.755449999999996,
                                         44.3473,
                                         44.93125,
                                         45.5152,
                                         46.1044,
                                         46.6936,
                                         47.2743,
                                         47.855,
                                         48.4486,
                                         49.0422,
                                         49.62495,
                                         50.2077,
                                         50.8043,
                                         51.4009,
                                         51.99535,
                                         52.5898,
                                         53.174899999999994,
                                         53.76,
                                         54.340149999999994,
                                         54.9203,
                                         55.5141,
                                         56.1079,
                                         56.6999,
                                         57.2919,
                                         57.88375,
                                         58.4756,
                                         59.067750000000004,
                                         59.6599,
                                         60.24275,
                                         60.8256,
                                         61.40495,
                                         61.9843,
                                         62.585300000000004,
                                         63.1863,
                                         63.76695,
                                         64.3476,
                                         64.94980000000001,
                                         65.552,
                                         66.14875,
                                         66.7455,
                                         67.3294,
                                         67.9133,
                                         68.50285,
                                         69.0924,
                                         69.681,
                                         70.2696,
                                         70.8702,
                                         71.4708,
                                         72.05735,
                                         72.6439,
                                         73.2438,
                                         73.8437,
                                         74.4267,
                                         75.0097,
                                         75.5901,
                                         76.1705,
                                         76.77355,
                                         77.3766,
                                         77.9667,
                                         78.5568,
                                         79.1522,
                                         79.7476,
                                         80.3483,
                                         80.949,
                                         81.53495000000001,
                                         82.1209,
                                         82.7061,
                                         83.2913,
                                         83.89,
                                         84.4887,
                                         85.07714999999999,
                                         85.6656,
                                         86.2646,
                                         86.8636,
                                         87.46685,
                                         88.0701,
                                         88.6566,
                                         89.2431,
                                         89.8309,
                                         90.4187,
                                         91.0137,
                                         91.6087,
                                         92.2045,
                                         92.8003,
                                         93.39599999999999,
                                         93.9917,
                                         94.5935,
                                         95.1953,
                                         95.78415000000001,
                                         96.373,
                                         96.9545,
                                         97.536,
                                         98.14065,
                                         98.7453,
                                         99.33855,
                                         99.9318,
                                         100.5334,
                                         101.135,
                                         101.729,
                                         102.323,
                                         102.9185,
                                         103.514,
                                         104.108,
                                         104.702,
                                         105.2975,
                                         105.893,
                                         106.4925,
                                         107.092,
                                         107.6875,
                                         108.283,
                                         108.888,
                                         109.493,
                                         110.0825,
                                         110.672,
                                         111.2635,
                                         111.855,
                                         112.4535,
                                         113.052,
                                         113.6495,
                                         114.247,
                                         114.8475,
                                         115.448,
                                         116.049,
                                         116.65,
                                         117.2425,
                                         117.835,
                                         118.4285,
                                         119.022,
                                         119.622,
                                         120.222,
                                         120.826,
                                         121.43,
                                         122.027,
                                         122.624,
                                         123.225,
                                         123.826,
                                         124.4225,
                                         125.019,
                                         125.612,
                                         126.205,
                                         126.8065,
                                         127.408,
                                         128.008,
                                         128.608,
                                         129.2105,
                                         129.813,
                                         130.41899999999998,
                                         131.025,
                                         131.6165,
                                         132.208,
                                         132.809,
                                         133.41,
                                         134.00799999999998,
                                         134.606,
                                         135.20999999999998,
                                         135.814,
                                         136.413,
                                         137.012,
                                         137.6};
  streamer.StreamRadialMomentumProfile(radialMomentumStream, 200, 500, 0,
                                       bubble.getRadius(), 0, 0.2, particles,
                                       kernels.getCommandQueue());
  /*
   *
   *
   *
   * Run simulation
   *
   *
   *
   */

  auto streamEndTime = high_resolution_clock::now();
  auto streamStartTime = high_resolution_clock::now();
  for (int i = 1; simulation.getTime() <= config.maxTime; i++) {
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
    timeSinceLastStream += simulation.get_dt_currentStep();

    if (counter < 231) {
      if (simulation.getTime() > fullStreamTime[counter]) {
        streamer.StreamRadialMomentumProfile(
            radialMomentumStream, 200, 500, 0, bubble.getRadius(), 0, 0.2,
            particles, kernels.getCommandQueue());
        counter += 1;
      }
    }

    if (timeSinceLastStream >= config.streamTime) {
      streamEndTime = high_resolution_clock::now();

      std::cout << std::setprecision(5) << std::fixed << std::showpoint;
      std::cout << "Time: " << simulation.getTime()
                << ", R: " << bubble.getRadius() << ", V: " << bubble.getSpeed()
                << ", dP: " << simulation.get_dP() << std::endl;
      std::cout << "Time taken (steps): "
                << std::chrono::duration_cast<std::chrono::milliseconds>(
                       streamEndTime - streamStartTime)
                       .count()
                << " ms." << std::endl;

      if (config.streamOn) {
        streamer.stream(simulation, particles, bubble,
                        kernels.getCommandQueue());
      }
      timeSinceLastStream = 0.;
      streamStartTime = high_resolution_clock::now();
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
   *
   *
   *
   * End simualtion
   *
   *
   *
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
