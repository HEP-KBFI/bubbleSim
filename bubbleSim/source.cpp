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
                "T-,T+,N-,N+,dt,interactionOn,dV,n,rho,deltaN,runtime"
             << std::endl;
  infoStream << filePath.filename() << "," << t_config.m_seed << ","
             << t_config.m_alpha << "," << t_config.m_eta << ",";
  infoStream << t_config.m_upsilon << "," << t_config.m_coupling << ",";
  infoStream << t_config.m_initialBubbleRadius << ","
             << t_config.m_initialBubbleSpeed << ",";
  infoStream << t_config.m_massFalse << "," << t_config.m_massTrue << ",";
  infoStream << t_config.m_temperatureFalse << "," << t_config.m_temperatureTrue
             << ",";
  infoStream << t_config.m_countParticlesFalse << ","
             << t_config.m_countParticlesTrue << ",";
  infoStream << t_config.m_dt << "," << t_config.m_interactionsOn << ",";
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

  std::string configPath = argv[1];
  std::string kernelPath = argv[2];  //"kernel.cl";

  std::array<std::string, 4> collisionKernelNames = {
      "assign_cell_index_to_particle", "label_particles_position_by_coordinate",
      "label_particles_position_by_mass", "transform_momentum"};

  std::cout << "Config path: " << configPath << std::endl;
  std::cout << "Kernel path: " << kernelPath << std::endl;

  using std::chrono::duration;
  using std::chrono::duration_cast;
  using std::chrono::high_resolution_clock;
  using std::chrono::milliseconds;
  auto programStartTime = high_resolution_clock::now();

  ConfigReader config(configPath);
  /*
          ===============  ===============
  */
  // If seed = 0 then it generates random seed.

  /*
    =============== Initialization ===============
    === TODO ===
 */

  // 1) Initialize random number generators
  RandomNumberGenerator generator_initialization(config.m_seed);
  RandomNumberGenerator generator_collision(config.m_seed);

  // 2) Initialize openCL
  OpenCLLoader kernels(kernelPath);
  // 3) Define required physical parameters
  /*
  alpha = dV/rho, eta = sqrt(M+^2 - m-^2)/T, Upsilon = sigma/(dV * R_0)

  Define M+ -> Get T -> Get rho -> get dV -> get sigma
  M+ is defined in a config
  */
  numType temperatureFalse = std::sqrt(std::pow(config.m_massTrue, 2) -
                                       std::pow(config.m_massFalse, 2)) /
                             config.m_eta;
  numType temperatureTrue = 0;  // -> No particles generated in true vacuum

  // 4) Generate particles

  /*ParticleGenerator particleGenerator1(config.m_massFalse, temperatureFalse,
                                       30 * temperatureFalse,
                                       1e-5 * temperatureFalse);*/
  ParticleGenerator particleGenerator1(config.m_massFalse, 3.);
  ParticleCollection particles1(
      config.m_massTrue, config.m_massFalse, temperatureTrue, temperatureFalse,
      config.m_countParticlesTrue, config.m_countParticlesFalse,
      config.m_coupling, config.m_isBubbleTrueVacuum, kernels.getContext());
  /* particles1.add_to_total_initial_energy(
      particleGenerator1.generateNParticlesInSphere(
          config.m_initialBubbleRadius,
          (u_int)particles1.getParticleCountTotal(), generator_initialization,
          particles1.getParticles()));*/
  particles1.add_to_total_initial_energy(
      particleGenerator1.generateNParticlesInBox(
          config.m_initialBubbleRadius,
          (u_int)particles1.getParticleCountTotal(), generator_initialization,
          particles1.getParticles()));
  // 5) Initialize simulation object

  Simulation simulation(config.m_seed, config.m_dt, kernels.getContext());

  // 6) Initialize bubble object
  numType bubbleVolume =
      4 * M_PI / 3 * std::pow(config.m_initialBubbleRadius, 3);

  numType initialEnergyDensityFalse =
      particles1.countParticlesEnergy() / bubbleVolume;
  numType initialNumberDensityFalse =
      particles1.getParticleCountFalse() / bubbleVolume;
  numType Tn = initialNumberDensityFalse * temperatureFalse;     // > 0
  numType dV = config.m_alpha * initialEnergyDensityFalse + Tn;  // > 0
  numType sigma = config.m_upsilon * dV * config.m_initialBubbleRadius;

  /*
   *
   * NB! Look bubble integration over. dP might have wrong sign.
   *
   */
  if (!config.m_isBubbleTrueVacuum) {
    // Expanding (ture vacuum) bubble -> dV>0 -> False vacuum bubble change the
    // sign.
    dV = -dV;
  }
  PhaseBubble bubble(config.m_initialBubbleRadius, config.m_initialBubbleSpeed,
                     dV, sigma, kernels.getContext());

  CollisionCellCollection cells(5., 31, false, kernels.getContext());

  /*simulation.set_bubble_interaction_buffers(particles1, bubble,
                                           kernels.getKernel());*/
  simulation.set_particle_interaction_buffers(particles1, cells,
                                              kernels.m_cellAssignmentKernel,
                                              kernels.m_rotationKernel);
  /*
   * Remove later
   */
  simulation.set_particle_step_buffers(particles1, cells,
                                       kernels.m_particleStepKernel);
  simulation.set_particle_bounce_buffers(particles1, cells,
                                         kernels.m_particleBounceKernel);

  simulation.addTotalEnergy(particles1.getInitialTotalEnergy());
  // simulation.addTotalEnergy(bubble.calculateEnergy());

  // 8) Streaming object

  /*
          =============== Display text ===============
  */
  numType numberDensityParam =
      3 * config.m_countParticlesFalse /
      (4 * M_PI * std::pow(config.m_initialBubbleRadius, 3));
  numType energyDensityParam = 3 * temperatureFalse * numberDensityParam;
  numType mu = std::log(bubbleVolume * std::pow(temperatureFalse, 3) /
                        (config.m_countParticlesFalse * std::pow(M_PI, 2)));
  std::cout << std::endl;
  config.print_info();
  std::cout << std::endl;
  particles1.print_info(config, bubble);
  std::cout << std::endl;
  bubble.print_info(config);
  std::cout << std::endl;
  // 9) Streams

  particles1.writeAllBuffers(kernels.getCommandQueue());
  simulation.writeAllBuffers(kernels.getCommandQueue());
  bubble.writeAllBuffers(kernels.getCommandQueue());
  cells.writeAllBuffers(kernels.getCommandQueue());

  std::string dataFolderName = createFileNameFromCurrentDate();

  std::filesystem::path filePath =
      createSimulationFilePath(config.m_dataSavePath, dataFolderName);
  DataStreamer streamer(filePath.string());

  streamer.initData();
  streamer.initDensityProfile(config.m_densityBinsCount, 150);
  streamer.initMomentumProfile(config.m_momentumBinsCount,
                               30 * temperatureFalse);
  streamer.stream(simulation, particles1, bubble, kernels.getCommandQueue());

  /*
   * TODO!
   * 1) Kernel for bouncing back from some wall +
   * 2) Test bouncing back kernel
   * 3) Take out the bubble from simulation
   * 4) Generate particles with all same momenta
   * 5) Simulate the system and see if it relaxes to boltzmann distribution
   */

  /*
  // Calculate collision cell index for each particle
  cells.generateShiftVector(generator_collision);
  cells.writeShiftVectorBuffer(kernels.getCommandQueue());
  kernels.getCommandQueue().enqueueNDRangeKernel(
      kernels.m_cellAssignmentKernel, cl::NullRange,
      cl::NDRange(particles1.getParticleCountTotal()));
  // Update data on CPU
  particles1.readParticlesBuffer(kernels.getCommandQueue());
  // Calculate zero momentum frames and generate axises of rotation
  cells.recalculate_cells(particles1.getParticles(), generator_collision);
  // Update data on GPU
  for (Particle p : particles1.getParticles()) {
    std::cout << p.p_x << ", " << p.p_y << ", " << p.p_z << std::endl;
  }
  std::cout << std::endl;

  particles1.writeParticlesBuffer(kernels.getCommandQueue());
  cells.writeCollisionCellBuffer(kernels.getCommandQueue());
  // Rotate momentums
  kernels.getCommandQueue().enqueueNDRangeKernel(
      kernels.m_rotationKernel, cl::NullRange,
      cl::NDRange(particles1.getParticleCountTotal()));
  // Update data on CPU
  particles1.readParticlesBuffer(kernels.getCommandQueue());

  for (Particle p : particles1.getParticles()) {
    std::cout << p.p_x << ", " << p.p_y << ", " << p.p_z << ", "
              << p.idxCollisionCell << std::endl;
  }
  */
  // exit(0);

  std::fstream infoStream(filePath / "info.txt",
                          std::ios::out | std::ios::trunc);
  createSimulationInfoFile(infoStream, filePath, config, dV,
                           initialNumberDensityFalse,
                           initialEnergyDensityFalse);

  /* std::cout << std::setprecision(10) << std::fixed;
  std::cout << "===== STARTING SIMULATION =====" << std::endl;
  std::cout << "t: " << simulation.getTime() << ", R: " << bubble.getRadius()
            << ", V: " << bubble.getSpeed() << std::endl;*/

  /*for (Particle p : particles1.getParticles()) {
    std::cout << p.x << ", " << p.y << ", " << p.z << ", " << p.E << ", "
              << p.p_x << ", " << p.p_y << ", " << p.p_z << std::endl;
  }
  for (Particle p : particles1.getParticles()) {
    std::cout << p.idxCollisionCell << ",";
  }
  std::cout << std::endl;
  std::cout << std::endl;

  simulation.step(particles1, cells, generator_collision,
                  kernels.m_particleStepKernel, kernels.m_cellAssignmentKernel,
                  kernels.m_rotationKernel, kernels.getCommandQueue());
  particles1.readParticlesBuffer(kernels.getCommandQueue());

  for (Particle p : particles1.getParticles()) {
    std::cout << p.x << ", " << p.y << ", " << p.z << ", " << p.E << ", "
              << p.p_x << ", " << p.p_y << ", " << p.p_z << std::endl;
  }
  for (Particle p : particles1.getParticles()) {
    std::cout << p.idxCollisionCell << ",";
  }
  std::cout << std::endl;
  std::cout << std::endl;

  simulation.step(particles1, cells, generator_collision,
                  kernels.m_particleStepKernel, kernels.m_cellAssignmentKernel,
                  kernels.m_rotationKernel, kernels.getCommandQueue());

  particles1.readParticlesBuffer(kernels.getCommandQueue());

  for (Particle p : particles1.getParticles()) {
    std::cout << p.x << ", " << p.y << ", " << p.z << ", " << p.E << ", "
              << p.p_x << ", " << p.p_y << ", " << p.p_z << std::endl;
  }
  for (Particle p : particles1.getParticles()) {
    std::cout << p.idxCollisionCell << ",";
  }
  std::cout << std::endl;
  std::cout << std::endl;

  exit(0);*/
  /*
  std::fstream debug(filePath / "data.txt", std::ios::out | std::ios::trunc);

  std::fstream debug2(filePath / "data2.txt", std::ios::out | std::ios::trunc);
  debug << std::setprecision(6) << std::fixed << std::showpoint;
  debug << simulation.getTime() << std::endl;
  particles1.readParticlesBuffer(kernels.getCommandQueue());
  for (Particle p : particles1.getParticles()) {
    debug << p.x << "," << p.y << "," << p.z << "," << p.E << "," << p.p_x
          << "," << p.p_y << "," << p.p_z << "," << p.idxCollisionCell
          << std::endl;
  }
  debug << std::endl;

  debug2 << std::setprecision(6) << std::fixed << std::showpoint;
  debug2 << simulation.getTime() << std::endl;
  for (size_t i = 0; i < cells.getCollisionCells().size(); i++) {
    auto c = cells.getCollisionCells()[i];
    if (c.particle_count > 1) {
      debug2 << i << ": " << c.gamma << ", " << c.v2 << ", " << c.particle_count
             << ", " << c.total_mass << "," << c.x << "," << c.y << "," << c.z
             << "," << c.theta << std::endl;
    }
  }
  debug2 << std::endl;
  */
  for (int i = 1; i <= config.m_maxSteps; i++) {
    /* if (config.m_interactionsOn) {
      // kernels.test();
      simulation.step(particles1, bubble, kernels.getKernel(),
                      kernels.getCommandQueue());
    } else {
      simulation.step(bubble, 0);
    }*/

    simulation.step(particles1, cells, generator_collision,
                    kernels.m_particleStepKernel,
                    kernels.m_cellAssignmentKernel, kernels.m_rotationKernel,
                    kernels.m_particleBounceKernel, kernels.getCommandQueue());
    /*
    debug << std::setprecision(6) << std::fixed << std::showpoint;
    debug << simulation.getTime() << std::endl;
    particles1.readParticlesBuffer(kernels.getCommandQueue());
    for (Particle p : particles1.getParticles()) {
      debug << p.x << "," << p.y << "," << p.z << "," << p.E << "," << p.p_x
            << "," << p.p_y << "," << p.p_z << "," << p.idxCollisionCell
            << std::endl;
    }
    debug << std::endl;

    debug2 << std::setprecision(6) << std::fixed << std::showpoint;
    debug2 << simulation.getTime() << std::endl;
    for (size_t i = 0; i < cells.getCollisionCells().size(); i++) {
      auto c = cells.getCollisionCells()[i];
      if (c.particle_count > 1) {
        debug2 << i << ": " << c.gamma << ", " << c.v2 << ", "
               << c.particle_count << ", " << c.total_mass << "," << c.x << ","
               << c.y << "," << c.z << "," << c.theta << std::endl;
      }
    }
    debug2 << std::endl;

    std::cout << cells.getCollisionCells()[27].particle_count << ", "
              << std::cos(cells.getCollisionCells()[27].theta) << std::endl;

    for (CollisionCell c : cells.getCollisionCells()) {
      if (std::isnan(c.gamma)) {
        std::terminate();
      }
    }
    */
    if (i % config.m_streamFreq == 0) {
      std::cout << std::setprecision(10) << std::fixed << std::showpoint;
      std::cout << "Step: " << simulation.getTime() / simulation.get_dt()
                << std::endl;
      /*
      std::cout << "t: " << simulation.getTime()
                << ", R: " << bubble.getRadius() << ", V: " << bubble.getSpeed()
                << ", dP: " << simulation.get_dP() << std::endl;*/
      // auto writeStartTime = high_resolution_clock::now();
      streamer.stream(simulation, particles1, bubble,
                      kernels.getCommandQueue());
      // auto writeEndTime = high_resolution_clock::now();
      /*std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(
                       writeEndTime - writeStartTime)
                       .count()
                << " ms." << std::endl;
      ;*/
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
  streamer.stream(simulation, particles1, bubble, kernels.getCommandQueue());
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
