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

void createSimulationInfoFile(
    std::filesystem::path filePath, int t_seed, numType t_alpha, numType t_eta,
    numType t_upsilon, numType t_radius, numType t_speed, numType t_m_false,
    numType t_m_true, numType t_temperatureFalse, numType t_temperatureTrue,
    u_int t_countFalse, u_int t_countTrue, numType t_coupling, numType t_dV,
    numType t_n, numType t_rho, numType t_dt, bool t_interactionOn,
    int t_postionDifference, int t_programRuntime) {
  std::fstream simulationListStream;

  simulationListStream = std::fstream(
      filePath / "info.txt", std::ios::out | std::ios::in | std::ios::trunc);
  simulationListStream
      << "file_name,seed,alpha,eta,upsilon,radius,speed,m-,m+,"
         "T-,T+,N-,N+,coupling,dV,n,rho,dt,interactionOn,deltaN,runtime"
      << std::endl;
  simulationListStream << filePath.filename() << "," << t_seed << "," << t_alpha
                       << "," << t_eta << "," << t_upsilon << "," << t_radius
                       << "," << t_speed << "," << t_m_false << "," << t_m_true
                       << "," << t_temperatureFalse << "," << t_temperatureTrue
                       << "," << t_countFalse << "," << t_countTrue << ","
                       << t_coupling << "," << t_dV << "," << t_n << ","
                       << t_rho << "," << t_dt << "," << t_interactionOn << ","
                       << t_postionDifference << "," << t_programRuntime
                       << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: bubbleSim.exe config.json" << std::endl;
    exit(0);
  }

  std::string configPath = argv[1];
  std::string kernelPath = "kernel.cl";
  std::string kernelName = "step_double";

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
    1) Define simulation parameters (alpha, eta, upsilon, M+)

    2) Define simulation object

    3) Generate particles (coordinates and momentum) and set their masses

    4) Get simulation parameters (number/energy desnity) and find dV and sigma

    5) Set dt size.

    6) Define bubble parameters (initial radius, initial speed, dV, sigma)

    7) Define OpenCLKernelLoader -> give data structure references

    8) Define streamer object

    9) Set up fstream objects
 */

  // 1) Defining simulation parameters if required
  /*
  alpha = dV/rho, eta = M+/T, Upsilon = sigma/(dV * R_0)

  Define M+ -> Get T -> Get rho -> get dV -> get sigma
  M+ is defined in a config
  */
  OpenCLKernelLoader kernels(kernelPath, kernelName);

  numType temperatureFalse = std::sqrt(std::pow(config.m_massTrue, 2) -
                                       std::pow(config.m_massFalse, 2)) /
                             config.m_eta;
  numType temperatureTrue = 0;  // -> No particles generated in true vacuum

  RandomNumberGenerator generator(config.m_seed);
  ParticleGenerator particleGenerator1(config.m_massFalse, temperatureFalse,
                                       30 * temperatureFalse,
                                       1e-5 * temperatureFalse);

  numType bubbleVolume =
      4 * M_PI / 3 * std::pow(config.m_initialBubbleRadius, 3);
  numType mu = std::log(bubbleVolume * std::pow(temperatureFalse, 3) /
                        (config.m_countParticlesFalse * std::pow(M_PI, 2)));

  // 2) ParticleCollection definition
  ParticleCollection particles1(
      config.m_massTrue, config.m_massFalse, temperatureTrue, temperatureFalse,
      config.m_countParticlesTrue, config.m_countParticlesFalse,
      config.m_coupling, config.m_isBubbleTrueVacuum, kernels.getContext());

  // 3) Generating particles (and add their energy to total initial)
  particles1.add_to_total_initial_energy(
      particleGenerator1.generateNParticlesInSphere(
          config.m_initialBubbleRadius, particles1.getParticleCountTotal(),
          generator, particles1.getParticles()));
  Simulation simulation(config.m_seed, config.m_dt, kernels.getContext());
  // exit(0);

  // 4) dV and sigma

  numType initialEnergyDensityFalse =
      particles1.countParticlesEnergy() / bubbleVolume;
  numType initialNumberDensityFalse =
      particles1.getParticleCountFalse() / bubbleVolume;

  numType Tn = initialNumberDensityFalse * temperatureFalse;
  numType dV = config.m_alpha * initialEnergyDensityFalse - Tn;
  numType sigma = config.m_upsilon * dV * config.m_initialBubbleRadius;

  if (!config.m_dVisPositive) {
    // Expanding bubble -> dV < 0
    dV = -dV;
  }

  // 5) dt definition
  simulation.set_dt(config.m_dt);

  // 6) PhaseBubble
  PhaseBubble bubble(config.m_initialBubbleRadius, config.m_initialBubbleSpeed,
                     dV, sigma, kernels.getContext());
  simulation.addTotalEnergy(bubble.calculateEnergy());
  simulation.set_bubble_interaction_buffers(particles1, bubble,
                                            kernels.getKernel());
  // 7) OpenCL wrapper

  // 8) Streaming object
  // DataStreamer dataStreamer(sim, bubble, kernels);

  /*
          =============== Display text ===============
  */
  numType numberDensityParam =
      3 * config.m_countParticlesFalse /
      (4 * M_PI * std::pow(config.m_initialBubbleRadius, 3));
  numType energyDensityParam = 3 * temperatureFalse * numberDensityParam;

  std::cout << std::endl
            << "=============== Config ===============" << std::endl;
  std::cout << "===== Simulation:" << std::endl;
  std::cout << "seed: " << config.m_seed << ", max_steps: " << config.m_maxSteps
            << ", dt: " << config.m_dt
            << ", dV_isPositive: " << config.m_dVisPositive << std::endl;
  std::cout << std::setprecision(5)
            << "===== Unitless parameters: " << std::endl;
  std::cout << "alpha: " << config.m_alpha << ", eta: " << config.m_eta
            << ", upsilon: " << config.m_upsilon << std::endl;
  std::cout << "===== Bubble:" << std::endl;
  std::cout << "Radius: " << bubble.getRadius()
            << ", Speed: " << bubble.getSpeed()
            << ", VacuumInBubble: " << config.m_isBubbleTrueVacuum << std::endl;
  std::cout << "M(true): " << config.m_massTrue
            << ", M(false): " << config.m_massFalse
            << ", N(true): " << config.m_countParticlesTrue
            << ", N(false): " << config.m_countParticlesFalse << std::endl;
  std::cout << std::endl
            << "=============== Initialization ===============" << std::endl;
  std::cout << "===== Bubble:" << std::endl;
  std::cout << "Particle's interaction with bubble on: "
            << config.m_interactionsOn << std::endl;
  std::cout << std::setprecision(10) << "dV: " << bubble.getdV()
            << ", dV(param): "
            << config.m_countParticlesFalse * config.m_massTrue *
                   (3 * config.m_alpha - 1) /
                   (config.m_eta * bubble.calculateVolume())
            << ", Ratio: "
            << bubble.getdV() / (config.m_countParticlesFalse *
                                 config.m_massTrue * (3 * config.m_alpha - 1) /
                                 (config.m_eta * bubble.calculateVolume()))
            << std::endl;
  std::cout << "sigma: " << bubble.getSigma() << ", sigma(param): "
            << bubble.getRadius() * config.m_countParticlesFalse *
                   config.m_upsilon * config.m_massTrue *
                   (3 * config.m_alpha - 1) /
                   (config.m_eta * bubble.calculateVolume())
            << ", Ratio: "
            << bubble.getSigma() /
                   (bubble.getRadius() * config.m_countParticlesFalse *
                    config.m_upsilon * config.m_massTrue *
                    (3 * config.m_alpha - 1) /
                    (config.m_eta * bubble.calculateVolume()))
            << std::endl;
  std::cout << "Bubble energy: " << bubble.calculateEnergy()
            << ", Bubble energy(param): "
            << config.m_countParticlesFalse * config.m_massTrue / config.m_eta *
                   (3 * config.m_alpha - 1) *
                   (1 + 3 * config.m_upsilon /
                            std::sqrt(1 - std::pow(bubble.getSpeed(), 2)))
            << ", Ratio: "
            << bubble.calculateEnergy() /
                   (config.m_countParticlesFalse * config.m_massTrue /
                    config.m_eta * (3 * config.m_alpha - 1) *
                    (1 + 3 * config.m_upsilon /
                             std::sqrt(1 - std::pow(bubble.getSpeed(), 2))))
            << std::endl;
  std::cout << "===== Thermodynamcis:" << std::endl;
  std::cout << "n(param): "
            << config.m_countParticlesFalse / bubble.calculateVolume()
            << ", n(theor): "
            << std::pow(temperatureFalse, 3) / std::pow(M_PI, 2) * std::exp(-mu)
            << ", n(sim): "
            << particles1.getParticleCountTotal() / bubble.calculateVolume()
            << ", Ratio: "
            << (particles1.getParticleCountTotal() / bubble.calculateVolume()) /
                   (config.m_countParticlesFalse / bubble.calculateVolume())

            << std::endl;
  std::cout << "rho(param): "
            << particles1.getParticleCountTotal() * 3 * config.m_massTrue /
                   (config.m_eta * bubble.calculateVolume())
            << ", rho(theor): "
            << 3 * std::pow(temperatureFalse, 4) / std::pow(M_PI, 2) *
                   std::exp(-mu)
            << ", rho(sim): "
            << particles1.countParticlesEnergy() / bubble.calculateVolume()
            << ", Ratio: "
            << (particles1.countParticlesEnergy() / bubble.calculateVolume()) /
                   (particles1.getParticleCountTotal() * 3 * config.m_massTrue /
                    (config.m_eta * bubble.calculateVolume()))
            << std::endl;
  std::cout << "<E>(param): " << 3 * config.m_massTrue / config.m_eta
            << ", <E>(theor): " << 3 * temperatureFalse << ", <E>(sim): "
            << particles1.countParticlesEnergy() / config.m_countParticlesFalse
            << ", Ratio: "
            << (particles1.countParticlesEnergy() /
                config.m_countParticlesFalse) /
                   (3 * config.m_massTrue / config.m_eta)
            << std::endl;

  // 9) Streams
  std::fstream pStreamIn, pStreamOut, nStream, rhoStream, dataStream;
  std::string dataFolderName = createFileNameFromCurrentDate();
  std::filesystem::path filePath;
  /*
  if (config.m_toStream) {
    filePath = createSimulationFilePath(config.m_dataSavePath, dataFolderName);
    std::cout << "Files saved to: " << filePath << std::endl;

    if (config.m_streamData) {
      dataStream = std::fstream(filePath / "data.csv",
                                std::ios::out | std::ios::in | std::ios::trunc);
      dataStream << "time,dP,R_b,V_b,E_b,E_p,E_f,E,C_f,C_if,Cpf,C_it"
                 << std::endl;
      dataStreamer.streamBaseData(dataStream, config.m_isBubbleTrueVacuum);
    }
    if (config.m_streamProfiles) {
      pStreamIn = std::fstream(filePath / "pIn.csv",
                               std::ios::out | std::ios::in | std::ios::trunc);
      pStreamOut = std::fstream(filePath / "pOut.csv",
                                std::ios::out | std::ios::in | std::ios::trunc);
      nStream = std::fstream(filePath / "n.csv",
                             std::ios::out | std::ios::in | std::ios::trunc);
      rhoStream = std::fstream(filePath / "rho.csv",
                               std::ios::out | std::ios::in | std::ios::trunc);
      dataStreamer.streamProfiles(
          nStream, rhoStream, pStreamIn, pStreamOut, config.m_densityBinsCount,
          config.m_momentumBinsCount, 2 * config.m_initialBubbleRadius,
          temperatureFalse * 30, initialEnergyDensityFalse);
    }
    dataStreamer.reset();
  }
  */
#ifdef LOG_DEBUG
  int particleIndex1 = 0;
  int particleIndex2 = sim.getParticleCountTotal() - 1;
  std::cout << std::endl
            << std::setprecision(15)
            << "=============== DEBUG ===============" << std::endl;
  std::cout << "Particle " << particleIndex1 << ": " << std::endl;
  std::cout << "X: " << sim.getReferenceX()[3 * particleIndex1] << ", "
            << sim.getReferenceX()[3 * particleIndex1 + 1] << ", "
            << sim.getReferenceX()[3 * particleIndex1 + 2] << std::endl;
  std::cout << "P: " << sim.getReferenceP()[3 * particleIndex1] << ", "
            << sim.getReferenceP()[3 * particleIndex1 + 1] << ", "
            << sim.getReferenceP()[3 * particleIndex1 + 2] << std::endl;
  std::cout << "M: " << sim.getParticleMass(particleIndex1)
            << ", E: " << sim.getParticleEnergy(particleIndex1) << std::endl
            << std::endl;
  std::cout << "Particle " << particleIndex2 << ": " << std::endl;
  std::cout << "X: " << sim.getReferenceX()[3 * particleIndex2] << ", "
            << sim.getReferenceX()[3 * particleIndex2 + 1] << ", "
            << sim.getReferenceX()[3 * particleIndex2 + 2] << std::endl;
  std::cout << "P: " << sim.getReferenceP()[3 * particleIndex2] << ", "
            << sim.getReferenceP()[3 * particleIndex2 + 1] << ", "
            << sim.getReferenceP()[3 * particleIndex2 + 2] << std::endl;
  std::cout << "M: " << sim.getParticleMass(particleIndex2)
            << ", E: " << sim.getParticleEnergy(particleIndex2) << std::endl;
  std::cout << std::setprecision(9)
            << "=============== DEBUG END ===============" << std::endl
            << std::endl;
#endif

  std::cout << "===== STARTING SIMULATION =====" << std::endl;
  std::cout << "t: " << simulation.getTime() << ", R: " << bubble.getRadius()
            << ", V: " << bubble.getSpeed() << std::endl;

  for (int i = 1; i <= config.m_maxSteps; i++) {
    if (config.m_interactionsOn) {
      simulation.step(particles1, bubble, kernels.getKernel(),
                      kernels.getCommandQueue());
    } else {
      simulation.step(bubble, 0);
    }

    if (i % config.m_streamFreq == 0) {
      std::cout << "t: " << simulation.getTime()
                << ", R: " << bubble.getRadius() << ", V: " << bubble.getSpeed()
                << std::endl;
      /*if (config.m_toStream) {
        if (config.m_streamData) {
          dataStreamer.streamBaseData(dataStream, config.m_isBubbleTrueVacuum);
        }
        dataStreamer.reset();
      }*/
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
  /*
  if (config.m_toStream) {
    if (config.m_streamData) {
      dataStreamer.streamBaseData(dataStream, config.m_isBubbleTrueVacuum);
    }
    if (config.m_streamProfiles) {
      dataStreamer.streamProfiles(
          nStream, rhoStream, pStreamIn, pStreamOut, config.m_densityBinsCount,
          config.m_momentumBinsCount, 2 * config.m_initialBubbleRadius,
          temperatureFalse * 30, initialEnergyDensityFalse);
    }
    dataStreamer.reset();
  }
  */
  // Measure runtime
  auto programEndTime = high_resolution_clock::now();
  auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(
      programEndTime - programStartTime);
  int seconds = (int)(ms_int.count() / 1000) % 60;
  int minutes = ((int)(ms_int.count() / (1000 * 60)) % 60);
  int hours = ((int)(ms_int.count() / (1000 * 60 * 60)) % 24);

  // Stream simulation info
  /*
  if (config.m_toStream) {
    createSimulationInfoFile(
        filePath, config.m_seed, config.m_alpha, config.m_eta, config.m_upsilon,
        config.m_initialBubbleRadius, config.m_initialBubbleSpeed,
        config.m_massFalse, config.m_massTrue, temperatureFalse,
        temperatureTrue, config.m_countParticlesFalse,
        config.m_countParticlesTrue, config.m_coupling, dV,
        initialNumberDensityFalse, initialEnergyDensityFalse, sim.get_dt(),
        config.m_interactionsOn,
        dataStreamer.countMassRadiusDifference(config.m_isBubbleTrueVacuum),
        (int)ms_int.count());
  }
  */
  std::cout << std::endl
            << "Program run: " << hours << "h " << minutes << "m " << seconds
            << "s " << std::endl;
}
