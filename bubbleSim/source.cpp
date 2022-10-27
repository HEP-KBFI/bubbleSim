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

  std::ifstream configStream(configPath);
  std::cout << "Config path: " << configPath << std::endl;
  nlohmann::json config = nlohmann::json::parse(configStream);

  /*
          ===============  ===============
  */
  // If seed = 0 then it generates random seed.
  int seed = config["simulation"]["seed"];
  int i_maxSteps = config["simulation"]["max_steps"];
  if (i_maxSteps == 0) {
    i_maxSteps = std::numeric_limits<int>::max();
  } else if (i_maxSteps < 0) {
    std::cerr << "i_maxSteps is set negative. i_maxSteps >= 0." << std::endl;
  }

  numType dt = config["simulation"]["dt"];
  bool b_dVisPositive = config["simulation"]["dV_positive"];

  numType alpha = config["parameters"]["alpha"];
  numType eta = config["parameters"]["eta"];
  numType upsilon = config["parameters"]["upsilon"];

  numType massFalse = config["particles"]["mass_false"];
  numType massTrue = config["particles"]["mass_true"];
  numType temperatureFalse = config["particles"]["T_false"];
  numType temperatureTrue = config["particles"]["T_true"];
  unsigned int countParticlesFalse = config["particles"]["N_false"];
  unsigned int countParticlesTrue = config["particles"]["N_true"];
  numType coupling = config["particles"]["coupling"];

  numType initialBubbleRadius = config["bubble"]["initial_radius"];
  numType initialBubbleSpeed = config["bubble"]["initial_speed"];
  bool b_isBubbleTrueVacuum = config["bubble"]["is_true_vacuum"];
  bool b_interactionsOn = config["bubble"]["interaction"];

  bool b_toStream = config["stream"]["stream"];
  bool b_streamData = config["stream"]["data"];
  bool b_streamProfile = config["stream"]["profile"];
  std::string s_dataPath = config["stream"]["data_path"];
  int i_streamFreq = config["stream"]["stream_freq"];
  int i_densityBins = config["stream"]["profile_density_bins"];
  int i_momentumBins = config["stream"]["profile_momentum_bins"];

  /*
    =============== Initialization ===============
    1) Define simulation parameters (alpha, eta, upsilon, M+)

    2) Define simulation object

    3) Generate particles (coordinates and momentum) and set their masses

    4) Get simulation parameters (number/energy desnity) and find dV and sigma

    5) Set dt size.

    6) Define bubble parameters (initial radius, initial speed, dV, sigma)

    7) Define OpenCLWrapper -> give data structure references

    8) Define streamer object

    9) Set up fstream objects
 */

  // 1) Defining simulation parameters if required
  /*
  alpha = dV/rho, eta = M+/T, Upsilon = sigma/(dV * R_0)

  Define M+ -> Get T -> Get rho -> get dV -> get sigma
  M+ is defined in a config
  */
  temperatureFalse =
      std::sqrt(std::pow(massTrue, 2) - std::pow(massFalse, 2)) / eta;
  temperatureTrue = 0;  // -> No particles generated in true vacuum

  numType bubbleVolume = 4 * M_PI / 3 * std::pow(initialBubbleRadius, 3);
  numType mu = std::log(bubbleVolume * std::pow(temperatureFalse, 3) /
                        (countParticlesFalse * std::pow(M_PI, 2)));

  // 2) Simulation definition
  Simulation sim(seed, massTrue, massFalse, temperatureTrue, temperatureFalse,
                 countParticlesTrue, countParticlesFalse, coupling);

  // 3) Generating particles
  sim.generateNParticlesInSphere(massFalse, initialBubbleRadius,
                                 sim.getParticleCountFalseInitial(),
                                 sim.getRef_CPDFalse(), sim.getRef_PFalse());

  // 4) dV and sigma
  sim.setEnergyDensityFalseSimInitial(sim.countParticlesEnergy() /
                                      bubbleVolume);
  sim.setNumberDesnityFalseSimInitial(sim.getParticleCountFalseInitial() /
                                      bubbleVolume);

  numType Tn = sim.getNumberDensityFalseInitial() * temperatureFalse;
  numType dV = alpha * sim.getEnergyDensityFalseInitial() - Tn;
  numType sigma = upsilon * dV * initialBubbleRadius;

  if (!b_dVisPositive) {
    // Expanding bubble -> dV < 0
    dV = -dV;
  }

  // 5) dt definition
  sim.set_dt(dt);

  // 6) PhaseBubble
  PhaseBubble bubble(initialBubbleRadius, initialBubbleSpeed, dV, sigma);

  sim.setEnergyTotalInitial(sim.countParticlesEnergy() +
                            bubble.calculateEnergy());

  // 7) OpenCL wrapper
  OpenCLWrapper openCL(kernelPath, kernelName, sim.getRef_Particles(),
                       sim.getRef_dP(), sim.getRef_dt(), sim.getRef_MassTrue(),
                       sim.getRef_MassFalse(), sim.getRef_MassDelta2(),
                       bubble.getRef_Bubble(), sim.getRef_InteractedFalse(),
                       sim.getRef_PassedFalse(), sim.getRef_InteractedTrue(),
                       b_isBubbleTrueVacuum);

  // 8) Streaming object
  DataStreamer dataStreamer(sim, bubble, openCL);

  /*
          =============== Display text ===============
  */
  numType numberDensityParam =
      3 * countParticlesFalse / (4 * M_PI * std::pow(initialBubbleRadius, 3));
  numType energyDensityParam = 3 * temperatureFalse * numberDensityParam;

  std::cout << std::endl
            << "=============== Config ===============" << std::endl;
  std::cout << "===== Simulation:" << std::endl;
  std::cout << "seed: " << seed << ", max_steps: " << i_maxSteps
            << ", dt: " << dt << ", dV_isPositive: " << b_dVisPositive
            << std::endl;
  std::cout << std::setprecision(5)
            << "===== Unitless parameters: " << std::endl;
  std::cout << "alpha: " << alpha << ", eta: " << eta
            << ", upsilon: " << upsilon << std::endl;
  std::cout << "===== Bubble:" << std::endl;
  std::cout << "Radius: " << bubble.getRadius()
            << ", Speed: " << bubble.getSpeed()
            << ", VacuumInBubble: " << b_isBubbleTrueVacuum << std::endl;
  std::cout << "M(true): " << massTrue << ", M(false): " << massFalse
            << ", N(true): " << countParticlesTrue
            << ", N(false): " << countParticlesFalse << std::endl;
  std::cout << std::endl
            << "=============== Initialization ===============" << std::endl;
  std::cout << "===== Bubble:" << std::endl;
  std::cout << "Particle's interaction with bubble on: " << b_interactionsOn
            << std::endl;
  std::cout << std::setprecision(10) << "dV: " << bubble.getdV()
            << ", dV(param): "
            << countParticlesFalse * massTrue * (3 * alpha - 1) /
                   (eta * bubble.calculateVolume())
            << ", Ratio: "
            << bubble.getdV() /
                   (countParticlesFalse * massTrue * (3 * alpha - 1) /
                    (eta * bubble.calculateVolume()))
            << std::endl;
  std::cout << "sigma: " << bubble.getSigma() << ", sigma(param): "
            << bubble.getRadius() * countParticlesFalse * upsilon * massTrue *
                   (3 * alpha - 1) / (eta * bubble.calculateVolume())
            << ", Ratio: "
            << bubble.getSigma() / (bubble.getRadius() * countParticlesFalse *
                                    upsilon * massTrue * (3 * alpha - 1) /
                                    (eta * bubble.calculateVolume()))
            << std::endl;
  std::cout
      << "Bubble energy: " << bubble.calculateEnergy()
      << ", Bubble energy(param): "
      << countParticlesFalse * massTrue / eta * (3 * alpha - 1) *
             (1 + 3 * upsilon / std::sqrt(1 - std::pow(bubble.getSpeed(), 2)))
      << ", Ratio: "
      << bubble.calculateEnergy() /
             (countParticlesFalse * massTrue / eta * (3 * alpha - 1) *
              (1 + 3 * upsilon / std::sqrt(1 - std::pow(bubble.getSpeed(), 2))))
      << std::endl;
  std::cout << "===== Thermodynamcis:" << std::endl;
  std::cout << "n(param): " << countParticlesFalse / bubble.calculateVolume()
            << ", n(theor): "
            << std::pow(temperatureFalse, 3) / std::pow(M_PI, 2) * std::exp(-mu)
            << ", n(sim): "
            << sim.getParticleCountTotal() / bubble.calculateVolume()
            << ", Ratio: "
            << (sim.getParticleCountTotal() / bubble.calculateVolume()) /
                   (countParticlesFalse / bubble.calculateVolume())

            << std::endl;
  std::cout << "rho(param): "
            << sim.getParticleCountTotal() * 3 * massTrue /
                   (eta * bubble.calculateVolume())
            << ", rho(theor): "
            << 3 * std::pow(temperatureFalse, 4) / std::pow(M_PI, 2) *
                   std::exp(-mu)
            << ", rho(sim): "
            << sim.countParticlesEnergy() / bubble.calculateVolume()
            << ", Ratio: "
            << (sim.countParticlesEnergy() / bubble.calculateVolume()) /
                   (sim.getParticleCountTotal() * 3 * massTrue /
                    (eta * bubble.calculateVolume()))
            << std::endl;
  std::cout << "<E>(param): " << 3 * massTrue / eta
            << ", <E>(theor): " << 3 * temperatureFalse << ", <E>(sim): "
            << sim.countParticlesEnergy() / countParticlesFalse << ", Ratio: "
            << (sim.countParticlesEnergy() / countParticlesFalse) /
                   (3 * massTrue / eta)
            << std::endl;

  // 9) Streams
  std::fstream pStreamIn, pStreamOut, nStream, rhoStream, dataStream;
  std::string dataFolderName = createFileNameFromCurrentDate();
  std::filesystem::path filePath;

  if (b_toStream) {
    filePath = createSimulationFilePath(s_dataPath, dataFolderName);
    std::cout << "Files saved to: " << filePath << std::endl;

    if (b_streamData) {
      dataStream = std::fstream(filePath / "data.csv",
                                std::ios::out | std::ios::in | std::ios::trunc);
      dataStream << "time,dP,R_b,V_b,E_b,E_p,E_f,E,C_f,C_if,Cpf,C_it"
                 << std::endl;
      dataStreamer.streamBaseData(dataStream, b_isBubbleTrueVacuum);
    }
    if (b_streamProfile) {
      pStreamIn = std::fstream(filePath / "pIn.csv",
                               std::ios::out | std::ios::in | std::ios::trunc);
      pStreamOut = std::fstream(filePath / "pOut.csv",
                                std::ios::out | std::ios::in | std::ios::trunc);
      nStream = std::fstream(filePath / "n.csv",
                             std::ios::out | std::ios::in | std::ios::trunc);
      rhoStream = std::fstream(filePath / "rho.csv",
                               std::ios::out | std::ios::in | std::ios::trunc);
      dataStreamer.streamProfiles(
          nStream, rhoStream, pStreamIn, pStreamOut, i_densityBins,
          i_momentumBins, 2 * initialBubbleRadius, temperatureFalse * 30,
          sim.getEnergyDensityFalseSimInitial());
    }
    dataStreamer.reset();
  }

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
  std::cout << "t: " << sim.getTime() << ", R: " << bubble.getRadius()
            << ", V: " << bubble.getSpeed() << std::endl;

  for (int i = 1; i <= i_maxSteps; i++) {
    if (b_interactionsOn) {
      sim.step(bubble, openCL);
    } else {
      sim.step(bubble, 0);
    }

    if (i % i_streamFreq == 0) {
      std::cout << "t: " << sim.getTime() << ", R: " << bubble.getRadius()
                << ", V: " << bubble.getSpeed() << std::endl;
      if (b_toStream) {
        if (b_streamData) {
          dataStreamer.streamBaseData(dataStream, b_isBubbleTrueVacuum);
        }
        dataStreamer.reset();
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
  if (b_toStream) {
    if (b_streamData) {
      dataStreamer.streamBaseData(dataStream, b_isBubbleTrueVacuum);
    }
    if (b_streamProfile) {
      dataStreamer.streamProfiles(
          nStream, rhoStream, pStreamIn, pStreamOut, i_densityBins,
          i_momentumBins, 2 * initialBubbleRadius, temperatureFalse * 30,
          sim.getEnergyDensityFalseSimInitial());
    }
    dataStreamer.reset();
  }

  // Measure runtime
  auto programEndTime = high_resolution_clock::now();
  auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(
      programEndTime - programStartTime);
  int seconds = (int)(ms_int.count() / 1000) % 60;
  int minutes = ((int)(ms_int.count() / (1000 * 60)) % 60);
  int hours = ((int)(ms_int.count() / (1000 * 60 * 60)) % 24);

  // Stream simulation info
  if (b_toStream) {
    createSimulationInfoFile(
        filePath, seed, alpha, eta, upsilon, initialBubbleRadius,
        initialBubbleSpeed, massFalse, massTrue, temperatureFalse,
        temperatureTrue, countParticlesFalse, countParticlesTrue, coupling, dV,
        sim.getNumberDensityFalseInitial(), sim.getEnergyDensityFalseInitial(),
        sim.get_dt(), b_interactionsOn,
        dataStreamer.countMassRadiusDifference(b_isBubbleTrueVacuum),
        (int)ms_int.count());
  }

  std::cout << std::endl
            << "Program run: " << hours << "h " << minutes << "m " << seconds
            << "s " << std::endl;
}
