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
  ss << "-";
  for (i = 0; i < 4; i++) {
    ss << dis(gen);
  }
  ss << "-4";
  for (i = 0; i < 3; i++) {
    ss << dis(gen);
  }
  ss << "-";
  ss << dis2(gen);
  for (i = 0; i < 3; i++) {
    ss << dis(gen);
  }
  ss << "-";
  for (i = 0; i < 12; i++) {
    ss << dis(gen);
  };
  result = std::string(c_name) + "_" + ss.str();
  std::cout << result << std::endl;
  return result;
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: bubbleSim.exe config.json" << std::endl;
    exit(0);
  }
  // TODO
  /*
  std::filesystem::path a = std::filesystem::current_path();
  std::filesystem::path b = std::filesystem::relative(a);
  std::filesystem::path c = std::filesystem::absolute(a);

  std::cout << a << std::endl;
  std::cout << b << std::endl;
  std::cout << c << std::endl;
  */
  std::filesystem::path current_path = std::filesystem::current_path();
  std::cout << "File name: ";
  createFileNameFromCurrentDate();
  std::string configPath = argv[1];
  std::string kernelPath = "kernel.cl";
  std::string kernelName = "step_double";

  using std::chrono::duration;
  using std::chrono::duration_cast;
  using std::chrono::high_resolution_clock;
  using std::chrono::milliseconds;

  std::ifstream configStream(configPath);
  std::cout << "Config path: " << configPath << std::endl;
  nlohmann::json config = nlohmann::json::parse(configStream);
  /*
          ===============  ===============
  */

  // If seed = 0 then it generates random seed.
  int seed = config["seed"];

  numType alpha = config["alpha"];
  numType eta = config["eta"];
  numType upsilon = config["upsilon"];
  numType sigma = config["sigma"];

  numType massFalse = config["mass_false"];
  numType massTrue = config["mass_true"];
  numType temperatureFalse = config["T_false"];
  numType temperatureTrue = config["T_true"];
  unsigned int countParticlesFalse = config["N_false"];
  unsigned int countParticlesTrue = config["N_true"];
  numType coupling = config["coupling"];
  numType initialBubbleSpeed = config["bubble_speed"];
  numType initialRadius = config["initial_radius"];
  int i_maxSteps = config["max_steps"];

  /*
          =============== Initialization ===============
          1) Define simulation (calculates number and energy densities)
  */

  Simulation sim(seed, alpha, massTrue, massFalse, temperatureTrue,
                 temperatureFalse, countParticlesTrue, countParticlesFalse,
                 coupling);
  numType radius = (numType)std::cbrt(
      countParticlesFalse / (4 * sim.getNumberDensityFalse()) * 3 / M_PI);
  sim.generateNParticlesInSphere(massFalse, radius, countParticlesFalse,
                                 sim.getCPDFalseRef(), sim.getPFalseRef());
  sim.set_dt(radius / 1000);
  numType rho0 = sim.countParticleEnergyDensity(radius);
  // Delta V_T / rho = alpha, Delta V_T = Delta V - T * n -> Delta V = rho *
  // alpha + T * n
  numType dV = alpha * rho0 + temperatureFalse * sim.getNumberDensityFalse();

  Bubble bubble(radius, initialBubbleSpeed, dV, sigma);

  OpenCLWrapper openCL(
      kernelPath, kernelName, sim.getParticleCountTotal(), sim.getReferenceX(),
      sim.getReferenceP(), sim.getReferenceE(), sim.getReferenceM(),
      sim.getReference_dP(), sim.getReference_dt(), sim.getReferenceMassTrue(),
      sim.getReferenceMassFalse(), sim.getReferenceMassDelta2(),
      bubble.getRadiusRef(), bubble.getRadius2Ref(),
      bubble.getRadiusAfterDt2Ref(), bubble.getSpeedRef(), bubble.getGammaRef(),
      bubble.getGammaSpeedRef(), sim.getReferenceInteractedFalse(),
      sim.getReferencePassedFalse(), sim.getReferenceInteractedTrue(), false);

  DataStreamer dataStream(sim, bubble, openCL);

  /*
          =============== Display text ===============
  */
  std::cout << "=============== Text ===============" << std::endl;
  std::cout << "    ========== Bubble ==========" << std::endl;
  std::cout << "Initial bubble radius: " << bubble.getRadius()
            << ", Initial bubble speed: " << bubble.getSpeed() << std::endl;
  std::cout << "dV : " << bubble.getdV() << ", Sigma: " << bubble.getSigma()
            << std::endl;

  std::cout << "Total particle energy: " << sim.countParticlesEnergy()
            << std::endl;

  std::cout << "=============== Text end ===============" << std::endl;
  std::cout << "n = " << sim.getNumberDensityFalse()
            << ", rho = " << sim.getEnergyDensityFalse() << std::endl;
  std::cout << "dt: " << sim.get_dt() << std::endl;
  std::cout << std::setprecision(9) << radius << std::endl;

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
#ifdef MAX_STEPS
  std::cout << i_maxSteps << std::endl;
  for (int i = 0; i < i_maxSteps; i++) {
      sim.step(bubble, openCL);
      std::cout << std::setprecision(15) << "R: " << bubble.getRadius() << ", V: " << bubble.getSpeed() << ", dP: " << sim.getdPressureStep() << std::endl;
      dataStream.streamParticleInfo();
      if (std::isnan(bubble.getSpeed())) {
        std::cerr << "Abort due to nan" << std::endl;
        exit(1);
    }
  }
#endif
#ifndef MAX_STEPS
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      sim.step(bubble, openCL);
      if (std::isnan(bubble.getSpeed())) {
        std::cerr << "Abort due to nan" << std::endl;
        exit(1);
      }
    }
    std::cout << std::setprecision(15) << "R: " << bubble.getRadius()
              << ", V: " << bubble.getSpeed()
              << ", dP: " << sim.getdPressureStep() << std::endl;
  }
#endif  // !MAX_STEPS

  dataStream.streamMassRadiusDifference(false);
}
