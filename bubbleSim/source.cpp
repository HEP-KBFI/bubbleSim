#define _CRT_SECURE_NO_WARNINGS
#include "source.h"

void createFileNameFromCurrentDate(char* name) {
	std::time_t t_cur = time(NULL);
	const auto* t_local = localtime(&t_cur);
	std::strftime(name, 50, "%j_%Y_%H_%M_%S", t_local);
}

int main(int argc, char *argv[]) {

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

	std::string configPath = argv[1];
	std::string kernelPath = "kernel.cl";
	std::string kernelName = "step_double";

	using std::chrono::high_resolution_clock;
	using std::chrono::duration_cast;
	using std::chrono::duration;
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
	
	/*
		=============== Initialization ===============
		1) Define simulation (calculates number and energy densities)
	*/

	Simulation sim(
		seed, alpha, massTrue, massFalse,
		temperatureTrue, temperatureFalse,
		countParticlesTrue, countParticlesFalse,
		coupling
	);
	numType radius = (numType) std::cbrt(countParticlesFalse / (4 * sim.getNumberDensityFalse()) * 3 / M_PI);
	sim.generateNParticlesInSphere(
		massFalse, radius, countParticlesFalse, 
		sim.getCPDFalseRef(), sim.getPFalseRef()
	);
	sim.set_dt(radius / 1000);
	std::cout << "=============== Testing ===============" << std::endl;
	numType n = sim.calculateNumberDensity(0.01, 0.5, 1e-4 * 0.5, 30 * 0.5);
	std::cout << "n = " << n << std::endl;


	std::cout << "=============== Testing end ===============" << std::endl << std::endl;
	numType rho0 = sim.countParticleEnergyDensity(radius);
	// Delta V_T / rho = alpha, Delta V_T = Delta V - T * n -> Delta V = rho * alpha + T * n
	numType dV = alpha * rho0 + temperatureFalse * sim.getNumberDensityFalse();

	Bubble bubble(radius, initialBubbleSpeed, dV, sigma);

	OpenCLWrapper openCL(
		kernelPath, kernelName, sim.getParticleCountTotal(),
		sim.getReferenceX(), sim.getReferenceP(), sim.getReferenceE(),
		sim.getReferenceM(), sim.getReference_dP(), sim.getReference_dt(),
		sim.getReferenceMassTrue(), sim.getReferenceMassFalse(), sim.getReferenceMassDelta2(),
		bubble.getRadiusRef(), bubble.getRadius2Ref(), bubble.getRadiusAfterDt2Ref(),
		bubble.getSpeedRef(), bubble.getGammaRef(), bubble.getGammaSpeedRef(),
		sim.getReferenceInteractedFalse(), sim.getReferencePassedFalse(), 
		sim.getReferenceInteractedTrue(), false
		);

	/*
		=============== Display text ===============
	*/
	std::cout << "=============== Text ===============" << std::endl;
	std::cout << "    ========== Bubble ==========" << std::endl;
	std::cout << "Initial bubble radius: " << bubble.getRadius() << ", Initial bubble speed: " << bubble.getSpeed() << std::endl;
	std::cout << "dV : " <<  bubble.getdV() << ", Sigma: " << bubble.getSigma() << std::endl;


	std::cout << "Total particle energy: " << sim.countParticlesEnergy() << std::endl;

	std::cout << "=============== Text end ===============" << std::endl;
	std::cout << "n = " << sim.getNumberDensityFalse() << ", rho = " << sim.getEnergyDensityFalse() << std::endl;
	std::cout << "dt: " << sim.get_dt() << std::endl;
	std::cout << std::setprecision(9) << radius << std::endl;

	for (int i = 0; i < 1000; i++) {
		sim.step(bubble, openCL);
		sim.step(bubble, openCL);
		sim.step(bubble, openCL);
		sim.step(bubble, openCL);
		sim.step(bubble, openCL);
		sim.step(bubble, openCL);
		sim.step(bubble, openCL);
		sim.step(bubble, openCL);
		sim.step(bubble, openCL);
		sim.step(bubble, openCL);
		std::cout << std::setprecision(15) << "dP: " << sim.getdPressureStep() << std::endl;
		std::cout << "Bubble speed: " << bubble.getSpeed() << std::endl;
		if (std::isnan(bubble.getSpeed())) {
	        	std::cerr << "Abort due to nan" << std::endl;
			exit(1);
		}
	}

}
