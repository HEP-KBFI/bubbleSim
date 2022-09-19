#include "source.h"

MainType takeStep(Simulation& sim, Bubble& bubble, MultiprocessingCL& multiprocessing) {
	// Write (R_b + V_b * dt)^2 value to buffer 
	MainType cumulative_dP = 0;
	multiprocessing.writeBubbleRadiusSpeedDt2Buffer(sim.m_dt, bubble);
	// Run kernel once
	multiprocessing.runStep(sim.m_particleCount);
	// Read dP from the buffer
	multiprocessing.readStep(sim.m_particleCount, sim.m_dP);
	for (unsigned int i = 0; i < sim.m_particleCount; i++) {
		cumulative_dP -= sim.m_dP[i];
	}
	cumulative_dP /= bubble.m_area;
	bubble.evolveWall(sim.m_dt, cumulative_dP);

	// Update buffer's bubble parameters
	multiprocessing.writeStep(bubble);
	sim.m_volumeIn = 4. * M_PI * pow(bubble.m_radius, 3) / 3;

	return cumulative_dP;
}

void createFileNameFromCurrentDate(char* name) {
        std::time_t t_cur = time(NULL); 
	const auto* t_local = localtime(&t_cur);
	std::strftime(name, 50, "%j_%Y_%H_%M_%S", t_local);
}

void countParticleDifferenceRadiusMass(int& t_particlesInMass, int& t_particlesInRadius, int& t_particlesOutMass, int& t_particlesOutRadius, Bubble& t_bubble, Simulation& t_sim) {
	t_particlesInMass = 0;
	t_particlesInRadius = 0;
	t_particlesOutMass = 0;
	t_particlesOutRadius = 0;

	for (u_int i = 0; i < t_sim.m_particleCount; i++) {
		if (t_sim.getParticleRadius(i) > t_bubble.m_radius) {
			t_particlesOutRadius += 1;
		}
		else {
			t_particlesInRadius += 1;
		}

		if (t_sim.m_M[i] == t_sim.m_massOut) {
			t_particlesOutMass += 1;
		}
		else {
			t_particlesInMass += 1;
		}
	}
}

int main() {
	bool b_logMomentum = false;
	bool b_logNumberDensity = false;
	bool b_logEnergyDensity = false;
	bool b_logBasicData = true;
	bool b_countMassRadiusDifference = true;

	u_int logFrequency = 50;
	unsigned int sim_steps = 1'000; // Number of steps for simulation


	std::string main_path = "./";
	std::string data_path = main_path + "data/";
	// std::string config_path = "C:/dev/Bubble/config/";


	// Simulation variables
	MainType dP = (MainType) 0; // Energy change in one step


	// Counting variables
	int countParticlesMassIn = 0;
	int countParticlesMassOut = 0;
	int countParticlesRadiusIn = 0;
	int countParticlesRadiusOut = 0;

	/*
	=================== Initialization parameters ===================
	*/

	using std::chrono::high_resolution_clock;
	using std::chrono::duration_cast;
	using std::chrono::duration;
	using std::chrono::milliseconds;

	// Read in simulation parameters
	// std::ifstream f(config_path + "test2.json");
	// nlohmann::json j_file = nlohmann::json::parse(f);
	// Assignment
	MainType alpha = -0.6; // j_file["alpha"]; //0.6;
	MainType massInside = 0.01; // j_file["mass_inside"]; // 1.;
	MainType massOutside = 10.; // j_file["mass_outside"]; // 0.01;
	MainType temperatureIn = 0.5; // j_file["T_in"]; // 0.5;
	MainType temperatureOut = 0.5; // j_file["T_out"]; // 0.5;
	unsigned int countParticlesIn = 1'000'000; // j_file["N_in"]; // 0;
	unsigned int countParticlesOut = 0;// j_file["N"]; // 10'000'000;
	MainType coupling = 0.;// j_file["coupling"]; // 0;
	MainType initialBubbleSpeed = -0.2; // j_file["bubble_speed"]; // 0;



	/*
	=================== Used parameters ===================
	*/


	auto t1 = high_resolution_clock::now();

	/*
	=================== Build simulation ===================
		1. Define bubble radius (this might need to generate simulation size before) -> Use density parameter from initializing Simulation class
		2. Generate particles' momentums
		3. Generate particles (in and out of the bubble)
		4. Define bubble parameters (radius, speed, alpha, dVT, dV, sigma)
		5. Create bubble
		6. Define dt value
		7. Create MultiprocessingCL class variable for OpenCL buffers and other OpenCL variables
	*/

	Simulation sim(alpha, massInside, massOutside, temperatureIn, temperatureOut, countParticlesIn, countParticlesOut, coupling);
	/*
		Radius when excluding volume inside the true vacuum bubble
		n = N/V and V = R_max ^ 3 * [8 - 4*pi/3 * (radius_per_rc/max_radius_per_rc)^3]

		*****  Sphere radius when excluding bubble initial radius
		MainType temp_simulationMaxRadius = cbrt(sim.m_particleCountOut / ((8 - 4 * M_PI / 3 * pow(radiusBubblePerRc / radiusMaxPerRc, 3)) * sim.m_nOut));
		***** Cube radius for simulation
		MainType simulationMaxRadius = cbrt(sim.m_particleCountOut / sim.m_nOut) / 2; // Cube
	*/

	// =========== True vaccum inside the bubble ===========
	/*
	MainType simulationMaxRadius = cbrt(3*sim.m_particleCountOut / (4 * M_PI * sim.m_nOut)); // Sphere
	MainType simulationCriticalRadius = simulationMaxRadius / radiusMaxPerRc;
	MainType simulationInitialBubbleRadius = (MainType)(simulationCriticalRadius * radiusBubblePerRc);
	sim.m_volumeInInitial = 4. * M_PI * pow(simulationInitialBubbleRadius, 3) / 3;
	sim.m_volumeIn = sim.m_volumeInInitial;
	sim.m_volumeOutInitial = sim.m_particleCount / sim.m_nOut - sim.m_volumeInInitial;
	sim.m_volumeOut = sim.m_volumeOutInitial;
	sim.generateParticleMomentum(sim.m_particleCount, sim.m_massOut, sim.m_cpdOut, sim.m_pValuesOut);
	sim.generateParticleCoordinateSphere(sim.m_particleCountOut, simulationInitialBubbleRadius, simulationMaxRadius);
	sim.m_dt = simulationMaxRadius / 1000;

	sim.m_rhoOutSimInitial = sim.countParticleEnergyDensity(sim.m_particleCountIn, sim.m_particleCount, sim.m_volumeOutInitial);
	sim.m_nOutSimInitial = sim.countParticleNumberDensity(sim.m_particleCountOut, sim.m_volumeOutInitial);
	Bubble bubble(simulationInitialBubbleRadius, initialBubbleSpeed, alpha, sim.m_rhoOutSimInitial, sim.m_nOutSimInitial, simulationCriticalRadius, sim.m_temperatureOut);
	*/
	
	// =========== False vaccum inside the bubble ===========

	// 1. Calculates bubble radius
	MainType simulationInitialBubbleRadius = cbrt((double)(3 * sim.m_particleCountIn) / (4 * M_PI * sim.m_nIn));
	sim.m_volumeInInitial = sim.m_particleCountIn / sim.m_nIn;
	sim.m_volumeIn = sim.m_volumeInInitial;
	// 2. and 3. Generates particle momentums and coordinates
	sim.generateParticleMomentum(sim.m_particleCountIn, sim.m_massIn, sim.m_cpdIn, sim.m_pValuesIn);
	sim.generateParticleCoordinateSphere(sim.m_particleCountIn, 0., simulationInitialBubbleRadius);

	// 4. Define bubble parameters
	sim.m_rhoInSimInitial = sim.countParticleEnergyDensity(0, sim.m_particleCountIn, sim.m_volumeInInitial);
	sim.m_nInSimInitial = sim.countParticleNumberDensity(sim.m_particleCountIn, sim.m_volumeInInitial);
	MainType criticalRadius = simulationInitialBubbleRadius / 20;
	MainType dVT = -alpha * sim.m_rhoInSimInitial;
	MainType dV = dVT - sim.m_temperatureIn * sim.m_nInSimInitial;
	MainType sigma = abs(dV) * criticalRadius / (MainType)2.0;

	// 5. Create bubble
	// radius, speed, alpha, dVT, dV, sigma
	Bubble bubble(simulationInitialBubbleRadius, initialBubbleSpeed, dVT, dV, sigma);
	std::string kernelFile = "./bubbleSim/kernel.cl";

	// 6. Define dt value
	sim.m_dt = simulationInitialBubbleRadius / 1000;
	
	// 7. Class for OpenCL buffers and other variables
	MultiprocessingCL mainMP(kernelFile, "step_double", sim, bubble);

	/*
	=================== Display initial information ===================
	*/
	sim.calculateTotalEnergyInitial(bubble);

	std::cout << std::endl << "========== Initialization ==========" << std::endl;
	std::cout << std::endl << "***** Simulation paramters" << std::endl << std::endl;
	// std::cout << "Critical/Initial bubble/Max radius: " << simulationCriticalRadius << "/" << simulationInitialBubbleRadius << "/" << simulationMaxRadius << std::endl;
	std::cout << std::setprecision(10) << "Initial bubble radius: " << simulationInitialBubbleRadius << std::endl;
	std::cout << "dt: " << sim.m_dt << std::endl;
	std::cout << std::setprecision(6) << "Total particles energy: " << sim.m_energyParticles << ", Bubble energy: " << sim.m_energyBubble << ", Total energy: " << sim.m_energyTotal << std::endl;
	// std::cout << std::setprecision(6) << "Number density fraction (sim./theor.)" << sim.m_nOutSimInitial / sim.m_nOut << ", " << "Energy density fraction (sim./theor.)" << sim.m_rhoOutSimInitial / sim.m_rhoOut << std::endl;
	std::cout << std::setprecision(6) << "Number density fraction (sim./theor.) " << sim.m_nInSimInitial / sim.m_nIn << ", " << "Energy density fraction (sim./theor.) " << sim.m_rhoInSimInitial / sim.m_rhoIn << std::endl;
	std::cout << std::endl << "***** Particle parameters" << std::endl << std::endl;
	std::cout << "T(in/out): " << temperatureIn << " / " << temperatureOut << ", coupling: " << coupling << std::endl;
	std::cout << "Mass in: " << massInside << ", Mass out: " << massOutside << ", Delta m^2: " << sim.m_massDelta2 << std::endl;
	std::cout << "Number of particles (in/out/total): " << sim.m_particleCountIn << "/" << sim.m_particleCountOut << "/" << sim.m_particleCount << std::endl;
	std::cout << std::endl << "***** Bubble initial parameters" << std::endl << std::endl;
	std::cout << "Alpha: " << alpha << ", Sigma: " << bubble.m_sigma << ", dV: " << bubble.m_dV << ", dVT: " << bubble.m_dVT << std::endl;
	// std::cout << "Critical/Initial radius/Ratio: " << simulationCriticalRadius << "/" << bubble.m_radius << "/" << bubble.m_radius / simulationCriticalRadius << std::endl;
	std::cout << "Speed: " << bubble.m_speed << ", Energy: " << sim.m_energyBubble << std::endl;


	/*
	=================== Data streaming ===================
	*/
	MainType streamMaxMomentum = 15 * std::max(temperatureIn, temperatureOut);
	MainType streamMaxRadius = 2 * simulationInitialBubbleRadius;
	// Create new file name in a format: %j_%Y_%H_%M_%S_****.csv
	char id_name_string[100];
	createFileNameFromCurrentDate(id_name_string);
	// Log id into general file list
	std::fstream id_file(main_path + "data.txt", std::ios::app);
	// Log main parameters
	id_file << id_name_string << "," << alpha << "," << massInside << "," << massOutside << "," << temperatureIn << "," << temperatureOut << ",";
	id_file << countParticlesIn << countParticlesOut << "," << coupling << std::endl;
	std::cout << "File prefix: " << id_name_string << std::endl << std::endl;
	
	std::fstream dataStream;
	std::fstream numberDensityStream;
	std::fstream energyDensityStream;
	std::fstream momentumInStream;
	std::fstream momentumOutStream;

	std::vector<u_int> binsNumberDensity;
	std::vector<MainType> binsEnergyDensity;
	std::vector<u_int> binsMomentumIn;
	std::vector<u_int> binsMomentumOut;

	u_int countBinsNumberDensity;
	u_int countBinsEnergyDensity;
	u_int countBinsMomentum;

	MainType drNumberDensity;
	MainType rMaxNumberDensity;
	MainType drEnergyDensity;
	MainType rMaxEnergyDensity;
	MainType dpMomentum;
	MainType pMaxMomentum;
	
	MainType oldBubbleEnergy = bubble.m_energy;
	MainType oldParticleEnergy = sim.m_energyParticlesInitial;

	if (b_logBasicData) {
		// Log basic data
		/*
		time, dP, Bubble r, Bubble v,
Bubble energy, Bubble energy change, Particle energy, Particle energy change, Particle energy in, total_energy/initial_total_energy,
Particle count in, Particle count interacted in, Particle count interacted in and passed,
Particle count interacted out, Particle count interacted ot and passed
		*/
		dataStream = std::fstream(data_path + id_name_string + "_data.csv", std::ios::out | std::ios::in | std::ios::trunc);
		dataStream << "time,dP,Bubble r,Bubble v,Bubble E,Bubble dE,Particle E,Particle dE,Particle E in,E,";
		dataStream << "Count in,Interacted in,Passed in,Interacted out,Passsed out" << std::endl;
		dataStream << sim.m_time << "," << dP << "," << bubble.m_radius << "," << bubble.m_speed << ",";
		dataStream << bubble.m_energy << "," << 0 << "," << sim.m_energyParticlesInitial << "," << 0 << ",";
		dataStream << sim.countParticlesEnergyIn(bubble) << "," << 1 << ",";
		dataStream << sim.m_particleCountIn << "," << 0 << "," << 0 << ",";
		dataStream << 0 << "," << 0 << std::endl;
	}
	if (b_logNumberDensity) {
		// Log number density profile
		countBinsNumberDensity = 1000;
		rMaxNumberDensity = streamMaxRadius;
		drNumberDensity = rMaxNumberDensity / countBinsNumberDensity;
		binsNumberDensity = std::vector<u_int>(countBinsNumberDensity, 0);
		numberDensityStream = std::fstream(data_path + id_name_string + "_numberDensity.csv", std::ios::out | std::ios::in | std::ios::trunc);
		for (u_int i = 1; i <= countBinsNumberDensity; i++) {
			numberDensityStream << i * drNumberDensity;
			if (i != countBinsNumberDensity) {
				numberDensityStream << ",";
			}
		}
		numberDensityStream << std::endl;
		streamNumberDensityDistribution(numberDensityStream, binsNumberDensity, sim, drNumberDensity, rMaxNumberDensity);
	}
	if (b_logEnergyDensity) {
		// Log energy density profile
		countBinsEnergyDensity = 1000;
		rMaxEnergyDensity = streamMaxRadius;
		drEnergyDensity = rMaxEnergyDensity / countBinsEnergyDensity;
		binsEnergyDensity = std::vector<MainType>(countBinsEnergyDensity, 0.);
		energyDensityStream = std::fstream(data_path + id_name_string + "_energyDensity.csv", std::ios::out | std::ios::in | std::ios::trunc);
		for (u_int i = 1; i <= countBinsEnergyDensity; i++) {
			energyDensityStream << i * drEnergyDensity;
			if (i != countBinsEnergyDensity) {
				energyDensityStream << ",";
			}
		}
		energyDensityStream << std::endl;
		streamEnergyDensityDistribution(energyDensityStream, binsEnergyDensity, sim, drEnergyDensity, rMaxEnergyDensity);
	}
	if (b_logMomentum) {
		// Log In and Out of bubble momentum distributions
		countBinsMomentum = 1000;
		pMaxMomentum = 15 * std::max(sim.m_temperatureIn, sim.m_temperatureOut);
		dpMomentum = pMaxMomentum / countBinsMomentum;
		binsMomentumIn = std::vector<u_int>(countBinsMomentum, 0);
		binsMomentumOut = std::vector<u_int>(countBinsMomentum, 0);
		momentumInStream = std::fstream(data_path + id_name_string + "_momentumIn.csv", std::ios::out | std::ios::in | std::ios::trunc);
		momentumOutStream = std::fstream(data_path + id_name_string + "_momentumOut.csv", std::ios::out | std::ios::in | std::ios::trunc);
		for (u_int i = 1; i <= countBinsMomentum; i++) {
			momentumInStream << i * dpMomentum;
			momentumOutStream << i * dpMomentum;
			if (i != countBinsMomentum) {
				momentumInStream << ",";
				momentumOutStream << ",";
			}
		}
		momentumInStream << std::endl;
		momentumOutStream << std::endl;
		streamMomentumDistributionsInOut(
			momentumInStream, momentumOutStream,
			binsMomentumIn, binsMomentumOut,
			sim, bubble, dpMomentum, pMaxMomentum);
	}

	
	/*
	=================== Simulation ===================
	*/

	bool b_terminated = false;
	for (unsigned int i = 0; i <= sim_steps; i++) {
		sim.m_time += sim.m_dt;
		dP = takeStep(sim, bubble, mainMP);

		/*
		========== Read out process ==========
		*/
		// std::cout << "dP: " << dP << ", R" << bubble.m_radius << ", V" << bubble.m_speed << std::endl;
		// std::cout << "Radius: " << bubble.m_radius << ", Speed: " << bubble.m_speed << std::endl;
		// std::cout << "Bubble energy: " << bubbleE << ", Particle energy: " << particlesE << std::endl;

		
		// =================== Log data ===================
		if (i % logFrequency==0) {

			mainMP.m_queue.enqueueReadBuffer(mainMP.m_bufferX, CL_TRUE, 0, 3 * sim.m_particleCount * sizeof(MainType), sim.m_X.data());
			mainMP.m_queue.enqueueReadBuffer(mainMP.m_bufferE, CL_TRUE, 0, sim.m_particleCount * sizeof(MainType), sim.m_E.data());
			mainMP.m_queue.enqueueReadBuffer(mainMP.m_bufferInteractedFalse, CL_TRUE, 0, sim.m_particleCount * sizeof(int8_t), sim.m_InteractedFalse.data());
			mainMP.m_queue.enqueueReadBuffer(mainMP.m_bufferPassedFalse, CL_TRUE, 0, sim.m_particleCount * sizeof(int8_t), sim.m_PassedFalse.data());
			mainMP.m_queue.enqueueReadBuffer(mainMP.m_bufferInteractedTrue, CL_TRUE, 0, sim.m_particleCount * sizeof(int8_t), sim.m_InteractedTrue.data());
			mainMP.m_queue.enqueueReadBuffer(mainMP.m_bufferPassedTrue, CL_TRUE, 0, sim.m_particleCount * sizeof(int8_t), sim.m_PassedTrue.data());
			streamData(dataStream, sim, bubble, dP, oldBubbleEnergy, oldParticleEnergy);
			std::fill(sim.m_InteractedFalse.begin(), sim.m_InteractedFalse.end(), 0);
			std::fill(sim.m_PassedFalse.begin(), sim.m_PassedFalse.end(), 0);
			std::fill(sim.m_InteractedTrue.begin(), sim.m_InteractedTrue.end(), 0);
			std::fill(sim.m_PassedTrue.begin(), sim.m_PassedTrue.end(), 0);
			mainMP.m_queue.enqueueWriteBuffer(mainMP.m_bufferInteractedFalse, CL_TRUE, 0, sim.m_particleCount * sizeof(int8_t), sim.m_InteractedFalse.data());
			mainMP.m_queue.enqueueWriteBuffer(mainMP.m_bufferPassedFalse, CL_TRUE, 0, sim.m_particleCount * sizeof(int8_t), sim.m_PassedFalse.data());
			mainMP.m_queue.enqueueWriteBuffer(mainMP.m_bufferInteractedTrue, CL_TRUE, 0, sim.m_particleCount * sizeof(int8_t), sim.m_InteractedTrue.data());
			mainMP.m_queue.enqueueWriteBuffer(mainMP.m_bufferPassedTrue, CL_TRUE, 0, sim.m_particleCount * sizeof(int8_t), sim.m_PassedTrue.data());

			if (b_countMassRadiusDifference) {
				mainMP.m_queue.enqueueReadBuffer(mainMP.m_bufferM, CL_TRUE, 0, sim.m_particleCount * sizeof(MainType), sim.m_M.data());
				countParticleDifferenceRadiusMass(countParticlesMassIn, countParticlesRadiusIn, countParticlesMassOut, countParticlesRadiusOut, bubble, sim);
				std::cout << "Count difference inside/outside: " << (countParticlesMassIn - countParticlesRadiusIn) << " / " << (countParticlesMassOut - countParticlesRadiusOut) << std::endl;
			}
			
			
			if (b_logEnergyDensity) {
				streamEnergyDensityDistribution(energyDensityStream, binsEnergyDensity, sim, drEnergyDensity, rMaxEnergyDensity);
			}
			if (b_logNumberDensity) {
				streamNumberDensityDistribution(numberDensityStream, binsNumberDensity, sim, drNumberDensity, rMaxNumberDensity);
			}
			if (b_logMomentum) {
				mainMP.m_queue.enqueueReadBuffer(mainMP.m_bufferP, CL_TRUE, 0, sim.m_particleCount * sizeof(MainType), sim.m_P.data());
				streamMomentumDistributionsInOut(
					momentumInStream, momentumOutStream,
				binsMomentumIn, binsMomentumOut,
					sim, bubble, dpMomentum, pMaxMomentum);
			}
		}

		// =================== Catch undefined values and end simulation if unphysical condition is met ===================
		if (std::isnan(sim.m_energyParticles)) {
			std::cout << std::endl << std::endl << "Particle energy is not defined." << std::endl;
			std::cout << "E_p: " << sim.m_energyParticles << ", E_b: " << sim.m_energyBubble << ", dP: " << dP << ", R_b: " << bubble.m_radius << ", V_b: " << bubble.m_speed << std::endl;
			std::cout << "Stopping simulation." << std::endl;
			b_terminated = true;
			break;
		}
		else if (std::isnan(sim.m_energyBubble)) {
			std::cout << std::endl << std::endl << "Bubble energy is not defined." << std::endl;
			std::cout << "E_p: " << sim.m_energyParticles << ", E_b: " << sim.m_energyBubble << ", dP: " << dP << ", R_b: " << bubble.m_radius << ", V_b: " << bubble.m_speed << std::endl;
			std::cout << "Stopping simulation." << std::endl;
			b_terminated = true;
			break;
		}
		else if (std::isnan(dP)) {
			std::cout << std::endl << std::endl << "Energy change is not defined." << std::endl;
			std::cout << "E_p: " << sim.m_energyParticles << ", E_b: " << sim.m_energyBubble << ", dP: " << dP << ", R_b: " << bubble.m_radius << ", V_b: " << bubble.m_speed << std::endl;
			std::cout << "Stopping simulation." << std::endl;
			b_terminated = true;
			break;
		}
		else if (std::isnan(bubble.m_radius) || bubble.m_radius < 0.) {
			std::cout << std::endl << std::endl << "Bubble radius is not defined or negative." << std::endl;
			std::cout << "E_p: " << sim.m_energyParticles << ", E_b: " << sim.m_energyBubble << ", dP: " << dP << ", R_b: " << bubble.m_radius << ", V_b: " << bubble.m_speed << std::endl;
			std::cout << "Stopping simulation." << std::endl;
			b_terminated = true;
			break;
		}
		else if (std::isnan(bubble.m_speed)) {
			std::cout << std::endl << std::endl << "Bubble speed is not defined." << std::endl;
			std::cout << "E_p: " << sim.m_energyParticles << ", E_b: " << sim.m_energyBubble << ", dP: " << dP << ", R_b: " << bubble.m_radius << ", V_b: " << bubble.m_speed << std::endl;
			std::cout << "Stopping simulation." << std::endl;
			b_terminated = true;
			break;
		}
	}
	
	mainMP.m_queue.enqueueReadBuffer(mainMP.m_bufferM, CL_TRUE, 0, sim.m_particleCount * sizeof(MainType), sim.m_M.data());
	countParticleDifferenceRadiusMass(countParticlesMassIn, countParticlesRadiusIn, countParticlesMassOut, countParticlesRadiusOut, bubble, sim);

	if (b_terminated) {
		std::cout << std::endl << "========== Final results ==========" << std::endl;
		std::cout << "Simulation was terminated as undefined value occured." << std::endl;

		std::cout << "Count particles by mass and distance:" << std::endl;
		std::cout << "Count difference inside/outside: " << ((int)countParticlesMassIn - (int)countParticlesRadiusIn) << " / " << ((int)countParticlesMassOut - (int)countParticlesRadiusOut) << std::endl;
	}
	else {
		std::cout << std::endl << "========== Final results ==========" << std::endl;
		std::cout << "Count particles by mass and distance:" << std::endl;
		std::cout << "Count difference inside/outside: " << (countParticlesMassIn - countParticlesRadiusIn) << " / " << (countParticlesMassOut - countParticlesRadiusOut) << std::endl;
		std::cout << "Total particles energy: " << sim.m_energyParticles << ", Bubble energy: " << sim.m_energyBubble << ", Total energy: " << sim.m_energyTotal << std::endl;
		std::cout << "Energy/Initial Energy: " << sim.m_energyTotal / sim.m_energyTotalInitial << std::endl;		
	}

	auto t2 = std::chrono::high_resolution_clock::now();
	auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
	int seconds = (int)(ms_int.count() / 1000) % 60;
	int minutes = ((int)(ms_int.count() / (1000 * 60)) % 60);
	int hours = ((int)(ms_int.count() / (1000 * 60 * 60)) % 24);

	std::cout << std::endl << "Time taken: " << hours << " h " << minutes << " m " << seconds << " s " << std::endl << std::endl;
}


