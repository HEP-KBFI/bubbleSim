#pragma once
#include "base.h"
#include <random>
#include "bubble.h"

class Simulation {
	/*
	False at the end of variable means False vaccum (lower mass)
	True at the end of variable means False vaccum (higher mass)
	*/

	// Simulation parameters
	numType m_alpha;
	numType m_coupling;
	// Masses of particles in true and false vacuum
	numType m_massTrue, m_massFalse, m_massDelta2;
	// Temperatures in true and false vacuum
	numType m_temperatureTrue, m_temperatureFalse;
	// Particle counts total / true vacuum / false vacuum
	int m_particleCountTotal, m_particleCountTrue, m_particleCountFalse;
	// Energy and number desnities (calculated from distribution)
	numType m_rhoTrue, m_rhoFalse, m_nTrue, m_nFalse;
	// Cumulative probability densities
	// std::vector<numType> m_cpdTrue, m_cpdFalse, m_pValuesTrue, m_pValuesFalse;

	// Simulation density parameters:
	numType m_rhoTrueSim, m_rhoFalseSim, m_nTrueSim, m_nFalseSim;
	numType m_rhoTrueSimInitial, m_rhoFalseSimInitial, m_nTrueSimInitial, m_nFalseSimInitial;
	
	// Simulation energy parameters
	numType m_energyTotalInitial, m_energyTotal, m_energyParticlesInitial, m_energyParticles, m_energyBubble, m_energyBubbleInitial;
	// Sim time paramters:
	// Cumulative time
	numType m_time;
	// One step time length
	numType m_dt;
	
	// Particle info
	std::vector<numType> m_X;
	std::vector<numType> m_P;
	std::vector<numType> m_E;
	std::vector<numType> m_M;
	// Pressure from particle-bubble collisions
	std::vector<numType> m_dP;

	// Logging parameters
	std::vector<int8_t> m_InteractedFalse;
	std::vector<int8_t> m_PassedFalse;
	// True vaccum interaction also means that the particle gets through
	std::vector<int8_t> m_InteractedTrue;

	// Random number generator
	int m_seed;
	std::random_device m_randDev;
	std::mt19937_64 m_generator;
	std::uniform_real_distribution<numType> m_distribution;


	Simulation();
	Simulation(
		int t_seed, numType t_alpha, numType t_massTrue, numType t_massFalse, numType t_temperatureTrue, numType t_temperatureFalse,
		unsigned int t_particleCountTrue, unsigned int t_particleCountFalse, numType t_coupling
	);

	void set_dt(numType t_dt);

	// Particle functions
	numType getParticleEnergy(u_int i) { return m_E[i]; }
	numType getParticleMass(u_int i) { return m_M[i]; }
	numType calculateParticleRadius(u_int i);
	numType calculateParticleMomentum(u_int i);
	numType calculateParticleEnergy(u_int i);
	
	// Calculate distributions
	void calculateCPD(numType t_mass, numType t_temperature, numType t_dp, numType t_pMax, int t_vectorSize, std::vector<numType>& t_cpd, std::vector<numType>& t_p);
	numType calculateNumberDensity(numType t_mass, numType t_temperature, numType t_dp, numType t_pMax);
	numType calculateEnergyDensity(numType t_mass, numType t_temperature, numType t_dp, numType t_pMax);

	// Sampling and generating
	numType interp(numType t_value, numType& result, std::vector<numType>& t_x, std::vector<numType>& t_y);

	void generateRandomDirectionPush(numType& t_radius, std::vector<numType>& t_resultVector);
	void generateRandomDirectionReplace(numType& t_radius, std::vector<numType>& t_resultVector);
	void generateParticleMomentum(std::vector<numType>& t_cpd, std::vector<numType>& t_p, numType& t_pResult, std::vector<numType>& t_resultPushVector);
	void generatePointInBoxPush(numType& t_SideHalf, std::vector<numType>& t_result);
	void generatePointInBoxReplace(numType& t_SideHalf, std::vector<numType>& t_result);
	void generatePointInBoxPush(numType& t_xSideHalf, numType& t_ySideHalf, numType& t_zSideHalf, std::vector<numType>& t_result);
	void generatePointInBoxReplace(numType& t_xSideHalf, numType& t_ySideHalf, numType& t_zSideHalf, std::vector<numType>& t_result);

	void generateNParticlesInBox(numType t_mass, numType& t_sideHalf, u_int t_N, std::vector<numType>& t_cpd, std::vector<numType>& t_p);
	void generateNParticlesInBox(numType t_mass, numType& t_radiusIn, numType& t_sideHalf, u_int t_N, std::vector<numType>& t_cpd, std::vector<numType>& t_p);
	void generateNParticlesInBox(numType t_mass, numType& t_xSideHalf, numType& t_ySideHalf, numType& t_zSideHalf, u_int t_N, std::vector<numType>& t_cpd, std::vector<numType>& t_p);
	void generateNParticlesInBox(numType t_mass, numType& t_radiusIn, numType& t_xSideHalf, numType& t_ySideHalf, numType& t_zSideHalf, u_int t_N, std::vector<numType>& t_cpd, std::vector<numType>& t_p);
	void generateNParticlesInSphere(numType t_mass, numType& t_radius1, u_int t_N, std::vector<numType>& t_cpd, std::vector<numType>& t_p);
	void generateNParticlesInSphere(numType t_mass, numType& t_radius1, numType t_radius2, u_int t_N, std::vector<numType>& t_cpd, std::vector<numType>& t_p);


	// Get values from the simulation
	numType countParticleNumberDensity(numType t_radius1);
	numType countParticleNumberDensity(numType t_radius1, numType t_radius2);
	numType countParticleEnergyDensity(numType t_radius1);
	numType countParticleEnergyDensity(numType t_radius1, numType t_radius2);
	numType countParticlesEnergy(numType t_radius1);
	numType countParticlesEnergy(numType t_radius1, numType t_radius2);

	void step(Bubble bubble, OpenCLWrapper openCLWrapper, std::string device);
	/*
		Runs one time on GPU or CPU.
	*/

	void stepCPU(Bubble bubble);

	void stepGPU(Bubble bubble, OpenCLWrapper openCLWrapper);

	

}; 