#pragma once
#include "base.h"
#include "bubble.h"
#include "openclwrapper.h"

#include <random>


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

	// Simulation density and distribuion parameters:
	numType m_rhoTrueSim, m_rhoFalseSim, m_nTrueSim, m_nFalseSim;
	numType m_rhoTrueSimInitial, m_rhoFalseSimInitial, m_nTrueSimInitial, m_nFalseSimInitial;
	std::vector<numType> m_cpdFalse, m_pFalse, m_cpdTrue, m_pTrue;


	// Simulation energy parameters
	numType m_energyTotalInitial, m_energyTotal, m_energyParticlesInitial, m_energyParticles, m_energyBubble, m_energyBubbleInitial;
	// Simulation last step
	numType m_energyBubbleLastStep, m_energyParticlesLastStep;

	// Sim time paramters:
	// Cumulative time
	numType m_time;
	// One step time length
	numType m_dt;
	numType m_dPressureStep;
	
	// Particle info
	std::vector<numType> m_X;
	std::vector<numType> m_P;
	std::vector<numType> m_E;
	std::vector<numType> m_M;
	// Pressure from particle-bubble collisions
	std::vector<numType> m_dP;

	// Logging parameters
	std::vector<int8_t> m_interactedFalse;
	std::vector<int8_t> m_passedFalse;
	// True vaccum interaction also means that the particle gets through
	std::vector<int8_t> m_interactedTrue;

	// Random number generator
	int m_seed;
	std::random_device m_randDev;
	std::mt19937_64 m_generator;
	std::uniform_real_distribution<numType> m_distribution;

public:
	Simulation() {}
	Simulation(
		int t_seed, numType t_alpha, numType t_massTrue, numType t_massFalse, numType t_temperatureTrue, numType t_temperatureFalse,
		unsigned int t_particleCountTrue, unsigned int t_particleCountFalse, numType t_coupling
	);
	Simulation& operator=(const Simulation& t) { return *this; }
	
	void set_dt(numType t_dt);

	std::vector<numType>& getReferenceX() { return m_X; }
	std::vector<numType>& getReferenceP() { return m_P; }
	std::vector<numType>& getReferenceM() { return m_M; }
	std::vector<numType>& getReferenceE() { return m_E; }
	std::vector<numType>& getReference_dP() { return m_dP; }
	std::vector<int8_t>& getReferenceInteractedFalse() { return m_interactedFalse; }
	std::vector<int8_t>& getReferencePassedFalse() { return m_passedFalse; }
	std::vector<int8_t>& getReferenceInteractedTrue() { return m_interactedTrue; }
	numType& getReference_dt() { return m_dt; }
	numType& getReferenceMassFalse() { return m_massFalse; }
	numType& getReferenceMassTrue() { return m_massTrue; }
	numType& getReferenceMassDelta2() { return m_massDelta2; }

	
	numType getNumberDensityFalse() { return m_nFalse; }
	numType getEnergyDensityFalse() { return m_rhoFalse; }
	numType getNumberDensityTrue() { return m_nTrue; }
	numType getEnergyDensityTrue() { return m_rhoTrue; }

	numType getMassFalse() { return m_massFalse; }
	numType getMassTrue() { return m_massTrue; }
	numType getTime() { return m_time; }
	numType get_dt() { return m_dt; }
	numType getdPressureStep() { return m_dPressureStep; }
	int getParticleCountTotal() { return m_particleCountTotal; }

	numType getBubbleEnergy() { return m_energyBubble; }
	numType getBubbleEnergyLastStep() { return m_energyBubbleLastStep; }

	numType getParticlesEnergy() { return m_energyParticles; }
	numType getParticlesEnergyLastStep() { return m_energyParticlesLastStep; }
	
	numType getTotalEnergy() { return m_energyTotal; }
	numType getTotalEnergyInitial() { return m_energyTotalInitial; }

	std::vector<numType>& getCPDFalseRef() { return m_cpdFalse; }
	std::vector<numType>& getPFalseRef() { return m_pFalse; }
	std::vector<numType>& getCPDTrueRef() { return m_cpdTrue; }
	std::vector<numType>& getPTrueRef() { return m_pTrue; }

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
	numType interp(numType t_value, std::vector<numType>& t_x, std::vector<numType>& t_y);

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
	numType countParticlesEnergy();
	numType countParticlesEnergy(numType t_radius1);
	numType countParticlesEnergy(numType t_radius1, numType t_radius2);

	void step(Bubble& bubble, OpenCLWrapper& openCLWrapper);
	/* 
	void step(Bubble bubble, OpenCLWrapper openCLWrapper, std::string device);
	

	void stepCPU(Bubble bubble);

	void stepGPU(Bubble bubble, OpenCLWrapper openCLWrapper);
	*/
	/*
		Runs one time on GPU or CPU.
	*/
	

}; 