#pragma once
#include "base.h"

#include <ios>
#include <random>
#include "opencl.h"


class Bubble {

public:
	MainType m_radius;
	MainType m_speed;
	MainType m_gamma;
	MainType m_dV;
	MainType m_dVT;
	MainType m_sigma;
	MainType m_area;
	MainType m_volume;
	MainType m_gammaSpeed;
	MainType m_radius2;
	MainType m_energy;
	MainType m_radiusSpeedDt2;

	Bubble();
	Bubble(MainType t_radius, MainType t_speed, MainType t_dVT, MainType t_dV, MainType t_sigma);
	MainType calculateRadius2() { return m_radius * m_radius; };
	MainType calculateRadiusSpeedDt2(MainType dt);
	MainType calculate_gammaSpeed() { return m_gamma * m_speed; };
	MainType calculateEnergy();

	void evolveWall(MainType dt, MainType dP);
	// MainType getEnergy();
};

class Simulation {

public:
	MainType m_alpha;
	MainType m_massIn, m_massOut, m_massDelta2;
	// Particle temperatures for generating distributions:
	MainType m_temperatureIn, m_temperatureOut;
	// Number of particles
	u_int m_particleCount, m_particleCountIn, m_particleCountOut;
	// Generated volume sizes for the 'bubble' and 'outside':
	MainType m_volumeIn, m_volumeOut, m_volumeInInitial, m_volumeOutInitial;
	// Density parameters (density / number density):
	MainType m_rhoIn, m_rhoOut, m_nIn, m_nOut;
	// Cumulative probability density
	std::vector<MainType> m_cpdIn, m_cpdOut, m_pValuesIn, m_pValuesOut;
	// Simulation density parameters:
	MainType m_rhoInSim, m_rhoOutSim, m_nInSim, m_nOutSim;
	MainType m_rhoInSimInitial, m_rhoOutSimInitial, m_nInSimInitial, m_nOutSimInitial;
	// Bubble radius parameters
	MainType m_radiusInitial, m_radius, m_radiusCritical, m_radiusSim;
	// Simulation energy parameters
	MainType m_energyTotalInitial, m_energyTotal, m_energyParticlesInitial, m_energyParticles, m_energyBubble, m_energyBubbleInitial;
	MainType m_energyParticlePrevStep;
	// Sim paramters:
	MainType m_time;
	MainType m_dt;
	// Physical parameters
	MainType m_coupling;

	std::vector<MainType> m_X;
	std::vector<MainType> m_P;
	std::vector<MainType> m_E;
	std::vector<MainType> m_M;
	std::vector<MainType> m_dP;

	// False vacuum and True vacuum
	std::vector<int8_t> m_InteractedFalse;
	std::vector<int8_t> m_PassedFalse;
	std::vector<int8_t> m_InteractedTrue;
	std::vector<int8_t> m_PassedTrue;

	Simulation(
		MainType t_alpha, MainType t_massIn, MainType t_massOut, MainType t_temperatureIn, MainType t_temperatureOut,
		u_int t_particleCountIn, unsigned int t_particleCountOut, MainType t_coupling
	);
	
	MainType getParticleRadius(u_int& i);
	MainType getParticleMomentum(u_int& i);
	MainType getParticleEnergy1(u_int& i);
	MainType getParticleEnergy2(u_int& i);
	void setVolumeIn(MainType& t_newVolume) { m_volumeIn = t_newVolume; }
	void setVolumeOut(MainType& t_newVolume) { m_volumeOut = t_newVolume; }
	Bubble createBubble();
	MainType countParticlesEnergy();
	MainType countParticlesEnergyIn(Bubble& bubble);
	MainType calculateTotalEnergy(Bubble& bubble);
	MainType calculateTotalEnergyInitial(Bubble& bubble);
	MainType countParticleEnergyDensity(u_int t_startParticleIndex, u_int t_endParticleIndex, MainType t_volume);
	MainType countParticleNumberDensity(u_int t_particleCount, MainType t_volume);
	MainType calculateNumberDensity(MainType& t_mass, MainType& t_temperature, MainType& t_dp, MainType& t_pUpperLimit);
	MainType calculateEnergyDensity(MainType& t_mass, MainType& t_temperature, MainType& t_dp, MainType& t_pUpperLimit);
	MainType interp(MainType& t_value, std::vector<MainType>& t_x, std::vector<MainType>& t_y);
	void calculateCPD(MainType& t_mass, MainType& t_temperature, const int& t_size, std::vector<MainType>& v_cpd, std::vector<MainType>& v_p, MainType& t_dp, MainType& t_pUpperLimit);
	std::array<MainType, 3> generateRandomDirectionArray(MainType& t_magnitude, std::uniform_real_distribution<MainType>& t_uniformDistribution, std::mt19937_64& t_generator);
	std::array<MainType, 3> generatePointInBoxArray(MainType& t_lengthX, MainType& t_lengthY, MainType& t_lengthZ, std::uniform_real_distribution<MainType>& t_uniformDistribution, std::mt19937_64& t_generator);
	void generateRandomDirection(MainType& t_magnitude, std::vector < MainType>& t_vector, std::uniform_real_distribution<MainType>& t_uniformDistribution, std::mt19937_64& t_generator);
	void generatePointInBox(MainType& t_lengthX, MainType& t_lengthY, MainType& t_lengthZ, std::vector < MainType>& t_vector, std::uniform_real_distribution<MainType>& t_uniformDistribution, std::mt19937_64& t_generator);
	void generateParticleMomentum(u_int t_particleCount, MainType t_mass, std::vector<MainType>& t_cpd, std::vector<MainType>& t_pValues);
	void generateParticleCoordinateCube(u_int t_particleCount, MainType t_radiusSphere, MainType t_radiusCube);
	void generateParticleCoordinateSphere(u_int t_particleCount, MainType t_radiusSphere1, MainType t_radiusSphere2);	
};	

class MultiprocessingCL {

public:
	std::vector<cl::Device> m_devices;
	cl::Platform m_platform;
	cl::Context m_context;
	cl::Program m_program;
	cl::Kernel m_kernel;
	cl::CommandQueue m_queue;

	cl::Buffer m_bufferX;
	cl::Buffer m_bufferP;
	cl::Buffer m_bufferE;
	cl::Buffer m_bufferM;
	cl::Buffer m_bufferDP;

	cl::Buffer m_bufferDt;
	cl::Buffer m_bufferMassIn;
	cl::Buffer m_bufferMassOut;
	cl::Buffer m_bufferMassDelta2;

	cl::Buffer m_bufferBubbleRadius;
	cl::Buffer m_bufferBubbleRadius2;
	cl::Buffer m_bufferBubbleRadiusSpeedDt2;
	cl::Buffer m_bufferBubbleSpeed;
	cl::Buffer m_bufferBubbleGamma;
	cl::Buffer m_bufferBubbleGammaSpeed;

	cl::Buffer m_bufferInteractedFalse;
	cl::Buffer m_bufferPassedFalse;
	cl::Buffer m_bufferInteractedTrue;
	cl::Buffer m_bufferPassedTrue;

	MultiprocessingCL(std::string fileName, std::string kernelName, Simulation& sim, Bubble& bubble);

	void runStep(u_int particleCount);
	void readStep(u_int particleCount, std::vector<MainType>& dP);
	void writeStep(Bubble& bubble);
	void writeBubbleRadiusSpeedDt2Buffer(MainType& dt, Bubble& bubble);
};