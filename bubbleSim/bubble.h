#pragma once
#include "base.h"

class Bubble {
	// Input parameters
	numType m_radius; // 1)
	numType m_speed; // 1)
	
	numType m_dV;
	numType m_sigma;
	// Calculated parameters from input
	numType m_area;
	numType m_volume;

	numType m_gamma; // 1
	numType m_gammaSpeed; // 1)
	numType m_radius2; // 1)
	numType m_energy;
	numType m_radiusAfterDt2; // 1)
	
	// 1) radius, speed, gamma, gamma*speed, radius^2, (radius*v*dt)^2 are used by GPU 

public:
	numType getRadius() { return m_radius; }
	numType getSpeed() { return m_speed; }
	numType getGamma() { return m_gamma; }
	numType getdV() { return m_dV; }
	numType getSigma() { return m_sigma; }
	numType getArea() { return m_area; }
	numType getVolume() { return m_volume; }
	numType getGammaSpeed() { return m_gammaSpeed; }
	numType getRadius2() { return m_radius2; }
	numType getEnergy() { return m_energy; }
	numType getRadiusAfterDt2() { return m_radiusAfterDt2; }

	numType& getRadiusRef() { return m_radius; }
	numType& getSpeedRef() { return m_speed; }
	numType& getGammaRef() { return m_gamma; }
	numType& getdVRef() { return m_dV; }
	numType& getSigmaRef() { return m_sigma; }
	numType& getAreaRef() { return m_area; }
	numType& getVolumeRef() { return m_volume; }
	numType& getGammaSpeedRef() { return m_gammaSpeed; }
	numType& getRadius2Ref() { return m_radius2; }
	numType& getEnergyRef() { return m_energy; }
	numType& getRadiusAfterDt2Ref() { return m_radiusAfterDt2; }

	Bubble(){}
	Bubble(numType t_initialRadius, numType t_initialSpeed, numType t_dV, numType t_sigma);
	
	void evolveWall(numType dt, numType dP);
	numType calculateRadiusAfterDt2(numType dt);
	numType calculateEnergy();
};