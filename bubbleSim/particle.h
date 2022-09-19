#pragma once
#include "base.h"

class Particle {
	numType& m_x;
	numType& m_y;
	numType& m_z;
	numType& m_px;
	numType& m_py;
	numType& m_pz;
	
	numType& m_energy;
	numType& m_mass;

public:
	Particle();
	Particle(std::vector<numType>& t_x, std::vector<numType>& t_p, numType& t_energy, numType& t_mass);

	numType getRadius() { return std::sqrt(std::fma(m_x, m_x, std::fma(m_y, m_y, m_z * m_z))); }
	numType getMomentum() { return std::sqrt(std::fma(m_px, m_px, std::fma(m_py, m_py, m_pz * m_pz))); }

	numType& refMass(){ return m_mass; }
	numType& refEnergy() { return m_energy; }
};