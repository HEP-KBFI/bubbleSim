#include "particle.h"

Particle::Particle(std::vector<numType>& t_x, std::vector<numType>& t_p, numType& t_energy, numType& t_mass) {
	m_x = t_x[0];
	m_y = t_x[1];
	m_z = t_x[2];

	m_px = t_p[0];
	m_py = t_p[1];
	m_pz = t_p[2];

	m_energy = t_energy;
	m_mass = t_mass;
}
