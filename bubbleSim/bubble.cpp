#include "bubble.h"


Bubble::Bubble(numType t_initialRadius, numType t_initialSpeed, numType t_dV, numType t_dVT, numType t_sigma) {
	m_radius = t_initialRadius;
	m_speed = t_initialSpeed;
	m_dV = t_dV;
	m_dVT = t_dVT;
	m_sigma = t_sigma;

	// m_gamma = 1/sqrt(1-v^2) = 1/exp(log1p(-v^2)*0.5)
	m_gamma = 1.0 / std::exp((std::log1p((-m_speed * m_speed)) * 0.5));
	m_radius2 = std::pow(m_radius, 2);
	m_area = 4 * M_PI * std::pow(m_radius, m_radius);
	m_volume = 4 * (pow(m_radius, 3) * M_PI) / 3;
	m_gammaSpeed = m_speed * m_gamma;
	m_energy = calculateEnergy();
	m_radiusAfterDt2 = m_radius2;
}
//{ return pow(m_radius + dt * m_speed, 2); }
numType Bubble::calculateEnergy() {
	m_energy = m_area * m_sigma / sqrt(1 - m_speed * m_speed) + m_volume * m_dV;
	return m_energy;
}
numType Bubble::calculateRadiusAfterDt2Ref(numType dt) {
	m_radiusAfterDt2 = pow(std::fma(m_speed, dt, m_radius), 2);
	return m_radiusAfterDt2;
}
void Bubble::evolveWall(numType dt, numType dP) {
	numType newRadius = m_radius + dt * m_speed;
	numType velocityElement = std::fma(-m_speed, m_speed, 1);
	// m_speed +=  sqrt(pow(velocity_elem, 3)) * (std::fma( - m_dV, t_dt, t_dP)) / m_sigma - 2 * velocity_elem * t_dt / m_radius;
	m_speed += std::sqrt(pow(velocityElement, 3)) * (std::fma(-m_dV, dt, dP)) / m_sigma - 2 * velocityElement * dt / m_radius;
	m_radius = newRadius;

	// update other parameters as well
	m_gamma = 1.0 / std::exp((std::log1p((-m_speed * m_speed)) * 0.5));
	m_radius2 = std::pow(m_radius, 2);
	m_area = 4 * M_PI * std::pow(m_radius, m_radius);
	m_volume = 4 * (pow(m_radius, 3) * M_PI) / 3;
	m_gammaSpeed = m_speed * m_gamma;
	m_energy = calculateEnergy();
}