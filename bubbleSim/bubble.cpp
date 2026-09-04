#include "bubble.h"

PhaseBubble::PhaseBubble(sim::real t_initialRadius,
                         sim::real t_initialSpeed,
                         sim::real t_dV,
                         sim::real t_sigma,
                         ) {
  m_bubble = Bubble{t_initialRadius, t_initialSpeed, gamma};
  m_dV = t_dV;
  m_sigma = t_sigma;
  m_initialRadius = t_initialRadius;

  if (m_sigma < 0) {
    std::cerr << "sigma < 0" << std::endl;
    std::terminate();
  }
}

sim::real PhaseBubble::Area() const {
  return 4 * M_PI * m_bubble.radius * m_bubble.radius;
}

sim::real PhaseBubble::Volume() const {
  return (4.0 / 3.0) * M_PI * std::pow(m_bubble.radius, 3);
}

sim::real PhaseBubble::Energy() const {
  return m_sigma * Area() / std::sqrt(1 - m_bubble.speed * m_bubble.speed) - m_dV * Volume();
}

void PhaseBubble::evolveWall(sim::real dt, sim::real dP) {
  sim::real newRadius;  // R_(i+1)
  sim::real newSpeed;   // V_(i+1)
  sim::real newGamma;   // gamma_(i+1)
  sim::real gammaChange;
  sim::real sgn = ((0 < m_bubble.speed) - (m_bubble.speed < 0));  // sign of speed

  // Use one method to evolve wall, if gamma is large enough, otherwise use another method.
  if (m_bubble.gamma >= 10) {
    newRadius = m_bubble.radius + dt * m_bubble.speed;

    gammaChange = (std::fma(m_dV, dt, dP) / m_sigma *
                       std::sqrt((m_bubble.gamma - 1) / m_bubble.gamma) -
                   2 * std::sqrt((m_bubble.gamma - 1) * m_bubble.gamma) /
                       m_bubble.radius * dt) *
                  sgn;
    newGamma = m_bubble.gamma + gammaChange;
    newSpeed = std::sqrt(1 - 1 / std::pow(newGamma, 2)) * sgn;

  } else {
    newRadius = m_bubble.radius + dt * m_bubble.speed;

    sim::real velocityElement = std::fma(-m_bubble.speed, m_bubble.speed, 1);

    newSpeed =
        m_bubble.speed +
        std::sqrt(pow(velocityElement, 3)) * std::fma(m_dV, dt, dP) / m_sigma -
        2 * velocityElement * dt / m_bubble.radius;
    newGamma = 1.0 / std::exp((std::log1p((-newSpeed * newSpeed)) * 0.5));
  }
  m_bubble.radius = newRadius;
  m_bubble.speed = newSpeed;
  m_bubble.gamma = newGamma;
}