#include "bubble.h"

PhaseBubble::PhaseBubble(numType t_initialRadius, numType t_initialSpeed,
                         numType t_dV, numType t_sigma,
                         cl::Context cl_context) {
  /*
  cl_double radius;
   cl_double radius2;  // Squared
   cl_double radiusAfterStep2;  // (radius + speed * dt)^2

   cl_double speed;

   cl_double gamma;
   cl_double gammaXspeed;  // gamma * speed
  */
  int openCLerrNum;

  // m_gamma = 1/sqrt(1-v^2) = 1/exp(log1p(-v^2)*0.5)
  numType radius2 = std::pow(t_initialRadius, 2);
  numType gamma =
      1.0 / std::exp((std::log1p((-t_initialSpeed * t_initialSpeed)) * 0.5));
  numType gammaXspeed = t_initialSpeed * gamma;
  m_bubble = Bubble{t_initialRadius, radius2, radius2,
                    t_initialSpeed,  gamma,   gammaXspeed};
  m_bubbleBuffer =
      cl::Buffer(cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 sizeof(Bubble), &m_bubble, &openCLerrNum);

  m_dV = t_dV;
  m_sigma = t_sigma;
  // m_area = 4 * M_PI * std::pow(t_initialRadius, 2);
  // m_volume = 4 * (pow(t_initialRadius, 3) * M_PI) / 3;
  // m_energy = calculateEnergy();
}
//{ return pow(m_radius + dt * m_speed, 2); }
numType PhaseBubble::calculateArea() {
  return 4. * M_PI * std::pow(m_bubble.radius, 2);
}
numType PhaseBubble::calculateVolume() {
  return 4. * (std::pow(m_bubble.radius, 3) * M_PI) / 3;
}

numType PhaseBubble::calculateEnergy() {
  return calculateArea() * m_sigma / sqrt(1 - m_bubble.speed * m_bubble.speed) +
         calculateVolume() * m_dV;
}
numType PhaseBubble::calculateRadiusAfterStep2(numType dt) {
  m_bubble.radiusAfterStep2 =
      pow(std::fma(m_bubble.speed, dt, m_bubble.radius), 2);
  return m_bubble.radiusAfterStep2;
}
void PhaseBubble::evolveWall(numType dt, numType dP) {
  numType newRadius = m_bubble.radius + dt * m_bubble.speed;

  numType velocityElement = std::fma(-m_bubble.speed, m_bubble.speed, 1);

  m_bubble.speed +=
      std::sqrt(pow(velocityElement, 3)) * std::fma(-m_dV, dt, dP) / m_sigma -
      2 * velocityElement * dt / m_bubble.radius;
  m_bubble.radius = newRadius;

  m_bubble.gamma =
      1.0 / std::exp((std::log1p((-m_bubble.speed * m_bubble.speed)) * 0.5));
  m_bubble.radius2 = std::pow(m_bubble.radius, 2);
  m_bubble.gammaXspeed = m_bubble.speed * m_bubble.gamma;
}