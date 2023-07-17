#include "bubble.h"

PhaseBubble::PhaseBubble(numType t_initialRadius, numType t_initialSpeed,
                         numType t_dV, numType t_sigma,
                         cl::Context& cl_context) {
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
  m_bubble = Bubble{t_initialRadius, radius2, t_initialSpeed,
                    gamma,           gammaXspeed};
  m_bubble_copy = m_bubble;
  m_bubbleBuffer =
      cl::Buffer(cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 sizeof(Bubble), &m_bubble, &openCLerrNum);
  m_dV = t_dV;
  m_sigma = t_sigma;
  m_initialRadius = t_initialRadius;

  if (m_sigma < 0) {
    std::cerr << "sigma < 0" << std::endl;
    std::terminate();
  }

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
  return calculateArea() * m_sigma * m_bubble.gamma - calculateVolume() * m_dV;
}

void PhaseBubble::evolveWall(numType dt, numType dP) {
  numType newRadius;  // R_(i+1)
  numType newSpeed;   // V_(i+1)
  numType newGamma;   // gamma_(i+1)
  numType gammaChange;
  numType sgn = ((0 < m_bubble.speed) - (m_bubble.speed < 0));  // sign of speed

  newRadius = m_bubble.radius + dt * m_bubble.speed;

  if (m_bubble.gamma >= 20) {
    gammaChange = (std::fma(m_dV, dt, dP) / m_sigma *
                       std::sqrt((m_bubble.gamma - 1) / m_bubble.gamma) -
                   2 * std::sqrt((m_bubble.gamma - 1) * m_bubble.gamma) /
                       m_bubble.radius * dt) *
                  sgn;
    newGamma = m_bubble.gamma + gammaChange;
    newSpeed = std::sqrt(1 - 1 / std::pow(newGamma, 2)) * sgn;
  } else {
    numType velocityElement = std::fma(-m_bubble.speed, m_bubble.speed, 1);

    newSpeed =
        m_bubble.speed +
        std::sqrt(pow(velocityElement, 3)) * std::fma(m_dV, dt, dP) / m_sigma -
        2 * velocityElement * dt / m_bubble.radius;
    newGamma = 1.0 / std::exp((std::log1p((-newSpeed * newSpeed)) * 0.5));
  }

  m_bubble.radius = newRadius;
  m_bubble.speed = newSpeed;
  m_bubble.gamma = newGamma;

  m_bubble.radius2 = std::pow(m_bubble.radius, 2);
  m_bubble.gammaXspeed = m_bubble.speed * m_bubble.gamma;
}

void PhaseBubble::print_info(ConfigReader& t_config) {
  std::string sublabel_prefix = "==== ";
  std::string sublabel_sufix = " ====";
  std::cout << std::setprecision(6);
  std::cout << "=============== Bubble ===============" << std::endl;
  std::cout << sublabel_prefix + "dV: " << m_dV << sublabel_sufix << std::endl;
  std::cout << sublabel_prefix + "Sigma: " << m_sigma << sublabel_sufix
            << std::endl;
  std::cout << sublabel_prefix + "Energy: " << calculateEnergy()
            << sublabel_sufix << std::endl;
}