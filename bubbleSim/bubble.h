#pragma once
#include "base.h"

class PhaseBubble {
 public:
  PhaseBubble(sim::real t_initialRadius, sim::real t_initialSpeed,
              sim::real t_dV, sim::real t_sigma);      // no cl::Context

  
  const sim::Bubble& state() const { return m_bubble; }
  
  sim::real Radius()        const { return m_bubble.radius; }
  sim::real Speed()         const { return m_bubble.speed; }
  sim::real dV()            const { return m_dV; }
  sim::real InitialRadius() const { return m_initialRadius; }

  void      evolveWall(sim::real dt, sim::real dP);
  sim::real Area()   const;
  sim::real Volume() const;
  sim::real Energy() const;

 private:
  sim::Bubble m_bubble;
  sim::real   m_dV;
  sim::real   m_sigma;
  sim::real   m_initialRadius;
};