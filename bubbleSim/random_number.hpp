#pragma once
#include <random>

#include "base.h"

class RandomNumberGenerator {
 public:
  RandomNumberGenerator() {}
  RandomNumberGenerator(int t_seed) {
    m_seed = t_seed;
    if (t_seed == 0) {
      m_generator = std::mt19937_64(m_randDev());
    } else {
      m_generator = std::mt19937_64(t_seed);
    }
    m_distribution = std::uniform_real_distribution<numType>(0, 1);
  }

  numType generate_number() { return m_distribution(m_generator); }

 private:
  int m_seed = 1;
  std::random_device m_randDev;
  std::mt19937_64 m_generator;
  std::uniform_real_distribution<numType> m_distribution;
};