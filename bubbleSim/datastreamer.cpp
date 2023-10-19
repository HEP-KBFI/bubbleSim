#include "datastreamer.h"

DataStreamerBinary::DataStreamerBinary(std::string t_file_path) {
  m_file_path = t_file_path;
}

numType DataStreamerBinary::calculateParticleRadialMomentum(
    ParticleCollection& particles, numType& radius, size_t i) {
  return (particles.returnParticleX(i) * particles.returnParticlepX(i) +
          particles.returnParticleY(i) * particles.returnParticlepY(i) +
          particles.returnParticleZ(i) * particles.returnParticlepZ(i)) /
         radius;
}

numType DataStreamerBinary::calculateParticlePolarMomentum(
    ParticleCollection& particles, numType& theta, numType& phi, size_t i) {
  return cos(theta) * cos(phi) * particles.returnParticlepX(i) +
         cos(theta) * sin(phi) * particles.returnParticlepY(i) -
         sin(theta) * particles.returnParticlepZ(i);
}

numType DataStreamerBinary::calculateParticleAzimuthalMomentum(
    ParticleCollection& particles, numType& phi, size_t i) {
  return cos(phi) * particles.returnParticlepY(i) -
         sin(phi) * particles.returnParticlepX(i);
}

void DataStreamerBinary::initStream_Data() {
  m_stream_data.open(m_file_path / "data.bin",
                     std::ios::out | std::ios::binary);
  m_stream_data_time = 0.;
  m_stream_data_dP = 0.;
  m_stream_data_radius = 0.;
  m_stream_data_velocity = 0.;
  m_stream_data_bubble_energy = 0.;
  m_stream_data_particle_energy = 0.;
  m_stream_data_particle_in_energy = 0.;
  m_stream_data_energy_conservation = 0.;
  m_stream_data_particle_count_in = (uint32_t)0;
  m_stream_data_particle_interacted_false_count = (uint32_t)0;
  m_stream_data_particle_passed_false_count = (uint32_t)0;
  m_stream_data_particle_interacted_true_count = (uint32_t)0;
}

void DataStreamerBinary::initialize_profile_streaming(uint32_t t_N_bins_in,
                                                      uint32_t t_N_bins_out) {
  m_N_bins_in_profile = t_N_bins_in;
  m_N_bins_out_profile = t_N_bins_out;
  m_stream_profile.open(m_file_path / "profile.bin",
                        std::ios::out | std::ofstream::binary);
  std::ofstream profile_guide(m_file_path / "README.md",
                              std::ios::out | std::ios_base::app);
  profile_guide << "Profile: N_profile_bins_in=" << m_N_bins_in_profile
                << "; N_profile_bins_out=" << m_N_bins_out_profile
                << "; dtypes=[uint32, float64, float64, float64, "
                   "float64, float64, float64, "
                   "float64, float64, float64, float64, float64]"
                << std::endl;
  profile_guide.close();

  m_particle_count.resize(m_N_bins_in_profile + m_N_bins_out_profile,
                          (uint32_t)0);
  m_T00.resize(m_N_bins_in_profile + m_N_bins_out_profile, (numType)0.);
  m_T01.resize(m_N_bins_in_profile + m_N_bins_out_profile, (numType)0.);
  m_T02.resize(m_N_bins_in_profile + m_N_bins_out_profile, (numType)0.);
  m_T03.resize(m_N_bins_in_profile + m_N_bins_out_profile, (numType)0.);
  m_T11.resize(m_N_bins_in_profile + m_N_bins_out_profile, (numType)0.);
  m_T22.resize(m_N_bins_in_profile + m_N_bins_out_profile, (numType)0.);
  m_T33.resize(m_N_bins_in_profile + m_N_bins_out_profile, (numType)0.);
  m_T12.resize(m_N_bins_in_profile + m_N_bins_out_profile, (numType)0.);
  m_T13.resize(m_N_bins_in_profile + m_N_bins_out_profile, (numType)0.);
  m_T23.resize(m_N_bins_in_profile + m_N_bins_out_profile, (numType)0.);
  m_radial_velocity.resize(m_N_bins_in_profile + m_N_bins_out_profile,
                           (numType)0.);
}

void DataStreamerBinary::initialize_momentum_streaming(uint32_t t_N_bins_in,
                                                       uint32_t t_N_bins_out,
                                                       numType energy_scale) {
  b_stream_momentum_in = t_N_bins_in > 0;
  b_stream_momentum_out = t_N_bins_out > 0;
  m_stream_momentum.open(m_file_path / "momentum.bin",
                         std::ios::out | std::ofstream::binary);

  m_N_bins_momentum_in = t_N_bins_in;
  m_N_bins_momentum_out = t_N_bins_out;
  // Currently 3 orders up (down) are max (min) values.
  numType scale = pow(10., floor(log10(energy_scale)));
  m_p_min = scale * m_p_min_factor;
  m_p_max = scale * m_p_max_factor;
  if (b_stream_momentum_in) {
    m_dp_in =
        (std::log10(m_p_max) - std::log10(m_p_min)) / m_N_bins_momentum_in;
  }
  if (b_stream_momentum_out) {
    m_dp_out =
        (std::log10(m_p_max) - std::log10(m_p_min)) / m_N_bins_momentum_out;
  }
  m_momentum.resize(m_N_bins_momentum_in + m_N_bins_momentum_out, (uint32_t)0);
  std::ofstream momentum_info(m_file_path / "README.md",
                              std::ios::out | std::ios_base::app);
  momentum_info << "Momentum: N_momentum_bins_in=" << t_N_bins_in
                << "; N_momentum_bins_out=" << t_N_bins_out
                << "; scale=" << scale << "; min_order=" << -3
                << "; max_order=" << 3 << std::endl;
  momentum_info.close();
}

void DataStreamerBinary::initialize_momentum_radial_profile_streaming(
    uint32_t t_N_momentum_bins, uint32_t t_N_radius_bins_in,
    uint32_t t_N_radius_bins_out, numType energy_scale) {
  b_stream_momentum_radial_profile = true;

  m_stream_momentum_radial_profile.open(m_file_path / "momentum_profile.bin",
                                        std::ios::out | std::ofstream::binary);

  // Currently 3 orders up (down) are max (min) values.
  m_N_bins_momentum_pr = t_N_momentum_bins;
  m_N_bins_in_profile_pr = t_N_radius_bins_in;
  m_N_bins_out_profile_pr = t_N_radius_bins_out;
  numType scale = pow(10., floor(log10(energy_scale)));

  m_p_min_pr = scale * m_p_min_factor_pr;
  m_p_max_pr = scale * m_p_max_factor_pr;

  m_dp_pr =
      (std::log10(m_p_max_pr) - std::log10(m_p_min_pr)) / m_N_bins_momentum_pr;

  m_momentum_radius_profile.resize(
      t_N_momentum_bins * (t_N_radius_bins_in + t_N_radius_bins_out),
      (uint32_t)0);

  std::ofstream momentum_info(m_file_path / "README.md",
                              std::ios::out | std::ios_base::app);
  momentum_info << "Momentum_profile: Momentum_profile_momentum_N_bins="
                << t_N_momentum_bins
                << "; Momentum_profile_radius_in_N_bins=" << t_N_radius_bins_in
                << "; Momentum_profile_radius_out_N_bins="
                << t_N_radius_bins_out << "; scale=" << scale
                << "; min_order=" << -3 << "; max_order=" << 3 << std::endl;
  momentum_info.close();
}

void DataStreamerBinary::stream(Simulation& simulation,
                                ParticleCollection& particleCollection,
                                PhaseBubble& bubble,
                                SimulationSettings& settings,
                                cl::CommandQueue& cl_queue) {
  numType bubble_radius = bubble.getRadius();
  numType boundary_radius =
      simulation.getSimulationParameters().getBoundaryRadius();

  // Read data from GPU
  particleCollection.readParticleXBuffer(cl_queue);
  particleCollection.readParticleYBuffer(cl_queue);
  particleCollection.readParticleZBuffer(cl_queue);
  particleCollection.readParticleEBuffer(cl_queue);
  particleCollection.readParticlepXBuffer(cl_queue);
  particleCollection.readParticlepYBuffer(cl_queue);
  particleCollection.readParticlepZBuffer(cl_queue);

  // Reset data
  m_stream_data_bubble_energy = 0.;
  m_stream_data_particle_energy = 0.;
  m_stream_data_particle_in_energy = 0.;
  m_stream_data_energy_conservation = 0.;
  m_stream_data_particle_count_in = (uint32_t)0;
  m_stream_data_particle_interacted_false_count = (uint32_t)0;
  m_stream_data_particle_passed_false_count = (uint32_t)0;
  m_stream_data_particle_interacted_true_count = (uint32_t)0;
  m_stream_data_active_particles_in_collision =
      simulation.getActiveCollidingParticleCount();
  m_stream_data_active_cells_in_collision =
      simulation.getActiveCollisionCellCount();

  if (settings.isFlagSet(BUBBLE_INTERACTION_ON)) {
    particleCollection.readInteractedBubbleFalseStateBuffer(cl_queue);
    particleCollection.readPassedBubbleFalseStateBuffer(cl_queue);
    particleCollection.readInteractedBubbleTrueStateBuffer(cl_queue);
  }

  m_dr_in = bubble_radius / m_N_bins_in_profile;
  m_dr_out = (boundary_radius - bubble_radius) / m_N_bins_out_profile;

  // Reset all profile values to zero
  memset(&m_particle_count[0], 0,
         sizeof(m_particle_count[0]) * m_particle_count.size());
  memset(&m_T00[0], 0, sizeof(m_T00[0]) * m_T00.size());
  memset(&m_T01[0], 0, sizeof(m_T01[0]) * m_T01.size());
  memset(&m_T02[0], 0, sizeof(m_T02[0]) * m_T02.size());
  memset(&m_T03[0], 0, sizeof(m_T03[0]) * m_T03.size());
  memset(&m_T11[0], 0, sizeof(m_T11[0]) * m_T11.size());
  memset(&m_T22[0], 0, sizeof(m_T22[0]) * m_T22.size());
  memset(&m_T33[0], 0, sizeof(m_T33[0]) * m_T33.size());
  memset(&m_T12[0], 0, sizeof(m_T12[0]) * m_T12.size());
  memset(&m_T13[0], 0, sizeof(m_T13[0]) * m_T13.size());
  memset(&m_T23[0], 0, sizeof(m_T23[0]) * m_T23.size());
  memset(&m_radial_velocity[0], 0,
         sizeof(m_radial_velocity[0]) * m_radial_velocity.size());

  if (b_stream_momentum_in || b_stream_momentum_out) {
    memset(&m_momentum[0], 0, sizeof(m_momentum[0]) * m_momentum.size());
  }

  if (b_stream_momentum_radial_profile) {
    memset(&m_momentum_radius_profile[0], 0,
           sizeof(m_momentum_radius_profile[0]) *
               m_momentum_radius_profile.size());
    m_dr_in_pr = bubble_radius / m_N_bins_in_profile_pr;
    m_dr_out_pr = (boundary_radius - bubble_radius) / m_N_bins_out_profile_pr;
  }

  numType energy = 0.;
  numType radius = 0.;
  numType momentum = 0.;
  numType theta = 0.;
  numType phi = 0;
  uint64_t index = 0;
  int index_p = 0;

  uint64_t index_radius = 0;
  int index_momentum = 0;

  numType p_R = 0.;
  numType p_phi = 0.;
  numType p_theta = 0.;

  // Collect data

  for (size_t i = 0; i < particleCollection.getParticleCountTotal(); i++) {
    radius = particleCollection.calculateParticleRadius(i);
    if (b_stream_momentum_in || b_stream_momentum_out ||
        b_stream_momentum_radial_profile) {
      momentum = particleCollection.calculateParticleMomentum(i);
    }

    if (settings.isFlagSet(BUBBLE_INTERACTION_ON)) {
      if (radius <= bubble.getRadius()) {
        m_stream_data_particle_count_in += 1;
        m_stream_data_particle_in_energy +=
            particleCollection.returnParticleE(i);
      }
      m_stream_data_particle_interacted_false_count +=
          particleCollection.getInteractedFalse()[i];
      m_stream_data_particle_interacted_true_count +=
          particleCollection.getInteractedTrue()[i];
      m_stream_data_particle_passed_false_count +=
          particleCollection.getPassedFalse()[i];
    }
    m_stream_data_particle_energy += particleCollection.returnParticleE(i);

    if (b_stream_momentum_in) {
      if (radius < bubble.getRadius()) {
        if (momentum >= m_p_min && momentum < m_p_max) {
          index_p = int(std::log10(momentum / m_p_min) / m_dp_in);
          m_momentum[index_p] += (uint32_t)1;
        }
      }
    }
    if (b_stream_momentum_out) {
      if (radius > bubble.getRadius()) {
        if (momentum >= m_p_min && momentum < m_p_max) {
          index_p = int(std::log10(momentum / m_p_min) / m_dp_out);
          m_momentum[index_p + m_N_bins_momentum_in] += (uint32_t)1;
        }
      }
    }

    if (radius > boundary_radius) {
      continue;
    } else if (radius > bubble_radius) {
      index =
          m_N_bins_in_profile + uint64_t((radius - bubble_radius) / m_dr_out);
      if (b_stream_momentum_radial_profile) {
        index_radius = m_N_bins_in_profile_pr +
                       uint64_t((radius - bubble_radius) / m_dr_out_pr);
      }
    } else {
      index = uint64_t(radius / m_dr_in);
      if (b_stream_momentum_radial_profile) {
        index_radius = uint64_t(radius / m_dr_in_pr);
      }
    }

    if (b_stream_momentum_radial_profile) {
      if (momentum >= m_p_min_pr && momentum < m_p_max_pr) {
        index_momentum = int(std::log10(momentum / m_p_min_pr) / m_dp_pr);
        m_momentum_radius_profile[index_radius * m_N_bins_momentum_pr +
                                  index_momentum] += (uint32_t)1;
      }
    }
    energy = particleCollection.returnParticleE(i);
    phi = atan2(particleCollection.returnParticleY(i),
                particleCollection.returnParticleX(i));
    theta = acos(particleCollection.returnParticleZ(i) / radius);

    


    p_R = calculateParticleRadialMomentum(particleCollection, radius, i);
    p_theta = calculateParticlePolarMomentum(particleCollection, theta, phi, i);
    p_phi = calculateParticleAzimuthalMomentum(particleCollection, phi, i);

    // Calculate energy momentum tensor:
    m_particle_count[index] += 1;
    m_T00[index] += energy;
    m_T01[index] += p_R;
    m_T02[index] += theta;
    m_T03[index] += phi;
    m_T11[index] += pow(p_R, 2.) / energy;
    m_T22[index] += pow(theta, 2.) / energy;
    m_T33[index] += pow(phi, 2.) / energy;
    m_T12[index] += p_R * theta / energy;
    m_T13[index] += p_R * phi / energy;
    m_T23[index] += phi * theta / energy;
    /*m_T02[index] += p_theta;
    m_T03[index] += p_phi;
    m_T11[index] += pow(p_R, 2.) / energy;
    m_T22[index] += pow(p_theta, 2.) / energy;
    m_T33[index] += pow(p_phi, 2.) / energy;
    m_T12[index] += p_R * p_theta / energy;
    m_T13[index] += p_R * p_phi / energy;
    m_T23[index] += p_phi * p_theta / energy;*/
    m_radial_velocity[index] += p_R / energy;
  }

  m_stream_data_time = simulation.getTime();
  if (settings.isFlagSet(BUBBLE_INTERACTION_ON)) {
    m_stream_data_dP = simulation.get_dP();
    m_stream_data_radius = bubble.getRadius();
    m_stream_data_velocity = bubble.getSpeed();
    m_stream_data_bubble_energy = bubble.calculateEnergy();
    m_stream_data_energy_conservation =
        (m_stream_data_particle_energy + bubble.calculateEnergy()) /
        simulation.getInitialTotalEnergy();
  } else {
    m_stream_data_energy_conservation =
        m_stream_data_particle_energy / simulation.getInitialTotalEnergy();
  }

  // Write data to file
  write_data();
  write_profile();
  if (b_stream_momentum_in || b_stream_momentum_out) {
    write_momentum();
  }
  if (b_stream_momentum_radial_profile) {
    write_momentum_profile();
  }
}
void DataStreamerBinary::write_data() {
  m_stream_data.write(reinterpret_cast<char*>(&m_stream_data_time),
                      sizeof(numType));
  m_stream_data.write(reinterpret_cast<char*>(&m_stream_data_dP),
                      sizeof(numType));
  m_stream_data.write(reinterpret_cast<char*>(&m_stream_data_radius),
                      sizeof(numType));
  m_stream_data.write(reinterpret_cast<char*>(&m_stream_data_velocity),
                      sizeof(numType));
  m_stream_data.write(reinterpret_cast<char*>(&m_stream_data_bubble_energy),
                      sizeof(numType));
  m_stream_data.write(reinterpret_cast<char*>(&m_stream_data_particle_energy),
                      sizeof(numType));
  m_stream_data.write(
      reinterpret_cast<char*>(&m_stream_data_particle_in_energy),
      sizeof(numType));
  m_stream_data.write(
      reinterpret_cast<char*>(&m_stream_data_energy_conservation),
      sizeof(numType));
  m_stream_data.write(reinterpret_cast<char*>(&m_stream_data_particle_count_in),
                      sizeof(uint32_t));
  m_stream_data.write(
      reinterpret_cast<char*>(&m_stream_data_particle_interacted_false_count),
      sizeof(uint32_t));
  m_stream_data.write(
      reinterpret_cast<char*>(&m_stream_data_particle_passed_false_count),
      sizeof(uint32_t));
  m_stream_data.write(
      reinterpret_cast<char*>(&m_stream_data_particle_interacted_true_count),
      sizeof(uint32_t));
  m_stream_data.write(
      reinterpret_cast<char*>(&m_stream_data_active_cells_in_collision),
      sizeof(uint32_t));
  m_stream_data.write(
      reinterpret_cast<char*>(&m_stream_data_active_particles_in_collision),
      sizeof(uint32_t));
}

void DataStreamerBinary::write_profile() {
  m_stream_profile.write(reinterpret_cast<const char*>(m_particle_count.data()),
                         m_particle_count.size() * sizeof(uint32_t));
  m_stream_profile.write(reinterpret_cast<const char*>(m_T00.data()),
                         m_T00.size() * sizeof(numType));
  m_stream_profile.write(reinterpret_cast<const char*>(m_T01.data()),
                         m_T01.size() * sizeof(numType));
  m_stream_profile.write(reinterpret_cast<const char*>(m_T02.data()),
                         m_T02.size() * sizeof(numType));
  m_stream_profile.write(reinterpret_cast<const char*>(m_T03.data()),
                         m_T03.size() * sizeof(numType));
  m_stream_profile.write(reinterpret_cast<const char*>(m_T11.data()),
                         m_T11.size() * sizeof(numType));
  m_stream_profile.write(reinterpret_cast<const char*>(m_T22.data()),
                         m_T22.size() * sizeof(numType));
  m_stream_profile.write(reinterpret_cast<const char*>(m_T33.data()),
                         m_T33.size() * sizeof(numType));
  m_stream_profile.write(reinterpret_cast<const char*>(m_T12.data()),
                         m_T12.size() * sizeof(numType));
  m_stream_profile.write(reinterpret_cast<const char*>(m_T13.data()),
                         m_T13.size() * sizeof(numType));
  m_stream_profile.write(reinterpret_cast<const char*>(m_T23.data()),
                         m_T23.size() * sizeof(numType));
  m_stream_profile.write(
      reinterpret_cast<const char*>(m_radial_velocity.data()),
      m_radial_velocity.size() * sizeof(numType));
}

void DataStreamerBinary::write_momentum() {
  m_stream_momentum.write(reinterpret_cast<const char*>(m_momentum.data()),
                          m_momentum.size() * sizeof(uint32_t));
}

void DataStreamerBinary::write_momentum_profile() {
  m_stream_momentum_radial_profile.write(
      reinterpret_cast<const char*>(m_momentum_radius_profile.data()),
      m_momentum_radius_profile.size() * sizeof(uint32_t));
}

DataStreamer::DataStreamer(std::string filePath) { m_filePath = filePath; }

void DataStreamer::initStream_Data() {
  m_initialized_Data = true;
  m_stream_data.open(m_filePath / "data.csv", std::ios::out);
  /*
   * time
   * dP - Pressure/energy change between particles and bubble
   * R_b - bubble radius
   * V_b - bubble speed
   * E_b - bubble energy
   * E_p - particles' energy
   * E_f - particles' energy which are inside the bubble
   * E - total energy / initial total energy (checking energy conservation)
   * C_f - particles inside the bubble count
   * C_if - particles which interacted with bubble form false vacuum (and did
   not get through) count
   * C_pf - particles which interacted with bubble form false vacuum (and got
   through) count
   * C_it - particles which interacted with bubble from true vacuum count
   * C - compactness
   */
  m_stream_data << "time,dP,R_b,V_b,E_b,E_p,E_f,E,C_f,C_if,C_pf,C_it,C"
                << std::endl;
}

void DataStreamer::initStream_Momentum(size_t t_binsCount,
                                       numType t_maxMomentumValue) {
  m_initialized_Momentum = true;
  m_binsCount_Momentum = t_binsCount;
  m_maxMomentum_Momentum = t_maxMomentumValue;
  m_dp_Momentum = t_maxMomentumValue / t_binsCount;

  m_stream_MomentumX.open(m_filePath / "pX.csv", std::ios::out);
  m_stream_MomentumY.open(m_filePath / "pY.csv", std::ios::out);
  m_stream_MomentumZ.open(m_filePath / "pZ.csv", std::ios::out);

  m_stream_MomentumX << t_binsCount << "," << t_maxMomentumValue << "\n";
  m_stream_MomentumY << t_binsCount << "," << t_maxMomentumValue << "\n";
  m_stream_MomentumZ << t_binsCount << "," << t_maxMomentumValue << "\n";
  for (size_t i = 1; i <= t_binsCount; i++) {
    if (i == t_binsCount) {
      m_stream_MomentumX << i << std::endl;
      m_stream_MomentumY << i << std::endl;
      m_stream_MomentumZ << i << std::endl;
    } else {
      m_stream_MomentumX << i << ",";
      m_stream_MomentumY << i << ",";
      m_stream_MomentumZ << i << ",";
    }
  }
}

void DataStreamer::initStream_MomentumIn(size_t t_binsCount,
                                         numType t_minMomentumValue,
                                         numType t_maxMomentumValue,
                                         bool t_log_scale_on) {
  if ((t_log_scale_on) && (t_minMomentumValue <= 0)) {
    std::cout << "Log scale is on. Minimum momentum value must be > 0.";
    std::exit(0);
  }

  m_initialized_MomentumIn = true;
  m_binsCount_MomentumIn = t_binsCount;
  m_minMomentum_MomentumIn = t_minMomentumValue;
  m_maxMomentum_MomentumIn = t_maxMomentumValue;
  if (t_log_scale_on) {
    m_dp_MomentumIn =
        (std::log10(t_maxMomentumValue) - std::log10(t_minMomentumValue)) /
        t_binsCount;

  } else {
    m_dp_MomentumIn = t_maxMomentumValue / t_binsCount;
  }

  m_stream_MomentumIn.open(m_filePath / "pIn.csv", std::ios::out);
  m_stream_MomentumIn << t_binsCount << "," << t_minMomentumValue << ","
                      << t_maxMomentumValue << "\n";
  for (size_t i = 1; i <= t_binsCount; i++) {
    if (i == t_binsCount) {
      m_stream_MomentumIn << i << std::endl;
    } else {
      m_stream_MomentumIn << i << ",";
    }
  }
}

void DataStreamer::initStream_MomentumOut(size_t t_binsCount,
                                          numType t_minMomentumValue,
                                          numType t_maxMomentumValue,
                                          bool t_log_scale_on) {
  if ((t_log_scale_on) && (t_minMomentumValue <= 0)) {
    std::cout << "Log scale is on. Minimum momentum value must be > 0.";
    std::exit(0);
  }
  m_initialized_MomentumOut = true;
  m_binsCount_MomentumOut = t_binsCount;
  m_minMomentum_MomentumOut = t_minMomentumValue;
  m_maxMomentum_MomentumOut = t_maxMomentumValue;
  if (t_log_scale_on) {
    m_dp_MomentumOut =
        (std::log10(t_maxMomentumValue) - std::log10(t_minMomentumValue)) /
        t_binsCount;

  } else {
    m_dp_MomentumOut = t_maxMomentumValue / t_binsCount;
  }

  m_stream_MomentumOut.open(m_filePath / "pOut.csv", std::ios::out);
  m_stream_MomentumOut << t_binsCount << "," << t_minMomentumValue << ","
                       << t_maxMomentumValue << "\n";
  for (size_t i = 1; i <= t_binsCount; i++) {
    if (i == t_binsCount) {
      m_stream_MomentumOut << i << std::endl;
    } else {
      m_stream_MomentumOut << i << ",";
    }
  }
}

void DataStreamer::initStream_Density(size_t t_binsCount,
                                      numType t_maxRadiusValue) {
  m_initialized_Density = true;
  m_binsCount_Density = t_binsCount;
  m_maxRadius_Density = t_maxRadiusValue;
  m_dr_Density = t_maxRadiusValue / t_binsCount;

  m_stream_Density.open(m_filePath / "numberDensity.csv", std::ios::out);
  m_stream_Density << t_binsCount << "," << t_maxRadiusValue << "\n";
  for (size_t i = 1; i <= t_binsCount; i++) {
    if (i == t_binsCount) {
      m_stream_Density << i << std::endl;
    } else {
      m_stream_Density << i << ",";
    }
  }
}

void DataStreamer::initStream_EnergyDensity(size_t t_binsCount,
                                            numType t_maxRadiusValue) {
  m_initialized_EnergyDensity = true;
  m_binsCount_EnergyDensity = t_binsCount;
  m_maxRadius_EnergyDensity = t_maxRadiusValue;
  m_dr_EnergyDensity = t_maxRadiusValue / t_binsCount;

  m_stream_EnergyDensity.open(m_filePath / "energyDensity.csv", std::ios::out);
  m_stream_EnergyDensity << t_binsCount << "," << t_maxRadiusValue << "\n";
  for (size_t i = 1; i <= t_binsCount; i++) {
    if (i == t_binsCount) {
      m_stream_EnergyDensity << i << std::endl;
    } else {
      m_stream_EnergyDensity << i << ",";
    }
  }
}

void DataStreamer::initStream_RadialVelocity(size_t t_binsCount,
                                             numType t_maxRadiusValue) {
  m_initialized_RadialVelocity = true;
  m_binsCount_RadialVelocity = t_binsCount;
  m_maxRadius_RadialVelocity = t_maxRadiusValue;
  m_dr_RadialVelocity = t_maxRadiusValue / t_binsCount;

  m_stream_RadialVelocity.open(m_filePath / "radialVelocity.csv",
                               std::ios::out);
  m_stream_RadialVelocity << t_binsCount << "," << t_maxRadiusValue << "\n";
  for (size_t i = 1; i <= t_binsCount; i++) {
    if (i == t_binsCount) {
      m_stream_RadialVelocity << i << std::endl;
    } else {
      m_stream_RadialVelocity << i << ",";
    }
  }
}

void DataStreamer::initStream_TangentialVelocity(size_t t_binsCount,
                                                 numType t_maxRadiusValue) {
  m_initialized_TangentialVelocity = true;
  m_binsCount_TangentialVelocity = t_binsCount;
  m_maxRadius_TangentialVelocity = t_maxRadiusValue;
  m_dr_TangentialVelocity = t_maxRadiusValue / t_binsCount;

  m_stream_TangentialVelocity.open(m_filePath / "tangentialVelocity.csv",
                                   std::ios::out);
  m_stream_TangentialVelocity << t_binsCount << "," << t_maxRadiusValue << "\n";
  for (size_t i = 1; i <= t_binsCount; i++) {
    if (i == t_binsCount) {
      m_stream_TangentialVelocity << i << std::endl;
    } else {
      m_stream_TangentialVelocity << i << ",";
    }
  }
}

void DataStreamer::stream(Simulation& simulation,
                          ParticleCollection& particleCollection,
                          PhaseBubble& bubble, bool t_log_scale_on,
                          cl::CommandQueue& cl_queue) {
  // auto programStartTime = std::chrono::high_resolution_clock::now();
  /*
   * Do only initialized streams.
   * 1) Read in necessary buffers
   * 2) Do for cycle over all particles and count/calculate profiles
   * 3) Stream into files
   */

  std::cout << std::setprecision(8) << std::fixed << std::showpoint;

  /*
   * Read particle buffers to get if particle interacted with the bubble or not
   * Don't read bubble buffer as it is not updated. Use PhaseBubble object.
   */
  if (m_initialized_Data) {
    particleCollection.readInteractedBubbleFalseStateBuffer(cl_queue);
    particleCollection.readPassedBubbleFalseStateBuffer(cl_queue);
    particleCollection.readInteractedBubbleTrueStateBuffer(cl_queue);
  }
  // Save general data
  size_t particleInCount;
  size_t particleInteractedFalseCount;
  size_t particlePassedFalseCount;
  size_t particleInteractedTrueCount;
  numType particleInEnergy;
  numType particleTotalEnergy;
  numType totalEnergy;

  // Save momentum data
  std::vector<u_int> bins_MomentumX;
  std::vector<u_int> bins_MomentumY;
  std::vector<u_int> bins_MomentumZ;
  std::vector<u_int> bins_MomentumIn;
  std::vector<u_int> bins_MomentumOut;

  // Save density data
  std::vector<u_int> bins_Density;
  std::vector<numType> bins_EnergyDensity;

  // Save velocity data
  std::vector<numType> bins_RadialVelocity;
  std::vector<numType> bins_TangentialVelocity;
  std::vector<u_int> bins_RadialVelocityCount;
  std::vector<u_int> bins_TangentialVelocityCount;

  // Bin indexes
  u_int bin_index_radial_velocity;
  u_int bin_index_tangential_velocity;

  // Read data from buffer:
  particleCollection.readParticleCoordinatesBuffer(cl_queue);
  particleCollection.readParticleMomentumsBuffer(cl_queue);
  particleCollection.readParticleEBuffer(cl_queue);

  // Initialize variables
  if (m_initialized_Momentum) {
    bins_MomentumX.resize(m_binsCount_Momentum, 0);
    bins_MomentumY.resize(m_binsCount_Momentum, 0);
    bins_MomentumZ.resize(m_binsCount_Momentum, 0);
  }
  if (m_initialized_MomentumIn) {
    bins_MomentumIn.resize(m_binsCount_MomentumIn, 0);
  }
  if (m_initialized_MomentumOut) {
    bins_MomentumOut.resize(m_binsCount_MomentumOut, 0);
  }
  if (m_initialized_Density) {
    bins_Density.resize(m_binsCount_Density, 0);
  }
  if (m_initialized_EnergyDensity) {
    bins_EnergyDensity.resize(m_binsCount_EnergyDensity, (numType)0.);
  }
  if (m_initialized_RadialVelocity) {
    bins_RadialVelocity.resize(m_binsCount_RadialVelocity, (numType)0.);
    bins_RadialVelocityCount.resize(m_binsCount_RadialVelocity, 0);
  }
  if (m_initialized_TangentialVelocity) {
    bins_TangentialVelocity.resize(m_binsCount_TangentialVelocity, (numType)0.);
    bins_TangentialVelocityCount.resize(m_binsCount_TangentialVelocity, 0);
  }
  if (m_initialized_Data) {
    particleInCount = 0;
    particleInteractedFalseCount = 0;
    particlePassedFalseCount = 0;
    particleInteractedTrueCount = 0;
    particleInEnergy = 0.;
    particleTotalEnergy = 0.;
  }

  numType particleRadius;
  numType particleMomentum;
  numType particleRadialVelocity;
  numType particleTangentialVelocity;
  for (u_int i = 0; i < particleCollection.getParticleCountTotal(); i++) {
    particleRadius = particleCollection.calculateParticleRadius(i);
    particleMomentum = particleCollection.calculateParticleMomentum(i);

    if (m_initialized_Momentum) {
      bins_MomentumX[std::clamp(
          (int)(abs(particleCollection.returnParticlepX(i) / m_dp_Momentum)), 0,
          (int)m_binsCount_Momentum - 1)] += 1;
      bins_MomentumY[std::clamp(
          (int)(abs(particleCollection.returnParticlepY(i) / m_dp_Momentum)), 0,
          (int)m_binsCount_Momentum - 1)] += 1;
      bins_MomentumZ[std::clamp(
          (int)(abs(particleCollection.returnParticlepZ(i) / m_dp_Momentum)), 0,
          (int)m_binsCount_Momentum - 1)] += 1;
    }
    if (m_initialized_Density && (particleRadius < m_maxRadius_Density)) {
      bins_Density[(int)(particleRadius / m_dr_Density)] += 1;
    }
    if (m_initialized_EnergyDensity &&
        (particleRadius < m_maxRadius_EnergyDensity)) {
      bins_EnergyDensity[(int)(particleRadius / m_dr_EnergyDensity)] +=
          particleCollection.returnParticleE(i);
    }
    if (m_initialized_RadialVelocity &&
        (particleRadius < m_maxRadius_RadialVelocity)) {
      // Average A(N) = [A(N-1) * (N-1) + a(N) ]/N
      particleRadialVelocity =
          particleCollection.calculateParticleRadialVelocity(i);
      bin_index_radial_velocity = (u_int)(particleRadius / m_dr_RadialVelocity);

      bins_RadialVelocity[bin_index_radial_velocity] =
          bins_RadialVelocity[bin_index_radial_velocity] +
          (particleRadialVelocity -
           bins_RadialVelocity[bin_index_radial_velocity]) /
              (bins_RadialVelocityCount[bin_index_radial_velocity] + 1);

      bins_RadialVelocityCount[(int)(particleRadius / m_dr_RadialVelocity)] +=
          1;
    }
    if (m_initialized_TangentialVelocity &&
        (particleRadius < m_maxRadius_TangentialVelocity)) {
      bin_index_tangential_velocity =
          (u_int)(particleRadius / m_dr_TangentialVelocity);
      if (m_initialized_RadialVelocity) {
        particleTangentialVelocity = std::sqrt(1 - particleRadialVelocity);
      } else {
        particleTangentialVelocity =
            particleCollection.calculateParticleTangentialVelocity(i);
      }
      bins_TangentialVelocity[bin_index_tangential_velocity] =
          bins_TangentialVelocity[bin_index_tangential_velocity] +
          (particleTangentialVelocity -
           bins_TangentialVelocity[bin_index_tangential_velocity]) /
              (bins_TangentialVelocityCount[bin_index_tangential_velocity] + 1);

      bins_TangentialVelocityCount[(int)(particleRadius /
                                         m_dr_TangentialVelocity)] += 1;
    }
    if (m_initialized_MomentumIn && (particleRadius < bubble.getRadius()) &&
        (particleMomentum < m_maxMomentum_MomentumIn) &&
        (particleMomentum >= m_minMomentum_MomentumIn)) {
      if (t_log_scale_on) {
        bins_MomentumIn[(
            int)(std::log10(particleMomentum / m_minMomentum_MomentumIn) /
                 m_dp_MomentumIn)] += 1;
      } else {
        bins_MomentumIn[(int)(particleMomentum / m_dp_MomentumIn)] += 1;
      }
    }
    if (m_initialized_MomentumOut && (particleRadius > bubble.getRadius()) &&
        (particleMomentum < m_maxMomentum_MomentumOut) &&
        (particleMomentum >= m_minMomentum_MomentumOut)) {
      if (t_log_scale_on) {
        bins_MomentumOut[(
            int)(std::log10(particleMomentum / m_minMomentum_MomentumOut) /
                 m_dp_MomentumOut)] += 1;
      } else {
        bins_MomentumOut[(int)(particleMomentum / m_dp_MomentumOut)] += 1;
      }
    }
    if (m_initialized_Data) {
      if (particleRadius <= bubble.getRadius()) {
        particleInCount += 1;
        particleInEnergy += particleCollection.returnParticleE(i);
      }
      particleInteractedFalseCount +=
          particleCollection.getInteractedFalse()[i];
      particlePassedFalseCount += particleCollection.getPassedFalse()[i];
      particleInteractedTrueCount += particleCollection.getInteractedTrue()[i];
      particleTotalEnergy += particleCollection.returnParticleE(i);
    }
  }

  // std::cout << "Data collection: successful!" << std::endl;
  //  std::cout << "Max momentum: " << maxMomentum << std::endl;
  if (m_initialized_Data) {
    totalEnergy = particleTotalEnergy + bubble.calculateEnergy();
    std::cout << std::fixed << std::noshowpoint << std::setprecision(8);
    m_stream_data << simulation.getTime() << ","
                  << simulation.get_dP() / simulation.get_dt_currentStep()
                  << ",";
    m_stream_data << bubble.getRadius() << "," << bubble.getSpeed() << ",";
    m_stream_data << bubble.calculateEnergy() << "," << particleTotalEnergy
                  << ",";
    m_stream_data << particleInEnergy << ","
                  << totalEnergy / simulation.getInitialTotalEnergy() << ",";
    m_stream_data << particleInCount << "," << particleInteractedFalseCount
                  << ",";
    m_stream_data << particlePassedFalseCount << ","
                  << particleInteractedTrueCount << ","
                  << (particleInEnergy + bubble.calculateEnergy()) /
                         bubble.getRadius() / simulation.getInitialCompactnes()
                  << std::endl;

    particleCollection.resetAndWriteInteractedBubbleFalseState(cl_queue);
    particleCollection.resetAndWritePassedBubbleFalseState(cl_queue);
    particleCollection.resetAndWriteInteractedBubbleTrueState(cl_queue);

    std::cout << std::noshowpoint;
  }
  if (m_initialized_Density) {
    for (size_t i = 0; i < m_binsCount_Density - 1; i++) {
      m_stream_Density << bins_Density[i] << ",";
    }
    m_stream_Density << bins_Density[m_binsCount_Density - 1] << "\n";
  }
  if (m_initialized_EnergyDensity) {
    for (size_t i = 0; i < m_binsCount_EnergyDensity - 1; i++) {
      m_stream_EnergyDensity << bins_EnergyDensity[i] << ",";
    }
    m_stream_EnergyDensity << bins_EnergyDensity[m_binsCount_EnergyDensity - 1]
                           << "\n";
  }
  if (m_initialized_MomentumIn) {
    for (size_t i = 0; i < m_binsCount_MomentumIn - 1; i++) {
      m_stream_MomentumIn << bins_MomentumIn[i] << ",";
    }
    m_stream_MomentumIn << bins_MomentumIn[m_binsCount_MomentumIn - 1] << "\n";
  }
  if (m_initialized_MomentumOut) {
    for (size_t i = 0; i < m_binsCount_MomentumOut - 1; i++) {
      m_stream_MomentumOut << bins_MomentumOut[i] << ",";
    }
    m_stream_MomentumOut << bins_MomentumOut[m_binsCount_MomentumOut - 1]
                         << "\n";
  }
  if (m_initialized_RadialVelocity) {
    for (size_t i = 0; i < m_binsCount_RadialVelocity - 1; i++) {
      m_stream_RadialVelocity << bins_RadialVelocity[i] << ",";
    }
    m_stream_RadialVelocity
        << bins_RadialVelocity[m_binsCount_RadialVelocity - 1] << "\n";
  }
  if (m_initialized_TangentialVelocity) {
    for (size_t i = 0; i < m_binsCount_TangentialVelocity - 1; i++) {
      m_stream_TangentialVelocity << bins_TangentialVelocity[i] << ",";
    }
    m_stream_TangentialVelocity
        << bins_TangentialVelocity[m_binsCount_TangentialVelocity - 1] << "\n";
  }
  if (m_initialized_Momentum) {
    for (size_t i = 0; i < m_binsCount_Momentum - 1; i++) {
      m_stream_MomentumX << bins_MomentumX[i] << ",";
      m_stream_MomentumY << bins_MomentumY[i] << ",";
      m_stream_MomentumZ << bins_MomentumZ[i] << ",";
    }
    m_stream_MomentumX << bins_MomentumX[m_binsCount_Momentum - 1] << "\n";
    m_stream_MomentumY << bins_MomentumY[m_binsCount_Momentum - 1] << "\n";
    m_stream_MomentumZ << bins_MomentumZ[m_binsCount_Momentum - 1] << "\n";
  }
  /*auto programEndTime = std::chrono::high_resolution_clock::now();
  std::cout << "Time taken (stream): "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   programEndTime - programStartTime)
                   .count()
            << " ms." << std::endl;
  ;*/
}

void DataStreamer::streamMomentumIn(std::ofstream& t_stream, size_t t_binsCount,
                                    numType t_minMomentumValue,
                                    numType t_maxMomentumValue,
                                    ParticleCollection& particleCollection,
                                    PhaseBubble& bubble,
                                    cl::CommandQueue& cl_queue) {
  std::cout << std::setprecision(8) << std::fixed << std::showpoint;
  numType dp = (t_maxMomentumValue - t_minMomentumValue) / t_binsCount;
  std::vector<u_int> bins(t_binsCount);
  numType particleMomentum;
  numType particleRadius;

  t_stream << t_binsCount << "," << t_minMomentumValue << ","
           << t_maxMomentumValue << "\n";
  for (size_t i = 0; i < t_binsCount - 1; i++) {
    t_stream << (i + 1) * dp << ",";
  }
  t_stream << t_binsCount * dp << "\n";

  for (u_int i = 0; i < particleCollection.getParticleCountTotal(); i++) {
    particleMomentum = particleCollection.calculateParticleMomentum(i);
    particleRadius = particleCollection.calculateParticleRadius(i);
    if (particleRadius < bubble.getRadius()) {
      bins[(int)((particleMomentum - t_minMomentumValue) / dp)] += 1;
    }
  }
  for (size_t i = 0; i < t_binsCount - 1; i++) {
    t_stream << bins[i] << ",";
  }
  t_stream << bins[t_binsCount - 1] << "\n";
}

void DataStreamer::streamMomentumOut(
    std::ofstream& t_stream, size_t t_binsCount, numType t_minMomentumValue,
    numType t_maxMomentumValue, ParticleCollection& particleCollection,
    PhaseBubble& bubble, cl::CommandQueue& cl_queue) {
  std::cout << std::setprecision(8) << std::fixed << std::showpoint;
  numType dp = (t_maxMomentumValue - t_minMomentumValue) / t_binsCount;
  std::vector<u_int> bins(t_binsCount);
  numType particleMomentum;
  numType particleRadius;

  t_stream << t_binsCount << "," << t_minMomentumValue << ","
           << t_maxMomentumValue << "\n";
  for (size_t i = 0; i < t_binsCount - 1; i++) {
    t_stream << (i + 1) * dp << ",";
  }
  t_stream << t_binsCount * dp << "\n";

  for (u_int i = 0; i < particleCollection.getParticleCountTotal(); i++) {
    particleMomentum = particleCollection.calculateParticleMomentum(i);
    particleRadius = particleCollection.calculateParticleRadius(i);
    if (particleRadius > bubble.getRadius()) {
      bins[(int)((particleMomentum - t_minMomentumValue) / dp)] += 1;
    }
  }
  for (size_t i = 0; i < t_binsCount - 1; i++) {
    t_stream << bins[i] << ",";
  }
  t_stream << bins[t_binsCount - 1] << "\n";
}

void DataStreamer::streamNumberDensity(std::ofstream& t_stream,
                                       size_t t_binsCount,
                                       numType t_minRadiusValue,
                                       numType t_maxRadiusValue,
                                       ParticleCollection& particleCollection,
                                       cl::CommandQueue& cl_queue) {
  std::cout << std::setprecision(8) << std::fixed << std::showpoint;
  particleCollection.readParticleCoordinatesBuffer(cl_queue);

  numType dr = (t_maxRadiusValue - t_minRadiusValue) / t_binsCount;
  std::vector<u_int> bins(t_binsCount);
  numType particleRadius;
  t_stream << t_binsCount << "," << t_minRadiusValue << "," << t_maxRadiusValue
           << "\n";
  for (size_t i = 0; i < t_binsCount - 1; i++) {
    t_stream << (i + 1) * dr << ",";
  }
  t_stream << t_binsCount * dr << "\n";

  for (size_t i = 0; i < particleCollection.getParticleCountTotal(); i++) {
    particleRadius = particleCollection.calculateParticleRadius(i);
    if ((particleRadius >= t_minRadiusValue) &&
        (particleRadius < t_maxRadiusValue)) {
      bins[(int)((particleRadius - t_minRadiusValue) / dr)] += 1;
    }
  }
  for (size_t i = 0; i < t_binsCount - 1; i++) {
    t_stream << bins[i] << ",";
  }
  t_stream << bins[t_binsCount - 1] << "\n";
}

void DataStreamer::streamEnergyDensity(std::ofstream& t_stream,
                                       size_t t_binsCount,
                                       numType t_minRadiusValue,
                                       numType t_maxRadiusValue,
                                       ParticleCollection& particleCollection,
                                       cl::CommandQueue& cl_queue) {
  particleCollection.readParticleCoordinatesBuffer(cl_queue);
  particleCollection.readParticleEBuffer(cl_queue);

  std::cout << std::setprecision(8) << std::fixed << std::showpoint;
  numType dr = (t_maxRadiusValue - t_minRadiusValue) / t_binsCount;
  std::vector<numType> bins(t_binsCount, 0.);
  numType particleRadius;
  t_stream << t_binsCount << "," << t_minRadiusValue << "," << t_maxRadiusValue
           << "\n";
  for (size_t i = 0; i < t_binsCount - 1; i++) {
    t_stream << (i + 1) * dr << ",";
  }
  t_stream << t_binsCount * dr << "\n";
  for (size_t i = 0; i < particleCollection.getParticleCountTotal(); i++) {
    particleRadius = particleCollection.calculateParticleRadius(i);
    if ((particleRadius >= t_minRadiusValue) &&
        (particleRadius < t_maxRadiusValue)) {
      bins[(int)((particleRadius - t_minRadiusValue) / dr)] +=
          particleCollection.returnParticleE(i);
    }
  }
  for (size_t i = 0; i < t_binsCount - 1; i++) {
    t_stream << bins[i] << ",";
  }
  t_stream << bins[t_binsCount - 1] << "\n";
}

void DataStreamer::streamRadialVelocity(std::ofstream& t_stream,
                                        size_t t_binsCount,
                                        numType t_minRadiusValue,
                                        numType t_maxRadiusValue,
                                        ParticleCollection& particleCollection,
                                        cl::CommandQueue& cl_queue) {
  std::cout << std::setprecision(8) << std::fixed << std::showpoint;
  particleCollection.readParticleCoordinatesBuffer(cl_queue);
  particleCollection.readParticleMomentumsBuffer(cl_queue);
  particleCollection.readParticleEBuffer(cl_queue);

  numType dr = (t_maxRadiusValue - t_minRadiusValue) / t_binsCount;
  std::vector<u_int> bins(t_binsCount);
  std::vector<numType> average_velocity(t_binsCount);
  numType particleRadius;
  numType particleRadialVelocity;
  t_stream << t_binsCount << "," << t_minRadiusValue << "," << t_maxRadiusValue
           << "\n";
  for (size_t i = 0; i < t_binsCount - 1; i++) {
    t_stream << (i + 1) * dr << ",";
  }
  t_stream << t_binsCount * dr << "\n";

  for (u_int i = 0; i < particleCollection.getParticleCountTotal(); i++) {
    particleRadius = particleCollection.calculateParticleRadius(i);
    particleRadialVelocity =
        particleCollection.calculateParticleRadialVelocity(i);

    if ((particleRadius >= t_minRadiusValue) &&
        (particleRadius < t_maxRadiusValue)) {
      average_velocity[(int)(particleRadius / dr)] =
          (average_velocity[(int)(particleRadius / dr)] *
               bins[(int)(particleRadius / dr)] +
           particleRadialVelocity) /
          (bins[(int)(particleRadius / dr)] + 1);
      bins[(int)((particleRadius - t_minRadiusValue) / dr)] += 1;
    }
  }
  for (size_t i = 0; i < t_binsCount - 1; i++) {
    t_stream << average_velocity[i] << ",";
  }
  t_stream << average_velocity[t_binsCount - 1] << "\n";
}

void DataStreamer::streamTangentialVelocity(
    std::ofstream& t_stream, size_t t_binsCount, numType t_minRadiusValue,
    numType t_maxRadiusValue, ParticleCollection& particleCollection,
    cl::CommandQueue& cl_queue) {
  particleCollection.readParticleCoordinatesBuffer(cl_queue);
  particleCollection.readParticleMomentumsBuffer(cl_queue);
  particleCollection.readParticleEBuffer(cl_queue);
  std::cout << std::setprecision(8) << std::fixed << std::showpoint;
  numType dr = (t_maxRadiusValue - t_minRadiusValue) / t_binsCount;
  std::vector<u_int> bins(t_binsCount);
  std::vector<numType> average_velocity(t_binsCount);
  numType particleRadius;
  numType particleTangentialVelocity;

  t_stream << t_binsCount << "," << t_minRadiusValue << "," << t_maxRadiusValue
           << "\n";

  for (size_t i = 0; i < t_binsCount - 1; i++) {
    t_stream << (i + 1) * dr << ",";
  }
  t_stream << t_binsCount * dr << "\n";

  for (u_int i = 0; i < particleCollection.getParticleCountTotal(); i++) {
    particleRadius = particleCollection.calculateParticleRadius(i);
    particleTangentialVelocity =
        particleCollection.calculateParticleTangentialVelocity(i);

    if ((particleRadius >= t_minRadiusValue) &&
        (particleRadius < t_maxRadiusValue)) {
      average_velocity[(int)(particleRadius / dr)] =
          (average_velocity[(int)(particleRadius / dr)] *
               bins[(int)(particleRadius / dr)] +
           particleTangentialVelocity) /
          (bins[(int)(particleRadius / dr)] + 1);
      bins[(int)((particleRadius - t_minRadiusValue) / dr)] += 1;
    }
  }
  for (size_t i = 0; i < t_binsCount - 1; i++) {
    t_stream << average_velocity[i] << ",";
  }
  t_stream << average_velocity[t_binsCount - 1] << "\n";
}

void DataStreamer::streamRadialMomentumProfile(
    std::ofstream& t_stream, size_t t_binsCountRadius,
    size_t t_binsCountMomentum, numType t_minRadiusValue,
    numType t_maxRadiusValue, numType t_minMomentumValue,
    numType t_maxMomentumValue, ParticleCollection& particleCollection,
    cl::CommandQueue& cl_queue) {
  particleCollection.readParticleCoordinatesBuffer(cl_queue);
  particleCollection.readParticleMomentumsBuffer(cl_queue);
  particleCollection.readParticleEBuffer(cl_queue);
  std::cout << std::setprecision(8) << std::fixed << std::showpoint;
  numType particleRadius;
  numType particleMomentum;
  numType dr = (t_maxRadiusValue - t_minRadiusValue) / t_binsCountRadius;
  numType dp = (t_maxMomentumValue - t_minMomentumValue) / t_binsCountMomentum;

  std::vector<std::vector<u_int>> radialMomentumBins(
      t_binsCountRadius, std::vector<u_int>(t_binsCountMomentum, 0));

  t_stream << t_binsCountRadius << "," << t_minRadiusValue << ","
           << t_maxRadiusValue << "," << t_binsCountMomentum << ","
           << t_minMomentumValue << "," << t_maxMomentumValue << "\n";
  for (size_t i = 0; i < t_binsCountRadius; i++) {
    for (size_t j = 0; j < t_binsCountMomentum - 1; j++) {
      t_stream << "(" << (i + 1) * dr << "; " << (j + 1) * dp << ")"
               << ",";
    }
    if (i == t_binsCountRadius - 1) {
      t_stream << "(" << (i + 1) * dr << "; " << t_binsCountMomentum * dp << ")"
               << "\n";
    } else {
      t_stream << "(" << (i + 1) * dr << "; " << t_binsCountMomentum * dp << ")"
               << ",";
    }
  }

  for (u_int i = 0; i < particleCollection.getParticleCountTotal(); i++) {
    particleRadius = particleCollection.calculateParticleRadius(i);
    particleMomentum = particleCollection.calculateParticleMomentum(i);

    if ((particleRadius >= t_minRadiusValue) &&
        (particleRadius < t_maxRadiusValue) &&
        (particleMomentum >= t_minMomentumValue) &&
        (particleMomentum < t_maxMomentumValue)) {
      radialMomentumBins[(int)((particleRadius - t_minRadiusValue) / dr)]
                        [(int)((particleMomentum - t_minMomentumValue) / dp)] +=
          1;
    }
  }

  for (size_t i = 0; i < t_binsCountRadius; i++) {
    for (size_t j = 0; j < t_binsCountMomentum - 1; j++) {
      t_stream << radialMomentumBins[i][j] << ",";
    }
    if (i == t_binsCountRadius - 1) {
      t_stream << radialMomentumBins[i][t_binsCountMomentum - 1] << std::endl;
    } else {
      t_stream << radialMomentumBins[i][t_binsCountMomentum - 1] << ",";
    }
  }
}