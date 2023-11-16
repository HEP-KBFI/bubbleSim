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
  m_stream_dt = 0.;
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
                   "float64, float64, float64, float64, float64, float64, "
                   "float64, float64, float64]"
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
  m_momentum_value.resize(m_N_bins_in_profile + m_N_bins_out_profile,
                          (numType)0.);
  m_momentum_change.resize(m_N_bins_in_profile + m_N_bins_out_profile,
                           (numType)0.);
  m_square_mean_velocity.resize(m_N_bins_in_profile + m_N_bins_out_profile,
                                (numType)0.);

  m_mean_velocity.resize(m_N_bins_in_profile + m_N_bins_out_profile,
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
  memset(&m_momentum_value[0], 0,
         sizeof(m_momentum_value[0]) * m_momentum_value.size());
  memset(&m_momentum_change[0], 0,
         sizeof(m_momentum_change[0]) * m_momentum_change.size());
  memset(&m_mean_velocity[0], 0,
         sizeof(m_mean_velocity) * m_mean_velocity.size());
  memset(&m_square_mean_velocity[0], 0,
         sizeof(m_square_mean_velocity) * m_square_mean_velocity.size());

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
  numType momentum_copy = 0.;
  numType theta = 0.;
  numType phi = 0;
  uint64_t index = 0;
  int index_p = 0;

  uint64_t index_radius = 0;
  int index_momentum = 0;

  numType p_R = 0.;
  numType p_phi = 0.;
  numType p_theta = 0.;

  numType deltaPx;
  numType deltaPy;
  numType deltaPz;
  // Collect data

  for (size_t i = 0; i < particleCollection.getParticleCountTotal(); i++) {
    radius = particleCollection.calculateParticleRadius(i);
    // if (b_stream_momentum_in || b_stream_momentum_out ||
    //     b_stream_momentum_radial_profile) {
    momentum = particleCollection.calculateParticleMomentum(i);
    //}

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
    // px value t - dt (last step)
    momentum_copy = particleCollection.calculateParticleMomentumCopy(i);

    deltaPx = particleCollection.getParticlepX()[i] -
              particleCollection.getParticlepXCopy()[i];
    deltaPy = particleCollection.getParticlepY()[i] -
              particleCollection.getParticlepYCopy()[i];
    deltaPz = particleCollection.getParticlepZ()[i] -
              particleCollection.getParticlepZCopy()[i];

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
    m_T02[index] += p_theta;
    m_T03[index] += p_phi;
    m_T11[index] += pow(p_R, 2.) / energy;
    m_T22[index] += pow(p_theta, 2.) / energy;
    m_T33[index] += pow(p_phi, 2.) / energy;
    m_T12[index] += p_R * p_theta / energy;
    m_T13[index] += p_R * p_phi / energy;
    m_T23[index] += p_phi * p_theta / energy;
    m_radial_velocity[index] += p_R / energy;
    m_momentum_value[index] += momentum_copy;
    m_momentum_change[index] += std::sqrt(
        std::pow(deltaPx, 2.) + std::pow(deltaPy, 2.) + std::pow(deltaPz, 2.));

    m_mean_velocity[index] += momentum / energy;
    m_square_mean_velocity[index] += std::pow(momentum / energy, 2.);
  }

  m_stream_data_time = simulation.getTime();
  m_stream_dt = simulation.get_dt_currentStep();
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
  m_stream_data.write(reinterpret_cast<char*>(&m_stream_dt), sizeof(numType));
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
  m_stream_data.flush();
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
  m_stream_profile.write(reinterpret_cast<const char*>(m_momentum_value.data()),
                         m_momentum_value.size() * sizeof(numType));
  m_stream_profile.write(
      reinterpret_cast<const char*>(m_momentum_change.data()),
      m_momentum_change.size() * sizeof(numType));
  m_stream_profile.write(reinterpret_cast<const char*>(m_mean_velocity.data()),
                         m_mean_velocity.size() * sizeof(numType));
  m_stream_profile.write(
      reinterpret_cast<const char*>(m_square_mean_velocity.data()),
      m_square_mean_velocity.size() * sizeof(numType));
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
