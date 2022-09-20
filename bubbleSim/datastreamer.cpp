#include "datastreamer.h"

DataStreamer::DataStreamer(Simulation& t_sim, Bubble& t_bubble, OpenCLWrapper& t_openCLWrapper) {
	m_sim = t_sim;
	m_bubble = t_bubble;
	m_openCLWrapper = t_openCLWrapper;
}

void DataStreamer::reset() {
	m_readBufferX = false;
	m_readBufferP = false;
	m_readBuffer_dP = false;
	m_readBufferM = false;
	m_readBufferE = false;

	m_readBufferInteractedFalse = false;
	m_readBufferPassedFalse = false;
	m_readBufferInteractedTrue = false;

	m_readBufferR = false;
	m_readBufferSpeed = false;
}

void DataStreamer::streamBaseData(std::fstream& t_stream, bool t_isBubbleTrueVacuum) {
	int countParticleFalse = 0, countParticleInteractedFalse = 0, countParticlePassedFalse = 0, countParticleInteratedTrue = 0;
	numType particleEnergy = 0., particleEnergyFalse = 0.;
	numType changeInPressure = 0.;

	if (!m_readBufferX) {
		m_openCLWrapper.readBufferX(m_sim.getReferenceX());
		m_readBufferX = true;
	}

	if (!m_readBufferInteractedFalse) {
		m_openCLWrapper.readBufferInteractedFalse(m_sim.getReferenceInteractedFalse());
		m_readBufferInteractedFalse = true;
	}

	if (!m_readBufferInteractedFalse) {
		m_openCLWrapper.readBufferPassedFalse(m_sim.getReferencePassedFalse());
		m_readBufferInteractedFalse = true;
	}

	if (!m_readBufferInteractedFalse) {
		m_openCLWrapper.readBufferInteractedFalse(m_sim.getReferenceInteractedFalse());
		m_readBufferInteractedFalse = true;
	}
	// If true vacuum is inside the bubble
	if (t_isBubbleTrueVacuum) {
		for (int i = 0; i < m_sim.getParticleCountTotal(); i++) {
			if (m_sim.calculateParticleRadius(i) > m_bubble.getRadius()) {
				countParticleFalse += 1;
				particleEnergyFalse += m_sim.getReferenceE()[i];
			}
			countParticleInteractedFalse += m_sim.getReferenceInteractedFalse()[i];
			countParticlePassedFalse += m_sim.getReferencePassedFalse()[i];
			countParticleInteratedTrue += m_sim.getReferenceInteractedTrue()[i];
		}
	}
	// If true vacuum is outside the bubble
	else {
		for (int i = 0; i < m_sim.getParticleCountTotal(); i++) {
			if (m_sim.calculateParticleRadius(i) < m_bubble.getRadius()) {
				countParticleFalse += 1;
				particleEnergyFalse += m_sim.getReferenceE()[i];
			}
			countParticleInteractedFalse += m_sim.getReferenceInteractedFalse()[i];
			countParticlePassedFalse += m_sim.getReferencePassedFalse()[i];
			countParticleInteratedTrue += m_sim.getReferenceInteractedTrue()[i];
		}
	}
	t_stream << m_sim.getTime() << "," << m_sim.getdPressureStep() << "," << m_bubble.getRadius() << "," << m_bubble.getSpeed() << ",";
	t_stream << m_sim.getBubbleEnergy() << "," << m_sim.getBubbleEnergy() - m_sim.getBubbleEnergyLastStep() << ",";
	t_stream << m_sim.getParticlesEnergy() << "," << m_sim.getParticlesEnergy() - m_sim.getParticlesEnergyLastStep() << ",";
	t_stream << particleEnergyFalse << "," << m_sim.getTotalEnergy() / m_sim.getTotalEnergyInitial() << ",";
	t_stream << countParticleFalse << "," << countParticleInteractedFalse << "," << countParticlePassedFalse << ",";
	t_stream << countParticleInteratedTrue;

	/*
		t_stream << sim.m_time << "," << dP << "," << bubble.m_radius << "," << bubble.m_speed << ",";
		t_stream << bubbleEnergy << "," << bubbleEnergy - bubbleEnergyOld << "," << particleEnergy << "," << particleEnergy - particleEnergyOld << ",";
		t_stream << particleEnergyIn << "," << (bubbleEnergy + particleEnergy) / sim.m_energyTotalInitial << ",";
		t_stream << countParticleIn << "," << countParticleInteractedIn << "," << countParticleInteractedPassedIn << ",";
		t_stream << countParticleInteratedOut << std::endl;
	*/
}