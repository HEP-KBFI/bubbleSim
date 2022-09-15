#include "stream_data.h"


/*
time, dP, Bubble r, Bubble v,
Bubble energy, Bubble energy change, Particle energy, Particle energy change, Particle energy in, total_energy/initial_total_energy,
Particle count in, Particle count interacted in, Particle count interacted in and passed,
Particle count interacted out, Particle count interacted ot and passed
*/
void streamData(std::fstream& t_stream, Simulation& sim, Bubble& bubble, MainType dP, MainType& bubbleEnergyOld, MainType& particleEnergyOld) {
	MainType bubbleEnergy = bubble.calculateEnergy();
	MainType particleEnergyIn = 0.;
	MainType particleEnergy = 0.;
	u_int countParticleIn = 0;
	u_int countParticleInteractedIn = 0;
	u_int countParticleInteractedPassedIn = 0;
	u_int countParticleInteratedOut = 0;
	u_int countParticleInteractedPassedOut = 0;
	for (u_int i = 0; i < sim.m_particleCount; i++) {
		particleEnergy += sim.m_E[i];
		if (sim.getParticleRadius(i) <= bubble.m_radius) {
			particleEnergyIn += sim.m_E[i];
			countParticleIn += 1;
		}
		countParticleInteratedOut += sim.m_InteractedTrue[i];
		countParticleInteractedPassedOut += sim.m_PassedTrue[i];
		countParticleInteractedIn += sim.m_InteractedFalse[i];
		countParticleInteractedPassedIn += sim.m_PassedFalse[i];
	}
	std::cout << countParticleInteractedIn << ", " << countParticleInteractedPassedIn << ", " << countParticleInteratedOut << ", " << countParticleInteractedPassedOut << std::endl;
	std::cout << "time: " << sim.m_time << ", R: " << bubble.m_radius << ", V: " << bubble.m_speed << ", E/E_init: " << (bubbleEnergy + particleEnergy) / sim.m_energyTotalInitial;
	std::cout << ", Count (In/Out): " << countParticleIn << " / " << sim.m_particleCount - countParticleIn << std::endl;

	t_stream << sim.m_time << "," << dP << "," << bubble.m_radius << "," << bubble.m_speed << ",";
	t_stream << bubbleEnergy << "," << bubbleEnergy - bubbleEnergyOld << "," << particleEnergy << "," << particleEnergy - particleEnergyOld << ",";
	t_stream << particleEnergyIn << "," << (bubbleEnergy + particleEnergy) / sim.m_energyTotalInitial << ",";
	t_stream << countParticleIn << "," << countParticleInteractedIn << "," << countParticleInteractedPassedIn << ",";
	t_stream << countParticleInteratedOut << "," << countParticleInteractedPassedOut << std::endl;
	
	bubbleEnergyOld = bubbleEnergy;
	particleEnergyOld = particleEnergy;

}

void streamNumberDensityDistribution(std::fstream& t_stream, std::vector<u_int>& densityBins, Simulation& sim, MainType& t_dr, MainType& t_rMax) {
	MainType r;
	int j;

	for (u_int i = 0; i < sim.m_particleCount; i++) {
		r = sim.getParticleRadius(i);
		if (r < t_rMax) {
			j = (int)(r / t_dr);
			densityBins[j] += 1;
		}
	}
	for (u_int i = 0; i < densityBins.size(); i++) {
		t_stream << densityBins[i] / (4 * (MainType)M_PI * pow(t_dr, 3) * (i * i + i + (MainType)1.0 / (MainType)3.0)) / sim.m_nInSimInitial;
		if (i != densityBins.size() - 1) { t_stream << ","; }
	}
	t_stream << std::endl;

	memset(&densityBins[0], 0, densityBins.size() * sizeof densityBins[0]);
}
void streamEnergyDensityDistribution(std::fstream& t_stream, std::vector<MainType>& densityBins, Simulation& sim, MainType& t_dr, MainType& t_rMax) {
	MainType r;
	int j;

	for (u_int i = 0; i < sim.m_particleCount; i++) {
		r = sim.getParticleRadius(i);
		if (r < t_rMax) {
			j = (int)(r / t_dr);
			densityBins[j] += sim.m_E[i];
		}
	}
	for (u_int i = 0; i < densityBins.size(); i++) {
		t_stream << densityBins[i] / (4 * (MainType)M_PI * pow(t_dr, 3) * (i * i + i + (MainType)1.0 / (MainType)3.0)) / sim.m_rhoInSimInitial;
		if (i != densityBins.size() - 1) { t_stream << ","; }
	}
	t_stream << std::endl;

	memset(&densityBins[0], 0, densityBins.size() * sizeof densityBins[0]);
}
void streamMomentumDistributionsInOut(std::fstream& t_streamIn, std::fstream& t_streamOut,
	std::vector<u_int>& inMomentumBins, std::vector<u_int>& outMomentumBins,
	Simulation& t_sim, Bubble& bubble, MainType& t_dp, MainType& t_pMax
) {
	MainType p;
	MainType r;
	int j;

	for (u_int i = 0; i < t_sim.m_particleCount; i++) {
		p = t_sim.getParticleMomentum(i);
		r = t_sim.getParticleRadius(i);
		if (p <= t_pMax) {
			j = (int)(p / t_dp);
			if (r > bubble.m_radius) {
				outMomentumBins[j] += 1;
			}
			else {
				inMomentumBins[j] += 1;
			}
		}
	}
	for (u_int i = 0; i < outMomentumBins.size(); i++) {
		t_streamIn << inMomentumBins[i];
		t_streamOut << outMomentumBins[i];
		if (i != outMomentumBins.size() - 1) {
			t_streamIn << ",";
			t_streamOut << ",";
		}
	}
	t_streamIn << std::endl;
	t_streamOut << std::endl;

	memset(&inMomentumBins[0], 0, inMomentumBins.size() * sizeof inMomentumBins[0]);
	memset(&outMomentumBins[0], 0, outMomentumBins.size() * sizeof outMomentumBins[0]);
}