#pragma once

#include "base.h"
#include "simulation.h"

// Stream time, Bubble E, Bubble dE,  dP,Bubble r,Bubble v

/*
time, dP, Bubble r, Bubble v,
Bubble energy, Bubble energy change, Particle energy, Particle energy change, Particle energy in, total_energy/initial_total_energy,
Particle count in, Particle count interacted in, Particle count interacted in and passed,
Particle count interacted out, Particle count interacted ot and passed
*/


void streamData(std::fstream& t_stream, Simulation& sim, Bubble& bubble, MainType dP, MainType& bubbleEnergyOld, MainType& particleEnergyOld);
void streamMomentumDistributionsInOut(std::fstream& t_streamIn, std::fstream& t_streamOut,
	std::vector<u_int>& inMomentumBins, std::vector<u_int>& outMomentumBins,
	Simulation& t_sim, Bubble& bubble, MainType& t_dp, MainType& t_pMax
);
void streamNumberDensityDistribution(std::fstream& t_stream, std::vector<u_int>& densityBins, Simulation& sim, MainType& t_dr, MainType& t_rMax);
void streamEnergyDensityDistribution(std::fstream& t_stream, std::vector<MainType>& densityBins, Simulation& sim, MainType& t_dr, MainType& t_rMax);
