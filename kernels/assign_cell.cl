#pragma OPENCL EXTENSION cl_khr_fp64 : enable

typedef struct Particle {
  % Particle physical paramters
  double x;
  double y;
  double z;

  double pX;
  double pY;
  double pZ;

  double E;
  double m;
  
  % Particle simulation parameters and labels
  char b_collide;
  char b_inBubble;
  int idxCollisionCell;
} Particle;

__kernel void assign_particle_cell_index(
	__global Particle *t_particles,
	__global const int *maxIndex,
	__global const double *cellLength;
	__global const double *cuboidShift
	
	){
	unsigned int gid = get_global_id(0);
	
	Particle particle = t_particles[gid];
	// Find cell numbers
	int a = (int) ((particle.x + cuboidShift[0]) / cellLength[0]);
	int b = (int) ((particle.y + cuboidShift[1]) / cellLength[1]);
	int c = (int) ((particle.z + cuboidShift[2]) / cellLength[2]);
	// Idx = 0 -> if particle is outside of the cuboid cell structure
	// Convert cell number into 1D vector
	if ((a < 0) || (a >= maxIndex[0])){
		t_particles[gid].idxCollisionCell = 0;
	}
	else if ((b < 0) || (b >= maxIndex[1])){
		t_particles[gid].idxCollisionCell = 0;
	}
	else if ((c < 0) || (c >= maxIndex[2])){
		t_particles[gid].idxCollisionCell = 0;
	}
	else {
		t_particles[gid].idxCollisionCell = 1 + a + b * maxIndex[0] + c * maxIndex[0] * maxIndex[1];
	}
}

__kernel void assign_particle_cell_index_two_phase(
	__global Particle *t_particles,
	__global const int *maxIndex,
	__global const double *cellLength;
	__global const double *cuboidShift
	
	){
	unsigned int gid = get_global_id(0);
	
	Particle particle = t_particles[gid];
	// Find cell numbers
	int a = (int) ((particle.x + cuboidShift[0]) / cellLength[0]);
	int b = (int) ((particle.y + cuboidShift[1]) / cellLength[1]);
	int c = (int) ((particle.z + cuboidShift[2]) / cellLength[2]);
	// Idx = 0 -> if particle is outside of the cuboid cell structure
	// Convert cell number into 1D vector. First half of the vector is for outisde the bubble and second half is inside the bubble
	if ((a < 0) || (a >= maxIndex[0])){
		t_particles[gid].idxCollisionCell = 0;
	}
	else if ((b < 0) || (b >= maxIndex[1])){
		t_particles[gid].idxCollisionCell = 0;
	}
	else if ((c < 0) || (c >= maxIndex[2])){
		t_particles[gid].idxCollisionCell = 0;
	}
	else {
		t_particles[gid].idxCollisionCell = 1 + a + b * maxIndex[0] + c * maxIndex[0] * maxIndex[1] + particle.b_inBubble * maxIndex[0] * maxIndex[1]*maxIndex[2];
	}
	
}