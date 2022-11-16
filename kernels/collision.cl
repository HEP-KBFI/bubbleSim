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


typedef struct CollisionCell {
  double vX;
  double vY;
  double vZ;
  
  double x;
  double y;
  double z;
  double theta;
  
  double v2; // v2 = Sum: v_i^2 
  double gamma;
  double mass;  
}

__kernel void collide(
	Particle *t_particles,
	CollisionCell *t_cells,
	unsigned int *number_of_cells;
	
	){
		double gammaMinusOne = gamma - 1;
		double cos_theta = cos(cell.theta);
		double sin_theta = sin(cell.theta);
		double p0 = 0, p1 = 0, p2 = 0, p3 = 0;
		unsigned int gid = get_global_id(0);
		Particle particle = t_particles[gid];
		// If in bubble then cell number is doubled and second half is in bubble cells.
		CollisionCell cell = t_cells[particle.idxCollisionCell + number_of_cells*particle.inBubble]);
		
		// Lorentz boost
		p0 = cell.gamma * (
			particle.E 
			- particle.pX * cell.vX 
			- particle.pY * cell.vY 
			- particle.pZ * cell.vZ
			);
		p1 = -gamma * particle.E * cell.vX 
			+ particle.pX * (1 + vX * vX * gammaMinusOne/cell.v2)
			+ particle.pY * cell.vX * cell.vY * gammaMinusOne/cell.v2
			+ particle.pZ * cell.vX * cell.vZ * gammaMinusOne/cell.v2;
		
		p2 = -gamma * particle.E * cell.vY 
			+ particle.pY * (1 + vY * vY * gammaMinusOne/cell.v2)
			+ particle.pX * cell.vX * cell.vY * gammaMinusOne/cell.v2
			+ particle.pZ * cell.vY * cell.vZ * gammaMinusOne/cell.v2;
		
		p3 = -gamma * particle.E * cell.vZ 
			+ particle.pZ * (1 + vZ * vZ * gammaMinusOne/cell.v2)
			+ particle.pX * cell.vX * cell.vZ * gammaMinusOne/cell.v2
			+ particle.pY * cell.vY * cell.vZ * gammaMinusOne/cell.v2;
		
		particle.E = p0;
		particle.pX = p1;
		particle.pY = p2;
		particle.pZ = p3;
		
		// Rotate momentum
		p1 = particle.pX * ( 1 + (cos_theta - 1) * (cell.y*cell.y + cell.z*cell.z)) -
			 particle.pY * (cell.x * cell.y * (cos_theta - 1) + cell.z * sin_theta) - 
			 particle.pZ * (cell.x * cell.z * (cos_theta - 1) - cell.y * sin_theta);
		p2 = particle.pY * ( 1 + (cos_theta - 1) * (cell.x*cell.x + cell.z*cell.z)) -
			 particle.pX * (cell.x * cell.y * (cos_theta - 1) - cell.z * sin_theta) - 
			 particle.pZ * (cell.y * cell.z * (cos_theta - 1) + cell.x * sin_theta);
		p3 = particle.pZ * ( 1 + (cos_theta - 1) * (cell.x*cell.x + cell.y*cell.y)) -
			 particle.pX * (cell.x * cell.z * (cos_theta - 1) + cell.y * sin_theta) - 
			 particle.pY * (cell.y * cell.z * (cos_theta - 1) - cell.x * sin_theta);
			 
		particle.pX = p1;
		particle.pY = p2;
		particle.pZ = p3;
		
		// Lorentz inverse transformation
		
		p0 = cell.gamma * (
			particle.E 
			+ particle.pX * cell.vX 
			+ particle.pY * cell.vY 
			+ particle.pZ * cell.vZ
			);
		p1 = +gamma * particle.E * cell.vX 
			+ particle.pX * (1 + vX * vX * gammaMinusOne/cell.v2)
			+ particle.pY * cell.vX * cell.vY * gammaMinusOne/cell.v2
			+ particle.pZ * cell.vX * cell.vZ * gammaMinusOne/cell.v2;
		
		p2 = +gamma * particle.E * cell.vY 
			+ particle.pY * (1 + vY * vY * gammaMinusOne/cell.v2)
			+ particle.pX * cell.vX * cell.vY * gammaMinusOne/cell.v2
			+ particle.pZ * cell.vY * cell.vZ * gammaMinusOne/cell.v2;
		
		p3 = +gamma * particle.E * cell.vZ 
			+ particle.pZ * (1 + vZ * vZ * gammaMinusOne/cell.v2)
			+ particle.pX * cell.vX * cell.vZ * gammaMinusOne/cell.v2
			+ particle.pY * cell.vY * cell.vZ * gammaMinusOne/cell.v2;
		
		particle.E = p0;
		particle.pX = p1;
		particle.pY = p2;
		particle.pZ = p3;
		
		t_particles[gid] = particle;
		
	}