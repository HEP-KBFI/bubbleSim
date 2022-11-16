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

typedef struct Bubble {
  double radius;
  double radius2;           // Squared
  double radiusAfterStep2;  // (radius + speed * dt)^2
  double speed;
  double gamma;
  double gammaXspeed;  // gamma * speed
} Bubble;


__kernel void label_particles_position_by_coordinate(
	__global Particle *t_particles,
	__global Bubble *t_bubble
	){
	unsigned int gid = get_global_id(0);
	
	% If R_b^2 > R_x^2 then particle is inside the bubble
	t_particles[gid].b_inBubble = fma(
									t_particles[gid].x, t_particles[gid].x,
										fma(t_particles[gid].y, t_particles[gid].y,
											t_particles[gid].z * t_particles[gid].z )) < t_bubble[0].radius2;
}

__kernel void label_particles_position_by_mass(
	__global Particle *t_particles,
	__global double *mass_in
	){
	unsigned int gid = get_global_id(0);
	
	% If R_b^2 > R_x^2 then particle is inside the bubble
	t_particles[gid].b_inBubble = t_particles[gid].m == mass_in[0];
}
