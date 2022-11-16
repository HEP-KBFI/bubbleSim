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
  double v_x;
  double v_y;
  double v_z;
  
  double x;
  double y;
  double z;
  double theta;
  
  double beta;
  double gamma;
  double mass;  
}


__kernel void calculate_center_of_mass_frame(
	__global Particle *t_particles,
	__global CollisionCell *t_collisionCells
	){
	% a + (b + c) != (a + b) + c 
	% atom_add has to be implemented by myself for double type?

}