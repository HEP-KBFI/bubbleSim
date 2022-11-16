#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define ULONG_MAX 0xffffffffffffffffUL

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
  
  // Rotation parameters
  
  double x;
  double y;
  double z;
  double theta;
  
  double beta;
  double gamma;
  double mass;  
}

double xorshift(unsigned long *seed){
	// Generate random number in range [0, 1)
	*seed ^= *seed >> 12;
    *seed ^= *seed << 25;
    *seed ^= *seed >> 27;
    *seed  = *seed * 0x2545F4914F6CDD1DULL;
	// seed value after xorshift can only be zero if seed = 0.
	return (double) (*seed -1) / ULONG_MAX;
}

__kernel void generate_rotation_axis_and_angles(
	CollisionCell *t_cells,
	unsigned long *cell_seed
){
	// If idx = 0 then let it stay same
	
	unsigned int gid = get_global_id(0);
	// Create unit vector
	CollisionCell cell = t_cells[gid];
	
	double unit_phi = acos(1. - 2.*xorshift(*cell_seed[gid]));
	double unit_theta = 2 * M_PI * xorshift(*cell_seed[gid]);
	
	t_cells[gid].x = sin(unit_phi) * cos(unit_theta);
	t_cells[gid].y = sin(unit_phi) * sin(unit_theta);
	t_cells[gid].z = cos(unit_phi);
	
	if (gid == 0){
		t_cells.theta = 0;
	}
	else {
		t_cells.theta = 2 * M_PI * xorshift(*cell_seed[gid]);
	}
}