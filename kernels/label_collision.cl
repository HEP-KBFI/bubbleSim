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

double xorshift(unsigned long *seed){
	// Generate random number in range [0, 1)
	*seed ^= *seed >> 12;
    *seed ^= *seed << 25;
    *seed ^= *seed >> 27;
    *seed  = *seed * 0x2545F4914F6CDD1DULL;
	// seed value after xorshift can only be zero if seed = 0.
	return (double) (*seed -1) / ULONG_MAX;
}


__kernel void label_collision(
	__global Particle *t_particles,
	__global unsigned long *t_seed,
	__global double *probability
){
	unsigned_int gid = get_global_id(0);
	t_particles[gid] = (char) (xorshift(t_seed[gid]) < probability);
}