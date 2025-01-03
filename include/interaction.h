#include "support/gdt/gdt/math/vec.h"
#include "USLaunchParams.h"
#include "math.h"

struct Ray {
	vec3f origin;
	vec3f direction;
	float tmax = FLT_MAX;
};

struct Interaction {
	float bias = 0.001f;
	float distance;
	vec3f position;
	vec3f geomNormal;
	int indexModelInt;
	__forceinline__ __device__ Ray gene_ray(const vec3f& wi) const {
		vec3f N = geomNormal;
		if (dot(wi, geomNormal) > 0.0f) {
			N = -geomNormal;
			printf("inver\n");
		}

		Ray ray;
		ray.origin = position + wi * bias;
		ray.direction = wi;
		return ray;
	}
};