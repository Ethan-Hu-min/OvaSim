#include "support/gdt/gdt/math/vec.h"
#include "USLaunchParams.h"
#include "math.h"

struct Ray {
	vec3f origin;
	vec3f direction;
	float tmax = FLT_MAX;
};

struct Interaction {
	vec3f position;
	vec3f geomNormal;
	vec3f next_dir;
	bool is_inside;
	bool is_stop;
	int indexModelInt;
	float intensity;
};