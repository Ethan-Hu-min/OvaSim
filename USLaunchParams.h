#pragma once
#include "optix.h"
#include "support/gdt/gdt/math/vec.h"
#define PI 3.14159

using namespace gdt;

struct  TriangleMeshSBTData
{
	vec3f color;
	vec3f* normal;
	vec3f* vertex;
	vec3i* index;
	int indexModelSBT;
};

struct segment {
	vec3f from, to;
	int material_id;
	segment(const vec3f& f, const vec3f& t, int id) : from(f), to(t), material_id(id) {};
};

struct USLaunchParams {
	struct {
		uint32_t* colorBuffer;
		vec2i  size;
	}frame;

	struct {
		vec3f position;
		vec3f direction;
		vec3f horizontal;
		vec3f vertical;
		int32_t nums;
		float angle;
		float width;
	}transducer;
	OptixTraversableHandle traversable;
	int maxBounce = 6;
};