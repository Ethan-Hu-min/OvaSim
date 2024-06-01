#pragma once
#include "optix.h"
#include "support/gdt/gdt/math/vec.h"

using namespace gdt;

struct  TriangleMeshSBTData
{
	vec3f color;
	vec3f* normal;
	vec3f* vertex;
	vec3i* index;
};

struct LaunchParams {
	struct {
		uint32_t* colorBuffer;
		vec2i  size;
	}frame;
	struct {
		vec3f position;
		vec3f direction;
		vec3f horizontal;
		vec3f vertical;
	}camera;
	OptixTraversableHandle traversable;
};