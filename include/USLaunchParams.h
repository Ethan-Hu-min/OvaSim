#pragma once
#include "optix.h"
#include "support/gdt/gdt/math/vec.h"
#include <include/GlobalConfig.h>

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

struct USLaunchParams {
	struct {
		uint32_t* colorBuffer;
		vec2i  size;
		float* intensityBuffer;
	}frame;

	struct {
		uint8_t* bgTexture;
		uint8_t* bladderTexture;
		uint8_t* uterusTexture;
		uint8_t* uterusinTexture;
		uint8_t* intestineTexture;
		uint8_t* ovaryTexture;
		uint8_t* ovamTexture;
	}textures;

	struct {
		vec3f position;
		vec3f direction;
		vec3f horizontal;
		vec3f vertical;
		int32_t nums;
		float angle;
		float width;
	}transducer;

	struct
	{
		float relaAngle;
		float relaDepth;
		uint8_t* collide_models_id;
		vec3f* collide_models_pos;
	}needle;
	OptixTraversableHandle traversable;
	int maxBounce = GlobalConfig::maxBounceNum;
	int numSamples = GlobalConfig::SampleNum;
};