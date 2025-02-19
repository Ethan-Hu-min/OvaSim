#pragma once
#include <vector>
#include <array>
#include "gdt/gdt/math/AffineSpace.h"

#define OVARY 1
#define UTERUS 2
#define INTESTINE 3
#define BLADDER 4
#define OVAM 5

using namespace gdt;

struct Material
{
	std::string mat_name;
	double impedance, attenuation, mu0, mu1, sigma, specularity;
};

struct  TriangleMesh
{
	std::vector<vec3f> vertex;
	std::vector<vec3f> normal;
	std::vector<vec3i> index;
	int indexMaterial;
	int indexModel;
};

class Model
{
public:
	Model(std::string fname, std::vector<double> d, std::vector<double> s, Material& inside, Material& outside)
		: filename(std::move(fname)), deltas(d), scaling(s), material_inside(inside), material_outside(outside)
	{
	}
	~Model() {
		for (auto mesh : meshes) delete mesh;
	}
	void loadOBJ();
public:
	std::vector<TriangleMesh*> meshes;
	std::string filename;
	int indexModel = -1;
	std::vector<double> deltas;
	std::vector<double> scaling;
	Material& material_inside;
	Material& material_outside;
	vec3f modelCenter;
};