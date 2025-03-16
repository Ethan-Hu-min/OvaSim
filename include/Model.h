#pragma once
#include <vector>
#include <array>
#include "gdt/gdt/math/AffineSpace.h"

#define BLADDER 1
#define UTERUS 2
#define UTERUSIN 3
#define INTESTINE 4
#define OVARY 5
#define OVAM 6

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
	Model(std::string fname, std::vector<double> d, std::vector<double> s, std::string& inside)
		: filename(std::move(fname)), deltas(d), scaling(s), material_inside(inside)
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
	std::string material_inside;
	//Material& material_outside;
	vec3f modelCenter;
};