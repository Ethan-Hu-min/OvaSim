#pragma once
#include <vector>
#include <array>
#include "gdt/gdt/math/AffineSpace.h"

using namespace gdt;

struct Material
{
	float impedance, attenuation, mu0, mu1, sigma, specularity;

};

struct  TriangleMesh
{
	std::vector<vec3f> vertex;
	std::vector<vec3f> normal;
	std::vector<vec3i> index;
	int indexModel;
};

class Model
{	
	public:
		Model(std::string fname, std::array<float, 3> d, std::array<float, 3> s, Material& inside, Material& outside)
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
		int indexModel;
		std::array<float, 3> deltas;
		std::array<float, 3> scaling;
		Material& material_inside;
		Material& material_outside;
};