#include <vector>
#include <unordered_map>
#include "gdt/gdt/gdt.h"
#include "Model.h"

using namespace gdt;

struct Transducer
{
	vec3f t_position;
	vec3f t_direction;
	vec3f t_vertical;
	int  t_nums;
	float  t_angle;
	float t_width;
	float t_depth;
};


class Scene {
public:
	Scene();
	void parseConfig(std::string config_dir);
	void setTransducer(vec3f pos, vec3f dir, vec3f ver, int nums, float angle, float depth, float width);
	void loadModels();
	void createWorldModels();
	void setExampleName(std::string name);
public:

	std::string exampleName;
  	std::vector<Model> models;
	std::vector<TriangleMesh*> worldmodel;
	Transducer transducer;
	std::unordered_map<std::string, Material> materials;
};