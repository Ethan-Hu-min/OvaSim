#include "Scene.h"
#include <nlohmann/json.hpp>
#include <cmath>
#include <iostream>
#include <fstream>

Scene::Scene(){}

void Scene::setSceneDir(std::string dir) {
	scene_dir = dir;
}

void Scene::parseConfig(std::string config_dir) {
	nlohmann::json json;
	std::ifstream infile{ (scene_dir + config_dir) };
	json << infile;
	const auto& mats = json.at("materials");
	if (mats.is_array()) {
		for (const auto& mat : mats) {
			materials[mat.at("name")] = {
					mat.at("impedance"),
					mat.at("attenuation"),
					mat.at("mu0"),
					mat.at("mu1"),
					mat.at("sigma"),
					mat.at("specularity")
			};
		}
	}
	else
	{
		throw std::runtime_error(" materials must be an array");
	}
	const auto& meshes_ = json.at("meshes");
	if (meshes_.is_array()) {
		for (const auto& mesh_ : meshes_) {
			const auto& deltas{ mesh_.at("deltas") };
			const auto& scalings{ mesh_.at("scaling") };
			//std::cout << mesh_.at("file") << std::endl;
			models.emplace_back(Model(scene_dir+std::string((mesh_.at("file"))),
				{ deltas[0], deltas[1],deltas[2] }, 
				{ scalings[0], scalings[1],scalings[2] }, 
				materials.at((mesh_.at("material"))),
				materials.at(mesh_.at("outsideMaterial"))));
			//std::cout << models[0].filename << std::endl;
		}
	}
	else
	{
		throw std::runtime_error("meshes must be an array");
	}
	for (int i = 0; i < models.size(); i++) {
		models[i].indexModel = i;
	}
}


void Scene::loadModels() {
	for (auto& model : models) {
		model.loadOBJ();
	}
}

void Scene::createWorldModels() {
	worldmodel.clear();
	for (auto& model : models) {
		worldmodel.insert(worldmodel.end(), model.meshes.begin(), model.meshes.end());
	}
}

void Scene::setTransducer(vec3f& pos, vec3f& dir,vec3f& ver, int nums,int angle, float depth, float width) {
	transducer.t_position = pos;
	transducer.t_direction = normalize(dir) ;
	transducer.t_vertical = normalize(ver);
	transducer.t_nums = nums;
	transducer.t_angle = angle;
	transducer.t_depth = depth;
	transducer.t_width = width;
}