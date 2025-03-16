#include "Model.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include"tiny_obj_loader.h"
#include<set>
#include<qdebug.h>

namespace std {
	inline bool operator<(const tinyobj::index_t& a,
		const tinyobj::index_t& b)
	{
		if (a.vertex_index < b.vertex_index)return true;
		if (a.vertex_index > b.vertex_index)return false;
		if (a.normal_index < b.normal_index)return true;
		if (a.normal_index > b.normal_index)return false;
		if (a.texcoord_index < b.texcoord_index)return true;
		if (a.texcoord_index < b.texcoord_index)return false;

		return false;
	}
}

int addVertex(TriangleMesh* mesh, tinyobj::attrib_t& attributes, const tinyobj::index_t& idx,
	std::map<tinyobj::index_t, int>& knownVertices) {
	if (knownVertices.find(idx) != knownVertices.end())return knownVertices[idx];
	const vec3f* vertex_array = (const vec3f*)attributes.vertices.data();
	const vec3f* normal_array = (const vec3f*)attributes.normals.data();
	const vec2f* texcoord_array = (const vec2f*)attributes.texcoords.data();
	int newID = mesh->vertex.size();
	knownVertices[idx] = newID;
	mesh->vertex.push_back(vertex_array[idx.vertex_index]);
	if (idx.normal_index >= 0) {
		while (mesh->normal.size() < mesh->vertex.size())
			mesh->normal.push_back(normal_array[idx.normal_index]);
	}
	// just for sanity's sake:
	if (mesh->normal.size() > 0)
		mesh->normal.resize(mesh->vertex.size());

	return newID;
}

vec3f calculCenter(tinyobj::attrib_t& attributes) {
	const int vertex_size = (const int)attributes.vertices.size();
	double vertex_sum_x = 0.0;
	double vertex_sum_y = 0.0;
	double vertex_sum_z = 0.0;
	for (int i = 0; i < vertex_size / 3; i++) {
		vertex_sum_x += attributes.vertices[i * 3];
		vertex_sum_y += attributes.vertices[i * 3 + 1];
		vertex_sum_z += attributes.vertices[i * 3 + 2];
	}
	return vec3f(vertex_sum_x / float(vertex_size / 3), vertex_sum_y / float(vertex_size / 3), vertex_sum_z / float(vertex_size / 3));
}

void Model::loadOBJ() {
	qDebug() << this->filename;
	tinyobj::attrib_t attributes;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string err = "";
	std::string mtl_dir = "";
	bool readOK = tinyobj::LoadObj(&attributes, &shapes, &materials, &err, &err, this->filename.c_str(), mtl_dir.c_str(), true);
	if (!readOK) {
		throw std::runtime_error("Could not read OBJ model from " + this->filename + ":" + err);
	}
	qDebug() << "model index: " << this->indexModel;
	qDebug() << "Done loading obj file - found " << shapes.size() << " shapes ";
	//qDebug() << shapes[0].mesh.indices[0].vertex_index << (vec3f)attributes.vertices[1];

	this->modelCenter = calculCenter(attributes);
	qDebug() << "modelCenter: " << this->modelCenter [0] << this->modelCenter[1]<< this->modelCenter[2];



	for (const auto& shape : shapes) {
		std::set<int> materialIDs;
		for (auto faceMatId : shape.mesh.material_ids)
			materialIDs.insert(faceMatId);
		for (int materialID : materialIDs) {
			std::map<tinyobj::index_t, int> knownVertices;
			TriangleMesh* mesh = new TriangleMesh;
			for (int faceID = 0; faceID < shape.mesh.material_ids.size(); faceID++) {
				if (shape.mesh.material_ids[faceID] != materialID)continue;
				tinyobj::index_t idx0 = shape.mesh.indices[3 * faceID + 0];
				tinyobj::index_t idx1 = shape.mesh.indices[3 * faceID + 1];
				tinyobj::index_t idx2 = shape.mesh.indices[3 * faceID + 2];
				vec3i idx(addVertex(mesh, attributes, idx0, knownVertices),
					addVertex(mesh, attributes, idx1, knownVertices),
					addVertex(mesh, attributes, idx2, knownVertices));
				mesh->index.push_back(idx);

			}
			mesh->indexModel = this->indexModel;
			if (material_inside == "Ovary")mesh->indexMaterial = OVARY;
			else if (material_inside == "Uterus")mesh->indexMaterial = UTERUS;
			else if (material_inside == "Uterusin")mesh->indexMaterial = UTERUSIN;
			else if (material_inside == "Intestine")mesh->indexMaterial = INTESTINE;
			else if (material_inside == "Bladder")mesh->indexMaterial = BLADDER;
			else if (material_inside == "Ovam")mesh->indexMaterial = OVAM;
			else 	mesh->indexMaterial = 999;
			if (mesh->vertex.empty())delete mesh;
			else this->meshes.push_back(mesh);
		}
	}

}