#include "Scene.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <GlobalConfig.h>
Scene::Scene() {}

// 读取JSON文件内容并返回QByteArray
QByteArray readJsonFile(const QString& filePath) {
	QFile file(filePath);
	if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
		qDebug() << "can't open file: " << filePath;
		return QByteArray();
	}
	QByteArray jsonData = file.readAll();
	file.close();
	return jsonData;
}

void	Scene::setExampleName(std::string name) {
	exampleName = name;
}


void Scene::parseConfig(std::string config_dir) {
	std::string nowExample = GlobalConfig::dataPath + exampleName;
	std::string nowScene = nowExample + "/" + config_dir;
	QByteArray jsonData = readJsonFile(QString::fromStdString(nowScene));
	QJsonDocument jsonDoc = QJsonDocument::fromJson(jsonData);

	if (jsonDoc.isObject()) {
		QJsonObject jsonObj = jsonDoc.object();
		QJsonValue matsValue = jsonObj.value("materials");
		if (matsValue.isArray()) {
			QJsonArray matsArray = matsValue.toArray();
			for (const QJsonValue& matValue : matsArray) {
				if (matValue.isObject()) {
					QJsonObject matObj = matValue.toObject();
					Material mat;
					mat.mat_name = std::string(matObj.value("name").toString().toLocal8Bit());
					mat.impedance = matObj.value("impedance").toDouble();
					mat.attenuation = matObj.value("attenuation").toDouble();
					mat.mu0 = matObj.value("mu0").toDouble();
					mat.mu1 = matObj.value("mu1").toDouble();
					mat.sigma = matObj.value("sigma").toDouble();
					mat.specularity = matObj.value("specularity").toDouble();
					materials[mat.mat_name] = mat;
				}
			}
		}
		else {
			qDebug() << "materials must be an array";
			throw std::runtime_error("materials must be an array");
		}

		QJsonValue meshesValue = jsonObj.value("meshes");
		if (meshesValue.isArray()) {
			QJsonArray meshesArray = meshesValue.toArray();
			for (const QJsonValue& meshValue : meshesArray) {
				if (meshValue.isObject()) {
					QJsonObject meshObj = meshValue.toObject();
					QJsonArray deltasArray = meshObj.value("deltas").toArray();
					QJsonArray scalingsArray = meshObj.value("scaling").toArray();
					std::string file = nowExample + "/" + std::string(meshObj.value("file").toString().toLocal8Bit());
					std::string material = std::string(meshObj.value("material").toString().toLocal8Bit());
					std::string outsideMaterial = std::string(meshObj.value("outsideMaterial").toString().toLocal8Bit());
					std::vector<double> deltas;
					std::vector<double> scalings;
					for (int i = 0; i < deltasArray.size(); ++i) {
						deltas.push_back(deltasArray[i].toDouble());
					}
					for (int i = 0; i < scalingsArray.size(); ++i) {
						scalings.push_back(scalingsArray[i].toDouble());
					}
					models.emplace_back(Model(file, deltas, scalings, materials.at(material), materials.at(outsideMaterial)));
				}
			}
		}
		else {
			qDebug() << "meshes must be an array";
			throw std::runtime_error("meshes must be an array");
		}
		for (int i = 0; i < models.size(); i++) {
			models[i].indexModel = i;
		}
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

void Scene::setTransducer(vec3f pos, vec3f dir, vec3f ver, int nums, int angle, float depth, float width) {
	transducer.t_position = pos;
	transducer.t_direction = normalize(dir);
	transducer.t_vertical = normalize(ver);
	transducer.t_nums = nums;
	transducer.t_angle = angle;
	transducer.t_depth = depth;
	transducer.t_width = width;
}