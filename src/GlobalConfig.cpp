#include "GlobalConfig.h"

#ifdef CMAKE_ROOT_PATH
	const std::string GlobalConfig::rootPath = std::string(CMAKE_ROOT_PATH);
	const std::string GlobalConfig::dataPath = std::string(CMAKE_ROOT_PATH) + "/data/";
	const std::string GlobalConfig::logPath = std::string(CMAKE_ROOT_PATH) + "/log/";
#endif

const int GlobalConfig::globalAddressVariable = 0;
const int GlobalConfig::frameSizeX = 800;
const int GlobalConfig::frameSizeY = 600;
const int GlobalConfig::transducerNums = 512;
const int GlobalConfig::maxBounceNum = 12;
const int GlobalConfig::SampleNum = 16;
const int GlobalConfig::maxFps = 30;

