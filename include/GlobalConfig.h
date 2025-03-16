#ifndef GLOBALCONFIG_H
#define GLOBALCONFIG_H
#include <string>

class GlobalConfig {
	public :
		static const int globalAddressVariable;
		static const int frameSizeX;
		static const int frameSizeY;
		static const int transducerNums;
		static const int maxBounceNum;
		static const int OvamNums;
		static const int SampleNum;
		static const int maxFps;
		static const float needleAngle;
		static const float needleDepth;


		static const std::string dataPath;
		static const std::string rootPath;
		static const std::string logPath;
		static const std::string berlinNoisePath;
};

#endif // GLOBALCONFIG_H
