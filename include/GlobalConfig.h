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
		static const int SampleNum;
		static const int maxFps;
		static const std::string dataPath;
		static const std::string rootPath;
		static const std::string logPath;
};

#endif // GLOBALCONFIG_H
