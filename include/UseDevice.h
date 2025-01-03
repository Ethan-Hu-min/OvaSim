#include <HD/hd.h>
//#include <HDU/hduError.h>
#include <HDU/hduVector.h>

struct DeviceInfo {
	hduVector3Dd position;
	hduVector3Dd angles;
};

HDCallbackCode HDCALLBACK GetDeviceInfoCallback(void* data);