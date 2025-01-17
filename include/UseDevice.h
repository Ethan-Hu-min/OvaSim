#ifndef  USEDEVICE_H
#define USEDEVICE_H


#include <HD/hd.h>
//#include <HDU/hduError.h>
#include <HDU/hduVector.h>

struct DeviceInfo {

    HDdouble dampingForce;
    HDdouble dampingTorque;

    hduVector3Dd originPos;
    hduVector3Dd originAngle;

    hduVector3Dd position;
    hduVector3Dd angles;
    hduVector3Dd position_force;
    hduVector3Dd angles_torque;
};

HDCallbackCode HDCALLBACK GetDeviceInfoCallback(void* data);
HDCallbackCode HDCALLBACK ForceDeviceCallback(void* data);
HDCallbackCode HDCALLBACK ForceGLCallback(void* data);

#endif