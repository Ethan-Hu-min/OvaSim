#include "UseDevice.h"

HDCallbackCode HDCALLBACK GetDeviceInfoCallback(void* data)
{
    DeviceInfo* nowInfo = static_cast<DeviceInfo*>(data);
    hdBeginFrame(hdGetCurrentDevice());

    // Get the position of the device.
    hduVector3Dd position;
    hduVector3Dd angles;
    hdGetDoublev(HD_CURRENT_POSITION, position);
    hdGetDoublev(HD_CURRENT_GIMBAL_ANGLES, angles);
    hdEndFrame(hdGetCurrentDevice());

    // In case of error, terminate the callback.
    HDErrorInfo error;
    if (HD_DEVICE_ERROR(error = hdGetError()))
    {
        //hduPrintError(stderr, &error, "Error detected during main scheduler callback\n");

        //if (hduIsSchedulerError(&error))
        //{
        //    return HD_CALLBACK_DONE;
        //}
    }

    nowInfo->position = position;
    nowInfo->angles[0] = angles[0] * 180.0;
    nowInfo->angles[1] = angles[1] * 180.0;
    nowInfo->angles[2] = angles[2] * 180.0;
    return HD_CALLBACK_CONTINUE;
}



HDCallbackCode HDCALLBACK ForceDeviceCallback(void* data)
{
    DeviceInfo* nowInfo = static_cast<DeviceInfo*>(data);
    //const hduVector3Dd spherePosition(0, -30, -80);
    //const hduVector3Dd sphereAngle(0, 0, 0);
    hduVector3Dd spherePosition = nowInfo->originPos;
    hduVector3Dd sphereAngle = nowInfo->originAngle;

    hdBeginFrame(hdGetCurrentDevice());

    // Get the position of the device.
    hduVector3Dd position;
    hdGetDoublev(HD_CURRENT_POSITION, position);

    hduVector3Dd angles;
    hdGetDoublev(HD_CURRENT_GIMBAL_ANGLES, angles);

    nowInfo->position = position;
    nowInfo->angles[0] = angles[0] * 180.0;
    nowInfo->angles[1] = angles[1] * 180.0;
    nowInfo->angles[2] = angles[2] * 60.0 ;
    // Find the distance between the device and the center of the
    // sphere.
    double distance = (position - spherePosition).magnitude();
    hduVector3Dd forceDirection = (spherePosition - position) / distance;
    hduVector3Dd x;
    if (distance < 10) x = distance * forceDirection;
    else x = 10 * forceDirection;
    hduVector3Dd f = nowInfo -> dampingForce * x;


    hduVector3Dd diffAngles = sphereAngle - angles;
    hduVector3Dd t = nowInfo -> dampingTorque *1000*  diffAngles;

    nowInfo->position_force = f;
    nowInfo->angles_torque = t / 1000.0;

    hdSetDoublev(HD_CURRENT_FORCE, f);
    hdSetDoublev(HD_CURRENT_GIMBAL_TORQUE, t);

    hdEndFrame(hdGetCurrentDevice());

    HDErrorInfo error;
    if (HD_DEVICE_ERROR(error = hdGetError()))
    {
        //hduPrintError(stderr, &error, "Error during main scheduler callback\n");

        //if (hduIsSchedulerError(&error))
        //{
        //    return HD_CALLBACK_DONE;
        //}
    }

    return HD_CALLBACK_CONTINUE;
}

HDCallbackCode HDCALLBACK ForceGLCallback(void* data)
{
    DeviceInfo* nowInfo = static_cast<DeviceInfo*>(data);
    hduVector3Dd spherePosition = nowInfo->originPos;
    hduVector3Dd sphereAngle = nowInfo->originAngle;

    hdBeginFrame(hdGetCurrentDevice());

    // Get the position of the device.
    hduVector3Dd position;
    hdGetDoublev(HD_CURRENT_POSITION, position);

    hduVector3Dd angles;
    hdGetDoublev(HD_CURRENT_GIMBAL_ANGLES, angles);

    nowInfo->position = position;
    nowInfo->angles[0] = angles[0] * 180.0;
    nowInfo->angles[1] = angles[1] * 180.0;
    nowInfo->angles[2] = angles[2] * 60.0;
    // Find the distance between the device and the center of the
    // sphere.
    double distance = (position - spherePosition).magnitude();
    hduVector3Dd forceDirection = (spherePosition - position) / distance;
    hduVector3Dd x;
    if (distance < 30) x = distance * forceDirection;
    else x = 30 * forceDirection;
    hduVector3Dd f = nowInfo->dampingForce * x;
    if (f[2] < 0)f[2] = 0;
   
    

    hduVector3Dd diffAngles = sphereAngle - angles;
    hduVector3Dd t = nowInfo->dampingTorque * 1000 * diffAngles;

    nowInfo->position_force = f;
    nowInfo->angles_torque = t / 1000.0;

    hdSetDoublev(HD_CURRENT_FORCE, f);
    //hdSetDoublev(HD_CURRENT_GIMBAL_TORQUE, t);

    hdEndFrame(hdGetCurrentDevice());

    HDErrorInfo error;
    if (HD_DEVICE_ERROR(error = hdGetError()))
    {
        //hduPrintError(stderr, &error, "Error during main scheduler callback\n");

        //if (hduIsSchedulerError(&error))
        //{
        //    return HD_CALLBACK_DONE;
        //}
    }

    return HD_CALLBACK_CONTINUE;
}