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