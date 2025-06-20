# OvaSim
## 取卵手术操作仿真系统
本系统用于取卵手术的训练模拟，主要有实时超声成像，硬件设备，用户界面三个模块

## 配置流程
### 1 下载代码

https://github.com/Ethan-Hu-min/OvaSim

### 2 下载VisualStudio 2022
https://visualstudio.microsoft.com/zh-hans/
### 3 下载OptiX 8.1.0
https://developer.nvidia.com/designworks/optix/downloads/legacy
### 4 下载CUDA 12.5
https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local
### 5 下载Cmake 3.29.5
https://cmake.org/download/
### 6 下载QT 6.5.3
https://www.qt.io/download-qt-installer-oss
### 7 下载touch驱动
https://support.3dsystems.com/s/article/Haptic-Device-Drivers?language=en_US&_gl=1*1rp5vse*_gcl_au*NzI4OTAxNzIyLjE3NDcwNTA2MzI.*_ga*OTQ0MDQwODQ2LjE3MjMyNzM4NDM.*_ga_9G6ZBH6DZM*czE3NDcwNTA2MzIkbzUkZzEkdDE3NDcwNTA2NjAkajMyJGwwJGgzMDIwMDkxMjE.
### 8 下载OpenHaptics库
https://support.3dsystems.com/s/article/OpenHaptics-for-Windows-Developer-Edition-v35?language=en_US
### 9 下载脚踏开关插件
链接：https://pan.baidu.com/s/1AfxiKo6meJvbJaWBS9w8CQ?pwd=8520 
提取码：8520


## 代码说明

1. 需要针对CMakeLists.txt文件进行修改以满足库管理

2. src文件夹包含cpp源码，GlobalConfig.cpp中可以修改部分参数

3. shaders文件夹中的USdevicePrograms.cu为核心超声渲染代码

4. kernels文件夹包含后处理代码

5. cuda\support\sutil中包含了工具类代码

6. data中包含了一例案例资源，其中器官模型可通过3Dslicer使用MRI影像生成，卵泡模型可以使用Blender中的泡沫膨胀能力生成

## 使用流程

1.点击开始模拟后选择案例与模式

2.进入模拟界面后点击开始进行模拟操作

3.点击结束模拟完成操作

## 文档信息
日期：2025年6月16日   

联系人：hjm18729889855@163.com