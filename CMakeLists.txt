cmake_minimum_required(VERSION 3.7 FATAL_ERROR)
project(OvaSim LANGUAGES CXX C) 

message("now CMake：${CMAKE_VERSION}")
include(qt.cmake)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(OH_SDK_BASE "C:/OpenHaptics/Developer/3.5.0")
set(Platform "x64")


find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Core SerialPort)

qt_standard_project_setup()

if( POLICY CMP0072 )
  # FindOpenGL prefers GLVND by default when available
  cmake_policy(SET CMP0072 NEW)
endif()

if( POLICY CMP0074 )
  # find_package uses <PackageName>_ROOT variables.
  cmake_policy(SET CMP0074 NEW)
endif()

set(LOCAL_3RDPARTY "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty")
message("LOCAL_3RDPARTY = " "${LOCAL_3RDPARTY}")

# Add paths to our CMake code to the module path, so they can be found automatically by
# CMake.
set(CMAKE_MODULE_PATH
  "${CMAKE_SOURCE_DIR}/CMake"
  ${CMAKE_MODULE_PATH}
  )
 message("CMAKE_MODULE_PATH = " "${CMAKE_MODULE_PATH}")


add_definitions(-DCMAKE_ROOT_PATH="${CMAKE_CURRENT_SOURCE_DIR}")

# Set the default build to Release.  Note this doesn't do anything for the VS
# default build target which defaults to Debug when you first start it.
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

 message("CMAKE_BUILD_TYPE = " "${CMAKE_BUILD_TYPE}")

# Tells CMake to build all the libraries as shared libraries by default.  This can be
# overrided by individual libraries later.
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

# add openHaptics
 include_directories(
	${OH_SDK_BASE}/include
	${OH_SDK_BASE}/utilities/include 
	${OH_SDK_BASE}/QuickHaptics/header
)

link_directories(
	${OH_SDK_BASE}/lib/${Platform}/${CMAKE_BUILD_TYPE}
	${OH_SDK_BASE}/utilities/lib/${Platform}/${CMAKE_BUILD_TYPE}
	${OH_SDK_BASE}/QuickHaptics/lib/${Platform}/${CMAKE_BUILD_TYPE}
)
# add QT
find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS        
        Core
        Gui
        Widgets
        OpenGLWidgets
        Serialport
)

# Include all CMake Macros.
include(Macros)
# Determine information about the compiler
include (CompilerInfo)
# Check for specific machine/compiler options.
include (ConfigCompilerFlags)

# Turn off the warning that NVCC issues when generating PTX from our CUDA samples.  This
# is a custom extension to the FindCUDA code distributed by CMake.
OPTION(CUDA_REMOVE_GLOBAL_MEMORY_SPACE_WARNING "Suppress the \"Advisory: Cannot tell what pointer points to, assuming global memory space\" warning nvcc makes." ON)

# For Xcode 5, gcc is actually clang, so we have to tell CUDA to treat the compiler as
# clang, so that it doesn't mistake it for something else.
if(USING_CLANG_C)
  set(CUDA_HOST_COMPILER "clang" CACHE FILEPATH "Host side compiler used by NVCC")
endif()

# CUDA 8 is broken for generating dependencies during configure
option(CUDA_GENERATE_DEPENDENCIES_DURING_CONFIGURE "Generate dependencies during configure time instead of only during build time." OFF)

# Passing the --use-local-env option to NVCC can dramatically speed up CUDA compilation
if(WIN32)
  option(CUDA_USE_LOCAL_ENV "Pass the '--use-local-env' option to NVCC; only applies to initial configure" ON)
endif()

# Find at least a 5.0 version of CUDA.
find_package(CUDA 5.0 REQUIRED)

# If NVRTC was enabled/disabled, reset OPTIXIR/PTX to default
set( CUDA_NVRTC_ENABLED OFF CACHE BOOL "Use NVRTC to compile OPTIXIR/PTX at run-time instead of NVCC at build-time" )
if( NOT ( NOT ( NOT CUDA_NVRTC_ENABLED ) ) EQUAL ( NOT ( NOT CUDA_NVRTC_ENABLED_INTERNAL ) ) )
  message( STATUS "Resetting OPTIXIR/PTX support" )
  unset( SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT CACHE )
endif()
set( CUDA_NVRTC_ENABLED_INTERNAL ${CUDA_NVRTC_ENABLED} CACHE INTERNAL "Previous configured value (NVRTC)" FORCE )

if( CUDA_NVRTC_ENABLED )
  if( CUDA_VERSION VERSION_LESS 12.0 )
    if( SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT )
      message( SEND_ERROR "CUDA_NVRTC_ENABLED is not compatible with SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT in CUDA versions less than 12.0" )
    else()
      option( SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT "Enable support for generating OptiX-IR targetted input files" OFF )
    endif()
  else()
    option( SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT "Enable support for generating OptiX-IR targetted input files" ON )
  endif()
else()
  if( CUDA_VERSION VERSION_LESS 11.7 )
    if( SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT )
      message( SEND_ERROR "SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT is not supported in CUDA versions less than 11.7" )
    else()
      option( SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT "Enable support for generating OptiX-IR targetted input files" OFF )
    endif()
  else()
    option( SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT "Enable support for generating OptiX-IR targetted input files" ON )
  endif()
endif()

# This code looks funny, but CMake doesn't have an equality operator for boolean types
# (only integer and string). By doing NOT NOT VAL, you can force the original value into 0
# or 1 and allow the EQUAL operator to function correctly.
if( NOT ( NOT ( NOT SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT ) ) EQUAL ( NOT ( NOT SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT_INTERNAL ) ) )
  message( STATUS "Unsetting values associated with OptiX code generation" )
  # This allows us to reset dependent options if you change it.
  unset( SAMPLES_INPUT_GENERATE_OPTIXIR CACHE )
  unset( SAMPLES_INPUT_GENERATE_PTX CACHE )
  unset( GENERATE_DEBUG_DEVICE_CODE CACHE )
endif()
set(SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT_INTERNAL ${SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT} CACHE INTERNAL "Previous configured value (OPTIXIR)" FORCE)

if( SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT )
  option( SAMPLES_INPUT_GENERATE_OPTIXIR "Generate Optix-IR OptiX shaders" ON )
  option( SAMPLES_INPUT_GENERATE_PTX "Generate PTX OptiX shaders" OFF )
else()
  option( SAMPLES_INPUT_GENERATE_OPTIXIR "Generate Optix-IR OptiX shaders" OFF )
  option( SAMPLES_INPUT_GENERATE_PTX "Generate PTX OptiX shaders" ON )
endif()

# Determine if we are going to use the static CRT on windows.
if(WIN32)
  option(RELEASE_USE_STATIC_CRT "Build using the static CRT library" ON)
endif()

# Helper for fixing compiler flags
function(replace_flag var old_flag new_flag)
  string(REPLACE "${old_flag}" "${new_flag}" ${var} ${${var}})
  set(${var} "${${var}}" CACHE STRING "Default compiler flags" FORCE)
endfunction()

function(replace_flags old_flag new_flag)
  foreach(build "" _DEBUG _MINSIZEREL _RELEASE _RELWITHDEBINFO)
    replace_flag(CMAKE_C_FLAGS${build}   "${old_flag}" "${new_flag}")
    replace_flag(CMAKE_CXX_FLAGS${build} "${old_flag}" "${new_flag}")
  endforeach()
endfunction()

if(WIN32)
  if(RELEASE_USE_STATIC_CRT)
    replace_flags("/MD" "/MT")
  else()
    replace_flags("/MT" "/MD")
  endif()
endif(WIN32)

# Present the CUDA_64_BIT_DEVICE_CODE on the default set of options.
mark_as_advanced(CLEAR CUDA_64_BIT_DEVICE_CODE)

set(CUDA_MIN_SM_TARGET sm_50 CACHE STRING "Minimum CUDA SM architecture to use for compilation.")

function(optix_add_cuda_flag_config config flag)
  string(TOUPPER "${config}" config)
  list(FIND CUDA_NVCC_FLAGS${config} ${flag} index)
  if(index EQUAL -1)
    list(APPEND CUDA_NVCC_FLAGS${config} ${flag})
    set(CUDA_NVCC_FLAGS${config} ${CUDA_NVCC_FLAGS${config}} CACHE STRING ${CUDA_NVCC_FLAGS_DESCRIPTION} FORCE)
  endif()
endfunction()

function(optix_add_cuda_flag flag)
  optix_add_cuda_flag_config( "" ${flag} )
endfunction()

# Add some useful default arguments to the NVCC and NVRTC flags.  This is an example of
# how we use PASSED_FIRST_CONFIGURE.  Once you have configured, this variable is TRUE
# and following block of code will not be executed leaving you free to edit the values
# as much as you wish from the GUI or from ccmake.
if( NOT PASSED_FIRST_CONFIGURE )
  set(CUDA_NVCC_FLAGS_DESCRIPTION "Semi-colon delimit multiple arguments.")
  string(REPLACE "sm_" "compute_" CUDA_MIN_SM_COMPUTE_TARGET ${CUDA_MIN_SM_TARGET})

  list(FIND CUDA_NVCC_FLAGS "-arch" index)
  if(index EQUAL -1)
    list(APPEND CUDA_NVCC_FLAGS -arch ${CUDA_MIN_SM_TARGET})
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE STRING "Semi-colon delimit multiple arguments." FORCE)
  endif()

  optix_add_cuda_flag("--use_fast_math")
  optix_add_cuda_flag("-lineinfo")

  # CMAKE_CONFIGURATION_TYPES is only defined for multi-config build systems like
  # MSVC and Ninja, but we need to generate flags for each configuration
  # regardless.
  if( DEFINED CMAKE_CONFIGURATION_TYPES )
    set( OPTIX_CONFIGURATION_TYPES ${CMAKE_CONFIGURATION_TYPES} )
  else()
    set( OPTIX_CONFIGURATION_TYPES "Debug" "Release" "RelWithDebInfo" "MinSizeRel" )
  endif()

  foreach( config ${OPTIX_CONFIGURATION_TYPES} )
    if( ${config} STREQUAL "Debug" )
      optix_add_cuda_flag_config( _${config} "-G" )
      optix_add_cuda_flag_config( _${config} "-O0" )
    endif()
  endforeach()

  if( CUDA_VERSION VERSION_LESS "3.0" )
    optix_add_cuda_flag("--keep")
  endif()

  # Some CUDA 11.x toolkits erroneously complain about sm_50 being deprecated
  if(CUDA_VERSION VERSION_GREATER "11.0")
    optix_add_cuda_flag("-Wno-deprecated-gpu-targets")
  endif()

  if(CUDA_USE_LOCAL_ENV)
    optix_add_cuda_flag("--use-local-env")
  endif()

  if(CMAKE_CXX_STANDARD EQUAL 11)
    set(SAMPLES_NVRTC_CXX "-std=c++11")
  else()
    set(SAMPLES_NVRTC_CXX "")
  endif()

  if( NOT DEFINED CMAKE_CONFIGURATION_TYPES )
    if( NOT CMAKE_BUILD_TYPE STREQUAL CMAKE_BUILD_TYPE_PREVIOUS )
      message( STATUS "Resetting CUDA_NVRTC_FLAGS" )
      unset( CUDA_NVRTC_FLAGS CACHE )
    endif()
    set( CMAKE_BUILD_TYPE_PREVIOUS ${CMAKE_BUILD_TYPE} CACHE INTERNAL "Previous configured value (CMAKE_BUILD_TYPE)" FORCE )

    set( configs "Debug" "Release" "RelWithDebInfo" "MinSizeRel" )
    foreach( config ${configs} )
      if( ${config} STREQUAL "Debug" )
        set( SAMPLES_NVRTC_DEBUG "-G" )
      else()
        set( SAMPLES_NVRTC_DEBUG "-lineinfo" )
      endif()

      string( TOUPPER ${config} config_upper )
      set( CUDA_NVRTC_FLAGS_${config_upper} ${SAMPLES_NVRTC_CXX} -arch ${CUDA_MIN_SM_COMPUTE_TARGET} ${SAMPLES_NVRTC_DEBUG} -use_fast_math -default-device -rdc true -D__x86_64 CACHE STRING "List of NVRTC options just for the samples" FORCE )

      if( ${config} STREQUAL ${CMAKE_BUILD_TYPE} )
        set( CUDA_NVRTC_FLAGS ${CUDA_NVRTC_FLAGS_${config_upper}} CACHE STRING "List of NVRTC options just for the samples" )
      endif()
    endforeach()
  else()
    set( CUDA_NVRTC_FLAGS ${SAMPLES_NVRTC_CXX} -arch ${CUDA_MIN_SM_COMPUTE_TARGET} -lineinfo -use_fast_math -default-device -rdc true -D__x86_64 CACHE STRING "List of NVRTC options just for the samples" FORCE )
    set( CUDA_NVRTC_FLAGS_DEBUG ${SAMPLES_NVRTC_CXX} -arch ${CUDA_MIN_SM_COMPUTE_TARGET} -G -use_fast_math -default-device -rdc true -D__x86_64 CACHE STRING "List of NVRTC options just for the samples" FORCE )
  endif()
endif()

mark_as_advanced(CUDA_NVRTC_FLAGS)

# This passes a preprocessor definition to cl.exe when processing CUDA code.
if(USING_WINDOWS_CL)
  list(APPEND CUDA_NVCC_FLAGS --compiler-options /D_USE_MATH_DEFINES)
endif()

# Put all the runtime stuff in the same directory.  By default, CMake puts each targets'
# output into their own directory.  We want all the targets to be put in the same
# directory, and we can do this by setting these variables.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# Create a flag for mac which will allow apps to add the local cuda toolkit
# install path to the app's rpath.
if( APPLE )
  set( CUDA_TOOLKIT_RPATH_FLAG "-Wl,-rpath,${CUDA_TOOLKIT_ROOT_DIR}/lib" )
endif()

# Locate the NVRT distribution.  Search the SDK first, then look in the system.
set(OptiX_INSTALL_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.1.0" CACHE PATH "Path to OptiX installed location.")

# Search for the OptiX libraries and include files.
find_package(OptiX REQUIRED)

# Add the path to the OptiX headers to our include paths.
include_directories(
  "${OptiX_INCLUDE}"
  "${CMAKE_CURRENT_SOURCE_DIR}/cuda"
  )

# Select whether to use NVRTC or NVCC to generate PTX
if( NOT SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT AND SAMPLES_INPUT_GENERATE_OPTIXIR )
  message( SEND_ERROR "Must enable SAMPLES_INPUT_ENABLE_OPTIXIR_SUPPORT to enable SAMPLES_INPUT_GENERATE_OPTIXIR" )
endif()


##################################################################
# SUtil compilation

set(SAMPLES_PTX_DIR "${CMAKE_BINARY_DIR}/lib/ptx")
set(SAMPLES_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

set(CUDA_GENERATED_OUTPUT_DIR ${SAMPLES_PTX_DIR})

if( WIN32 )
  string(REPLACE "/" "\\\\" SAMPLES_PTX_DIR ${SAMPLES_PTX_DIR})
else( WIN32 )
  if( USING_GNU_C AND NOT APPLE )
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DM_PI=3.14159265358979323846" )
  endif()
endif( WIN32 )

set(SAMPLES_CUDA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cuda")

set(SAMPLES_SUPPORT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/support")

# NVRTC include paths relative to the sample path
set(SAMPLES_RELATIVE_INCLUDE_DIRS "\\
  \"cuda\", \\
  \"sutil\", \\
  \"shaders\",\\
  \"kernels\",\\
  \".\", ")

# NVRTC absolute include paths to the headers used to build the samples
set(SAMPLES_ABSOLUTE_INCLUDE_DIRS "\\
  \"${OptiX_INCLUDE}\", \\
  \"${CUDA_INCLUDE_DIRS}\", ")

# Build a null-terminated option list for NVRTC
set( config_suffixes "_RELEASE" "_DEBUG" )
foreach( config_suffix ${config_suffixes} )
  # CMake doesn't allow empty strings in lists, so use a dummy suffix
  if( ${config_suffix} STREQUAL "_RELEASE" )
    set( config_suffix "" )
  endif()
  set(CUDA_NVRTC_OPTIONS${config_suffix})
  foreach(flag ${CUDA_NVRTC_FLAGS${config_suffix}})
    set(CUDA_NVRTC_OPTIONS${config_suffix} "${CUDA_NVRTC_OPTIONS${config_suffix}} \\\n  \"${flag}\",")
  endforeach()
  set(CUDA_NVRTC_OPTIONS${config_suffix} "${CUDA_NVRTC_OPTIONS${config_suffix}}")
endforeach()

configure_file(sampleConfig.h.in sampleConfig.h @ONLY)

# Path to sutil.h that all the samples need
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}
                     "${CMAKE_BINARY_DIR}/include"
                     ${CMAKE_CURRENT_BINARY_DIR}
                     ${CUDA_INCLUDE_DIRS}
                     )

set(SAMPLES_CUDA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cuda)

# Helper macro to generate PTX from the CUDA files in sutil.
macro(OPTIX_sutil_compile_to_optix_input generated_files)
  if( NOT CUDA_NVRTC_ENABLED )
    if( SAMPLES_INPUT_GENERATE_OPTIXIR )
      message("sutil OPTIXIR")
      CUDA_WRAP_SRCS( sutil OPTIXIR generated_files2 ${ARGN} )
      list(APPEND ${generated_files} ${generated_files2})
    endif()
    if( SAMPLES_INPUT_GENERATE_PTX )
      message("sutil PTX")
      CUDA_WRAP_SRCS( sutil PTX generated_files3 ${ARGN} )
      list(APPEND ${generated_files} ${generated_files3})
    endif()
  endif()
  message("${generated_files} = ${${generated_files}}")
endmacro()

# Helper macro to generate PTX from the CUDA files in shaders.
macro(OPTIX_shader_compile_to_optix_input generated_files)
  if( NOT CUDA_NVRTC_ENABLED )
    if( SAMPLES_INPUT_GENERATE_OPTIXIR )
      message("shader OPTIXIR")
      CUDA_WRAP_SRCS( ${PROJECT_NAME} OPTIXIR generated_files2 ${ARGN} )
      list(APPEND ${generated_files} ${generated_files2})
    endif()
    if( SAMPLES_INPUT_GENERATE_PTX )
      message("shader PTX")
      CUDA_WRAP_SRCS( ${PROJECT_NAME} PTX generated_files3 ${ARGN} )
      list(APPEND ${generated_files} ${generated_files3})
    endif()
  endif()
  message("${generated_files} = ${${generated_files}}")
endmacro()


# These calls will group PTX and CUDA files into their own directories in the Visual
# Studio projects.
macro(OPTIX_add_source_groups)
  if( NOT CUDA_NVRTC_ENABLED )
    if( SAMPLES_INPUT_GENERATE_PTX )
      source_group("PTX Files"  REGULAR_EXPRESSION ".+\\.ptx$")
    endif()
    if( SAMPLES_INPUT_GENERATE_OPTIXIR )
      source_group("OptixIR Files"  REGULAR_EXPRESSION ".+\\.optixir$")
    endif()
  endif()
  source_group("CUDA Files" REGULAR_EXPRESSION ".+\\.cu$")
endmacro()


#########################################################
#  List of samples found in subdirectories.
#
# If you wish to start your own sample, you can copy one of the sample's directories.
# Just make sure you rename all the occurances of the sample's name in the C code as well
# and the CMakeLists.txt file.

# Our sutil library.  The rules to build it are found in the subdirectory.
add_subdirectory(sutil)
# Third-party support libraries.
add_subdirectory(support)
#################################################################


file(GLOB HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/include/*.ui")
file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB SHADERS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.cu")
file(GLOB KERNELS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cu" "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.h"  )


set_source_files_properties(
${CMAKE_CURRENT_SOURCE_DIR}/kernels/ChangeVertices_cuda.cu
${CMAKE_CURRENT_SOURCE_DIR}/kernels/ImageProcess_cuda.cu
${CMAKE_CURRENT_SOURCE_DIR}/kernels/CreateTexture_cuda.cu
PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ
)

get_source_file_property(format1 ${CMAKE_CURRENT_SOURCE_DIR}/kernels/ImageProcess_cuda.cu CUDA_SOURCE_PROPERTY_FORMAT)
message("ImageProcess_cuda.cu format: ${format1}")
get_source_file_property(format2 ${CMAKE_CURRENT_SOURCE_DIR}/kernels/ChangeVertices_cuda.cu CUDA_SOURCE_PROPERTY_FORMAT)
message("ChangeVertices_cuda.cu format: ${format2}")

set(ARGN      
    ${SOURCES}
     ${HEADERS}
     ${SHADERS}
     ${KERNELS})

OPTIX_add_source_groups()

CUDA_GET_SOURCES_AND_OPTIONS(source_files cmake_options options ${ARGN})
message("source_files: ${source_files}")

set(cu_obj_source_files)
set(cu_optix_source_files)
message("obj_input_files: ${source_files}")
foreach(file ${source_files})
    get_source_file_property(_cuda_source_format ${file} CUDA_SOURCE_PROPERTY_FORMAT)
    if(${_cuda_source_format} MATCHES "OBJ")
        list(APPEND cu_obj_source_files ${file})
        message("!!!!!!!!obj_source_files: ${file}")
    else()
        list(APPEND cu_optix_source_files ${file})
    endif()
endforeach()

CUDA_WRAP_SRCS( ${PROJECT_NAME} OBJ genegrated_files ${cu_obj_source_files}  ${cmake_options}  OPTIONS ${options} )
message("obj_output_files = ${genegrated_files}")

CUDA_WRAP_SRCS(${PROJECT_NAME} OPTIXIR genegrated_files2 ${cu_optix_source_files}  ${cmake_options}  OPTIONS ${options} )
list(APPEND generated_files ${genegrated_files2})



#OPTIX_shader_compile_to_optix_input(shader_ptx_file ${SHADERS})

qt_add_executable(OvaSim
    MANUAL_FINALIZATION
      ${source_files}
     ${genegrated_files}
     ${cmake_options}
)



set_target_properties( ${PROJECT_NAME} PROPERTIES
COMPILE_DEFINITIONS
"OPTIX_SAMPLE_NAME_DEFINE=${PROJECT_NAME};OPTIX_SAMPLE_DIR_DEFINE=${PROJECT_NAME}" )

if( UNIX AND NOT APPLE )
# Force using RPATH instead of RUNPATH on Debian
target_link_libraries( ${PROJECT_NAME} "-Wl,--disable-new-dtags" )
endif()

if(USING_GNU_CXX)
target_link_libraries( ${PROJECT_NAME} m ) # Explicitly link against math library (C samples don't do that by default)
endif()

target_link_libraries( ${PROJECT_NAME} ${CUDA_LIBRARY})

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/data)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/kernels)

include(GNUInstallDirs)
install(TARGETS OvaSim
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

set_target_properties(${PROJECT_NAME}
    PROPERTIES
        WIN32_EXECUTABLE TRUE
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        Qt::Core
        Qt::Gui
        Qt::Widgets
        Qt::OpenGLWidgets
        Qt::SerialPort
        hd.lib 
	    hl.lib
	    hdu.lib
	    hlu.lib
        ${GLFW_LIB_NAME}
        sutil_7_sdk
        imgui
        ${CUDA_LIBRARIES}

)
