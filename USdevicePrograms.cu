#include <optix_device.h>
#include "USLaunchParams.h"
#include "interaction.h"
#include "random.h"

extern "C" __constant__ USLaunchParams optixLaunchParams;
 

// for this simple example, we have a single ray type
enum { SURFACE_RAY_TYPE = 0, RAY_TYPE_COUNT };

static __forceinline__ __device__
void* unpackPointer(uint32_t i0, uint32_t i1)
{
    const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}

static __forceinline__ __device__
void  packPointer(void* ptr, uint32_t& i0, uint32_t& i1)
{
    const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

template<typename T>
static __forceinline__ __device__ T* getPRD()
{
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T*>(unpackPointer(u0, u1));
}

//------------------------------------------------------------------------------
// closest hit and anyhit programs for radiance-type rays.
//
// Note eventually we will have to create one pair of those for each
// ray type and each geometry type we want to render; but this
// simple example doesn't use any actual geometries yet, so we only
// create a single, dummy, set of them (we do have to have at least
// one group of them to set up the SBT)
//------------------------------------------------------------------------------


extern "C" __global__ void __closesthit__radiance()
{   
    const vec3f ray_orig = optixGetWorldRayOrigin();
    const vec3f ray_dir = optixGetWorldRayDirection();
    float t_hit = optixGetRayTmax();
    const vec3f world_raypos = ray_orig + t_hit * ray_dir;
    const auto& transducer = optixLaunchParams.transducer;
    const TriangleMeshSBTData& sbtData = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();
    const int ray_i = optixGetLaunchIndex().x;
    const vec3i index = sbtData.index[ray_i];
    const int indexModel = sbtData.indexModelSBT;
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;
    const vec3f& A = sbtData.vertex[index.x];
    const vec3f& B = sbtData.vertex[index.y];
    const vec3f& C = sbtData.vertex[index.z];
    vec3f Ng = cross(B - A, C - A);
    vec3f Ns = (sbtData.normal)
        ? ((1.f - u - v) * sbtData.normal[index.x]
            + u * sbtData.normal[index.y]
            + v * sbtData.normal[index.z])
        : Ng;
    const vec3f pos = (1.f - u - v) * A + u * B + v * C;
    //printf("%f %f %f %f %f %f\n", world_raypos.x, world_raypos.y, world_raypos.z, pos.x, pos.y, pos.z);
    //printf("%f %f %f\n", pos.x, pos.y, pos.z);
    //printf("%f %f %f\n", world_raypos.x, world_raypos.y, world_raypos.z);
    uint32_t isectPtr0 = optixGetPayload_0();
    uint32_t isectPtr1 = optixGetPayload_1();
    Interaction* isect = reinterpret_cast<Interaction*>(unpackPointer(isectPtr0, isectPtr1));
    const vec3f lastinter = isect->position;
    //printf("%f %f %f\n", lastinter.x, lastinter.y, lastinter.z);
    vec3f relative_last;
    vec3f relative_now;
    int relative_dis;
    vec3f relative_dir;
    vec3f relative_vec;
    int relative_x;
    int relative_y;
    if ((abs(lastinter.x) > 0.01f || abs(lastinter.y) > 0.01f) && 
        ((isect->indexModelInt == indexModel)||
            ((isect->indexModelInt >= 4)&&(indexModel>=4)))){
        relative_last = lastinter - transducer.position;
        relative_now  = world_raypos - transducer.position; 
        relative_dis = int(length(relative_last - relative_now));
        relative_dir = normalize(relative_now - relative_last);
        for (int pace = 0; pace < relative_dis; pace++) {
            relative_vec = relative_last + pace*1.0f * relative_dir;
            relative_x = dot(relative_vec, transducer.horizontal);
            relative_y = dot(relative_vec, transducer.direction);
            relative_x += optixLaunchParams.frame.size.x / 2;
            if (relative_x >= 0 && relative_x < optixLaunchParams.frame.size.x
                && relative_y >= 0 && relative_y < optixLaunchParams.frame.size.y) {
                unsigned int seed = tea<16>(pace, pace* ray_i);
                float rand_1 = rnd(seed)+ 0.001;
                float rand_2 = rnd(seed)+ 0.001;
                float rand_R = sqrt(-2.0 * log(rand_1));
                float rand_theta = 2.0 * M_PI * rand_2;
                float var_color = 0;
                float mean_color = 0;
                //bladder
                if (indexModel == 1) {
                    var_color = 20;
                    mean_color = 30;
                }
                //uterus
                else if (indexModel == 2) {
                    var_color = 50;
                    mean_color = 150;

                }
                //intestine
                else if (indexModel == 3) {
                    var_color = 100;
                    mean_color = 110;

                }
                //ovary
                else if (isect->indexModelInt >= 4 && indexModel >=4){
                    var_color = 20;
                    mean_color = 100;
                }
                //ovam
                if ((isect->indexModelInt >=6) && (isect->indexModelInt == indexModel))
                {
                    var_color = 20;
                    mean_color = 30;
                }
                float rand_color = ((rand_R * cos(rand_theta)) * var_color - mean_color) * ((pace / relative_dis) / 5 + 0.8);
                int colorvalue = int(abs(rand_color)) % 255;
                //printf("%d", colorvalue);
                //const int randcolor = rand_color * 255;
                uint32_t now_color = 0xff000000
                    | (colorvalue << 0) | (colorvalue << 8) | (colorvalue << 16);
                //0xff505050
                const uint32_t fbIndex = relative_x + relative_y * optixLaunchParams.frame.size.x;
                //printf("%d %d\n", relative_x, relative_y);
                optixLaunchParams.frame.colorBuffer[fbIndex] = now_color;
            }
        }
    }
    isect->indexModelInt = indexModel;
    isect->position = world_raypos;
    isect->geomNormal = Ns;
    //if (ray_i == 100)printf("%f %f %f %f %f %f\n", transducer.direction.x, transducer.direction.y, transducer.direction.z,
    //    transducer.vertical.x, transducer.vertical.y, transducer.vertical.z);
}

extern "C" __global__ void __anyhit__radiance()
{ /*! for this simple example, this will remain empty */


}

//------------------------------------------------------------------------------
// miss program that gets called for any ray that did not have a
// valid intersection
//
// as with the anyhit/closest hit programs, in this example we only
// need to have _some_ dummy function to set up a valid SBT
// ------------------------------------------------------------------------------

extern "C" __global__ void __miss__radiance()
{
    uint32_t isectPtr0 = optixGetPayload_0();
    uint32_t isectPtr1 = optixGetPayload_1();
    Interaction* interaction = reinterpret_cast<Interaction*>(unpackPointer(isectPtr0, isectPtr1));
    interaction->distance = FLT_MAX;
}

//------------------------------------------------------------------------------
// ray gen program - the actual rendering happens in here
//------------------------------------------------------------------------------
extern "C" __global__ void __raygen__renderFrame()
{   
    //printf("texturecolor %d", optixLaunchParams.texture.texture1[100]);
    const int ray_i = optixGetLaunchIndex().x;
    const auto& transducer = optixLaunchParams.transducer;
    int totalnums = transducer.nums;
    float value_hor = (float(ray_i  - totalnums / 2) / totalnums)*2;
    Ray ray_t;
    ray_t.origin = transducer.position;
    const float this_angle = (transducer.angle / transducer.nums) * (ray_i - transducer.nums/2)*PI/180.0;
    ray_t.direction = normalize(cos(this_angle)*transducer.direction+sin(this_angle) * transducer.horizontal);
    vec3f relative_vec;
    int relative_x;
    int relative_y;
    for (int pace = 0; pace < 400; pace++) {
        relative_vec = pace * 1.0f * ray_t.direction;
        relative_x = dot(relative_vec, transducer.horizontal);
        relative_y = dot(relative_vec, transducer.direction);
        relative_x += optixLaunchParams.frame.size.x / 2;
        if (relative_x >= 0 && relative_x < optixLaunchParams.frame.size.x
            && relative_y >= 0 && relative_y < optixLaunchParams.frame.size.y) {
            const uint32_t fbIndex = relative_x + relative_y * optixLaunchParams.frame.size.x;
            unsigned int seed = tea<16>(ray_i, pace);
            float rand_color = rnd(seed);
            const int randcolor = rand_color * 255 * ((400.0 - pace) / pace);
            uint32_t color_value = 0xff000000
                | (randcolor << 0) | (randcolor << 8) | (randcolor << 16);
            optixLaunchParams.frame.colorBuffer[fbIndex] = color_value;
        }
    }
    Interaction isect;  
    isect.distance = 0;
    isect.indexModelInt = 0;
    for (int bounces = 0; bounces < optixLaunchParams.maxBounce; ++bounces) {
        vec3f wi = ray_t.direction;
        uint32_t isectPtr0, isectPtr1;
        packPointer(&isect, isectPtr0, isectPtr1);
        optixTrace(
            optixLaunchParams.traversable,
            ray_t.origin,
            ray_t.direction,
            0,
            1e20f,
            0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            SURFACE_RAY_TYPE,
            RAY_TYPE_COUNT,
            SURFACE_RAY_TYPE,
            isectPtr0, isectPtr1);   
        if (isect.distance == FLT_MAX)break;
        relative_vec = isect.position - transducer.position;
        relative_x = dot(relative_vec, transducer.horizontal);
        relative_y = dot(relative_vec, transducer.direction);
        relative_x += optixLaunchParams.frame.size.x / 2;
        if (relative_x >= 0 && relative_x < optixLaunchParams.frame.size.x
            && relative_y >= 0 && relative_y < optixLaunchParams.frame.size.y) {
            const uint32_t fbIndex = relative_x + relative_y * optixLaunchParams.frame.size.x;
            optixLaunchParams.frame.colorBuffer[fbIndex] = 0xffffffff;
        }          
        ray_t.origin = isect.position + ray_t.direction * 0.001f;
    }

}
