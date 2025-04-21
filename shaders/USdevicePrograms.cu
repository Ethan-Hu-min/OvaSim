#include <optix_device.h>

#include <include/interaction.h>
#include <cuda/random.h>
#include <include/USLaunchParams.h>

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

__constant__ struct {
    float impetance_in[6]; // bladder, uterus, uterusin, intestine, ovary, ovam
    float impetance_out[6];
    float thickness[6];
}  impetanceParams = {
    {1.49, 1.62, 1.49, 1.99, 1.62, 1.49},
    {1.99, 1.99, 1.62, 1.99, 1.99, 1.49},
    { 5.0,  2.0,  2.0,  2.0,  2.0,  2.0},
};

static __forceinline__ __device__
vec3f cosDiffuse(const vec3f& in, int no_sample) {
    vec3f axis = vec3f(0.0f, 0.0f, 1.0f);
    vec3f tangent = normalize(cross(in, axis));
    vec3f bitangent = cross(in, tangent);
    unsigned int seed1 = tea<16>(int(in.x * 68329), int(no_sample * 52398));
    unsigned int seed2 = tea<16>(int(in.z * 23725), int(no_sample * 72934));
    float theta = rnd(seed1) / 2.0;
    float phi = rnd(seed2) / 2.0;

    return normalize(in + theta * tangent + phi * bitangent);
}

static __forceinline__ __device__
vec3f USreflect(const vec3f& incident, const vec3f& normal, int no_sample) {
    vec3f _in = incident - 2.0f * dot(incident, normal) * normal;
    return cosDiffuse(_in, no_sample);
}

static __forceinline__ __device__
bool USrefract(const vec3f& incident, const vec3f& normal,
    float n1, float n2, vec3f& refracted_dir, int no_sample) {
    float eta = n1 / n2;
    float cos_theta = fminf(dot(-incident, normal), 1.0f);
    float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

    if (eta * sin_theta > 1.0f) {
        //全内反射
        return false;
    }
    vec3f r_out_perp = eta * (incident + cos_theta * normal);
    vec3f r_out_parallel = -sqrtf(fabs(1.0f - dot(r_out_perp, r_out_perp))) * normal;
    vec3f _in = r_out_perp + r_out_parallel;
    refracted_dir = cosDiffuse(_in, no_sample);
    return true;
}

static __forceinline__ __device__
uint32_t calculate_color(const float attenuation) {
    int intensity = static_cast<int>(attenuation * 255);
    intensity = clamp(intensity, 0, 255);
    return 0xff000000 | (intensity << 16) | (intensity << 8) | intensity;
}


__device__
void render_border(const vec3f& touch, const vec3f& dir, const int materialID, float color_intensity) {
    unsigned int nowseed = tea<16>(int(touch.x * 10000), materialID);
    float offset = rnd(nowseed) * 2.0 * impetanceParams.thickness[materialID];
    vec3f now_pos;
    vec3f rela_pos;
    int rela_x;
    int rela_y;
    now_pos = touch + offset * dir;
    rela_pos = now_pos - optixLaunchParams.transducer.position;
    rela_x = int(dot(rela_pos, optixLaunchParams.transducer.horizontal));
    rela_y = int(dot(rela_pos, optixLaunchParams.transducer.direction));
    rela_x += optixLaunchParams.frame.size.x / 2;
    uint32_t fbIndex = rela_x + rela_y * optixLaunchParams.frame.size.x;
    if (rela_x >= 0 && rela_x < optixLaunchParams.frame.size.x && rela_y >= 0 && rela_y < optixLaunchParams.frame.size.y)
        atomicAdd(&optixLaunchParams.frame.intensityBuffer[fbIndex], color_intensity);
}

__device__
void render_fragment(const vec3f& start, const vec3f& dir, const int distance, int textureID, float attenuation, float intensity) {
    vec3f now_pos;
    vec3f rela_pos;
    int rela_x;
    int rela_y;
    int rela_z;
    uint32_t fbIndex;
    uint8_t* now_texture = nullptr;
    unsigned int nowseed = tea<16>(distance, textureID);
    int rand_offset = int(rnd(nowseed) * 256);

    //if (textureID == 4)printf("I!: %f \n",intensity);
    if (textureID == 0) now_texture = optixLaunchParams.textures.bgTexture;
    else if (textureID == 1) now_texture = optixLaunchParams.textures.bladderTexture;
    else if (textureID == 2) now_texture = optixLaunchParams.textures.uterusTexture;
    else if (textureID == 3) now_texture = optixLaunchParams.textures.uterusinTexture;
    else if (textureID == 4) now_texture = optixLaunchParams.textures.intestineTexture;
    else if (textureID == 5) now_texture = optixLaunchParams.textures.ovaryTexture;
    else if (textureID == 6) now_texture = optixLaunchParams.textures.ovamTexture;

    for (volatile int path = 0; path < distance; path++) {
        //printf("path: %d \n", path);
        now_pos = start + float(path) * dir;
        rela_pos = now_pos - optixLaunchParams.transducer.position;
        rela_x = int(dot(rela_pos, optixLaunchParams.transducer.horizontal));
        rela_y = int(dot(rela_pos, optixLaunchParams.transducer.direction));

        rela_x += optixLaunchParams.frame.size.x / 2;
        if (rela_x >= 0 && rela_x < optixLaunchParams.frame.size.x && rela_y >= 0 && rela_y < optixLaunchParams.frame.size.y) {
            fbIndex = rela_x + rela_y * optixLaunchParams.frame.size.x;
            uint8_t color_texture = now_texture[fbIndex + rand_offset];

            float attenuation_dis = (distance - path * attenuation) / float(distance);
            float color_intensity = (static_cast<float>(color_texture) / 255.0) * attenuation_dis * intensity;
            //if(textureID == 1)printf("bladder!: %p %p %p\n", now_texture, optixLaunchParams.textures.bgTexture,  optixLaunchParams.textures.bladderTexture);

            atomicAdd(&optixLaunchParams.frame.intensityBuffer[fbIndex], color_intensity);
        }
    }
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
    const int ray_sample = optixGetLaunchIndex().y;
    if (ray_i + ray_sample != 0) {
        const vec3i index = sbtData.index[ray_i];
        const int indexModel = sbtData.indexModelSBT;
        const int indexMaterial = sbtData.materialID;
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
        bool is_inside = (dot(ray_dir, Ns) > 0.0f);
        vec3f shading_normal = is_inside ? -Ns : Ns; // 确保法线指向射线来源的外侧
        float Z1 = is_inside ? impetanceParams.impetance_in[indexMaterial] : impetanceParams.impetance_out[indexMaterial];
        float Z2 = is_inside ? impetanceParams.impetance_out[indexMaterial] : impetanceParams.impetance_in[indexMaterial];

        // 计算反射系数
        float R = fabsf((Z2 - Z1) / (Z2 + Z1));
        
        //生成随机数决定反射或折射
        unsigned int seed = tea<16>(optixGetLaunchIndex().x, optixGetRayTmax());
        float rand_val = rnd(seed);

        uint32_t isectPtr0 = optixGetPayload_0();
        uint32_t isectPtr1 = optixGetPayload_1();
        Interaction* isect = reinterpret_cast<Interaction*>(unpackPointer(isectPtr0, isectPtr1));

        //计算新方向
        vec3f new_dir;
        if (rand_val < R) {
            // 反射
            new_dir = USreflect(ray_dir, shading_normal, ray_sample);
           // isect->intensity *= R;
        }
        else {
            // 折射
            vec3f refracted_dir;
            bool has_refracted = USrefract(ray_dir, shading_normal, Z1, Z2, refracted_dir, ray_sample);
            new_dir = has_refracted ? refracted_dir : USreflect(ray_dir, shading_normal, ray_sample);
            //isect->intensity *= (1.0f - R);
        }
        new_dir -= dot(new_dir, optixLaunchParams.transducer.vertical) * optixLaunchParams.transducer.vertical;


        if (isect->intensity < 0.01 / float(optixLaunchParams.numSamples))isect->is_stop = true;
        isect->next_dir = new_dir;
        isect->is_inside = !is_inside;
        int lastModel = isect->indexModelInt;
        int nowModel = indexModel;
        float now_attenuation = isect->intensity;
        const vec3f lastinter = isect->position;
        int render_textureID = 0;

        render_border(world_raypos, shading_normal, indexMaterial, now_attenuation + 0.1);

        if ((abs(lastinter.x) > 0.01f || abs(lastinter.y) > 0.01f)) {
            if (nowModel == lastModel) {
                render_textureID = indexMaterial;
                //if (nowModel == 0)render_textureID = 1;
                //else if (nowModel == 1)render_textureID = 2;
                //else if (nowModel == 2)render_textureID = 3;
                //else if (nowModel == 3)render_textureID = 4;
                //else if (nowModel == 4 || nowModel == 5)render_textureID = 5;
                //else render_textureID = 6;
            }
            else {
                if (nowModel > 6 || lastModel > 6)render_textureID = 5; //ovary
                else if (nowModel == 2 || lastModel == 2)render_textureID = 2; //uerusin
                else if (nowModel == 3 || lastModel == 3)render_textureID = 3;
                else render_textureID = 0; // bg
            }

            //if (ray_i == 100)render_fragment(ray_orig, ray_dir, int(length(t_hit * ray_dir)), 0, 0.1, now_attenuation);
            render_fragment(ray_orig, ray_dir, int(length(t_hit * ray_dir)), render_textureID, 0.1, now_attenuation);
        }
        isect->intensity *= 0.9;
        isect->indexModelInt = indexModel;
        isect->position = world_raypos;
        isect->geomNormal = Ns;
    }
    else
    {
        uint32_t isectPtr0 = optixGetPayload_0();
        uint32_t isectPtr1 = optixGetPayload_1();
        Interaction* isect = reinterpret_cast<Interaction*>(unpackPointer(isectPtr0, isectPtr1));
        isect->indexModelInt = sbtData.indexModelSBT;
        isect->position = world_raypos;
    }
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
    Interaction* isect = reinterpret_cast<Interaction*>(unpackPointer(isectPtr0, isectPtr1));
    isect->is_stop = true;
    const float now_attenuation = isect->intensity;
    const vec3f ray_orig = optixGetWorldRayOrigin();
    const vec3f ray_dir = optixGetWorldRayDirection();
    const int ray_i = optixGetLaunchIndex().x;
    //if(ray_i == 100)render_fragment(ray_orig, ray_dir, 400, 0, 0, now_attenuation);
    render_fragment(ray_orig, ray_dir, 600, 0,  0.2, 1.0);
}

//------------------------------------------------------------------------------
// ray gen program - the actual rendering happens in here
//------------------------------------------------------------------------------
extern "C" __global__ void __raygen__renderFrame()
{
    //printf("texturecolor %d", optixLaunchParams.texture.texture1[100]);
    const int ray_i = optixGetLaunchIndex().x;
    const int no_sample = optixGetLaunchIndex().y;
    const auto& transducer = optixLaunchParams.transducer;
    const auto& needle = optixLaunchParams.needle;
    int totalnums = transducer.nums;
    float value_hor = (float(ray_i - totalnums / 2) / totalnums) * 2;
    Ray ray_t;
    if ((ray_i + no_sample) != 0) {

        ray_t.origin = transducer.position;
        const float this_angle = (transducer.angle / transducer.nums) * (ray_i - transducer.nums / 2) * PI / 180.0;
        ray_t.direction = normalize(cos(this_angle) * transducer.direction + sin(this_angle) * transducer.horizontal);
        //ray_t.origin = transducer.position + float((ray_i - totalnums / 2.0) * 0.4) * transducer.horizontal;
        //ray_t.direction = transducer.direction;

        //render_fragment(ray_t.origin, ray_t.direction, 400, 0, 0.9);
        //迭代求交
        Interaction isect;
        isect.is_stop = false;
        isect.indexModelInt = -1;
        isect.intensity = 1.0;
        isect.position = transducer.position;
        for (int bounces = 0; bounces < optixLaunchParams.maxBounce; ++bounces) {
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
            if (isect.is_stop)break;
            ray_t.direction = isect.next_dir;
            ray_t.origin = isect.position + ray_t.direction * 0.0001f;
        }

    }
    else
    {
        ray_t.origin = transducer.position;
        const float this_angle = (needle.relaAngle / 180.0) * PI;
        ray_t.direction = normalize(cos(this_angle) * transducer.direction + sin(this_angle) * transducer.horizontal);
        Interaction isect;
        isect.is_stop = false;
        isect.indexModelInt = -1;
        for (int collide_model = 0; collide_model < optixLaunchParams.maxBounce; collide_model++) {
            needle.collide_models_id[collide_model] = 0;
            needle.collide_models_pos[collide_model] = vec3f(0.0, 0.0, 0.0);

        }
        for (int bounces = 0; bounces < optixLaunchParams.maxBounce; ++bounces) {
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
            if (isect.is_stop) {
                break;
            }
            needle.collide_models_id[bounces] = isect.indexModelInt;
            needle.collide_models_pos[bounces] = isect.position;
            ray_t.origin = isect.position + ray_t.direction * 0.0001f;
        }
    }
}
