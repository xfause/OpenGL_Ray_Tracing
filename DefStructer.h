#pragma once
#include <glm/glm.hpp>
using namespace glm;

// ray
typedef struct Ray {
    vec3 startPoint = vec3(0, 0, 0);
    vec3 direction = vec3(0, 0, 0);
}Ray;

// Material
typedef struct Material
{
    bool isEmissive = false;
    vec3 normal = vec3(0, 0, 0);
    vec3 color = vec3(0, 0, 0);
}Material;

// ray collision
typedef struct HitResult
{
    bool isHit = false;
    double distance = 0.0f;
    vec3 hitPoint = vec3(0, 0, 0);
    Material material;
}HitResult;
