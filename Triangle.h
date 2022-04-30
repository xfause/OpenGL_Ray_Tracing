#pragma once
#include "Shape.h"

class Triangle : public Shape
{
public:
    Triangle() {}
    Triangle(vec3 P1, vec3 P2, vec3 P3, vec3 C)
    {
        p1 = P1, p2 = P2, p3 = P3;
        material.normal = normalize(cross(p2 - p1, p3 - p1)); material.color = C;
    }
    vec3 p1, p2, p3;    // vertices
    Material material;  // material

    // 与光线求交
    HitResult intersect(Ray ray)
    {
        HitResult res;

        vec3 S = ray.startPoint;        // ray start point
        vec3 d = ray.direction;         // ray direction
        vec3 N = material.normal;       // normal
        if (dot(N, d) > 0.0f) N = -N;   // get correct normal

        // view horizen with triangle
        if (fabs(dot(N, d)) < 0.00001f) return res;

        // get distance
        float t = (dot(N, p1) - dot(S, N)) / dot(d, N);
        if (t < 0.0005f) return res;    // triangle behind camera

        // collision point
        vec3 P = S + d * t;

        // if hit point in triangle
        vec3 c1 = cross(p2 - p1, P - p1);
        vec3 c2 = cross(p3 - p2, P - p2);
        vec3 c3 = cross(p1 - p3, P - p3);
        vec3 n = material.normal;
        if (dot(c1, n) < 0 || dot(c2, n) < 0 || dot(c3, n) < 0) return res;

        // result
        res.isHit = true;
        res.distance = t;
        res.hitPoint = P;
        res.material = material;
        res.material.normal = N;
        return res;
    };
};