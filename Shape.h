#pragma once
#include "DefStructer.h"

class Shape
{
public:
    Shape() {}
    virtual HitResult intersect(Ray ray) { return HitResult(); }
};