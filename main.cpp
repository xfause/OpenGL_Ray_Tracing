#include <iostream>
#include <vector>
#include <random>
#include <stdlib.h>
// math
#include <glm/glm.hpp>
// pic output
#include "svpng.inc"
// multi-thread
#include <omp.h>

#include "DefStructer.h"
#include "Shape.h"
#include "Triangle.h"

using namespace glm;
using namespace std;

// output pic resolution
const int WIDTH = 256;
const int HEIGHT = 256;

// camera
const double SCREEN_Z = 1.1;        // view plane Z
const vec3 EYE = vec3(0, 0, 4.0);   // position

// color
const vec3 RED(1, 0.5, 0.5);
const vec3 GREEN(0.5, 1, 0.5);
const vec3 BLUE(0.5, 0.5, 1);
const vec3 YELLOW(1.0, 1.0, 0.1);
const vec3 CYAN(0.1, 1.0, 1.0);
const vec3 MAGENTA(1.0, 0.1, 1.0);
const vec3 GRAY(0.5, 0.5, 0.5);
const vec3 WHITE(1, 1, 1);

// sampling counts
// 4096
const int SAMPLE = 128;
// sampling lightness
const double BRIGHTNESS = (2.0f * 3.1415926f) * (1.0f / double(SAMPLE));

// output pixel to pic
void imshow(double* SRC, string name)
{

	unsigned char* image = new unsigned char[WIDTH * HEIGHT * 3];// image buffer
	unsigned char* p = image;
	double* S = SRC;    // 源数据

	FILE* fp;
	fopen_s(&fp, name.c_str(), "wb");

	for (int i = 0; i < HEIGHT; i++)
	{
		for (int j = 0; j < WIDTH; j++)
		{
			*p++ = (unsigned char)clamp(pow(*S++, 1.0f / 2.2f) * 255, 0.0, 255.0);  // R 通道
			*p++ = (unsigned char)clamp(pow(*S++, 1.0f / 2.2f) * 255, 0.0, 255.0);  // G 通道
			*p++ = (unsigned char)clamp(pow(*S++, 1.0f / 2.2f) * 255, 0.0, 255.0);  // B 通道
		}
	}

	svpng(fp, WIDTH, HEIGHT, image, 0);
}

// closest hit result
HitResult shoot(vector<Shape*>& shapes, Ray ray)
{
	HitResult res, r;
	res.distance = 1145141919.810f; // inf

	// traverse all shape
	for (auto& shape : shapes)
	{
		r = shape->intersect(ray);
		if (r.isHit && r.distance < res.distance) res = r;
	}

	return res;
}

// 0-1 random
std::uniform_real_distribution<> dis(0.0, 1.0);
random_device rd;
mt19937 gen(rd());
double randf()
{
	return dis(gen);
}

// random vector in unit sphere
vec3 randomVec3()
{
	vec3 d;
	do
	{
		d = 2.0f * vec3(randf(), randf(), randf()) - vec3(1, 1, 1);
	} while (dot(d, d) > 1.0);
	return normalize(d);
}

// random vector in normal half sphere
vec3 randomDirection(vec3 n)
{
	//vec3 d;
	//do
	//{
	//    d = randomVec3();
	//} while (dot(d, n) < 0.0f);
	//return d;

	return normalize(randomVec3() + n);
}

// path tracing
vec3 directLightPathTracing(vector<Shape*>& shapes, Ray ray)
{
	HitResult res = shoot(shapes, ray);

	if (!res.isHit) return vec3(0); // not hit

	// if emissive return color
	if (res.material.isEmissive) return res.material.color;

	// return directly
	return vec3(0);
}

vec3 indirectLightPathTracing(vector<Shape*>& shapes, Ray ray, int depth) {
	if (depth > 8) {
		return vec3(0);
	}
	HitResult res = shoot(shapes, ray);
	if (!res.isHit) {
		return vec3(0);
	}
	if (res.material.isEmissive) {
		return res.material.color;
	}

	// P probability end
	double r = randf();
	float P = 0.8;
	if (r > P) return vec3(0);

	Ray randomRay;
	randomRay.startPoint = res.hitPoint;
	randomRay.direction = randomDirection(res.material.normal);
	float cosine = dot(-ray.direction, res.material.normal);

	vec3 srcColor = res.material.color;
	vec3 ptColor = indirectLightPathTracing(shapes, randomRay, depth + 1) * cosine;
	vec3 color = ptColor * srcColor;    // 和原颜色混合

	return color / P;
}



void OutputTest() {
	double* image = new double[WIDTH * HEIGHT * 3];
	memset(image, 0.0, sizeof(double) * WIDTH * HEIGHT * 3);
	double* p = image;
	for (int i = 0; i < HEIGHT; i++)
	{
		for (int j = 0; j < WIDTH; j++)
		{
			// 像素坐标转投影平面坐标
			double x = 2.0 * double(j) / double(WIDTH) - 1.0;
			double y = 2.0 * double(HEIGHT - i) / double(HEIGHT) - 1.0;

			vec3 coord = vec3(x, y, SCREEN_Z);          // 计算投影平面坐标
			vec3 direction = normalize(coord - EYE);    // 计算光线投射方向

			vec3 color = direction;

			*p = color.x; p++;  // R 通道
			*p = color.y; p++;  // G 通道
			*p = color.z; p++;  // B 通道
		}
	}

	imshow(image, "test.png");
}

void TriangleTest() {
	vector<Shape*> shapes;
	shapes.push_back(new Triangle(vec3(-0.5, -0.5, 0), vec3(0.0, 0.5, 0), vec3(0.5, -0.5, 0), RED));

	double* image = new double[WIDTH * HEIGHT * 3];
	memset(image, 0.0, sizeof(double) * WIDTH * HEIGHT * 3);
	double* p = image;
	for (int i = 0; i < HEIGHT; i++)
	{
		for (int j = 0; j < WIDTH; j++)
		{
			// pixels position to projection position
			double x = 2.0 * double(j) / double(WIDTH) - 1.0;
			double y = 2.0 * double(HEIGHT - i) / double(HEIGHT) - 1.0;

			vec3 coord = vec3(x, y, SCREEN_Z);          // get projection plane position
			vec3 direction = normalize(coord - EYE);    // get ray direction

			// 生成光线
			Ray ray;
			ray.startPoint = coord;
			ray.direction = direction;

			// 找交点并输出交点的颜色
			HitResult res = shoot(shapes, ray);
			vec3 color = res.material.color;

			*p = color.x; p++;  // R
			*p = color.y; p++;  // G
			*p = color.z; p++;  // B
		}
	}

	imshow(image, "triangle.png");
}

void DirectLight() {
	vector<Shape*> shapes;

	// light
	Triangle l1 = Triangle(vec3(0.4, 0.99, 0.4), vec3(-0.4, 0.99, -0.4), vec3(-0.4, 0.99, 0.4), WHITE);
	Triangle l2 = Triangle(vec3(0.4, 0.99, 0.4), vec3(0.4, 0.99, -0.4), vec3(-0.4, 0.99, -0.4), WHITE);
	l1.material.isEmissive = true;
	l2.material.isEmissive = true;
	shapes.push_back(&l1);
	shapes.push_back(&l2);

	// object
	shapes.push_back(new Triangle(vec3(-0.9, 0.4, -0.8), vec3(-0.9, -0.9, -0.8), vec3(-0.3, -0.9, 0.5), WHITE));
	shapes.push_back(new Triangle(vec3(0.9, 0.4, -0.8), vec3(0.9, -0.9, -0.8), vec3(0.3, -0.9, 0.5), WHITE));

	// cornell box
	// bottom
	shapes.push_back(new Triangle(vec3(1, -1, 1), vec3(-1, -1, -1), vec3(-1, -1, 1), WHITE));
	shapes.push_back(new Triangle(vec3(1, -1, 1), vec3(1, -1, -1), vec3(-1, -1, -1), WHITE));
	// top
	shapes.push_back(new Triangle(vec3(1, 1, 1), vec3(-1, 1, 1), vec3(-1, 1, -1), WHITE));
	shapes.push_back(new Triangle(vec3(1, 1, 1), vec3(-1, 1, -1), vec3(1, 1, -1), WHITE));
	// back
	shapes.push_back(new Triangle(vec3(1, -1, -1), vec3(-1, 1, -1), vec3(-1, -1, -1), CYAN));
	shapes.push_back(new Triangle(vec3(1, -1, -1), vec3(1, 1, -1), vec3(-1, 1, -1), CYAN));
	// left
	shapes.push_back(new Triangle(vec3(-1, -1, -1), vec3(-1, 1, 1), vec3(-1, -1, 1), BLUE));
	shapes.push_back(new Triangle(vec3(-1, -1, -1), vec3(-1, 1, -1), vec3(-1, 1, 1), BLUE));
	// right
	shapes.push_back(new Triangle(vec3(1, 1, 1), vec3(1, -1, -1), vec3(1, -1, 1), RED));
	shapes.push_back(new Triangle(vec3(1, -1, -1), vec3(1, 1, 1), vec3(1, 1, -1), RED));

	double* image = new double[WIDTH * HEIGHT * 3];
	memset(image, 0.0, sizeof(double) * WIDTH * HEIGHT * 3);
	omp_set_num_threads(50); // 线程个数
#pragma omp parallel for
	for (int k = 0; k < SAMPLE; k++) {
		double* p = image;
		for (int i = 0; i < HEIGHT; i++) {
			for (int j = 0; j < WIDTH; j++) {
				double x = 2.0 * double(j) / double(WIDTH) - 1.0;
				double y = 2.0 * double(HEIGHT - i) / double(HEIGHT) - 1.0;

				vec3 coord = vec3(x, y, SCREEN_Z);
				vec3 direction = normalize(coord - EYE);

				Ray ray;
				ray.startPoint = coord;
				ray.direction = direction;

				HitResult res = shoot(shapes, ray);
				vec3 color = vec3(0, 0, 0);

				if (res.isHit) {
					if (res.material.isEmissive) {
						color = res.material.color;
					}
					else {
						Ray randomRay;
						randomRay.startPoint = res.hitPoint;
						randomRay.direction = randomDirection(res.material.normal);

						vec3 srcColor = res.material.color;
						vec3 ptColor = directLightPathTracing(shapes, randomRay);
						color = srcColor * ptColor;
						color *= BRIGHTNESS;
					}
				}
				*p += color.x; p++;
				*p += color.y; p++;
				*p += color.z; p++;
			}
		}
	}
	imshow(image, "direct_light.png");
}

void IndirectLight() {
	vector<Shape*> shapes;

	// light
	Triangle l1 = Triangle(vec3(0.4, 0.99, 0.4), vec3(-0.4, 0.99, -0.4), vec3(-0.4, 0.99, 0.4), WHITE);
	Triangle l2 = Triangle(vec3(0.4, 0.99, 0.4), vec3(0.4, 0.99, -0.4), vec3(-0.4, 0.99, -0.4), WHITE);
	l1.material.isEmissive = true;
	l2.material.isEmissive = true;
	shapes.push_back(&l1);
	shapes.push_back(&l2);

	// object
	shapes.push_back(new Triangle(vec3(-0.9, 0.4, -0.8), vec3(-0.9, -0.9, -0.8), vec3(-0.3, -0.9, 0.5), WHITE));
	shapes.push_back(new Triangle(vec3(0.9, 0.4, -0.8), vec3(0.9, -0.9, -0.8), vec3(0.3, -0.9, 0.5), WHITE));

	// cornell box
	// bottom
	shapes.push_back(new Triangle(vec3(1, -1, 1), vec3(-1, -1, -1), vec3(-1, -1, 1), WHITE));
	shapes.push_back(new Triangle(vec3(1, -1, 1), vec3(1, -1, -1), vec3(-1, -1, -1), WHITE));
	// top
	shapes.push_back(new Triangle(vec3(1, 1, 1), vec3(-1, 1, 1), vec3(-1, 1, -1), WHITE));
	shapes.push_back(new Triangle(vec3(1, 1, 1), vec3(-1, 1, -1), vec3(1, 1, -1), WHITE));
	// back
	shapes.push_back(new Triangle(vec3(1, -1, -1), vec3(-1, 1, -1), vec3(-1, -1, -1), CYAN));
	shapes.push_back(new Triangle(vec3(1, -1, -1), vec3(1, 1, -1), vec3(-1, 1, -1), CYAN));
	// left
	shapes.push_back(new Triangle(vec3(-1, -1, -1), vec3(-1, 1, 1), vec3(-1, -1, 1), BLUE));
	shapes.push_back(new Triangle(vec3(-1, -1, -1), vec3(-1, 1, -1), vec3(-1, 1, 1), BLUE));
	// right
	shapes.push_back(new Triangle(vec3(1, 1, 1), vec3(1, -1, -1), vec3(1, -1, 1), RED));
	shapes.push_back(new Triangle(vec3(1, -1, -1), vec3(1, 1, 1), vec3(1, 1, -1), RED));

	double* image = new double[WIDTH * HEIGHT * 3];
	memset(image, 0.0, sizeof(double) * WIDTH * HEIGHT * 3);
	omp_set_num_threads(50);
	#pragma omp parallel for
	for (int k = 0; k < SAMPLE; k++) {
		double* p = image;
		for (int i = 0; i < HEIGHT; i++) {
			for (int j = 0; j < WIDTH; j++) {
				double x = 2.0 * double(j) / double(WIDTH) - 1.0;
				double y = 2.0 * double(HEIGHT - i) / double(HEIGHT) - 1.0;

				vec3 coord = vec3(x, y, SCREEN_Z);
				vec3 direction = normalize(coord - EYE);

				Ray ray;
				ray.startPoint = coord;
				ray.direction = direction;

				HitResult res = shoot(shapes, ray);
				vec3 color = vec3(0, 0, 0);

				if (res.isHit) {
					if (res.material.isEmissive) {
						color = res.material.color;
					}
					else {
						Ray randomRay;
						randomRay.startPoint = res.hitPoint;
						randomRay.direction = randomDirection(res.material.normal);

						vec3 srcColor = res.material.color;
						vec3 ptColor = indirectLightPathTracing(shapes, randomRay, 0);
						color = srcColor * ptColor;
						color *= BRIGHTNESS;
					}
				}
				*p += color.x; p++;
				*p += color.y; p++;
				*p += color.z; p++;
			}
		}
	}
	imshow(image, "indirect_light.png");
}

int main() {

	// output test
	//OutputTest();

	// triangle test
	//TriangleTest();

	// direct light render test
	//DirectLight();

	// direct + indirect light render test
	//IndirectLight();

	return 0;
}