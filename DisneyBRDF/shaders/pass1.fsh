#version 330 core

in vec3 pix;
out vec4 fragColor;

uniform uint frameCounter;
uniform int nTriangles;
uniform int nNodes;
uniform int width;
uniform int height;

uniform samplerBuffer triangles;
uniform samplerBuffer nodes;

uniform sampler2D lastFrame;
uniform sampler2D hdrMap;

uniform vec3 eye;
uniform mat4 cameraRotate;

#define PI              3.1415926
#define INF             2147483647.0
#define SIZE_TRIANGLE   12
#define SIZE_BVHNODE    4

struct Triangle {
	vec3 p1,p2,p3;
	vec3 n1,n2,n3;
};

struct BVHNode{
	int left;
	int right;
	int n;
	int index;
	vec3 AA, BB;
};

struct Material{
	vec3 emissive;
    vec3 baseColor;
    float subsurface;
    float metallic;
    float specular;
    float specularTint;
    float roughness;
    float anisotropic;
    float sheen;
    float sheenTint;
    float clearcoat;
    float clearcoatGloss;
    float IOR;
    float transmission;
};

struct Ray{
    vec3 startPosition;
    vec3 direction;
};

struct HitResult{
    bool isHit;
    bool isInside;
    float distance;
    vec3 hitPoint;
    vec3 normal;
    vec3 viewDir;
    Material material;
};

uint seed = uint(
    uint((pix.x * 0.5 + 0.5) * width)  * uint(1973) + 
    uint((pix.y * 0.5 + 0.5) * height) * uint(9277) + 
    uint(frameCounter) * uint(26699)) | uint(1);

uint wang_hash(inout uint seed) {
    seed = uint(seed ^ uint(61)) ^ uint(seed >> uint(16));
    seed *= uint(9);
    seed = seed ^ (seed >> 4);
    seed *= uint(0x27d4eb2d);
    seed = seed ^ (seed >> 15);
    return seed;
}
 
float rand() {
    return float(wang_hash(seed)) / 4294967296.0;
}

// sample on half sphere
vec3 SampleHemisphere() {
    float z = rand();
    float r = max(0, sqrt(1.0 - z*z));
    float phi = 2.0 * PI * rand();
    return vec3(r * cos(phi), r * sin(phi), z);
}

/*
vec3 toNormalHemisphere(vec3 v, vec3 N) {
    vec3 tangent = vec3(0);
    if(N.yz==vec2(0)) tangent = vec3(0, 0, -N.x);
    else if(N.xz==vec2(0)) tangent = vec3(0, 0, N.y);
    else if(N.xy==vec2(0)) tangent = vec3(-N.z, 0, 0);
    else if(abs(N.x)>abs(N.y)) tangent = normalize(vec3(0, N.z, -N.y));
    else tangent = normalize(vec3(-N.z, 0, N.x)); 
    vec3 bitangent = cross(N, tangent);
    return normalize(v.x * tangent + v.y * bitangent + v.z * N);
}
*/

// ?????? v ?????? N ??????????
vec3 toNormalHemisphere(vec3 v, vec3 N) {
    vec3 helper = vec3(1, 0, 0);
    if(abs(N.x)>0.999) helper = vec3(0, 0, 1);
    vec3 tangent = normalize(cross(N, helper));
    vec3 bitangent = normalize(cross(N, tangent));
    return v.x * tangent + v.y * bitangent + v.z * N;
}

void getTangent(vec3 N, inout vec3 tangent, inout vec3 bitangent) {
    /*
    vec3 helper = vec3(0, 0, 1);
    if(abs(N.z)>0.999) helper = vec3(0, -1, 0);
    tangent = normalize(cross(N, helper));
    bitangent = normalize(cross(N, tangent));
    */
    vec3 helper = vec3(1, 0, 0);
    if(abs(N.x)>0.999) helper = vec3(0, 0, 1);
    bitangent = normalize(cross(N, helper));
    tangent = normalize(cross(N, bitangent));
}

// vec3 v to HDR map's texture coordinate uv
vec2 SampleSphericalMap(vec3 v){
    vec2 uv = vec2(atan(v.z, v.x), asin(v.y));
    uv /= vec2(2.0 * PI, PI);
    uv += 0.5;
    uv.y = 1.0 - uv.y;
    return uv;
}

// get HDR environment color
vec3 sampleHdr(vec3 v){
    vec2 uv = SampleSphericalMap(normalize(v));
    vec3 color = texture2D(hdrMap, uv).rgb;
    color = min(color, vec3(10));
    return color;
}

Triangle getTriangle(int i){
    int offset = i * SIZE_TRIANGLE;
    Triangle t;

    // vertex
    t.p1 = texelFetch(triangles, offset + 0).xyz;
    t.p2 = texelFetch(triangles, offset + 1).xyz;
    t.p3 = texelFetch(triangles, offset + 2).xyz;
    // normal
    t.n1 = texelFetch(triangles, offset + 3).xyz;
    t.n2 = texelFetch(triangles, offset + 4).xyz;
    t.n3 = texelFetch(triangles, offset + 5).xyz;

    return t;
}

Material getMaterial(int i){
    Material m;

    int offset = i * SIZE_TRIANGLE;
    vec3 param1 = texelFetch(triangles, offset + 8).xyz;
    vec3 param2 = texelFetch(triangles, offset + 9).xyz;
    vec3 param3 = texelFetch(triangles, offset + 10).xyz;
    vec3 param4 = texelFetch(triangles, offset + 11).xyz;
    
    m.emissive = texelFetch(triangles, offset + 6).xyz;
    m.baseColor = texelFetch(triangles, offset + 7).xyz;
    m.subsurface = param1.x;
    m.metallic = param1.y;
    m.specular = param1.z;
    m.specularTint = param2.x;
    m.roughness = param2.y;
    m.anisotropic = param2.z;
    m.sheen = param3.x;
    m.sheenTint = param3.y;
    m.clearcoat = param3.z;
    m.clearcoatGloss = param4.x;
    m.IOR = param4.y;
    m.transmission = param4.z;

    return m;
}

BVHNode getBVHNode(int i){
    BVHNode node;
    int offset = i * SIZE_BVHNODE;

    ivec3 childs = ivec3(texelFetch(nodes, offset + 0).xyz);
    ivec3 leaveInfo = ivec3(texelFetch(nodes, offset + 1).xyz);
    node.left = int(childs.x);
    node.right = int(childs.y);
    node.n = int(leaveInfo.x);
    node.index = int(leaveInfo.y);

    node.AA = texelFetch(nodes, offset + 2).xyz;
    node.BB = texelFetch(nodes, offset + 3).xyz;

    return node;
}

HitResult hitTriangle(Triangle triangle, Ray ray) {
    HitResult res;
    res.distance = INF;
    res.isHit = false;
    res.isInside = false;

    vec3 p1 = triangle.p1;
    vec3 p2 = triangle.p2;
    vec3 p3 = triangle.p3;

    vec3 S = ray.startPosition;
    vec3 d = ray.direction;
    vec3 N = normalize(cross(p2-p1, p3-p1));

    // inside model
    if (dot(N, d) > 0.0f) {
        N = -N;   
        res.isInside = true;
    }

    if (abs(dot(N, d))<0.00001f) return res;

    float t = (dot(N, p1) - dot(S, N)) / dot(d, N);
    if (t<0.0005f) return res;

    vec3 P = S + d * t;

    // if hit point inside triangle
    vec3 c1 = cross(p2 - p1, P - p1);
    vec3 c2 = cross(p3 - p2, P - p2);
    vec3 c3 = cross(p1 - p3, P - p3);
    bool r1 = (dot(c1, N) > 0 && dot(c2, N) > 0 && dot(c3, N) > 0);
    bool r2 = (dot(c1, N) < 0 && dot(c2, N) < 0 && dot(c3, N) < 0);

    if (r1 || r2) {
        res.isHit = true;
        res.hitPoint = P;
        res.distance = t;
        res.normal = N;
        res.viewDir = d;

        // vertices normal interpolation 
        float alpha = (-(P.x-p2.x)*(p3.y-p2.y) + (P.y-p2.y)*(p3.x-p2.x)) / (-(p1.x-p2.x-0.00005)*(p3.y-p2.y+0.00005) + (p1.y-p2.y+0.00005)*(p3.x-p2.x+0.00005));
        float beta  = (-(P.x-p3.x)*(p1.y-p3.y) + (P.y-p3.y)*(p1.x-p3.x)) / (-(p2.x-p3.x-0.00005)*(p1.y-p3.y+0.00005) + (p2.y-p3.y+0.00005)*(p1.x-p3.x+0.00005));
        float gama  = 1.0 - alpha - beta;
        vec3 Nsmooth = alpha * triangle.n1 + beta * triangle.n2 + gama * triangle.n3;
        Nsmooth = normalize(Nsmooth);
        res.normal = (res.isInside) ? (-Nsmooth) : (Nsmooth);
    }
    return res;
}

float hitAABB(Ray r, vec3 AA, vec3 BB){
    vec3 invdir = 1.0/r.direction;

    vec3 f = (BB - r.startPosition) * invdir;
    vec3 n = (AA - r.startPosition) * invdir;

    vec3 tmax = max(f, n);
    vec3 tmin = min(f, n);

    float t1 = min(tmax.x, min(tmax.y, tmax.z));
    float t0 = max(tmin.x, max(tmin.y, tmin.z));

    return (t1 >= t0) ? ((t0 > 0.0) ? (t0) : (t1)) : (-1);
}

// force
HitResult hitArray(Ray ray, int l, int r) {
    HitResult res;
    res.isHit = false;
    res.distance = INF;
    for(int i=l; i<=r; i++) {
        Triangle triangle = getTriangle(i);
        HitResult r = hitTriangle(triangle, ray);
        if(r.isHit && r.distance<res.distance) {
            res = r;
            res.material = getMaterial(i);
        }
    }
    return res;
}

// BVH
HitResult hitBVH(Ray ray){
    HitResult res;
    res.isHit = false;
    res.distance = INF;

    int stack[256];
    int sp = 0;

    stack[sp++] = 1;
    while (sp>0){
        int top = stack[--sp];
        BVHNode node = getBVHNode(top);

        if (node.n>0){
            int L = node.index;
            int R = node.index + node.n - 1;
            HitResult r = hitArray(ray, L, R);
            if(r.isHit && r.distance<res.distance) res = r;
            continue;
        }

        float d1 = INF;
        float d2 = INF;
        if (node.left > 0) {
            BVHNode leftNode = getBVHNode(node.left);
            d1 = hitAABB(ray, leftNode.AA, leftNode.BB);
        }
        if (node.right > 0) {
            BVHNode rightNode = getBVHNode(node.right);
            d2 = hitAABB(ray, rightNode.AA, rightNode.BB);
        }

        if (d1>0 && d2>0) {
            if(d1<d2) {
                stack[sp++] = node.right;
                stack[sp++] = node.left;
            } else { 
                stack[sp++] = node.left;
                stack[sp++] = node.right;
            }
        } else if (d1>0){
            stack[sp++] = node.left;
        } else if (d2>0){
            stack[sp++] = node.right;
        }
    }
    return res;
}

// =============================== BRDF

float sqr(float x) { 
    return x*x; 
}

float SchlickFresnel(float u) {
    float m = clamp(1-u, 0, 1);
    float m2 = m*m;
    return m2*m2*m; // pow(m,5)
}

float GTR1(float NdotH, float a) {
    if (a >= 1) return 1/PI;
    float a2 = a*a;
    float t = 1 + (a2-1)*NdotH*NdotH;
    return (a2-1) / (PI*log(a2)*t);
}

float GTR2(float NdotH, float a) {
    float a2 = a*a;
    float t = 1 + (a2-1)*NdotH*NdotH;
    return a2 / (PI * t*t);
}

float GTR2_aniso(float NdotH, float HdotX, float HdotY, float ax, float ay) {
    return 1 / (PI * ax*ay * sqr( sqr(HdotX/ax) + sqr(HdotY/ay) + NdotH*NdotH ));
}

float smithG_GGX(float NdotV, float alphaG) {
    float a = alphaG*alphaG;
    float b = NdotV*NdotV;
    return 1 / (NdotV + sqrt(a + b - a*b));
}

float smithG_GGX_aniso(float NdotV, float VdotX, float VdotY, float ax, float ay) {
    return 1 / (NdotV + sqrt( sqr(VdotX*ax) + sqr(VdotY*ay) + sqr(NdotV) ));
}

vec3 BRDF_Evaluate(vec3 V, vec3 N, vec3 L, vec3 X, vec3 Y, in Material material){
    float NdotL = dot(N, L);
    float NdotV = dot(N, V);
    if(NdotL < 0 || NdotV < 0) return vec3(0);

    vec3 H = normalize(L + V);
    float NdotH = dot(N, H);
    float LdotH = dot(L, H);

    vec3 Cdlin = material.baseColor;
    float Cdlum = 0.3 * Cdlin.r + 0.6 * Cdlin.g  + 0.1 * Cdlin.b;
    vec3 Ctint = (Cdlum > 0) ? (Cdlin/Cdlum) : (vec3(1));  
    vec3 Cspec = material.specular * mix(vec3(1), Ctint, material.specularTint); // ????????????????
    vec3 Cspec0 = mix(0.08*Cspec, Cdlin, material.metallic); // 0?? ????????????
    vec3 Csheen = mix(vec3(1), Ctint, material.sheenTint);   // ????????

    // diffuse
    float Fd90 = 0.5 + 2.0 * LdotH * LdotH * material.roughness;
	float FL = SchlickFresnel(NdotL);
	float FV = SchlickFresnel(NdotV);
	float Fd = mix(1.0, Fd90, FL) * mix(1.0, Fd90, FV);

    // subsurface
    float Fss90 = LdotH * LdotH * material.roughness;
    float Fss = mix(1.0, Fss90, FL) * mix(1.0, Fss90, FV);
    float ss = 1.25 * (Fss * (1.0 / (NdotL + NdotV) - 0.5) + 0.5);

    // specular
	//float alpha = material.roughness * material.roughness;
	//float Ds = GTR2(NdotH, alpha);
	//float FH = SchlickFresnel(LdotH);
	//vec3 Fs = mix(Cspec0, vec3(1), FH);
	//float Gs = smithG_GGX(NdotL, material.roughness);
	//Gs *= smithG_GGX(NdotV, material.roughness);
	//vec3 specular = Gs * Fs * Ds;

    // specular
    float aspect = sqrt(1.0 - material.anisotropic * 0.9);
    float ax = max(0.001, sqr(material.roughness)/aspect);
    float ay = max(0.001, sqr(material.roughness)*aspect);
    float Ds = GTR2_aniso(NdotH, dot(H, X), dot(H, Y), ax, ay);
    float FH = SchlickFresnel(LdotH);
    vec3 Fs = mix(Cspec0, vec3(1), FH);
    float Gs;
    Gs  = smithG_GGX_aniso(NdotL, dot(L, X), dot(L, Y), ax, ay);
    Gs *= smithG_GGX_aniso(NdotV, dot(V, X), dot(V, Y), ax, ay);
    vec3 specular = Gs * Fs * Ds;

    // ????
	float Dr = GTR1(NdotH, mix(0.1, 0.001, material.clearcoatGloss));
	float Fr = mix(0.04, 1.0, FH);
	float Gr = smithG_GGX(NdotL, 0.25) * smithG_GGX(NdotV, 0.25);
	vec3 clearcoat = vec3(0.25 * Gr * Fr * Dr * material.clearcoat);

    // sheen
    vec3 Fsheen = FH * material.sheen * Csheen;

    vec3 diffuse = (1.0/PI) * mix(Fd, ss, material.subsurface) * Cdlin + Fsheen;
    return diffuse * (1.0 - material.metallic) + specular + clearcoat;
}


vec3 pathTracing(HitResult hit, int maxBounce){
    vec3 Lo = vec3(0); // final color
    vec3 history = vec3(1);

    for (int bounce = 0; bounce<maxBounce; bounce++){
        vec3 V = -hit.viewDir;
        vec3 N = hit.normal;
        vec3 L = toNormalHemisphere(SampleHemisphere(), hit.normal); // random bounce direction
        float pdf = 1.0 / (2.0 * PI);         // ????????????????????
        float cosine_o = max(0, dot(V, N));   // ????????????????????
        float cosine_i = max(0, dot(L, N));   // ????????????????????
        vec3 tangent, bitangent;
        getTangent(N, tangent, bitangent);
        vec3 f_r = BRDF_Evaluate(V, N, L, tangent, bitangent, hit.material);    // BRDF

        // diffuse, random ray
        Ray randomRay;
        randomRay.startPosition = hit.hitPoint;
        randomRay.direction = L;
        HitResult newHit = hitBVH(randomRay);

        // ??????
        if(!newHit.isHit) {
            vec3 skyColor = sampleHdr(randomRay.direction);
            Lo += history * skyColor * f_r * cosine_i / pdf;
            break;
        }
        
        // hit light add color
        vec3 Le = newHit.material.emissive;
        Lo += history * Le * f_r * cosine_i / pdf;
        
        // add color
        hit = newHit;
        history *= f_r * cosine_i / pdf;
    }

    return Lo;
}


void main(){
    Ray ray;

    ray.startPosition = eye;
    vec2 AA = vec2((rand()-0.5)/float(width), (rand()-0.5)/float(height));
    vec4 dir = cameraRotate * vec4(pix.xy+AA, -1.5, 0.0);
    ray.direction = normalize(dir.xyz);

    // primary hit
    HitResult firstHit = hitBVH(ray);
    // HitResult firstHit = hitArray(ray, 0, nTriangles - 1);
    vec3 color;

    if (!firstHit.isHit){
        color = vec3(0);
        color = sampleHdr(ray.direction);
    } else {
        vec3 Le = firstHit.material.emissive;
        vec3 Li = pathTracing(firstHit, 5);
        color = Le + Li;
    }

    vec3 lastColor = texture2D(lastFrame, pix.xy*0.5 + 0.5).rgb;
    color = mix(lastColor, color, 1.0/float(frameCounter+1));

    gl_FragData[0] = vec4(color, 1.0);
}