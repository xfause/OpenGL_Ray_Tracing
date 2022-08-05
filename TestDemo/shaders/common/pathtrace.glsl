/*
 * MIT License
 *
 * Copyright(c) 2019-2021 Asif Ali
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

void GetMaterials(inout State state, in Ray r)
{
    int index = state.matID * 7;
    Material mat;

    vec4 param1 = texelFetch(materialsTex, ivec2(index + 0, 0), 0);
    vec4 param2 = texelFetch(materialsTex, ivec2(index + 1, 0), 0);
    vec4 param3 = texelFetch(materialsTex, ivec2(index + 2, 0), 0);
    vec4 param4 = texelFetch(materialsTex, ivec2(index + 3, 0), 0);
    vec4 param5 = texelFetch(materialsTex, ivec2(index + 4, 0), 0);
    vec4 param6 = texelFetch(materialsTex, ivec2(index + 5, 0), 0);
    vec4 param7 = texelFetch(materialsTex, ivec2(index + 6, 0), 0);

    mat.baseColor          = param1.xyz;
                           
    mat.emission           = param2.xyz;
    mat.anisotropic        = param2.w;
                           
    mat.metallic           = param3.x;
    mat.roughness          = max(param3.y, 0.001);
                           
    mat.subsurface         = param3.z;
    mat.specularTint       = param3.w;
                           
    mat.sheen              = param4.x;
    mat.sheenTint          = param4.y;
    mat.clearcoat          = param4.z;
    mat.clearcoatRoughness = mix(0.1, 0.001, param4.w); // Remapping from gloss to roughness

    mat.specTrans          = param5.x;
    mat.ior                = param5.y;
    mat.atDistance         = param5.z;
                           
    mat.extinction         = param6.xyz;
                           
    ivec4 texIDs           = ivec4(param7);

    vec2 texUV = state.texCoord;
    texUV.y = 1.0 - texUV.y;

    // Albedo Map
    if (texIDs.x >= 0)
        mat.baseColor *= pow(texture(textureMapsArrayTex, vec3(texUV, texIDs.x)).rgb, vec3(2.2));

    // Metallic Roughness Map
    if (texIDs.y >= 0)
    {
        vec2 matRgh = texture(textureMapsArrayTex, vec3(texUV, texIDs.y)).rg;
        mat.metallic = matRgh.x;
        mat.roughness = max(matRgh.y * matRgh.y, 0.001);
    }

    // Normal Map
    if (texIDs.z >= 0)
    {
        vec3 texNormal = texture(textureMapsArrayTex, vec3(texUV, texIDs.z)).rgb;
        texNormal = normalize(texNormal * 2.0 - 1.0);

        vec3 origNormal = state.normal;
        state.normal = normalize(state.tangent * texNormal.x + state.bitangent * texNormal.y + state.normal * texNormal.z);
        state.ffnormal = dot(origNormal, r.direction) <= 0.0 ? state.normal : -state.normal;
    }

    // Emission Map
    if (texIDs.w >= 0)
        mat.emission = pow(texture(textureMapsArrayTex, vec3(texUV, texIDs.w)).rgb, vec3(2.2));

    // Commented out the following as anisotropic param is temporarily unused.
    // Calculate anisotropic roughness along the tangent and bitangent directions
    // float aspect = sqrt(1.0 - mat.anisotropic * 0.9);
    // mat.ax = max(0.001, mat.roughness / aspect);
    // mat.ay = max(0.001, mat.roughness * aspect);

    state.mat = mat;
    state.eta = dot(r.direction, state.normal) < 0.0 ? (1.0 / mat.ior) : mat.ior;
}

vec3 DirectLight(in Ray r, in State state)
{
    vec3 Li = vec3(0.0);
    vec3 surfacePos = state.fhp + state.normal * EPS;

    BsdfSampleRec bsdfSampleRec;

    // Environment Light
#ifdef ENVMAP
#ifndef CONSTANT_BG
    {
        vec3 color;
        vec4 dirPdf = SampleEnvMap(color);
        vec3 lightDir = dirPdf.xyz;
        float lightPdf = dirPdf.w;

        Ray shadowRay = Ray(surfacePos, lightDir);
        bool inShadow = AnyHit(shadowRay, INF - EPS);

        if (!inShadow)
        {
            bsdfSampleRec.f = DisneyEval(state, -r.direction, state.ffnormal, lightDir, bsdfSampleRec.pdf);

            if (bsdfSampleRec.pdf > 0.0)
            {
                float misWeight = PowerHeuristic(lightPdf, bsdfSampleRec.pdf);
                if (misWeight > 0.0)
                    Li += misWeight * bsdfSampleRec.f * color / lightPdf;
            }
        }
    }
#endif
#endif

    // Analytic Lights 
#ifdef LIGHTS
    {
        LightSampleRec lightSampleRec;
        Light light;

        //Pick a light to sample
        int index = int(rand() * float(numOfLights)) * 5;

        // Fetch light Data
        vec3 position = texelFetch(lightsTex, ivec2(index + 0, 0), 0).xyz;
        vec3 emission = texelFetch(lightsTex, ivec2(index + 1, 0), 0).xyz;
        vec3 u        = texelFetch(lightsTex, ivec2(index + 2, 0), 0).xyz; // u vector for rect
        vec3 v        = texelFetch(lightsTex, ivec2(index + 3, 0), 0).xyz; // v vector for rect
        vec3 params   = texelFetch(lightsTex, ivec2(index + 4, 0), 0).xyz;
        float radius  = params.x;
        float area    = params.y;
        float type    = params.z; // 0->Rect, 1->Sphere, 2->Distant

        light = Light(position, emission, u, v, radius, area, type);
        SampleOneLight(light, surfacePos, lightSampleRec);

        if (dot(lightSampleRec.direction, lightSampleRec.normal) < 0.0) // Required for quad lights with single sided emission
        {
            Ray shadowRay = Ray(surfacePos, lightSampleRec.direction);
            bool inShadow = AnyHit(shadowRay, lightSampleRec.dist - EPS);

            if (!inShadow)
            {
                bsdfSampleRec.f = DisneyEval(state, -r.direction, state.ffnormal, lightSampleRec.direction, bsdfSampleRec.pdf);

                float weight = 1.0;
                if(light.area > 0.0) // No MIS for distant light
                    weight = PowerHeuristic(lightSampleRec.pdf, bsdfSampleRec.pdf);

                if (bsdfSampleRec.pdf > 0.0)
                    Li += weight * bsdfSampleRec.f * lightSampleRec.emission / lightSampleRec.pdf;
            }
        }
    }
#endif

    return Li;
}

vec3 PathTrace(Ray r)
{
    vec3 radiance = vec3(0.0);
    vec3 throughput = vec3(1.0);
    State state;
    LightSampleRec lightSampleRec;
    BsdfSampleRec bsdfSampleRec;
    vec3 absorption = vec3(0.0);
    
    for (int depth = 0; depth < maxDepth; depth++)
    {
        state.depth = depth;
        bool hit = ClosestHit(r, state, lightSampleRec);

        if (!hit)
        {
#ifdef CONSTANT_BG
            radiance += bgColor * throughput;
#else
#ifdef ENVMAP
            {
                float misWeight = 1.0f;
                vec2 uv = vec2((PI + atan(r.direction.z, r.direction.x)) * INV_TWO_PI, acos(r.direction.y) * INV_PI);

                if (depth > 0)
                {
                    // TODO: Fix NaNs when using certain HDRs
                    float lightPdf = EnvMapPdf(r);
                    misWeight = PowerHeuristic(bsdfSampleRec.pdf, lightPdf);
                }
                radiance += misWeight * texture(hdrTex, uv).xyz * throughput * hdrMultiplier;
            }
#endif
#endif
            return radiance;
        }

        GetMaterials(state, r);

        // Reset absorption when ray is going out of surface
        if (dot(state.normal, state.ffnormal) > 0.0)
            absorption = vec3(0.0);

        radiance += state.mat.emission * throughput;

#ifdef LIGHTS
        if (state.isEmitter)
        {
            radiance += EmitterSample(r, state, lightSampleRec, bsdfSampleRec) * throughput;
            break;
        }
#endif

        // Add absoption
        throughput *= exp(-absorption * state.hitDist);

        radiance += DirectLight(r, state) * throughput;

        bsdfSampleRec.f = DisneySample(state, -r.direction, state.ffnormal, bsdfSampleRec.L, bsdfSampleRec.pdf);

        // Set absorption only if the ray is currently inside the object.
        if (dot(state.ffnormal, bsdfSampleRec.L) < 0.0)
            absorption = -log(state.mat.extinction) / state.mat.atDistance;

        if (bsdfSampleRec.pdf > 0.0)
            throughput *= bsdfSampleRec.f / bsdfSampleRec.pdf;
        else
            break;

#ifdef RR
        // Russian roulette
        if (depth >= RR_DEPTH)
        {
            float q = min(max(throughput.x, max(throughput.y, throughput.z)) + 0.001, 0.95);
            if (rand() > q)
                break;
            throughput /= q;
        }
#endif

        r.direction = bsdfSampleRec.L;
        r.origin = state.fhp + r.direction * EPS;
    }

    return radiance;
}