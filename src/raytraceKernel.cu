// CIS565 CUDA Raytracer: A parallel raytracer for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania
// This file includes code from:
//       Rob Farber for CUDA-GL interop, from CUDA Supercomputing For The Masses: http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
//       Peter Kutz and Yining Karl Li's GPU Pathtracer: http://gpupathtracer.blogspot.com/
//       Yining Karl Li's TAKUA Render, a massively parallel pathtracing renderer: http://www.yiningkarlli.com

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "sceneStructs.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include "raytraceKernel.h"
#include "intersections.h"
#include "interactions.h"
#include <vector>
/*

#include "utils/cuPrintf.cuh"
#include "utils/cuPrintf.cu"*/

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 

//LOOK: This function demonstrates how to use thrust for random number generation on the GPU!
//Function that generates static.
__host__ __device__ glm::vec3 generateRandomNumberFromThread(glm::vec2 resolution, float time, int x, int y){
  int index = x + (y * resolution.x);
   
  thrust::default_random_engine rng(hash(index*time));
  thrust::uniform_real_distribution<float> u01(0,1);

  return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

//TODO: IMPLEMENT THIS FUNCTION
//Function that does the initial raycast from the camera
__host__ __device__ ray raycastFromCameraKernel(glm::vec2 resolution, float time, int x, int y, glm::vec3 eye, glm::vec3 view, glm::vec3 up, glm::vec2 fov){
    glm::vec3 M, H, V, A, B, P;
	ray r;
    M = eye + view;
	A = glm::cross(view, up);
	B = glm::cross(A, view);
	V = B * (float)(tan(fov[1] / 180.0 * PI) * (glm::length(view) / glm::length(B)));
	H = A * (float)(tan(fov[0] / 180.0 * PI) * (glm::length(view) / glm::length(A)));

	P = M + H * (2.0f * x/(resolution[0]-1.0f) - 1.0f) + V * (2.0f * (1.0f-y/(resolution[1]-1.0f))-1.0f);
	r.origin = eye;
	r.direction = glm::normalize(P - eye);
     
    return r;
}

//Kernel that blacks out a given image buffer
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      image[index] = glm::vec3(0,0,0);
    }
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;
      color.x = image[index].x*255.0;
      color.y = image[index].y*255.0;
      color.z = image[index].z*255.0;

      if(color.x>255){
        color.x = 255;
      }

      if(color.y>255){
        color.y = 255;
      }

      if(color.z>255){
        color.z = 255;
      }
      
      // Each thread writes one pixel location in the texture (textel)
      PBOpos[index].w = 0;
      PBOpos[index].x = color.x;
      PBOpos[index].y = color.y;
      PBOpos[index].z = color.z;
  }
}

__host__ __device__ int rayIntersect(const ray& r, staticGeom* geoms, int numberOfGeoms, glm::vec3& intersectionPoint, glm::vec3& intersectionNormal, material* materials/*, int numberOfMaterials*/){
	float distance = 8000.0f;
	float tempDistance = -1.0f;
	glm::vec3 tempIntersctionPoint(0.0f), tempIntersectionNormal(0.0f);
	int intersIndex = -1;
//	printf("%f\n", r.origin.z);
//	if(abs(r.origin.z + 5.0f) < 0.1f)
//	printf("%f %f %f\n", r.direction.x, r.direction.y, r.direction.z);
	for(int i = 0; i < numberOfGeoms; i++)
	{
		tempDistance = -1.0f;
//		if(materials[geoms[i].materialid].emittance > 0.0f) continue; // do not test intesection with light source
		if(geoms[i].type == GEOMTYPE::SPHERE){
			tempDistance = sphereIntersectionTest(geoms[i], r, tempIntersctionPoint, tempIntersectionNormal);
		}
		else if(geoms[i].type == GEOMTYPE::CUBE){
			tempDistance = boxIntersectionTest(geoms[i], r, tempIntersctionPoint, tempIntersectionNormal);
		}
		else 
			continue;
		if(!(abs(tempDistance + 1.0f) < EPSILON))
		{
			if(tempDistance < distance) {
				intersIndex = i;
				distance = tempDistance;
				intersectionPoint = tempIntersctionPoint;
				intersectionNormal = tempIntersectionNormal;
			}
		}
	}
//	printf("%d", intersIndex);
	return intersIndex;
}
__host__ __device__ bool ShadowRayUnblocked(glm::vec3 surfacePoint,glm::vec3 lightPosition, staticGeom* geoms, int numberOfGeoms, material* materials) // return true if unblocked
{
//	printf("in shadow test\n");
	glm::vec3 rayDir = glm::normalize(lightPosition - surfacePoint);
//	ray shadowRay(surfacePoint + 0.01f * rayDir, rayDir); 
	ray shadowRay;
	shadowRay.origin = surfacePoint + 0.01f * rayDir;
	shadowRay.direction = rayDir;
	glm::vec3 intersPoint, intersNormal;
//	if(abs(surfacePoint.y) > 8.0f)
//	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
 //   int y = (blockIdx.y * blockDim.y) + threadIdx.y;
//	printf("%d %d\n", (blockIdx.x * blockDim.x), (blockIdx.y * blockDim.y) + threadIdx.y);
//	printf("%f %f %f\n", shadowRay.origin.x, shadowRay.origin.y, shadowRay.origin.z);
	int intersIndex = rayIntersect(shadowRay, geoms, numberOfGeoms, intersPoint, intersNormal, materials); 
//	printf("%d", intersIndex);
	if(intersIndex == -1) return false;
	else if(materials[geoms[intersIndex].materialid].emittance > 0.0f) return true;
	else return false;

}
__host__ __device__ glm::vec3 raytraceRecursive(const ray &r, int depth, int maximumDepth, staticGeom* geoms, int numberOfGeoms, material* materials, int numberOfMaterials, light* lightSources, int numberOfLights){

	glm::vec3 bgColor(0.0f);
	glm::vec3 ambientColor(1.0f);
	glm::vec3 phongColor(0.0f), reflColor(0.0f), refraColor(0.0f);;
	glm::vec3 returnColor(0.0f);
	float ka = 0.2f;


/*
	if(depth > maximumDepth)
		return bgColor;*/

	// intersection test	
	glm::vec3 intersectionPoint, intersectionNormal;
	int intersIndex = rayIntersect(r, geoms, numberOfGeoms, intersectionPoint, intersectionNormal, materials);
//	printf("%f %f %f\n", intersectionPoint.x, intersectionPoint.y, intersectionPoint.z);
//	printf("%d\n", intersIndex);
	/*glm::vec3 intersectionPoint, intersectionNormal;
	float distance = 8000.0f;
	float tempDistance = -1.0f;
	glm::vec3 tempIntersctionPoint(0.0f), tempIntersectionNormal(0.0f);
	int intersIndex = -1;
	printf("%f %f %f\n", tempIntersectionNormal.x, tempIntersectionNormal.y, tempIntersectionNormal.z);
	for(int i = 0; i < numberOfGeoms; i++)
	{
		tempDistance = -1.0f;
//		if(materials[geoms[i].materialid].emittance > 0.0f) continue; // do not test intesection with light source
		if(geoms[i].type == GEOMTYPE::SPHERE){
			tempDistance = sphereIntersectionTest(geoms[i], r, tempIntersctionPoint, tempIntersectionNormal);
		}
		else if(geoms[i].type == GEOMTYPE::CUBE){
			tempDistance = boxIntersectionTest(geoms[i], r, tempIntersctionPoint, tempIntersectionNormal);
		}
		else 
			continue;
		if(!(abs(tempDistance + 1.0f) < EPSILON))
		{
			if(tempDistance < distance) {
				intersIndex = i;
				distance = tempDistance;
				intersectionPoint = tempIntersctionPoint;
				intersectionNormal = tempIntersectionNormal;
			}
		}
	}*/
	


	if(intersIndex == -1) return bgColor;
	else if(materials[geoms[intersIndex].materialid].emittance > 0.0f) // intersected with light source geometry
		return materials[geoms[intersIndex].materialid].color;
	else // intersected with actual geometry
	{
		returnColor = ka * ambientColor * materials[geoms[intersIndex].materialid].color;

		if(materials[geoms[intersIndex].materialid].hasReflective == 1)
		{
			glm::vec3 reflDir = ReflectedRay(r.direction, intersectionNormal);
			ray reflRay;
			reflRay.origin = intersectionPoint + 0.01f * reflDir;
			reflRay.direction = reflDir;
			reflColor = raytraceRecursive(reflRay, depth + 1, maximumDepth, geoms, numberOfGeoms, materials, numberOfMaterials, lightSources, numberOfLights);
		}
		if(materials[geoms[intersIndex].materialid].hasRefractive == 1)
		{
//				refraColor = bgColor;
		}
//		printf("before shadow test\n");
		/*if(ShadowRayUnblocked(intersectionPoint, lightSources[0].position, geoms, numberOfGeoms, materials))
		{
		glm::vec3 L = glm::normalize(lightSources[0].position - intersectionPoint);
//		if(abs(intersectionPoint.y) > 5.0f)
//				printf("%f %f %f\n", intersectionPoint.x, intersectionPoint.y, intersectionPoint.z);
				float dot1 = glm::clamp(glm::dot(intersectionNormal, L), 0.0f, 1.0f);
//				printf("%f\n", dot1);
				float dot2 = glm::dot(ReflectedRay(-L, intersectionNormal) ,-r.direction);
				glm::vec3 diffuse = lightSources[0].color * 0.5f * materials[geoms[intersIndex].materialid].color * dot1;
				returnColor = diffuse;
		}
	}*/
		//printf("before shadow test\n");
			for(int i = 0; i < numberOfLights; i++)
			if(ShadowRayUnblocked(intersectionPoint, lightSources[i].position, geoms, numberOfGeoms, materials))
			{
//					printf("shadow ray not blocked\n");
				glm::vec3 L = glm::normalize(lightSources[i].position - intersectionPoint);
//				printf("%f %f %f\n", intersectionNormal.x, intersectionNormal.y, intersectionNormal.z);
				float dot1 = glm::clamp(glm::dot(intersectionNormal, L), 0.0f, 1.0f);
//				printf("%f\n", dot1);
				float dot2 = glm::dot(ReflectedRay(-L, intersectionNormal) ,-r.direction);
				glm::vec3 diffuse = lightSources[i].color * 0.5f * materials[geoms[intersIndex].materialid].color * dot1;
				glm::vec3 specular;
				if(abs(materials[geoms[intersIndex].materialid].specularExponent) > 1e-6)
					specular = lightSources[i].color * 0.1f * pow(max(dot2, 0.0f), materials[geoms[intersIndex].materialid].specularExponent);
				phongColor += diffuse + specular;
				
			}

		returnColor += phongColor + 0.1f * reflColor + refraColor;
	}



	
	return returnColor;
}

//TODO: IMPLEMENT THIS FUNCTION
//Core raytracer kernel
__global__ void raytracePrimary(glm::vec2 resolution, float time, cameraData cam, int rayDepth, glm::vec3* colors,
                            staticGeom* geoms, int numberOfGeoms, material* materials, int numberOfMaterials, light* lightSources, int numberOfLights){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if((x<=resolution.x && y<=resolution.y)){
	  ray r = raycastFromCameraKernel(resolution, time, x, y, cam.position, cam.view, cam.up, cam.fov);
//    colors[index] = generateRandomNumberFromThread(resolution, time, x, y);
	  colors[index] = raytraceRecursive(r, 0, rayDepth, geoms, numberOfGeoms, materials, numberOfMaterials, lightSources, numberOfLights);
   }
}



//TODO: FINISH THIS FUNCTION
// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRaytraceCore(uchar4* PBOpos, camera* renderCam, int frame, int iterations, material* materials, int numberOfMaterials, geom* geoms, int numberOfGeoms){
  // frame: current frame, objects and cam move between frames in multi-frame mode
  // iterations: curent iteration, objects and cam do not move between iterations
  int traceDepth = 5; //determines how many bounces the raytracer traces

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(renderCam->resolution.x)/float(tileSize)), (int)ceil(float(renderCam->resolution.y)/float(tileSize)));
  
  //send image to GPU
  glm::vec3* cudaimage = NULL;
  cudaMalloc((void**)&cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3));
  cudaMemcpy( cudaimage, renderCam->image, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyHostToDevice);
  
  //package geometry and materials and sent to GPU
  staticGeom* geomList = new staticGeom[numberOfGeoms];
  for(int i=0; i<numberOfGeoms; i++){
    staticGeom newStaticGeom;
    newStaticGeom.type = geoms[i].type;
    newStaticGeom.materialid = geoms[i].materialid;
    newStaticGeom.translation = geoms[i].translations[frame];
    newStaticGeom.rotation = geoms[i].rotations[frame];
    newStaticGeom.scale = geoms[i].scales[frame];
    newStaticGeom.transform = geoms[i].transforms[frame];
    newStaticGeom.inverseTransform = geoms[i].inverseTransforms[frame];
    geomList[i] = newStaticGeom;
  }
  
  staticGeom* cudageoms = NULL;
  cudaMalloc((void**)&cudageoms, numberOfGeoms*sizeof(staticGeom));
  cudaMemcpy( cudageoms, geomList, numberOfGeoms*sizeof(staticGeom), cudaMemcpyHostToDevice);

  material* cudamaterials = NULL;
  cudaMalloc((void**)&cudamaterials, numberOfMaterials*sizeof(material));
  cudaMemcpy( cudamaterials, materials, numberOfMaterials*sizeof(material), cudaMemcpyHostToDevice);

  // generate light sources from light source geometry. Could be single point light or area light
  int maxNumOfLightSources = numberOfGeoms;
  light* lightSources = new light[maxNumOfLightSources];
  int numberOfLights = 0;
  int j = 0;
  for(int i = 0; i < numberOfGeoms; ++i){
	  if(materials[geoms[i].materialid].emittance > 0.0f)
	  {	// generate point sources based on light source geometry		   
		  lightSources[j].position = geomList[i].translation;
		  lightSources[j].color = materials[geoms[i].materialid].color;
		  lightSources[j].emittance = materials[geoms[i].materialid].emittance;
		  ++numberOfLights;
		  ++j;
	  }
  }

  light* cudalights = NULL;
  cudaMalloc((void**)&cudalights, numberOfLights*sizeof(light));
  cudaMemcpy( cudalights, lightSources, numberOfLights*sizeof(light), cudaMemcpyHostToDevice);
  
  //package camera
  cameraData cam;
  cam.resolution = renderCam->resolution;
  cam.position = renderCam->positions[frame];
  cam.view = renderCam->views[frame];
  cam.up = renderCam->ups[frame];
  cam.fov = renderCam->fov;

  size_t size;
  cudaDeviceSetLimit(cudaLimitStackSize, 10000 * sizeof(float));
  cudaDeviceGetLimit(&size, cudaLimitStackSize);
  
  
//  printf("Stack size found to be %d\n",(int)size);

  //kernel launches
  raytracePrimary<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->resolution, (float)iterations, cam, traceDepth, cudaimage, cudageoms, numberOfGeoms, cudamaterials, numberOfMaterials, cudalights, numberOfLights);
 // cudaDeviceSynchronize ();
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, renderCam->resolution, cudaimage);

  //retrieve image from GPU
  cudaMemcpy( renderCam->image, cudaimage, (int)renderCam->resolution.x*(int)renderCam->resolution.y*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

  //free up stuff, or else we'll leak memory like a madman
  cudaFree( cudaimage );
  cudaFree( cudageoms );
  cudaFree( cudamaterials);
  cudaFree( cudalights);
  delete geomList;

  // make certain the kernel has completed
  cudaThreadSynchronize();

  checkCUDAError("Kernel failed!");
}
