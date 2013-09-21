-------------------------------------------------------------------------------
CIS565: Project 1: CUDA Raytracer
-------------------------------------------------------------------------------
Fall 2013
-------------------------------------------------------------------------------
Due Thursday, 09/19/2013
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
PROJECT DESCRIPTION
-------------------------------------------------------------------------------
This is a GPU ray tracing program. Features implemented including:
* Basic features
	?Raycasting from a camera into a scene through a pixel grid
	?Phong lighting for one point light source
	?Diffuse lambertian surfaces
	?Raytraced shadows
	?Cube intersection testing
	?Sphere surface point sampling

* Additional features
	?Specular reflection 
	?Soft shadows and area lights 
	?Refraction

-------------------------------------------------------------------------------
SCREEN SHOTS AND VIDEOS
-------------------------------------------------------------------------------
* Project running

* Final renders
  ![ScreenShot](https://raw.github.com/wuhao1117/Project1-RayTracer/master/renders/MyRender.jpg)

* Video
  https://raw.github.com/wuhao1117/Project1-RayTracer/master/renders/GPU_raytracer.mp4
-------------------------------------------------------------------------------
HOW TO BUILD
-------------------------------------------------------------------------------
* Project tested in Visual Studio 2012 in Release(5.5) configuration with 
  compute_20,sm_21

-------------------------------------------------------------------------------
PERFORMANCE EVALUATION
-------------------------------------------------------------------------------
The performance evaluation is where you will investigate how to make your CUDA
programs more efficient using the skills you've learned in class. You must have
perform at least one experiment on your code to investigate the positive or
negative effects on performance. 

One such experiment would be to investigate the performance increase involved 
with adding a spatial data-structure to your scene data.

Another idea could be looking at the change in timing between various block
sizes.

A good metric to track would be number of rays per second, or frames per 
second, or number of objects displayable at 60fps.

We encourage you to get creative with your tweaks. Consider places in your code
that could be considered bottlenecks and try to improve them. 

Each student should provide no more than a one page summary of their
optimizations along with tables and or graphs to visually explain and
performance differences.

-------------------------------------------------------------------------------
THIRD PARTY CODE USED
-------------------------------------------------------------------------------
* An Efficient and Robust Ray¨CBox Intersection Algorithm, A. Williams, et al.  
  http://people.csail.mit.edu/amy/papers/box-jgt.pdf

-------------------------------------------------------------------------------
SELF-GRADING
-------------------------------------------------------------------------------

