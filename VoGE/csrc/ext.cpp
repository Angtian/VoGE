#include "rasterize_coarse/rasterize_coarse.h"
#include "ray_trace_voge/ray_trace_voge.h"
#include "sample_voge/sample_voge.h"
#include "voge_ray_tracing_ray/voge_ray_tracing_ray.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_points_coarse", &RasterizeEllipseCoarseCuda);
  m.def("ray_trace_voge_fine", &RayTraceFineVoge);
  m.def("ray_trace_voge_fine_backward", &RayTraceFineVogeBackward);
  m.def("ray_trace_voge_ray", &RayTraceVogeRay);
  m.def("ray_trace_voge_ray_backward", &RayTraceVogeRayBackward);
  m.def("find_nearest_k", &FindNearestK);
  m.def("sample_voge", &SampleVoge);
  m.def("sample_voge_backward", &SampleVogeBackward);
  m.def("scatter_max", &ScatterMax);
}
