#include "rasterize_coarse/rasterize_coarse.h"
#include "ray_trace_voge/ray_trace_voge.h"
#include "sample_voge/sample_voge.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_points_coarse", &RasterizeEllipseCoarseCuda);
  m.def("ray_trace_voge_fine", &RayTraceFineVoge);
  m.def("ray_trace_voge_fine_backward", &RayTraceFineVogeBackward);
  m.def("sample_voge", &SampleVoge);
  m.def("sample_voge_backward", &SampleVogeBackward);
}