#include "hilbertcurve.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("distances_from_points", &distances_from_points);
  m.def("points_from_distances", &points_from_distances);
}