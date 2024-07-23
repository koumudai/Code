#include "hilbertcurve.h"
#include "utils.h"

typedef int32_t point_type;
typedef int64_t distance_type;

void distances_from_points_kernel_wrapper(int b_s, int n_p, int d_p, int m_l,
                                          point_type * points, distance_type * distances);

at::Tensor distances_from_points(at::Tensor points, const int d_p, const int m_l) {
  CHECK_CONTIGUOUS(points);

  at::Tensor distances = torch::zeros({points.size(0), points.size(1), 1},
          at::device(points.device()).dtype(at::ScalarType::Long));
  if (points.is_cuda()) {
    distances_from_points_kernel_wrapper(points.size(0), points.size(1), d_p, m_l,
            points.data_ptr<point_type>(), distances.data_ptr<distance_type>());
  } else {
    AT_ASSERT(false, "CPU not supported");
  }

  return distances;
}

void points_from_distances_kernel_wrapper(int b_s, int n_p, int d_p, int m_l,
                                          distance_type * distances, point_type * points);

at::Tensor points_from_distances(at::Tensor distances, const int d_p, const int m_l) {
  CHECK_CONTIGUOUS(distances);

  at::Tensor points = torch::zeros({distances.size(0), distances.size(1), d_p},
          at::device(distances.device()).dtype(at::ScalarType::Int));

  if (distances.is_cuda()) {
    points_from_distances_kernel_wrapper(distances.size(0), distances.size(1), d_p, m_l,
            distances.data_ptr<distance_type>(), points.data_ptr<point_type>());
  } else {
    AT_ASSERT(false, "CPU not supported");
  }

  return points;
}