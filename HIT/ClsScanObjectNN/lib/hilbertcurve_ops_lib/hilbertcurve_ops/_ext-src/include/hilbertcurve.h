#pragma once
#include <torch/extension.h>

at::Tensor distances_from_points(at::Tensor points, const int d_p, const int m_l);
at::Tensor points_from_distances(at::Tensor distances, const int d_p, const int m_l);
