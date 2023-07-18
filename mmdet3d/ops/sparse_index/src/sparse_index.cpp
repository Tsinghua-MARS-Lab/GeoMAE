#include <torch/extension.h>
#include "sparse_index.h"

namespace sparse_index {


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sparse_index_test", &sparse_index_test, "sparse index");
  m.def("sparse_index_test_", &sparse_index_test_, "sparse index test");
  m.def("sparse_index_backward_test", &sparse_index_backward_test, "sparse index backward");
  m.def("sparse_index_wo_pos", &sparse_index_wo_pos, "sparse index wo pos");
  m.def("sparse_index_with_pos", &sparse_index_with_pos, "sparse index with pos");
  m.def("sparse_index_with_pos_half", &sparse_index_with_pos_half, "sparse index with pos half percision");
  m.def("sparse_index_backward_half", &sparse_index_backward_half, "sparse index backward");

}
}   // namespace sparse_index