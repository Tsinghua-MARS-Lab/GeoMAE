#include <torch/extension.h>
#include "nms.h"

namespace wnms {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("wnms_4c", &point4_wnms_4c<float>, "wnms_4c");

}
}
