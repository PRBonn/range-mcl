#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


namespace py = pybind11;

int wrap(int x, int dim) {
  int value = x;
  if (value >= dim)
    value = value - dim;
  if (value < 0)
    value = value + dim;
  return value;
}

//py::array_t<float> range_projection_vertex(py::array_t<float> vertex, float fov_up_deg, float fov_down_deg)
py::array_t<float> gen_depth_and_normal(py::array_t<float, py::array::c_style> virtual_scan,
                                        int H, int W, float fov_up_deg, float fov_down_deg,
                                        float max_range, float min_range){

  float fov_up = fov_up_deg / 180.0 * M_PI;
  float fov_down = fov_down_deg / 180.0 * M_PI;
  float fov = std::abs(fov_down) + std::abs(fov_up);

  /*  allocate the buffer */
  py::array_t<float> normal_and_range = py::array_t<float>(H * W * 4);

  auto buf1 = virtual_scan.request();
  auto buf2 = normal_and_range.request();

  float *ptr1 = (float *) buf1.ptr,
        *ptr2 = (float *) buf2.ptr;

  // initialize depth and normal map
#pragma omp parallel for schedule(static)
  for(int i=0; i<H*W*4; ++i){
    ptr2[i] = -1.0;
  }

#pragma omp parallel for schedule(static)
  for(int x=0; x<W; ++x) {
    for(int y=0; y<H-1; ++y) {
      float px = ptr1[y*W*4 + x*4];
      float py = ptr1[y*W*4 + x*4 + 1];
      float pz = ptr1[y*W*4 + x*4 + 2];
      float depth = ptr1[y*W*4 + x*4 + 3];

      if (depth > 0) {
        int wrap_x = wrap(x + 1, W);
        float ux = ptr1[y*W*4 + wrap_x*4 ];
        float uy = ptr1[y*W*4 + wrap_x*4 + 1];
        float uz = ptr1[y*W*4 + wrap_x*4 + 2];
        float u_depth = ptr1[y*W*4 + wrap_x*4 + 3];
        if (u_depth < 0)
          continue;

        float vx = ptr1[(y+1)*W*4 + x*4 ];
        float vy = ptr1[(y+1)*W*4 + x*4 + 1];
        float vz = ptr1[(y+1)*W*4 + x*4 + 2];
        float v_depth = ptr1[(y+1)*W*4 + x*4 + 3];
        if (v_depth < 0)
          continue;

        float u_normx = ux - px;
        float u_normy = uy - py;
        float u_normz = uz - pz;
        float l=std::sqrt(u_normx*u_normx+u_normy*u_normy+u_normz*u_normz);
        u_normx/=l;
        u_normy/=l;
        u_normz/=l;

        float v_normx = vx - px;
        float v_normy = vy - py;
        float v_normz = vz - pz;
        l=std::sqrt(v_normx*v_normx+v_normy*v_normy+v_normz*v_normz);
        v_normx/=l;
        v_normy/=l;
        v_normz/=l;

        float crossx = u_normz * v_normy - u_normy * v_normz;
        float crossy = u_normx * v_normz - u_normz * v_normx;
        float crossz = u_normy * v_normx - u_normx * v_normy;
        float norm = std::sqrt(crossx*crossx+crossy*crossy+crossz*crossz);

        if (norm > 0) {
          float normalx = crossx / norm;
          float normaly = crossy / norm;
          float normalz = crossz / norm;
          ptr2[y*W*4 + x*4] = normalx;
          ptr2[y*W*4 + x*4 + 1] = normaly;
          ptr2[y*W*4 + x*4 + 2] = normalz;
          ptr2[y*W*4 + x*4 + 3] = depth;
        }
      }
    }
  }

  // reshape array to match input shape
  normal_and_range.resize({H, W, 4});

  return normal_and_range;
}


PYBIND11_MODULE(c_gen_depth_and_normal, m) {
        m.doc() = "generate depth and normal map using pybind11"; // optional module docstring

        m.def("gen_depth_and_normal", &gen_depth_and_normal, "generate depth and normal map");
}
