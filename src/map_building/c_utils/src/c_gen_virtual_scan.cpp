#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


namespace py = pybind11;


py::array_t<float> gen_virtual_scan(py::array_t<float, py::array::c_style> points,
                                    int H, int W, float fov_up_deg, float fov_down_deg,
                                    float max_range, float min_range){
  float fov_up = fov_up_deg / 180.0 * M_PI;
  float fov_down = fov_down_deg / 180.0 * M_PI;
  float fov = std::abs(fov_down) + std::abs(fov_up);

  /*  allocate the buffer */
  py::array_t<float> virtual_scan = py::array_t<float>(H * W * 4);

  auto buf1 = points.request();
  auto buf2 = virtual_scan.request();

  float *ptr1 = (float *) buf1.ptr,
        *ptr2 = (float *) buf2.ptr;

  // initialize depth map
  for(int i=0; i<H*W*4; ++i){
    ptr2[i] = -1.0;
  }

  int num_points = buf1.size / 4;

#pragma omp parallel for schedule(static)
  for(int x=0; x<num_points; ++x) {
    float px = ptr1[x*4];
    float py = ptr1[x*4 + 1];
    float pz = ptr1[x*4 + 2];
    float intensity = ptr1[x*4 + 3];
    float depth = sqrt(px*px+py*py+pz*pz);

    // filter out the outliers
    if (depth > max_range || depth < min_range)
      continue;

    // get angles of the point
    float yaw = -std::atan2(py, px);
    float pitch = std::asin(pz / depth);

    // get projections in image coords in [0.0, 1.0]
    float proj_x = 0.5 * (yaw / M_PI + 1.0);
    float proj_y = 1.0 - (pitch + std::abs(fov_down)) / fov;

    // scale to image size using angular resolution
    proj_x *= W; // in [0.0, W]
    proj_y *= H; // in [0.0, H]

    // round and clamp for use as index
    proj_x = std::floor(proj_x);
    proj_x = std::min(W - 1, static_cast<int>(proj_x));
    proj_x = std::max(0, static_cast<int>(proj_x)); // in [0,W-1]

    proj_y = std::floor(proj_y);
    proj_y = std::min(H - 1, static_cast<int>(proj_y));
    proj_y = std::max(0, static_cast<int>(proj_y)); // in [0,H-1];

    // save only nearest point for each pixel
    int proj_index = static_cast<int>(proj_y)*W*4 + static_cast<int>(proj_x)*4;
    float old_depth = ptr2[proj_index + 3];
    if ((depth < old_depth && old_depth > 0) || old_depth < 0){
      ptr2[proj_index] = px;
      ptr2[proj_index + 1] = py;
      ptr2[proj_index + 2] = pz;
      ptr2[proj_index + 3] = depth;
    }
  }

  // reshape array to match input shape
  virtual_scan.resize({H, W, 4});

  return virtual_scan;
}


PYBIND11_MODULE(c_gen_virtual_scan, m) {
        m.doc() = "generate a virtual scan from map using pybind11"; // optional module docstring

        m.def("gen_virtual_scan", &gen_virtual_scan, "generate virtual scan");
}
