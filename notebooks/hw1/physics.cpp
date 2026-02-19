#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>

namespace py = pybind11;

struct Ball {
    double x, y, z, vx, vy, vz, radius, mass;
};

class cppSimulation {
public:
    std::vector<Ball> balls;
    double dt;

    cppSimulation(double time_step) : dt(time_step) {}

    void add_ball(double x, double y, double z, double vx, double vy, double vz, double r, double m) {
        balls.push_back({x, y, z, vx, vy, vz, r, m});
    }

    void step() {
        for (auto& b : balls) {
            b.vz -= 9.81 * dt; // Gravity
            b.x += b.vx * dt;
            b.y += b.vy * dt;
            b.z += b.vz * dt;

            // Simple floor bounce
            if (b.z < b.radius) {
                b.z = b.radius;
                b.vz *= -0.9; 
            }
        }
    }
};

PYBIND11_MODULE(physics_engine, m) {
    py::class_<cppSimulation>(m, "cppSimulation")
        .def(py::init<double>())
        .def("add_ball", &cppSimulation::add_ball)
        .def("step", &cppSimulation::step);
}