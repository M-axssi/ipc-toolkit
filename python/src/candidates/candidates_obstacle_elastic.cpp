#include <common.hpp>

#include <ipc/candidates/candidates_obstacle_elastic.hpp>

namespace py = pybind11;
using namespace ipc;

void define_obstacle_elastic_candidates(py::module_& m)
{
    py::class_<CandidatesObstacleElastic, Candidates>(
        m, "CandidatesObstacleElastic")
        .def(
            py::init<const CollisionObstacleElasticMesh&>(),
            R"ipc_Qu8mg5v7(
            Initialization from a CollisionObstacleElasticMesh mesh.

            Parameters:
                mesh: The surface of the CollisionObstacleElasticMesh mesh.
            )ipc_Qu8mg5v7",
            py::arg("mesh"))
        .def(
            "build",
            py::overload_cast<const Eigen::MatrixXd&, const double, bool>(
                &CandidatesObstacleElastic::build),
            R"ipc_Qu8mg5v7(
            Initialize the set of discrete collision detection candidates.

            Parameters:
                vertices: Surface vertex positions (rowwise).
                inflation_radius: Amount to inflate the bounding boxes.
                updateObstacleBVH: Force to update bvh of obstacle.
            )ipc_Qu8mg5v7",
            py::arg("vertices"),
            py::arg("inflation_radius") = 0,
            py::arg("updateObstacleBVH") = false)
        .def(
            "build",
            py::overload_cast<
                const Eigen::MatrixXd&, const Eigen::MatrixXd&, const double,
                bool>(&CandidatesObstacleElastic::build),
            R"ipc_Qu8mg5v7(
            Initialize the set of continuous collision detection candidates.

            Note:
                Assumes the trajectory is linear.

            Parameters:
                vertices_t0: Surface vertex starting positions (rowwise).
                vertices_t1: Surface vertex ending positions (rowwise).
                inflation_radius: Amount to inflate the bounding boxes.
                updateObstacleBVH: Force to update bvh of obstacle.
            )ipc_Qu8mg5v7",
            py::arg("vertices_t0"), py::arg("vertices_t1"),
            py::arg("inflation_radius") = 0,
            py::arg("updateObstacleBVH") = false);
}
