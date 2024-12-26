#include <common.hpp>

#include <ipc/collision_obstacle_elastic_mesh.hpp>

namespace py = pybind11;
using namespace ipc;

void define_obstacle_elastic_collision_mesh(py::module_& m)
{
    py::class_<CollisionObstacleElasticMesh,CollisionMesh>(
        m, "CollisionObstacleElasticMesh")
        .def(
            py::init<
                const Eigen::MatrixXd&, const Eigen::MatrixXd&,
                const Eigen::MatrixXi&, const Eigen::MatrixXi&,
                const Eigen::MatrixXi&, const Eigen::MatrixXi&,
                const std::vector<int>&>(),
            R"ipc_Qu8mg5v7(
            Construct a new CollisionObstacleElasticMesh Mesh object from obstacle meshes and elastic meshes.

            Parameters:
                rest_positions_obstacle: The vertices of the obstacle mesh at rest (#V × dim).
                rest_positions_elastic: The vertices of the elastic mesh at rest (#V × dim).
                edges_obstacle: The edges of the obstacle mesh (#E × 2).
                faces_obstacle: The faces of the obstacle mesh (#F × 3).
                edges_elastic: The edges of the elastic mesh (#E × 2).
                faces_elastic: The faces of the elastic mesh (#F × 3).
                bc_vids The vertex indices of the Dirichlet boundary of the elastic mesh.
            )ipc_Qu8mg5v7",
            py::arg("rest_positions_obstacle"),
            py::arg("rest_positions_elastic"),
            py::arg("edges_obstacle") = Eigen::MatrixXi(),
            py::arg("faces_obstacle") = Eigen::MatrixXi(),
            py::arg("edges_elastic") = Eigen::MatrixXi(),
            py::arg("faces_elastic") = Eigen::MatrixXi(),
            py::arg("bc_vids_elastic") = std::vector<int>());
}
