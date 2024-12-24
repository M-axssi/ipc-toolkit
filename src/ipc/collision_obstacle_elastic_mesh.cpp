#include "collision_obstacle_elastic_mesh.hpp"

#include <ipc/utils/unordered_map_and_set.hpp>
#include <ipc/utils/logger.hpp>
#include <ipc/utils/eigen_ext.hpp>
#include <ipc/utils/local_to_global.hpp>
#include <ipc/utils/area_gradient.hpp>

namespace ipc {
CollisionObstacleElasticMesh::CollisionObstacleElasticMesh(
    const Eigen::MatrixXd& rest_positions_obstacle,
    const Eigen::MatrixXd& rest_positions_elastic,
    const Eigen::MatrixXi& edges_obstacle,
    const Eigen::MatrixXi& faces_obstacle,
    const Eigen::MatrixXi& edges_elastic,
    const Eigen::MatrixXi& faces_elastic)
{
    m_edges_elastic = edges_elastic;
    m_faces_elastic = faces_elastic;
    m_edges_obstacle = edges_obstacle;
    m_faces_obstacle = faces_obstacle;

    m_obstacle_vnum = rest_positions_obstacle.rows();
    m_elastic_vnum = rest_positions_elastic.rows();
    m_obstacle_enum = edges_obstacle.rows();
    m_elastic_enum = edges_elastic.rows();
    m_obstacle_fnum = faces_obstacle.rows();
    m_elastic_fnum = faces_elastic.rows();

    // Merge to one mesh
    Eigen::MatrixXd rest_positions(
        rest_positions_obstacle.rows() + rest_positions_elastic.rows(),
        rest_positions_elastic.cols());
    rest_positions << rest_positions_obstacle, rest_positions_elastic;

    Eigen::MatrixXi edges(
        edges_obstacle.rows() + edges_elastic.rows(), edges_elastic.cols());
    edges << edges_obstacle,
        edges_elastic.array() + rest_positions_obstacle.rows();

    Eigen::MatrixXi faces(
        faces_obstacle.rows() + faces_elastic.rows(), faces_elastic.cols());
    faces << faces_obstacle,
        faces_elastic.array() + rest_positions_obstacle.rows();
    std::vector<bool> include_vertex(rest_positions.rows(), true);

    this->initialization(include_vertex, rest_positions, edges, faces);
}

} // namespace ipc
