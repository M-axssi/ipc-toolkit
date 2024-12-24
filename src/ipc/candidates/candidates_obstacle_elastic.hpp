#pragma once

#include "candidates.hpp"

#include <ipc/collision_obstacle_elastic_mesh.hpp>

#include <ipc/broad_phase/bvh.hpp>
#include <ipc/broad_phase/broad_phase.hpp>
#include <ipc/candidates/vertex_vertex.hpp>
#include <ipc/candidates/edge_vertex.hpp>
#include <ipc/candidates/edge_edge.hpp>
#include <ipc/candidates/face_vertex.hpp>

#include <Eigen/Core>

#include <vector>

namespace ipc {

class CandidatesObstacleElastic : public Candidates {
public:
    CandidatesObstacleElastic() = default;
    CandidatesObstacleElastic(const CollisionObstacleElasticMesh& mesh)
        : m_mesh(mesh)
    {
        m_dim = m_mesh.dim();
    };

    /// @brief Initialize the set of discrete collision detection candidates.
    /// @param vertices Surface vertex positions (rowwise).
    /// @param inflation_radius Amount to inflate the bounding boxes.
    /// @param updateObstacleBVH update bvh of obstacle.
    void build(
        const Eigen::MatrixXd& vertices,
        const double inflation_radius = 0,
        bool updateObstacleBVH = false);

    /// @brief Initialize the set of continuous collision detection candidates.
    /// @note Assumes the trajectory is linear.
    /// @param vertices_t0 Surface vertex starting positions (rowwise).
    /// @param vertices_t1 Surface vertex ending positions (rowwise).
    /// @param inflation_radius Amount to inflate the bounding boxes.
    void build(
        const Eigen::MatrixXd& vertices_t0,
        const Eigen::MatrixXd& vertices_t1,
        const double inflation_radius = 0,
        bool updateObstacleBVH = false);

    void detect_candidates();

protected:
    const CollisionObstacleElasticMesh& m_mesh;
    int m_dim;

    // Broad Phase Method : only support BVH now
    std::shared_ptr<BVH> broad_phase_elastic;
    std::shared_ptr<BVH> broad_phase_obstacle;
};

} // namespace ipc
