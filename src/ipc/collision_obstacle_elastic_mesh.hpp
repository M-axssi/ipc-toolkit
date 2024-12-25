#pragma once

#include "collision_mesh.hpp"

#include <ipc/utils/unordered_map_and_set.hpp>

#include <Eigen/Core>
#include <Eigen/Sparse>

namespace ipc {

/// @brief A class for encapsolating the transformation/selections needed to go from a volumetric FE mesh to a surface collision mesh.
class CollisionObstacleElasticMesh:public CollisionMesh {
public:
    /// @brief Construct a new Collision Mesh object.
    /// Collision Mesh objects are immutable, so use the other constructors.
    CollisionObstacleElasticMesh() = default;

    /// @brief Construct a new Collision Mesh object directly from the collision mesh vertices.
    /// @param rest_positions_obstacle The vertices of the obstacle mesh at rest (#V × dim).
    /// @param rest_positions_elastic The vertices of the elastic mesh at rest (#V × dim).
    /// @param edges_obstacle The edges of the obstacle mesh (#E × 2).
    /// @param faces_obstacle The faces of the obstacle mesh (#F × 3).
    /// @param edges_elastic The edges of the elastic mesh (#E × 2).
    /// @param faces_elastic The faces of the elastic mesh (#F × 3).
    CollisionObstacleElasticMesh(
        const Eigen::MatrixXd& rest_positions_obstacle,
        const Eigen::MatrixXd& rest_positions_elastic,
        const Eigen::MatrixXi& edges_obstacle = Eigen::MatrixXi(),
        const Eigen::MatrixXi& faces_obstacle = Eigen::MatrixXi(),
        const Eigen::MatrixXi& edges_elastic = Eigen::MatrixXi(),
        const Eigen::MatrixXi& faces_elastic = Eigen::MatrixXi()
        );

    /// @brief Destroy the Collision Mesh object
    ~CollisionObstacleElasticMesh() = default;

    // -----------------------------------------------------------------------
    const Eigen::MatrixXi& elastic_edges() const { return m_edges_elastic; }
    const Eigen::MatrixXi& obstacle_edges() const { return m_edges_obstacle; }
    const Eigen::MatrixXi& elastic_faces() const { return m_faces_elastic; }
    const Eigen::MatrixXi& obstacle_faces() const { return m_faces_obstacle; }

    // -----------------------------------------------------------------------
    const Eigen::MatrixXd extract_obstacle_vertices(const Eigen::MatrixXd& vertices) const
    {
        int dim = this->dim();
        return vertices.block(0, 0, m_obstacle_vnum, dim);
    }

    const Eigen::MatrixXd extract_elastic_vertices(const Eigen::MatrixXd& vertices) const 
    {
        int dim = this->dim();
        return vertices.block(m_obstacle_vnum, 0, m_elastic_vnum, dim);
    }

    long elastic_vertex_id_to_full_vertex_id(long vertex_id) const
    {
        return vertex_id + m_obstacle_vnum;
    }

    long obstacle_vertex_id_to_full_vertex_id(long vertex_id) const
    {
        return vertex_id;
    }

    long elastic_edge_id_to_full_edge_id(long edge_id) const
    {
        return edge_id + m_obstacle_enum;
    }

    long obstacle_edge_id_to_full_edge_id(long edge_id) const
    {
        return edge_id;
    }

    long elastic_face_id_to_full_face_id(long face_id) const
    {
        return face_id + m_obstacle_fnum;
    }

    long obstacle_face_id_to_full_face_id(long face_id) const
    {
        return face_id;
    }


protected:  
    int m_obstacle_vnum;
    int m_elastic_vnum;
    int m_obstacle_enum;
    int m_elastic_enum;
    int m_obstacle_fnum;
    int m_elastic_fnum;

    Eigen::MatrixXi m_edges_obstacle;
    Eigen::MatrixXi m_faces_obstacle;
    Eigen::MatrixXi m_edges_elastic;
    Eigen::MatrixXi m_faces_elastic;
};

} // namespace ipc
