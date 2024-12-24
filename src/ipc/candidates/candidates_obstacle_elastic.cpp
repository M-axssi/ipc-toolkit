#include "candidates_obstacle_elastic.hpp"

#include <ipc/ipc.hpp>
#include <ipc/utils/save_obj.hpp>
#include <ipc/utils/eigen_ext.hpp>

#include <ipc/config.hpp>

#include <igl/remove_unreferenced.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <shared_mutex>

#include <fstream>

namespace ipc {
void CandidatesObstacleElastic::build(
    const Eigen::MatrixXd& vertices,
    const double inflation_radius,
    bool updateObstacleBVH)
{
    const Eigen::MatrixXd& elastic_vertices =
        m_mesh.extract_elastic_vertices(vertices);

    // Construct bvh for elastic objects
    broad_phase_elastic = std::make_shared<BVH>();
    broad_phase_elastic->build(
        elastic_vertices, m_mesh.elastic_edges(), m_mesh.elastic_faces(),
        inflation_radius);

    // Construct bvh for obstacle objects
    if (updateObstacleBVH) {
        const Eigen::MatrixXd& obstacle_vertices =
            m_mesh.extract_obstacle_vertices(vertices);

        broad_phase_obstacle = std::make_shared<BVH>();
        broad_phase_obstacle->build(
            obstacle_vertices, m_mesh.obstacle_edges(), m_mesh.obstacle_faces(),
            inflation_radius);
    }

    detect_candidates();
}

void CandidatesObstacleElastic::build(
    const Eigen::MatrixXd& vertices_t0,
    const Eigen::MatrixXd& vertices_t1,
    const double inflation_radius,
    bool updateObstacleBVH)
{
    clear();

    // Construct bvh for elastic objects
    const Eigen::MatrixXd& elastic_vertices_t0 =
        m_mesh.extract_elastic_vertices(vertices_t0);
    const Eigen::MatrixXd& elastic_vertices_t1 =
        m_mesh.extract_elastic_vertices(vertices_t1);

    broad_phase_elastic = std::make_shared<BVH>();
    broad_phase_elastic->build(
        elastic_vertices_t0, elastic_vertices_t1, 
        m_mesh.elastic_edges(), m_mesh.elastic_faces(),
        inflation_radius);

    // Construct bvh for obstacle objects
    if (updateObstacleBVH) {
        const Eigen::MatrixXd& obstacle_vertices_t0 =
            m_mesh.extract_obstacle_vertices(vertices_t0);
        const Eigen::MatrixXd& obstacle_vertices_t1 =
            m_mesh.extract_obstacle_vertices(vertices_t1);

        broad_phase_obstacle = std::make_shared<BVH>();
        broad_phase_obstacle->build(
            obstacle_vertices_t0, obstacle_vertices_t1, 
            m_mesh.obstacle_edges(), m_mesh.obstacle_faces(),
            inflation_radius);
    }

    detect_candidates();
}

void CandidatesObstacleElastic::detect_candidates()
{
    clear();

    // Detect collision candidates between elastic objects
    broad_phase_elastic->detect_collision_candidates(m_dim, *this);

    tbb::parallel_for(
        tbb::blocked_range<size_t>(size_t(0), ee_candidates.size()),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i < r.end(); i++) {
                m_mesh.elastic_edge_id_to_full_edge_id(ee_candidates[i].edge0_id);
                m_mesh.elastic_edge_id_to_full_edge_id(ee_candidates[i].edge1_id);
            }
        });

    tbb::parallel_for(
        tbb::blocked_range<size_t>(size_t(0), fv_candidates.size()),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i < r.end(); i++) {
                m_mesh.elastic_face_id_to_full_face_id(fv_candidates[i].face_id);
                m_mesh.elastic_vertex_id_to_full_vertex_id(
                    fv_candidates[i].vertex_id);
            }
        });

    // Detect collision candidates between elastic objects and obstacles
    const std::vector<AABB>& elastic_edge_boxes =
        broad_phase_elastic->get_edge_boxes();
    const std::vector<AABB>& elastic_vertex_boxes =
        broad_phase_elastic->get_vertex_boxes();
    const std::vector<AABB>& obstacle_vertex_boxes =
        broad_phase_obstacle->get_vertex_boxes();

    std::vector<EdgeEdgeCandidate> elastic_obstacle_ee_candidates;
    std::vector<FaceVertexCandidate> elastic_obstacle_vf_candidates;
    std::vector<FaceVertexCandidate> elastic_obstacle_fv_candidates;

    broad_phase_obstacle->detect_input_edge_edge_candidates(
        elastic_obstacle_ee_candidates, elastic_edge_boxes);
    broad_phase_obstacle->detect_input_vertex_face_candidates(
        elastic_obstacle_vf_candidates, elastic_vertex_boxes);
    broad_phase_elastic->detect_input_vertex_face_candidates(
        elastic_obstacle_fv_candidates, obstacle_vertex_boxes);

    tbb::parallel_for(
        tbb::blocked_range<size_t>(
            size_t(0), elastic_obstacle_ee_candidates.size()),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i < r.end(); i++) {
                m_mesh.elastic_edge_id_to_full_edge_id(
                    elastic_obstacle_ee_candidates[i].edge0_id);
                m_mesh.obstacle_edge_id_to_full_edge_id(
                    elastic_obstacle_ee_candidates[i].edge1_id);
            }
        });

    tbb::parallel_for(
        tbb::blocked_range<size_t>(
            size_t(0), elastic_obstacle_vf_candidates.size()),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i < r.end(); i++) {
                m_mesh.elastic_vertex_id_to_full_vertex_id(
                    elastic_obstacle_vf_candidates[i].vertex_id);
                m_mesh.obstacle_face_id_to_full_face_id(
                    elastic_obstacle_vf_candidates[i].face_id);
            }
        });

    tbb::parallel_for(
        tbb::blocked_range<size_t>(
            size_t(0), elastic_obstacle_fv_candidates.size()),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i < r.end(); i++) {
                m_mesh.obstacle_vertex_id_to_full_vertex_id(
                    elastic_obstacle_fv_candidates[i].vertex_id);
                m_mesh.elastic_face_id_to_full_face_id(
                    elastic_obstacle_fv_candidates[i].face_id);
            }
        });

    // merge to final candidates
    ee_candidates.insert(
        ee_candidates.end(), elastic_obstacle_ee_candidates.begin(),
        elastic_obstacle_ee_candidates.end());

    fv_candidates.insert(
        fv_candidates.end(), elastic_obstacle_fv_candidates.begin(),
        elastic_obstacle_fv_candidates.end());

    fv_candidates.insert(
        fv_candidates.end(), elastic_obstacle_vf_candidates.begin(),
        elastic_obstacle_vf_candidates.end());
}

} // namespace ipc
