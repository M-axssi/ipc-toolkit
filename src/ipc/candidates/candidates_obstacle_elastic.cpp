#include "candidates_obstacle_elastic.hpp"

#include <ipc/ipc.hpp>
#include <ipc/utils/save_obj.hpp>
#include <ipc/utils/eigen_ext.hpp>

#include <ipc/config.hpp>

#include <igl/remove_unreferenced.h>
#include <ipc/utils/merge_thread_local.hpp>
#include <ipc/utils/intersection.hpp>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>

#include <shared_mutex>

#include <fstream>

namespace ipc {
void CandidatesObstacleElastic::build(
    const Eigen::MatrixXd& vertices,
    const double inflation_radius,
    bool updateObstacleBVH,
    bool detect_elastic_obstacle_coll,
    bool detect_elastic_elastic_coll)
{
    const Eigen::MatrixXd elastic_vertices =
        m_mesh.extract_elastic_vertices(vertices);

    // Construct bvh for elastic objects
    if (!broad_phase_elastic)
        broad_phase_elastic = std::make_shared<BVH>();
    broad_phase_elastic->build(
        elastic_vertices, m_mesh.elastic_edges(), m_mesh.elastic_faces(),
        inflation_radius);

    // Construct bvh for obstacle objects
    if (updateObstacleBVH) {
        const Eigen::MatrixXd obstacle_vertices =
            m_mesh.extract_obstacle_vertices(vertices);

        if (!broad_phase_obstacle)
            broad_phase_obstacle = std::make_shared<BVH>();
        broad_phase_obstacle->build(
            obstacle_vertices, m_mesh.obstacle_edges(), m_mesh.obstacle_faces(),
            inflation_radius);
    }

    detect_candidates(
        detect_elastic_obstacle_coll, detect_elastic_elastic_coll);
}

void CandidatesObstacleElastic::build(
    const Eigen::MatrixXd& vertices_t0,
    const Eigen::MatrixXd& vertices_t1,
    const double inflation_radius,
    bool updateObstacleBVH,
    bool detect_elastic_obstacle_coll,
    bool detect_elastic_elastic_coll)
{
    clear();

    // Construct bvh for elastic objects
    const Eigen::MatrixXd& elastic_vertices_t0 =
        m_mesh.extract_elastic_vertices(vertices_t0);
    const Eigen::MatrixXd& elastic_vertices_t1 =
        m_mesh.extract_elastic_vertices(vertices_t1);

    if (!broad_phase_elastic)
        broad_phase_elastic = std::make_shared<BVH>();
    broad_phase_elastic->build(
        elastic_vertices_t0, elastic_vertices_t1, m_mesh.elastic_edges(),
        m_mesh.elastic_faces(), inflation_radius);

    // Construct bvh for obstacle objects
    if (updateObstacleBVH) {
        const Eigen::MatrixXd& obstacle_vertices_t0 =
            m_mesh.extract_obstacle_vertices(vertices_t0);
        const Eigen::MatrixXd& obstacle_vertices_t1 =
            m_mesh.extract_obstacle_vertices(vertices_t1);

        if (!broad_phase_obstacle)
            broad_phase_obstacle = std::make_shared<BVH>();
        broad_phase_obstacle->build(
            obstacle_vertices_t0, obstacle_vertices_t1, m_mesh.obstacle_edges(),
            m_mesh.obstacle_faces(), inflation_radius);
    }

    detect_candidates(
        detect_elastic_obstacle_coll, detect_elastic_elastic_coll);
}

void CandidatesObstacleElastic::detect_candidates(
    bool detect_elastic_obstacle_coll, 
    bool detect_elastic_elastic_coll)
{
    clear();

    // Detect collision candidates between elastic objects
    if (detect_elastic_elastic_coll) {
        Candidates temp_cand;
        broad_phase_elastic->detect_collision_candidates(m_dim, temp_cand);

        tbb::enumerable_thread_specific<std::vector<EdgeEdgeCandidate>> ee_storage;
        tbb::enumerable_thread_specific<std::vector<FaceVertexCandidate>> fv_storage;

        tbb::parallel_for(
            tbb::blocked_range<size_t>(size_t(0), temp_cand.ee_candidates.size()),
            [&](const tbb::blocked_range<size_t>& r) {
                auto& local_candidates = ee_storage.local();
                for (size_t i = r.begin(); i < r.end(); i++) {
                    long& e0_id = temp_cand.ee_candidates[i].edge0_id;
                    long& e1_id = temp_cand.ee_candidates[i].edge1_id;
                    e0_id = m_mesh.elastic_edge_id_to_full_edge_id(e0_id);
                    e1_id = m_mesh.elastic_edge_id_to_full_edge_id(e1_id);
                    if (!(m_mesh.is_edge_bc(e0_id) && m_mesh.is_edge_bc(e1_id))) {
                        local_candidates.push_back(temp_cand.ee_candidates[i]);
                    }
                }
            });

        tbb::parallel_for(
            tbb::blocked_range<size_t>(size_t(0), temp_cand.fv_candidates.size()),
            [&](const tbb::blocked_range<size_t>& r) {
                auto& local_candidates = fv_storage.local();
                for (size_t i = r.begin(); i < r.end(); i++) {
                    long& f_id = temp_cand.fv_candidates[i].face_id;
                    long& v_id = temp_cand.fv_candidates[i].vertex_id;
                    f_id = m_mesh.elastic_face_id_to_full_face_id(f_id);
                    v_id = m_mesh.elastic_vertex_id_to_full_vertex_id(v_id);
                    if (!(m_mesh.is_face_bc(f_id) && m_mesh.is_vertex_bc(v_id))) {
                        local_candidates.push_back(temp_cand.fv_candidates[i]);
                    }
                }
            });

        merge_thread_local_vectors(ee_storage, ee_candidates);
        merge_thread_local_vectors(fv_storage, fv_candidates);
    }


    // Detect collision candidates between elastic objects and obstacles
    if (detect_elastic_obstacle_coll) {
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
           
        tbb::enumerable_thread_specific<std::vector<EdgeEdgeCandidate>> ee_storage;
        tbb::enumerable_thread_specific<std::vector<FaceVertexCandidate>> fv_storage;
        tbb::enumerable_thread_specific<std::vector<FaceVertexCandidate>> vf_storage;

        tbb::parallel_for(
            tbb::blocked_range<size_t>(size_t(0), elastic_obstacle_ee_candidates.size()),
            [&](const tbb::blocked_range<size_t>& r) {
                auto& local_candidates = ee_storage.local();
                for (size_t i = r.begin(); i < r.end(); i++) {
                    long& e0_id = elastic_obstacle_ee_candidates[i].edge0_id;
                    long& e1_id = elastic_obstacle_ee_candidates[i].edge1_id;
                    e0_id = m_mesh.elastic_edge_id_to_full_edge_id(e0_id);
                    e1_id = m_mesh.obstacle_edge_id_to_full_edge_id(e1_id);
                    if (!(m_mesh.is_edge_bc(e0_id) && m_mesh.is_edge_bc(e1_id))) {
                        local_candidates.push_back(elastic_obstacle_ee_candidates[i]);
                    }
                }
            });

        tbb::parallel_for(
            tbb::blocked_range<size_t>(size_t(0), elastic_obstacle_vf_candidates.size()),
            [&](const tbb::blocked_range<size_t>& r) {
                auto& local_candidates = vf_storage.local();
                for (size_t i = r.begin(); i < r.end(); i++) {
                    long& f_id = elastic_obstacle_vf_candidates[i].face_id;
                    long& v_id = elastic_obstacle_vf_candidates[i].vertex_id;
                    f_id = m_mesh.obstacle_face_id_to_full_face_id(f_id);
                    v_id = m_mesh.elastic_vertex_id_to_full_vertex_id(v_id);
                    if (!(m_mesh.is_face_bc(f_id) && m_mesh.is_vertex_bc(v_id))) {
                        local_candidates.push_back(elastic_obstacle_vf_candidates[i]);
                    }
                }
            });

        tbb::parallel_for(
            tbb::blocked_range<size_t>(size_t(0), elastic_obstacle_fv_candidates.size()),
            [&](const tbb::blocked_range<size_t>& r) {
                auto& local_candidates = fv_storage.local();
                for (size_t i = r.begin(); i < r.end(); i++) {
                    long& f_id = elastic_obstacle_fv_candidates[i].face_id;
                    long& v_id = elastic_obstacle_fv_candidates[i].vertex_id;
                    f_id = m_mesh.elastic_face_id_to_full_face_id(f_id);
                    v_id = m_mesh.obstacle_vertex_id_to_full_vertex_id(v_id);
                    if (!(m_mesh.is_face_bc(f_id) && m_mesh.is_vertex_bc(v_id))) {
                        local_candidates.push_back(elastic_obstacle_fv_candidates[i]);
                    }
                }
            });

        merge_thread_local_vectors(ee_storage, ee_candidates);
        merge_thread_local_vectors(vf_storage, fv_candidates);
        merge_thread_local_vectors(fv_storage, fv_candidates);

        //// merge to final candidates
        //ee_candidates.insert(
        //    ee_candidates.end(), elastic_obstacle_ee_candidates.begin(),
        //    elastic_obstacle_ee_candidates.end());

        //fv_candidates.insert(
        //    fv_candidates.end(), elastic_obstacle_fv_candidates.begin(),
        //    elastic_obstacle_fv_candidates.end());

        //fv_candidates.insert(
        //    fv_candidates.end(), elastic_obstacle_vf_candidates.begin(),
        //    elastic_obstacle_vf_candidates.end());
    }
}

bool CandidatesObstacleElastic::detect_elastic_collisions(
    const Eigen::MatrixXd& vertices,
    double inflation_radius,
    bool detect_elastic_obstacle_coll, bool detect_elastic_elastic_coll)
{
    m_intersect_elastic_edge_ids.clear();
    m_intersect_elastic_face_ids.clear();

    const Eigen::MatrixXd elastic_vertices =
        m_mesh.extract_elastic_vertices(vertices);

    // Construct bvh for elastic objects
    std::shared_ptr<BVH>  temp_broad_phase_elastic = std::make_shared<BVH>();
    temp_broad_phase_elastic->build(
        elastic_vertices, m_mesh.elastic_edges(), m_mesh.elastic_faces(),
        inflation_radius);

    // Construct bvh for obstacle objects
    const Eigen::MatrixXd obstacle_vertices =
        m_mesh.extract_obstacle_vertices(vertices);

    std::shared_ptr<BVH>  temp_broad_phase_obstacle = std::make_shared<BVH>();
    temp_broad_phase_obstacle->build(
        obstacle_vertices, m_mesh.obstacle_edges(), m_mesh.obstacle_faces(),
        inflation_radius);

    std::vector<EdgeFaceCandidate> ef_candidates;

    // Detect collision candidates between elastic objects
    if (detect_elastic_elastic_coll) {
        std::vector<EdgeFaceCandidate> elastic_ef_candidates;
        temp_broad_phase_elastic->detect_edge_face_candidates(elastic_ef_candidates);

        tbb::enumerable_thread_specific<std::vector<EdgeFaceCandidate>>
            ef_storage;

        tbb::parallel_for(
            tbb::blocked_range<size_t>(size_t(0), elastic_ef_candidates.size()),
            [&](const tbb::blocked_range<size_t>& r) {
                auto& local_candidates = ef_storage.local();
                for (size_t i = r.begin(); i < r.end(); i++) {
                    long& e_id = elastic_ef_candidates[i].edge_id;
                    long& f_id = elastic_ef_candidates[i].face_id;
                    e_id = m_mesh.elastic_edge_id_to_full_edge_id(e_id);
                    f_id = m_mesh.elastic_face_id_to_full_face_id(f_id);
                    if (!(m_mesh.is_edge_bc(e_id) && m_mesh.is_face_bc(f_id))) {
                        local_candidates.push_back(elastic_ef_candidates[i]);
                    }
                }
            });

        merge_thread_local_vectors(ef_storage, ef_candidates);

        for (const auto& [e_id, f_id] : ef_candidates) {
            if (is_edge_intersecting_triangle(
                    vertices.row(m_mesh.edges()(e_id, 0)),
                    vertices.row(m_mesh.edges()(e_id, 1)),
                    vertices.row(m_mesh.faces()(f_id, 0)),
                    vertices.row(m_mesh.faces()(f_id, 1)),
                    vertices.row(m_mesh.faces()(f_id, 2)))) {
                m_intersect_elastic_edge_ids.push_back(
                    m_mesh.elastic_full_edge_id_to_edge_id(e_id));
                m_intersect_elastic_face_ids.push_back(
                    m_mesh.elastic_full_face_id_to_face_id(f_id));
            }
        }
    }

    // Detect collision candidates between elastic objects and obstacles
    if (detect_elastic_obstacle_coll) {
        const std::vector<AABB>& elastic_face_boxes =
            temp_broad_phase_elastic->get_face_boxes();
        const std::vector<AABB>& obstacle_face_boxes =
            temp_broad_phase_obstacle->get_face_boxes();

        std::vector<EdgeFaceCandidate> elastic_obstacle_ef_candidates;
        std::vector<EdgeFaceCandidate> elastic_obstacle_fe_candidates;

        temp_broad_phase_elastic->detect_input_edge_face_candidates(
            elastic_obstacle_ef_candidates, obstacle_face_boxes);
        temp_broad_phase_obstacle->detect_input_edge_face_candidates(
            elastic_obstacle_fe_candidates, elastic_face_boxes);

        tbb::enumerable_thread_specific<std::vector<EdgeFaceCandidate>>
            ef_storage;
        tbb::enumerable_thread_specific<std::vector<EdgeFaceCandidate>>
            fe_storage;

        tbb::parallel_for(
            tbb::blocked_range<size_t>(
                size_t(0), elastic_obstacle_ef_candidates.size()),
            [&](const tbb::blocked_range<size_t>& r) {
                auto& local_candidates = ef_storage.local();
                for (size_t i = r.begin(); i < r.end(); i++) {
                    long& e_id = elastic_obstacle_ef_candidates[i].edge_id;
                    long& f_id = elastic_obstacle_ef_candidates[i].face_id;
                    e_id = m_mesh.elastic_edge_id_to_full_edge_id(e_id);
                    f_id = m_mesh.obstacle_face_id_to_full_face_id(f_id);
                    if (!(m_mesh.is_edge_bc(e_id) && m_mesh.is_face_bc(f_id))) {
                        local_candidates.push_back(
                            elastic_obstacle_ef_candidates[i]);
                    }
                }
            });

        tbb::parallel_for(
            tbb::blocked_range<size_t>(
                size_t(0), elastic_obstacle_fe_candidates.size()),
            [&](const tbb::blocked_range<size_t>& r) {
                auto& local_candidates = fe_storage.local();
                for (size_t i = r.begin(); i < r.end(); i++) {
                    long& e_id = elastic_obstacle_fe_candidates[i].edge_id;
                    long& f_id = elastic_obstacle_fe_candidates[i].face_id;
                    e_id = m_mesh.obstacle_edge_id_to_full_edge_id(e_id);
                    f_id = m_mesh.elastic_face_id_to_full_face_id(f_id);
                    if (!(m_mesh.is_edge_bc(e_id) && m_mesh.is_face_bc(f_id))) {
                        local_candidates.push_back(
                            elastic_obstacle_fe_candidates[i]);
                    }
                }
            });

        ef_candidates.clear();
        merge_thread_local_vectors(ef_storage, ef_candidates);
        for (const auto& [e_id, f_id] : ef_candidates) {
            if (is_edge_intersecting_triangle(
                    vertices.row(m_mesh.edges()(e_id, 0)),
                    vertices.row(m_mesh.edges()(e_id, 1)),
                    vertices.row(m_mesh.faces()(f_id, 0)),
                    vertices.row(m_mesh.faces()(f_id, 1)),
                    vertices.row(m_mesh.faces()(f_id, 2)))) {
                m_intersect_elastic_edge_ids.push_back(
                    m_mesh.elastic_full_edge_id_to_edge_id(e_id));
            }
        }

        ef_candidates.clear();
        merge_thread_local_vectors(fe_storage, ef_candidates);
        for (const auto& [e_id, f_id] : ef_candidates) {
            if (is_edge_intersecting_triangle(
                    vertices.row(m_mesh.edges()(e_id, 0)),
                    vertices.row(m_mesh.edges()(e_id, 1)),
                    vertices.row(m_mesh.faces()(f_id, 0)),
                    vertices.row(m_mesh.faces()(f_id, 1)),
                    vertices.row(m_mesh.faces()(f_id, 2)))) {
                m_intersect_elastic_face_ids.push_back(
                    m_mesh.elastic_full_face_id_to_face_id(f_id));
            }
        }
    }

    if (m_intersect_elastic_edge_ids.size() > 0
        || m_intersect_elastic_face_ids.size() > 0)
        return true;
    return false;
}

} // namespace ipc
