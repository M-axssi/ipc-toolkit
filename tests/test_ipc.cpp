#include <catch2/catch.hpp>

#include <iostream>

#include <finitediff.hpp>

#include <ipc.hpp>

#include "test_utils.hpp"

using namespace ipc;

TEST_CASE("Dummy test for IPC compilation", "[ipc]")
{
    double dhat_squared = -1;
    std::string mesh_name;

    SECTION("cube")
    {
        dhat_squared = 2.0;
        mesh_name = "cube.obj";
    }
    SECTION("bunny")
    {
        dhat_squared = 1e-4;
        mesh_name = "bunny.obj";
    }

    Eigen::MatrixXd V;
    Eigen::MatrixXi E, F;
    bool success = load_mesh(mesh_name, V, E, F);
    REQUIRE(success);

    ccd::Candidates constraint_set;
    ipc::construct_constraint_set(V, E, F, dhat_squared, constraint_set);
    CAPTURE(mesh_name, dhat_squared);
    CHECK(constraint_set.ee_candidates.size() > 0);
    CHECK(constraint_set.fv_candidates.size() > 0);

    double b = ipc::compute_barrier_potential(
        V, V, E, F, constraint_set, dhat_squared);
    Eigen::VectorXd grad_b = ipc::compute_barrier_potential_gradient(
        V, V, E, F, constraint_set, dhat_squared);
    Eigen::MatrixXd hess_b = ipc::compute_barrier_potential_hessian(
        V, V, E, F, constraint_set, dhat_squared);
}

TEST_CASE("Test IPC full gradient", "[ipc][grad]")
{
    double dhat_squared = -1;
    std::string mesh_name;

    SECTION("cube")
    {
        dhat_squared = 2.0;
        mesh_name = "cube.obj";
    }
    SECTION("two cubes far")
    {
        dhat_squared = 1e-2;
        mesh_name = "two-cubes-far.obj";
    }
    SECTION("two cubes close")
    {
        dhat_squared = 1e-2;
        mesh_name = "two-cubes-close.obj";
    }
    // SECTION("bunny")
    // {
    //     dhat_squared = 1e-4;
    //     mesh_name = "bunny.obj";
    // }

    Eigen::MatrixXd V;
    Eigen::MatrixXi E, F;
    bool success = load_mesh(mesh_name, V, E, F);
    REQUIRE(success);

    ccd::Candidates constraint_set;
    ipc::construct_constraint_set(V, E, F, dhat_squared, constraint_set);
    CAPTURE(mesh_name, dhat_squared);
    CHECK(constraint_set.ee_candidates.size() > 0);
    CHECK(constraint_set.fv_candidates.size() > 0);

    Eigen::VectorXd grad_b = ipc::compute_barrier_potential_gradient(
        V, V, E, F, constraint_set, dhat_squared);

    // Compute the gradient using finite differences
    auto f = [&](const Eigen::VectorXd& x) {
        return ipc::compute_barrier_potential(
            V, unflatten(x, V.cols()), E, F, constraint_set, dhat_squared);
    };
    Eigen::VectorXd fgrad_b;
    fd::finite_gradient(flatten(V), f, fgrad_b);

    REQUIRE(grad_b.squaredNorm() > 1e-8);
    CHECK(fd::compare_gradient(grad_b, fgrad_b));
}

TEST_CASE("Test IPC full hessian", "[ipc][hess]")
{
    double dhat_squared = -1;
    std::string mesh_name = "blah.obj";

    // SECTION("cube")
    // {
    //     dhat_squared = 2.0;
    //     mesh_name = "cube.obj";
    // }
    SECTION("two cubes far")
    {
        dhat_squared = 1e-2;
        mesh_name = "two-cubes-far.obj";
    }
    SECTION("two cubes close")
    {
        dhat_squared = 1e-2;
        mesh_name = "two-cubes-close.obj";
    }
    // WARNING: The bunny takes too long in debug.
    // SECTION("bunny")
    // {
    //     dhat_squared = 1e-4;
    //     mesh_name = "bunny.obj";
    // }

    Eigen::MatrixXd V;
    Eigen::MatrixXi E, F;
    bool success = load_mesh(mesh_name, V, E, F);
    REQUIRE(success);

    ccd::Candidates constraint_set;
    ipc::construct_constraint_set(V, E, F, dhat_squared, constraint_set);
    CAPTURE(mesh_name, dhat_squared);
    REQUIRE(constraint_set.ee_candidates.size() > 0);
    REQUIRE(constraint_set.fv_candidates.size() > 0);

    Eigen::MatrixXd hess_b = ipc::compute_barrier_potential_hessian(
        /*V_rest=*/V, V, E, F, constraint_set, dhat_squared);

    // Compute the gradient using finite differences
    auto f = [&](const Eigen::VectorXd& x) {
        return ipc::compute_barrier_potential_gradient(
            V, unflatten(x, V.cols()), E, F, constraint_set, dhat_squared);
    };
    Eigen::MatrixXd fhess_b;
    fd::finite_jacobian(flatten(V), f, fhess_b);

    REQUIRE(hess_b.squaredNorm() > 1e-3);
    CHECK(fd::compare_hessian(hess_b, fhess_b, 1e-3));
}
