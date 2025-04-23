#define BOOST_TEST_MODULE SM_INSTANCE_SCHEDULE
#include <boost/test/unit_test.hpp>

#include <filesystem>
#include <string>
#include <vector>

#include <Eigen/SparseCore>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/SparseExtra>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/OrderingMethods>

#include "file_interactions/FileReader.hpp"
#include "model/SmInstance.hpp"
#include "model/SmSchedule.hpp"

#include "scheduler/GreedySchedulers/SMGreedyBspGrowLocalAutoCoresParallel.hpp"
#include "scheduler/GreedySchedulers/SMGreedyVarianceFillupScheduler.hpp"
#include "scheduler/GreedySchedulers/SMFunGrowlv2.hpp"
#include "scheduler/GreedySchedulers/SMFunOriGrowlv2.hpp"


std::vector<std::string> test_graphs() {
    return {"data/mtx_tests/ErdosRenyi_100_1k_A.mtx", "data/mtx_tests/ErdosRenyi_200_5k_A.mtx", "data/mtx_tests/ErdosRenyi_500_8k_A.mtx", "data/mtx_tests/ErdosRenyi_2k_14k_A.mtx", "data/mtx_tests/RandomBand_p70_b12_500_4k_A.mtx", "data/mtx_tests/RandomBand_p80_b5_100_419_A.mtx", "data/mtx_tests/RandomBand_p80_b14_200_2k_A.mtx", "data/mtx_tests/RandomBand_p40_b30_2k_23k_A.mtx" };
}

std::vector<std::string> test_architectures() {
    return {"data/machine_params/p3.txt", "data/machine_params/p16_g1_l5_numa2.txt", "data/machine_params/p3_g2_l100.txt"};
}


BOOST_AUTO_TEST_CASE(SMGraph_test) {
    std::vector<std::string> filenames_graph = test_graphs();

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    for (auto &filename_graph : filenames_graph) {
        std::string name_graph = filename_graph.substr(filename_graph.find_last_of("/\\") + 1);
        name_graph = name_graph.substr(0, name_graph.find_last_of("."));

        std::cout << "Graph: " << name_graph << std::endl;

        auto [status_graph, graph] =
            FileReader::readComputationalDagMartixMarketFormat((cwd / filename_graph).string());

        if (!status_graph) {
            std::cout << "Reading files failed." << std::endl;
            BOOST_CHECK(false);
        }

        std::cout << "Vertices " << graph.numberOfVertices() << std::endl;
        std::cout << "Edges " << graph.numberOfEdges() << std::endl;

        using SM_csc = Eigen::SparseMatrix<double, Eigen::ColMajor, int32_t>; // Compressed Sparse Column format
        using SM_csr = Eigen::SparseMatrix<double, Eigen::RowMajor, int32_t>; // Compressed Sparse Row format

        SM_csc L_csc; // Initialize a sparse matrix in CSC format

        Eigen::loadMarket(L_csc, (cwd / filename_graph).string());

        SM_csr L_csr = L_csc;   // Reformat the sparse matrix from CSC to CSR format

        SparseMatrix mat{};
        mat.setCSR(&L_csr);
        mat.setCSC(&L_csc);

        BOOST_CHECK_EQUAL(mat.numberOfVertices(), graph.numberOfVertices());
        BOOST_CHECK_EQUAL(mat.numberOfEdges(), graph.numberOfEdges());

        for (VertexType vert = 0; vert < graph.numberOfVertices(); vert++) {
            BOOST_CHECK_EQUAL(graph.numberOfChildren(vert), mat.numberOfChildren(vert));
            BOOST_CHECK_EQUAL(graph.numberOfParents(vert), mat.numberOfParents(vert));

            std::set<VertexType> children;
            std::set<VertexType> parents;

            for (const VertexType &child : graph.children(vert)) {
                children.emplace(child);
            }

            SM_csc::InnerIterator c_it(*(mat.getCSC()), vert);
            ++c_it;

            const unsigned int number_of_children = mat.numberOfChildren(vert);
            auto succ = static_cast<VertexType>(c_it.index());

            for (unsigned i = 0; i < number_of_children; ++i){
                succ = static_cast<VertexType>(c_it.index());
                BOOST_CHECK(children.find(succ) != children.end());
            }

            for (const VertexType &parent : graph.parents(vert)) {
                parents.emplace(parent);
            }

            SM_csr::InnerIterator par_it(*(mat.getCSR()) , vert);
            unsigned int number_of_parents = mat.numberOfParents(vert);
            auto par = static_cast<VertexType>(par_it.index()); 
            for(unsigned int k=0; k<number_of_parents; ++k) {
                par = static_cast<VertexType>(par_it.index());
                BOOST_CHECK(parents.find(par) != parents.end());
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(SM_GrowLocalAutoCoresParallel_test) {
    SMGreedyBspGrowLocalAutoCoresParallel test_scheduler;


    std::vector<std::string> filenames_graph = test_graphs();
    std::vector<std::string> filenames_architectures = test_architectures();

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    for (auto &filename_graph : filenames_graph) {
        for (auto &filename_machine : filenames_architectures) {
            std::string name_graph = filename_graph.substr(filename_graph.find_last_of("/\\") + 1);
            name_graph = name_graph.substr(0, name_graph.find_last_of("."));
            std::string name_machine = filename_machine.substr(filename_machine.find_last_of("/\\") + 1);
            name_machine = name_machine.substr(0, name_machine.rfind("."));

            std::cout << std::endl << "Scheduler: " << test_scheduler.getScheduleName() << std::endl; 
            std::cout << "Graph: " << name_graph << std::endl;
            std::cout << "Architecture: " << name_machine << std::endl;

            auto [status_architecture, architecture] =
                FileReader::readBspArchitecture((cwd / filename_machine).string());

            using SM_csc = Eigen::SparseMatrix<double, Eigen::ColMajor, int32_t>; // Compressed Sparse Column format
            using SM_csr = Eigen::SparseMatrix<double, Eigen::RowMajor, int32_t>; // Compressed Sparse Row format
    
            SM_csc L_csc; // Initialize a sparse matrix in CSC format
    
            Eigen::loadMarket(L_csc, (cwd / filename_graph).string());
    
            SM_csr L_csr = L_csc;   // Reformat the sparse matrix from CSC to CSR format
    
            SparseMatrix mat{};
            mat.setCSR(&L_csr);
            mat.setCSC(&L_csc);

            if (!status_architecture) {

                std::cout << "Reading files failed." << std::endl;
                BOOST_CHECK(false);
            }

            std::cout << "Vertices " << mat.numberOfVertices() << std::endl;
            std::cout << "Edges " << mat.numberOfEdges() << std::endl;

            SmInstance instance(mat, architecture);

            for (unsigned threads = 1; threads < 11; threads++) {
                std::cout << "Number of threads: " << threads << std::endl;

                std::pair<RETURN_STATUS, SmSchedule> result = test_scheduler.computeScheduleParallel(instance, threads);

                BOOST_CHECK_EQUAL(SUCCESS, result.first);
                BOOST_CHECK_EQUAL(&result.second.getInstance(), &instance);
                BOOST_CHECK(result.second.noOutOfBounds());
                BOOST_CHECK(result.second.satisfiesPrecedenceConstraints());
            }
        }
    }
};


BOOST_AUTO_TEST_CASE(SM_GreedyVariance_test) {
    SMGreedyVarianceFillupScheduler test_scheduler;


    std::vector<std::string> filenames_graph = test_graphs();
    std::vector<std::string> filenames_architectures = test_architectures();

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    for (auto &filename_graph : filenames_graph) {
        for (auto &filename_machine : filenames_architectures) {
            std::string name_graph = filename_graph.substr(filename_graph.find_last_of("/\\") + 1);
            name_graph = name_graph.substr(0, name_graph.find_last_of("."));
            std::string name_machine = filename_machine.substr(filename_machine.find_last_of("/\\") + 1);
            name_machine = name_machine.substr(0, name_machine.rfind("."));

            std::cout << std::endl << "Scheduler: " << test_scheduler.getScheduleName() << std::endl; 
            std::cout << "Graph: " << name_graph << std::endl;
            std::cout << "Architecture: " << name_machine << std::endl;

            auto [status_architecture, architecture] =
                FileReader::readBspArchitecture((cwd / filename_machine).string());

            using SM_csc = Eigen::SparseMatrix<double, Eigen::ColMajor, int32_t>; // Compressed Sparse Column format
            using SM_csr = Eigen::SparseMatrix<double, Eigen::RowMajor, int32_t>; // Compressed Sparse Row format
    
            SM_csc L_csc; // Initialize a sparse matrix in CSC format
    
            Eigen::loadMarket(L_csc, (cwd / filename_graph).string());
    
            SM_csr L_csr = L_csc;   // Reformat the sparse matrix from CSC to CSR format
    
            SparseMatrix mat{};
            mat.setCSR(&L_csr);
            mat.setCSC(&L_csc);

            if (!status_architecture) {

                std::cout << "Reading files failed." << std::endl;
                BOOST_CHECK(false);
            }

            std::cout << "Vertices " << mat.numberOfVertices() << std::endl;
            std::cout << "Edges " << mat.numberOfEdges() << std::endl;

            SmInstance instance(mat, architecture);

            std::pair<RETURN_STATUS, SmSchedule> result = test_scheduler.computeSmSchedule(instance);

            BOOST_CHECK_EQUAL(SUCCESS, result.first);
            BOOST_CHECK_EQUAL(&result.second.getInstance(), &instance);
            BOOST_CHECK(result.second.noOutOfBounds());
            BOOST_CHECK(result.second.satisfiesPrecedenceConstraints());
        }
    }
};



BOOST_AUTO_TEST_CASE(SM_FunnelGrowlv2_test) {
    SMFunGrowlv2 test_scheduler;


    std::vector<std::string> filenames_graph = test_graphs();
    std::vector<std::string> filenames_architectures = test_architectures();

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    for (auto &filename_graph : filenames_graph) {
        for (auto &filename_machine : filenames_architectures) {
            std::string name_graph = filename_graph.substr(filename_graph.find_last_of("/\\") + 1);
            name_graph = name_graph.substr(0, name_graph.find_last_of("."));
            std::string name_machine = filename_machine.substr(filename_machine.find_last_of("/\\") + 1);
            name_machine = name_machine.substr(0, name_machine.rfind("."));

            std::cout << std::endl << "Scheduler: " << test_scheduler.getScheduleName() << std::endl; 
            std::cout << "Graph: " << name_graph << std::endl;
            std::cout << "Architecture: " << name_machine << std::endl;

            auto [status_architecture, architecture] =
                FileReader::readBspArchitecture((cwd / filename_machine).string());

            using SM_csc = Eigen::SparseMatrix<double, Eigen::ColMajor, int32_t>; // Compressed Sparse Column format
            using SM_csr = Eigen::SparseMatrix<double, Eigen::RowMajor, int32_t>; // Compressed Sparse Row format
    
            SM_csc L_csc; // Initialize a sparse matrix in CSC format
    
            Eigen::loadMarket(L_csc, (cwd / filename_graph).string());
    
            SM_csr L_csr = L_csc;   // Reformat the sparse matrix from CSC to CSR format
    
            SparseMatrix mat{};
            mat.setCSR(&L_csr);
            mat.setCSC(&L_csc);

            if (!status_architecture) {

                std::cout << "Reading files failed." << std::endl;
                BOOST_CHECK(false);
            }

            std::cout << "Vertices " << mat.numberOfVertices() << std::endl;
            std::cout << "Edges " << mat.numberOfEdges() << std::endl;

            SmInstance instance(mat, architecture);

            for (unsigned threads = 1; threads < 11; threads++) {
                std::cout << "Number of threads: " << threads << std::endl;

                std::pair<RETURN_STATUS, SmSchedule> result = test_scheduler.computeScheduleParallel(instance, threads);

                BOOST_CHECK_EQUAL(SUCCESS, result.first);
                BOOST_CHECK_EQUAL(&result.second.getInstance(), &instance);
                BOOST_CHECK(result.second.noOutOfBounds());
                BOOST_CHECK(result.second.satisfiesPrecedenceConstraints());
            }
        }
    }
};

BOOST_AUTO_TEST_CASE(SM_FunnelOriginalGrowlv2_test) {
    SMFunOriGrowlv2 test_scheduler;


    std::vector<std::string> filenames_graph = test_graphs();
    std::vector<std::string> filenames_architectures = test_architectures();

    // Getting root git directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << cwd << std::endl;
    while ((!cwd.empty()) && (cwd.filename() != "OneStopParallel")) {
        cwd = cwd.parent_path();
        std::cout << cwd << std::endl;
    }

    for (auto &filename_graph : filenames_graph) {
        for (auto &filename_machine : filenames_architectures) {
            std::string name_graph = filename_graph.substr(filename_graph.find_last_of("/\\") + 1);
            name_graph = name_graph.substr(0, name_graph.find_last_of("."));
            std::string name_machine = filename_machine.substr(filename_machine.find_last_of("/\\") + 1);
            name_machine = name_machine.substr(0, name_machine.rfind("."));

            std::cout << std::endl << "Scheduler: " << test_scheduler.getScheduleName() << std::endl; 
            std::cout << "Graph: " << name_graph << std::endl;
            std::cout << "Architecture: " << name_machine << std::endl;

            auto [status_architecture, architecture] =
                FileReader::readBspArchitecture((cwd / filename_machine).string());

            using SM_csc = Eigen::SparseMatrix<double, Eigen::ColMajor, int32_t>; // Compressed Sparse Column format
            using SM_csr = Eigen::SparseMatrix<double, Eigen::RowMajor, int32_t>; // Compressed Sparse Row format
    
            SM_csc L_csc; // Initialize a sparse matrix in CSC format
    
            Eigen::loadMarket(L_csc, (cwd / filename_graph).string());
    
            SM_csr L_csr = L_csc;   // Reformat the sparse matrix from CSC to CSR format
    
            SparseMatrix mat{};
            mat.setCSR(&L_csr);
            mat.setCSC(&L_csc);

            if (!status_architecture) {

                std::cout << "Reading files failed." << std::endl;
                BOOST_CHECK(false);
            }

            std::cout << "Vertices " << mat.numberOfVertices() << std::endl;
            std::cout << "Edges " << mat.numberOfEdges() << std::endl;

            SmInstance instance(mat, architecture);

            for (unsigned threads = 1; threads < 11; threads++) {
                std::cout << "Number of threads: " << threads << std::endl;

                std::pair<RETURN_STATUS, SmSchedule> result = test_scheduler.computeScheduleParallel(instance, threads);

                BOOST_CHECK_EQUAL(SUCCESS, result.first);
                BOOST_CHECK_EQUAL(&result.second.getInstance(), &instance);
                BOOST_CHECK(result.second.noOutOfBounds());
                BOOST_CHECK(result.second.satisfiesPrecedenceConstraints());
            }
        }
    }
};