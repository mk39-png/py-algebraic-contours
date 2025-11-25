
// These are .cpp files that were written to serialize various parts of the ASOC code for testing.


// This could be any matrix of any size. Scalable.
inline void serialize_eigen_matrix_f(std::string filename, Eigen::MatrixXf M) {
    spdlog::info("Writing matrix data to {}", filename);
    std::ofstream output_file(filename, std::ios::out | std::ios::trunc);

    int prec = 17;

    // Print out by row order
    for (Eigen::Index i = 0; i < M.rows(); ++i) {
        // Printing out the first element of the row separately since we do not want the comma at the start
        output_file << std::setprecision(prec) << M(i, 0);

        for (Eigen::Index j = 1; j < M.cols(); ++j) {
            output_file << std::setprecision(prec) << "," << M(i, j);
        }
        output_file << std::endl;
    }

    output_file.close();
}

