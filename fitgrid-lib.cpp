#include "hdcalib.h"

#include <opencv2/highgui.hpp>

namespace {

template<class T>
void rot_vec2mat(const T vec[], T mat[]) {
    T const theta = ceres::sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
    T const c = ceres::cos(theta);
    T const s = ceres::sin(theta);
    T const c1 = T(1) - c;

    // Calculate normalized vector.
    T const factor = (ceres::abs(theta) < std::numeric_limits<double>::epsilon() ? T(1) : T(1)/theta);
    T const vec_norm[3] = {factor * vec[0], factor * vec[1], factor * vec[2]};

    mat[0] = c + c1*vec_norm[0]*vec_norm[0];
    mat[1] = c1*vec_norm[0]*vec_norm[1] - s*vec_norm[2];
    mat[2] = c1*vec_norm[0]*vec_norm[2] + s*vec_norm[1];

    mat[3] = c1*vec_norm[0]*vec_norm[1] + s*vec_norm[2];
    mat[4] = c + c1*vec_norm[1]*vec_norm[1];
    mat[5] = c1*vec_norm[1]*vec_norm[2] - s*vec_norm[0];

    mat[6] = c1*vec_norm[0]*vec_norm[2] - s*vec_norm[1];
    mat[7] = c1*vec_norm[1]*vec_norm[2] + s*vec_norm[0];
    mat[8] = c + c1*vec_norm[2]*vec_norm[2];
}

template<class F, class T>
void rotate_translate(
        F const p[3],
T result[3],
const T r_vec[3],
const T t[3]
) {
    T R[9];
    rot_vec2mat(r_vec, R);
    T const X(p[0]), Y(p[1]), Z(p[2]);
    T& x = result[0];
    T& y = result[1];
    T& z = result[2];
    x = R[0]*X + R[1]*Y + R[2]*Z + t[0];
    y = R[3]*X + R[4]*Y + R[5]*Z + t[1];
    z = R[6]*X + R[7]*Y + R[8]*Z + t[2];
}

struct GridFitCost {
    cv::Point3f src, dst;

    GridFitCost(cv::Point3f _src, cv::Point3f _dst) : src(_src), dst(_dst) {}

    template<class T>
    bool operator()(T const * const scale,
                    T const * const rvec,
                    T const * const tvec,
                    T * residuals) const {
        T pt[3] = {scale[0]*T(src.x), scale[0]*T(src.y), scale[0]*T(src.z)};
        rotate_translate(pt, residuals, rvec, tvec);
        residuals[0] -= dst.x;
        residuals[1] -= dst.y;
        residuals[2] -= dst.z;
        return true;
    }
};

} // anonymous namespace


namespace hdcalib {

//                               grid prefix           suffix       vector of 3D marker locations
void FitGrid::findGrids(std::map<std::string, std::map<std::string, std::vector<cv::Point3f> > > & detected_grids,
                        GridDescription const& desc,
                        Calib & calib, CalibResult &calib_result,
                        std::vector<cv::Point3f> initial_points) {

    for (std::string const& filename : calib.getImageNames()) {
        for (GridPointDesc const& pt : desc.points) {
            if (filename.size() < pt.suffix.size()) {
                continue;
            }
            std::string const suffix = filename.substr(filename.size() - pt.suffix.size());
            if (suffix != pt.suffix) {
                continue;
            }
            std::string const prefix = filename.substr(0, filename.size() - pt.suffix.size());
            for (size_t ii = 0; ii < initial_points.size(); ++ii) {
                std::vector<cv::Point3f>& transformed_pts = detected_grids[prefix][suffix];
                transformed_pts.push_back(hdcalib::Calib::getTransformedPoint(calib_result, filename, initial_points[ii]));
            }
        }
    }
}

void FitGrid::runFit(Calib &calib, CalibResult& calib_result, const std::vector<GridDescription> &desc) {
    cv::Rect_<int> const area = calib.getIdRectangleUnion();

    // Number of grid points on x axis.
    double const num_x = 3;
    // Number of grid points on y axis. Total number of points is num_x * num_y
    double const num_y = 3;

    /**
     * @brief target_pts holds the initial 3D coordinates of the 3D markers used in the fit.
     */
    std::vector<cv::Point3f> known_grid_pts;
    for (double xx = 0; xx <= num_x+.1; ++xx) {
        for (double yy = 0; yy <= num_y+.1; ++yy) {
            hdmarker::Corner c;
            c.page = 0;
            c.id.x = std::round(area.x + area.width * xx / num_x);
            c.id.y = std::round(area.y + area.width * yy / num_y);
            known_grid_pts.push_back(calib.getInitial3DCoord(c));
        }
    }

    // desc index        grid prefix           suffix       vector of 3D marker locations
    std::vector<std::map<std::string, std::map<std::string, std::vector<cv::Point3f> > > > detected_grids(desc.size());

    // desc index        grid prefix  rvec
    std::vector<std::map<std::string, cv::Vec3d> > rvecs(desc.size());

    // desc index        grid prefix  vector of tvecs, one for each marker on the target
    std::vector<std::map<std::string, std::vector<cv::Vec3d> > > tvecs(desc.size());

    double scale = 102.24;

    for (size_t grid = 0; grid < desc.size(); ++grid) {
        findGrids(detected_grids[grid], desc[grid], calib, calib_result, known_grid_pts);
    }

    for (size_t ii = 0; ii < desc.size(); ++ii) {
        std::cout << "Grid name: " << desc[ii].name << std::endl
                  << "#points: " << desc[ii].points.size() << std::endl
                  << "#detected: " << detected_grids[ii].size() << std::endl
                  << "sizes: " << std::endl;
        for (auto const& it : detected_grids[ii]) {
            std::cout << it.second.size() << "\t" << it.first << std::endl;
        }
        std::cout << std::endl;
    }

    ceres::Problem problem;

    for (size_t ii = 0; ii < desc.size(); ++ii) { // Debug output only
        std::cout << "Grid name: " << desc[ii].name << std::endl
                  << "#points: " << desc[ii].points.size() << std::endl
                  << "#detected: " << detected_grids[ii].size() << std::endl
                  << "sizes: " << std::endl;
        for (auto const& it : detected_grids[ii]) {
            std::cout << it.second.size() << "\t" << it.first << std::endl;
        }
        std::cout << std::endl;
    }

    for (size_t ii = 0; ii < desc.size(); ++ii) { // Initialize all the rvecs and tvecs in the nested containers.
        GridDescription const& grid_desc = desc[ii];
        for (auto const& it : detected_grids[ii]) {
            std::string const& grid_prefix = it.first;
            std::map<std::string, std::vector<cv::Point3f> > const& grid = it.second;
            for (auto const& it2 : grid) {
                std::string const& suffix = it2.first;
                std::vector<cv::Point3f> const& target_pts = it2.second;

                GridPointDesc const& pt_desc = grid_desc.getDesc(suffix);

                rvecs[ii][grid_prefix] = cv::Vec3d(0.1, 0.1, 0.1);
                rvecs[ii][grid_prefix][0] = 0.11;
                rvecs[ii][grid_prefix][1] = 0.12;
                rvecs[ii][grid_prefix][2] = 0.13;
                tvecs[ii][grid_prefix] = std::vector<cv::Vec3d>(target_pts.size(), cv::Vec3d(0,0,0));
            }
        }
    }

    for (size_t ii = 0; ii < desc.size(); ++ii) { // Initialize all the rvecs and tvecs in the nested containers.
        GridDescription const& grid_desc = desc[ii];
        for (auto const& it : detected_grids[ii]) {
            std::string const& grid_prefix = it.first;
            std::map<std::string, std::vector<cv::Point3f> > const& grid = it.second;
            for (auto const& it2 : grid) {
                std::string const& suffix = it2.first;
                std::vector<cv::Point3f> const& target_pts = it2.second;

                GridPointDesc const& pt_desc = grid_desc.getDesc(suffix);

                assert(target_pts.size() == tvecs[ii][grid_prefix].size());
                auto const rvec_it = rvecs[ii].find(grid_prefix);
                assert(rvec_it != rvecs[ii].end());
                cv::Vec3d & r_vec = rvecs[ii][grid_prefix];
                assert(r_vec.dot(r_vec) > 0.0001);
                for (size_t jj = 0; jj < target_pts.size(); ++jj) {
                    problem.AddResidualBlock(
                                new ceres::AutoDiffCostFunction<GridFitCost, 3, 1, 3, 3>(
                                    new GridFitCost(target_pts[ii], pt_desc.getPt())
                                    ),
                                nullptr, // loss function
                                &scale,
                                r_vec.val,
                                tvecs[ii][grid_prefix][jj].val
                                );
                }
            }
        }
    }

    ceres::Solver::Options options;
    options.num_threads = int(calib.threads);
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.max_num_iterations = 150*1000;
    options.minimizer_progress_to_stdout = true;
    options.function_tolerance = ceres_tolerance;
    options.gradient_tolerance = ceres_tolerance;
    options.parameter_tolerance = ceres_tolerance;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    clog::L(__func__, 1) << summary.BriefReport() << "\n";
    clog::L(__func__, 1) << summary.FullReport() << "\n";

    std::vector<std::map<std::string, runningstats::QuantileStats<float> > > per_grid_stats_x(desc.size());
    std::vector<std::map<std::string, runningstats::QuantileStats<float> > > per_grid_stats_y(desc.size());
    std::vector<std::map<std::string, runningstats::QuantileStats<float> > > per_grid_stats_z(desc.size());

    std::vector<runningstats::QuantileStats<float> > per_grid_type_stats_x(desc.size());
    std::vector<runningstats::QuantileStats<float> > per_grid_type_stats_y(desc.size());
    std::vector<runningstats::QuantileStats<float> > per_grid_type_stats_z(desc.size());

    std::multimap<double, std::string> max_errors_per_image;

    for (size_t ii = 0; ii < desc.size(); ++ii) { // Evaluate residuals.
        GridDescription const& grid_desc = desc[ii];
        for (auto const& it : detected_grids[ii]) {
            std::string const& grid_prefix = it.first;
            std::map<std::string, std::vector<cv::Point3f> > const& grid = it.second;
            for (auto const& it2 : grid) {
                std::string const& suffix = it2.first;
                std::vector<cv::Point3f> const& target_pts = it2.second;

                GridPointDesc const& pt_desc = grid_desc.getDesc(suffix);

                double max_error_length = 0;
                auto const rvec_it = rvecs[ii].find(grid_prefix);
                assert(rvec_it != rvecs[ii].end());
                cv::Vec3d & r_vec = rvecs[ii][grid_prefix];
                for (size_t jj = 0; jj < target_pts.size(); ++jj) {
                    GridFitCost cost (target_pts[ii], pt_desc.getPt());
                    cv::Vec3d residual(0,0,0);
                    cost(&scale,
                         r_vec.val,
                         tvecs[ii][grid_prefix][jj].val,
                         residual.val
                         );
                    per_grid_stats_x[ii][grid_prefix].push_unsafe(residual[0]);
                    per_grid_stats_y[ii][grid_prefix].push_unsafe(residual[1]);
                    per_grid_stats_z[ii][grid_prefix].push_unsafe(residual[2]);

                    per_grid_type_stats_x[ii].push_unsafe(residual[0]);
                    per_grid_type_stats_y[ii].push_unsafe(residual[1]);
                    per_grid_type_stats_z[ii].push_unsafe(residual[2]);

                    double const error_length = std::sqrt(residual.dot(residual));
                    max_error_length = std::max(max_error_length, error_length);
                }
                max_errors_per_image.insert({max_error_length, grid_desc.name + ": " + grid_prefix + suffix});
            }

            /*
            std::cout << "Residual stats for grid " << grid_prefix << ": " << std::endl
                      << "x: " << per_grid_stats_x[ii][grid_prefix].print() << std::endl
                      << "y: " << per_grid_stats_y[ii][grid_prefix].print() << std::endl
                      << "z: " << per_grid_stats_z[ii][grid_prefix].print() << std::endl << std::endl;
                      */
        }
    }

    std::cout << "Max errors per image, sorted: " << std::endl;
    for (auto const& it : max_errors_per_image) {
        double const error = it.first;
        std::string const& filename = it.second;
        std::cout << std::setw(8) << error << "\t" << filename << std::endl;
    }
    std::cout << std::endl;

    for (size_t ii = 0; ii < desc.size(); ++ii) {
        std::cout << "Residual stats for grid type " << desc[ii].name << ": " << std::endl
                  << "x: " << per_grid_type_stats_x[ii].print() << std::endl
                  << "y: " << per_grid_type_stats_y[ii].print() << std::endl
                  << "z: " << per_grid_type_stats_z[ii].print() << std::endl << std::endl;
        runningstats::HistConfig conf;
        conf.setTitle(std::string("Residuals for Grid ") + desc[ii].name + ", x-axis")
                .setXLabel("Residual[mm]");
        per_grid_type_stats_x[ii].plotHist(desc[ii].name + "-stats-x", per_grid_type_stats_x[ii].FreedmanDiaconisBinSize(), conf);
        conf.setTitle(std::string("Residuals for Grid ") + desc[ii].name + ", y-axis");
        per_grid_type_stats_y[ii].plotHist(desc[ii].name + "-stats-y", per_grid_type_stats_y[ii].FreedmanDiaconisBinSize());
        conf.setTitle(std::string("Residuals for Grid ") + desc[ii].name + ", z-axis");
        per_grid_type_stats_z[ii].plotHist(desc[ii].name + "-stats-z", per_grid_type_stats_z[ii].FreedmanDiaconisBinSize());
    }

    std::cout << "estimated scale: " << scale << std::endl;

}

} // namespace hdcalib