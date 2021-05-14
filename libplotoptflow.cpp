#include "libplotoptflow.h"

#include <ceres/ceres.h>

#include "gnuplot-iostream.h"

namespace hdflow {
namespace  {
bool hasColorWheel = false;
// colorcode.cpp
//
// Color encoding of flow vectors
// adapted from the color circle idea described at
//   http://members.shaw.ca/quadibloc/other/colint.htm
//
// Daniel Scharstein, 4/2007
// added tick marks and out-of-range coding 6/05/07
#define MAXCOLS 60
int colorwheel[MAXCOLS][3];
int ncols = 0;

void setcols(int r, int g, int b, int k) {
    colorwheel[k][0] = r;
    colorwheel[k][1] = g;
    colorwheel[k][2] = b;
}

void makecolorwheel() {
    if (hasColorWheel) {
        return;
    }
    hasColorWheel = true;
    // relative lengths of color transitions:
    // these are chosen based on perceptual similarity
    // (e.g. one can distinguish more shades between red and yellow
    //  than between yellow and green)
    int RY = 15;
    int YG = 6;
    int GC = 4;
    int CB = 11;
    int BM = 13;
    int MR = 6;
    ncols = RY + YG + GC + CB + BM + MR;
    //printf("ncols = %d\n", ncols);
    if (ncols > MAXCOLS)
        exit(1);
    int i;
    int k = 0;
    for (i = 0; i < RY; i++)
        setcols(255,	   255*i/RY,	 0,	       k++);
    for (i = 0; i < YG; i++)
        setcols(255-255*i/YG, 255,		 0,	       k++);
    for (i = 0; i < GC; i++)
        setcols(0,		   255,		 255*i/GC,     k++);
    for (i = 0; i < CB; i++)
        setcols(0,		   255-255*i/CB, 255,	       k++);
    for (i = 0; i < BM; i++)
        setcols(255*i/BM,	   0,		 255,	       k++);
    for (i = 0; i < MR; i++)
        setcols(255,	   0,		 255-255*i/MR, k++);
}


void computeColor(float fx, float fy, cv::Vec3b &pix) {
    if (!std::isfinite(fx) || !std::isfinite(fy) || std::abs(fx) > 1e6 || std::abs(fy) > 1e6) {
        pix[0] = pix[1] = pix[2] = 0;
        return;
    }

    float rad = sqrt(fx * fx + fy * fy);
    float a = atan2(-fy, -fx) / M_PI;
    float fk = (a + 1.0) / 2.0 * (ncols-1);
    int k0 = (int)fk;
    int k1 = (k0 + 1) % ncols;
    float f = fk - k0;
    //f = 0; // uncomment to see original color wheel
    for (int b = 0; b < 3; b++) {
        float col0 = colorwheel[k0][b] / 255.0;
        float col1 = colorwheel[k1][b] / 255.0;
        float col = (1 - f) * col0 + f * col1;
        if (rad <= 1) {
            col = 1 - rad * (1 - col); // increase saturation with radius
        }
        else {
            col *= .75; // out of range
        }
        pix[2 - b] = (int)(255.0 * col);
    }
}

} // anonymous namespace

namespace  {
template<typename T>
void applySensorTilt(
        T& x, T& y,
        T const& tau_x, T const& tau_y
        ) {
    T const s_x = ceres::sin(tau_x);
    T const s_y = ceres::sin(tau_y);
    T const c_x = ceres::cos(tau_x);
    T const c_y = ceres::cos(tau_y);

    T const x1 = c_y*x + s_x*s_y*y - s_y*c_x;
    T const y1 = c_x*y + s_x;
    T const z1 = s_y*x - c_y*s_x*y + c_y*c_x;

    x = (c_y*c_x*x1 + s_y*c_x*z1)/z1;
    y = (c_y*c_x*y1 - s_x*z1)/z1;
}
} // anonymous namespace

struct DistortionFunctor {
    cv::Vec2f src, dst, center;

    DistortionFunctor(cv::Vec2f const& _src, cv::Vec2f const& _dst, cv::Vec2f const& _center) :
        src(_src),
        dst(_dst),
        center(_center) {}

    template<class T>
    bool operator()(T const * const dist, T * residuals) const {
        T const _src[2] = {T(src[0]), T(src[1])};
        T const _center[2] = {T(center[0]), T(center[1])};
        T _dst[2] = {T(0),T(0)};
        applyDist(_src, _dst, _center, dist);
        residuals[0] = _dst[0] - T(dst[0]);
        residuals[1] = _dst[1] - T(dst[1]);

        return true;
    }

    template<class T>
    cv::Vec2f apply(T const* const dist) {
        T const _src[2] = {T(src[0]), T(src[1])};
        T const _center[2] = {T(center[0]), T(center[1])};
        T _dst[2] = {T(0),T(0)};
        applyDist(_src, _dst, _center, dist);
        return {float(_dst[0]), float(_dst[1])};
    }

    template<class T>
    static void applyDist(
            T const src[2],
    T dst[2],
    T const center[2],
    T const dist[14]
    ) {

        //(k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]])
        T const& k1 = dist[0];
        T const& k2 = dist[1];
        T const& p1 = dist[2];
        T const& p2 = dist[3];
        T const& k3 = dist[4];
        T const& k4 = dist[5];
        T const& k5 = dist[6];
        T const& k6 = dist[7];
        T const& s1 = dist[8];
        T const& s2 = dist[9];
        T const& s3 = dist[10];
        T const& s4 = dist[11];
        T const& tau_x = dist[12];
        T const& tau_y = dist[13];

        T const x = src[0] - center[0];
        T const y = src[1] - center[1];

        T const r2 = x*x + y*y;
        T const r4 = r2*r2;
        T const r6 = r4*r2;

        T & x2 = dst[0];
        x2 = x*(T(1) + k1*r2 + k2*r4 + k3*r6)/(T(1) + k4*r2 + k5*r4 + k6*r6)
                + T(2)*x*y*p1 + p2*(r2 + T(2)*x*x) + s1*r2 + s2*r4;

        T & y2 = dst[1];
        y2 = y*(T(1) + k1*r2 + k2*r4 + k3*r6)/(T(1) + k4*r2 + k5*r4 + k6*r6)
                + T(2)*x*y*p2 + p1*(r2 + T(2)*y*y) + s3*r2 + s4*r4;

        applySensorTilt(x2, y2, tau_x, tau_y);

        x2 += center[0];
        y2 += center[1];

    }
};

cv::Mat_<cv::Vec3b> colorFlow(const cv::Mat_<cv::Vec2f>& flow,
                              double &factor,
                              const double scaleFactor) {
    makecolorwheel();
    cv::Mat_<cv::Vec3b> result(flow.rows, flow.cols, cv::Vec3b(0,0,0));
#pragma omp parallel for
    for (int ii = 0; ii < flow.rows; ++ii) {
        cv::Vec3b * resultRow = result.ptr<cv::Vec3b>(ii);
        const cv::Vec2f * flowRow = flow.ptr<cv::Vec2f>(ii);
        for (int jj = 0; jj < flow.cols; ++jj) {
            computeColor(
                        flowRow[jj][0]/factor,
                    flowRow[jj][1]/factor,
                    resultRow[jj]);
        }
    }
    if (scaleFactor > 0) {
        cv::resize(result, result, cv::Size(0,0), scaleFactor, scaleFactor, cv::INTER_CUBIC);
    }
    return result;
}

void fitDistortion(
        std::string const prefix,
        const cv::Mat_<cv::Vec2f> &flow,
        double factor,
        const double length_factor,
        const cv::Scalar &color) {

    runningstats::QuantileStats<float> motion_stats, abs_motion_stats, length_stats;
    for (cv::Vec2f const& it : flow) {
        if (std::isfinite(it[0]) && std::isfinite(it[1])) {
            motion_stats.push_unsafe(it[0]);
            motion_stats.push_unsafe(it[1]);
            abs_motion_stats.push_unsafe(std::abs(it[0]));
            abs_motion_stats.push_unsafe(std::abs(it[1]));
            length_stats.push_unsafe(std::sqrt(it[0]*it[0] + it[1]*it[1]));
        }
    }
    if (factor <= 0) {
        factor = length_stats.getQuantile(.999);
    }
    std::cout << "Factor: " << factor << std::endl;
    cv::Mat_<cv::Vec3b> result = colorFlow(flow, factor, 1);

    ceres::Problem problem;

    std::vector<double> dist(14, 0.0);

    try {
        cv::FileStorage input(prefix + "-dist.yaml", cv::FileStorage::READ);
        input["dist"] >> dist;
        if (dist.size() < 14) {
            dist.resize(14);
        }
    }
    catch (...) {

    }

    cv::Vec2f const center{float(result.cols-1)/2, float(result.rows-1)/2};
    for (int row = 0; row < result.rows; row++) {
        for (int col = 0; col < result.cols; col++) {
            cv::Vec2f dst{float(col), float(row)};
            cv::Vec2f src = dst + flow(row, col);
            problem.AddResidualBlock(
                        new ceres::AutoDiffCostFunction<DistortionFunctor, 2, 14>(
                            new DistortionFunctor(src, dst, center)
                            ),
                        nullptr,
                        dist.data()
                        );
        }
    }


    // Run the solver!
    ceres::Solver::Options options;
    options.num_threads = int(8);
    options.max_num_iterations = 500;
    double const ceres_tolerance = 1e-12;
    options.function_tolerance = ceres_tolerance;
    options.gradient_tolerance = ceres_tolerance;
    options.parameter_tolerance = ceres_tolerance;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    cv::FileStorage output(prefix + "-dist.yaml", cv::FileStorage::WRITE);
    output << "dist" << dist;

    std::cout << summary.BriefReport() << "\n";
    std::cout << summary.FullReport() << "\n";

    cv::Mat_<cv::Vec2f> corrected(flow.size());
    for (int row = 0; row < result.rows; row++) {
        for (int col = 0; col < result.cols; col++) {
            if (!std::isfinite(flow(row,col)[0]) || !std::isfinite(flow(row,col)[1])) {
                continue;
            }
            cv::Vec2f dst{float(col), float(row)};
            cv::Vec2f src = dst + flow(row, col);
            DistortionFunctor func(src, dst, center);
            corrected(row, col) = func.apply(dist.data()) - dst;
        }
    }

    cv::imwrite(prefix + "-orig.png", plotWithArrows(flow, factor, length_factor, color));
    cv::imwrite(prefix + "-corrected.png", plotWithArrows(corrected, factor, length_factor, color));

    gnuplotWithArrows(prefix + "-orig-gpl", flow, factor, length_factor);
    gnuplotWithArrows(prefix + "-corrected-gpl", corrected, factor, length_factor);
}

cv::Mat_<cv::Vec3b> plotWithArrows(
        const cv::Mat_<cv::Vec2f> &flow,
        double factor,
        const double length_factor,
        const cv::Scalar &color) {

    runningstats::QuantileStats<float> motion_stats, abs_motion_stats, length_stats;
    for (cv::Vec2f const& it : flow) {
        if (std::isfinite(it[0]) && std::isfinite(it[1])) {
            motion_stats.push_unsafe(it[0]);
            motion_stats.push_unsafe(it[1]);
            abs_motion_stats.push_unsafe(std::abs(it[0]));
            abs_motion_stats.push_unsafe(std::abs(it[1]));
            length_stats.push_unsafe(std::sqrt(it[0]*it[0] + it[1]*it[1]));
        }
    }
    if (factor <= 0) {
        factor = length_stats.getQuantile(.999);
    }
    std::cout << "Factor: " << factor << std::endl;
    cv::Mat_<cv::Vec3b> result = colorFlow(flow, factor, 1);

    int const grid = std::max<int>(10, std::ceil(length_factor * 2 * abs_motion_stats.getQuantile(.99)));
    std::cout << "Grid: " << grid << std::endl;

    for (int row = grid; row+grid/2 < result.rows; row += grid) {
        for (int col = grid; col+grid/2 < result.cols; col += grid) {
            cv::Point2f src = {float(col), float(row)};
            cv::Point2f dst = src + length_factor * cv::Point2f(flow(src)[0], flow(src)[1]);
            cv::line(result, src, dst, color, 1, cv::LINE_AA, 0);
            result(src) = cv::Vec3b{0,0,255};
        }
    }

    return result;
}

runningstats::QuantileStats<float> getLengthStats(cv::Mat_<cv::Vec2f> const& flow) {
    runningstats::QuantileStats<float> length_stats;
    for (cv::Vec2f const& it : flow) {
        if (std::isfinite(it[0]) && std::isfinite(it[1])) {
            length_stats.push_unsafe(std::sqrt(it[0]*it[0] + it[1]*it[1]));
        }
    }
    return length_stats;
}

double adaptiveRound(double const in) {
    int const digits = std::ceil(std::log10(in));
    int const factor = std::round(std::pow(10, 2 - digits));
    return std::round(in*factor)/factor;
}

void image2txtfile(cv::Mat_<float> const& img, std::string const& filename) {
    std::ofstream out(filename);
    for (int row = 0; row < img.rows; row++) {
        for (int col = 0; col < img.cols; col++) {
            out << row << "\t" << col << "\t" << img(row, col) << std::endl;
        }
        out << std::endl;
    }
}

cv::Mat_<float> flow2length(cv::Mat_<cv::Vec2f> const& flow) {
    cv::Mat_<float> result(flow.size());
    for (int row = 0; row < flow.rows; row++) {
        for (int col = 0; col < flow.cols; col++) {
            result(row, col) = std::sqrt(flow(row, col).dot(flow(row, col)));
        }
    }
    return result;
}

void gnuplotWithArrows(const std::string& prefix,
                       const cv::Mat_<cv::Vec2f> &flow,
                       double factor,
                       const double arrow_factor,
                       const cv::Scalar &color) {

    std::vector<std::tuple<double,double,double,double>> arrows;

    int const grid = 15;
    cv::Mat_<cv::Vec2f> flow_filtered;
    cv::GaussianBlur(flow, flow, cv::Size(), 1, 1);
    cv::GaussianBlur(flow, flow_filtered, cv::Size(), double(grid)/2, double(grid)/2);
    for (int row = grid; row+grid/2 < flow.rows; row += grid) {
        for (int col = grid; col+grid/2 < flow.cols; col += grid) {
            cv::Point2f src((float(col)), float(row));
            cv::Vec2f offset = flow(row, col);
            arrows.push_back({src.x, src.y, offset[0], offset[1]});
        }
    }
    std::stringstream settings;
    settings << "set lmargin 2.5;\n"
             << "set cbtics offset -.5,0;\n"
             << "set ytics out offset .9,0;\n"
             << "set xtics out offset 0,.5;\n"
             << "set view equal xy;\n"
             << "set bmargin 0;\n"
             << "set tmargin 2;\n"
             << "set size ratio -1;\n";
    if (factor <= 0) {
        runningstats::QuantileStats<float> length_stats = getLengthStats(flow);
        factor = length_stats.getQuantile(.999);
    }
    image2txtfile(flow2length(flow), prefix + "-flowlength.data");
    factor = adaptiveRound(factor);
    cv::imwrite(prefix + "-flow.png", colorFlow(flow, factor, 1));
    {
        gnuplotio::Gnuplot plt(std::string("tee ") + prefix + ".gpl | gnuplot -persist");
        plt << "set term svg enhanced background rgb 'white';\n"
        << "set output '" << prefix + ".svg';\n";
        plt << "set xrange [0:" << flow.cols << "];\n"
        << "set yrange [" << flow.rows << ":0];\n"
        << "set title 'Color threshold " << factor << "';\n" << settings.str();

        plt << "plot '" << prefix << "-flow.png' binary filetype=png w rgbimage notitle,"
      << plt.file1d(arrows, prefix + ".arrows")
        << " u 1:2:($3*" << arrow_factor << "):($4*" << arrow_factor << ") w vectors lc rgb 'black' notitle,"
      << std::endl;
    }
    {
        gnuplotio::Gnuplot plt(std::string("tee ") + prefix + "-length.gpl | gnuplot -persist");
        plt << "set term svg enhanced background rgb 'white';\n"
            << "set output '" << prefix + "-length.svg';\n";
        plt << "set xrange [0:" << flow.cols << "];\n"
            << "set yrange [" << flow.rows << ":0];\n"
            << "set cbrange[*:" << factor << "];\n" << settings.str();

        plt << "plot '" << prefix << "-flowlength.data' u 2:1:3 w image notitle,"
            << plt.file1d(arrows, prefix + ".arrows")
            << " u 1:2:($3*" << arrow_factor << "):($4*" << arrow_factor << ") w vectors lc rgb 'white' notitle,"
            << std::endl
            << "set term epslatex color size 2.1,1.5;\n"
            << "set output '" << prefix + "-length.tex';\n"
            << "replot;\n"
            << "set term png;\n"
            << "set output '" << prefix + "-length-png.png';\n"
            << "replot;";
    }
}

} // namespace hdcalib


