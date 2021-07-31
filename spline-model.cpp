#include <iostream>
#include <vector>
#include <ceres/ceres.h>
#include "gnuplot-iostream.h"

double const resolution = 1;

template<class T>
T evaluateSpline(T const x, int const POS, int const DEG) {
    T const pos(POS);
    if (0 == DEG) {
        return (x <= pos || x > T(POS+1)) ? T(0) : T(1);
    }
    T const deg(DEG);
    return (x-pos)/deg*evaluateSpline(x,POS,DEG-1) + (T(POS+DEG+1)-x)/deg*evaluateSpline(x, POS+1, DEG-1);
}

template<int N, int DEG>
struct SplineN {
    double const x;
    double const y;
    static size_t const num = N+DEG;

    SplineN(double const _x, double const _y) : x(_x), y(_y) {}

    template<class T>
    bool operator ()(
            T const * const data,
            T * residuals) const {
        residuals[0] = T(-y);
        for (int ii = 0; size_t(ii) < num; ++ii) {
            residuals[0] += data[ii] * evaluateSpline(x*T(N)/T(2), ii-DEG, DEG);
        }
        return true;
    }
};


template<class F>
std::vector<std::pair<double, double> > fit(std::vector<std::pair<double, double> > const& data) {
    ceres::Problem p;

    std::vector<double> params(F::num, 0.0);

    for (auto const& it : data) {
        p.AddResidualBlock(new ceres::AutoDiffCostFunction<F, 1, F::num>(new F(it.first, it.second)),
                           nullptr,
                           params.data()
                           );
    }

    double const ceres_tolerance = 1e-32;
    ceres::Solver::Options options;
    options.num_threads = int(8);
    options.max_num_iterations = 5000;
    options.function_tolerance = ceres_tolerance;
    options.gradient_tolerance = ceres_tolerance;
    options.parameter_tolerance = ceres_tolerance;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &p, &summary);

    std::cout << summary.BriefReport() << "\n";
    std::cout << summary.FullReport() << "\n";

    std::vector<std::pair<double, double> > res;
    for (auto const& it : data) {
        double residual = 0;
        F(it.first, it.second)(params.data(), &residual);
        res.push_back({it.first, residual*resolution});
    }
    return res;
}

void plot(std::string name,
          std::vector<std::pair<double, double> > const& data,
          std::vector<std::pair<double, double> > const& residuals,
          std::ofstream& log, int N) {
    double min = 0;
    double max = 0;
    double squaresum = 0;
    std::ofstream out(name + "-all.data");
    for (size_t ii = 0; ii < residuals.size(); ++ii) {
        auto const it = residuals[ii];
        min = std::min(min, it.second);
        max = std::max(max, it.second);
        squaresum += it.second*it.second;
        out << data[ii].first << "\t" << data[ii].second << "\t" << (data[ii].second + it.second) << std::endl;
    }
    double const absmax = std::max(std::abs(min), std::abs(max));
    double const rmse = std::sqrt(squaresum/residuals.size());

    log << N << "\t" << absmax << "\t" << rmse << "\t" << min << "\t" << max << std::endl;

    {
        gnuplotio::Gnuplot plt("tee " + name + ".gpl | gnuplot -persist");
        plt << "set term svg enhanced background rgb 'white';\n"
<< "set xrange [-.1:2.1];\n"
<< "set output '" << name << ".svg';\n"
<< "set title 'absmax: " << absmax << ", range: [" << min << ", " << max << "], RMSE: " << rmse << ";\n"
<< "plot " << plt.file1d(residuals, name + ".data") << " w l notitle";
    }
    {
        gnuplotio::Gnuplot plt("tee " + name + "-all.gpl | gnuplot -persist");
        plt << "set term svg enhanced background rgb 'white';\n"
<< "set xrange [-.1:2.1];\n"
<< "set output '" << name << "-all.svg';\n"
<< "set title 'absmax: " << absmax << ", range: [" << min << ", " << max << "], RMSE: " << rmse << ";\n"
<< "plot '" << name << "-all.data' u 1:2 w l title 'data', '" << name << "-all.data' u 1:2 w l title 'fit'";
    }
}

void run(std::string const& name, std::vector<std::pair<double, double> > const& data) {
    std::ofstream orig_log(name + "-orig.data");
    orig_log << "#N absmax rmse min max " << std::endl;
    std::ofstream new_log(name + "-new.data");
    std::ofstream poly_log(name + "-poly.data");

    plot(name + "-SplineN1D3", data, fit<SplineN<1,3> >(data), poly_log, 1);
    plot(name + "-SplineN2D3", data, fit<SplineN<2,3> >(data), poly_log, 2);
    plot(name + "-SplineN3D3", data, fit<SplineN<3,3> >(data), poly_log, 3);
    plot(name + "-SplineN4D3", data, fit<SplineN<4,3> >(data), poly_log, 4);
    plot(name + "-SplineN5D3", data, fit<SplineN<5,3> >(data), poly_log, 5);
    plot(name + "-SplineN6D3", data, fit<SplineN<6,3> >(data), poly_log, 6);
    plot(name + "-SplineN7D3", data, fit<SplineN<7,3> >(data), poly_log, 7);
    plot(name + "-SplineN8D3", data, fit<SplineN<8,3> >(data), poly_log, 8);
    plot(name + "-SplineN9D3", data, fit<SplineN<9,3> >(data), poly_log, 9);
    plot(name + "-SplineN10D3", data, fit<SplineN<10,3> >(data), poly_log, 10);
    plot(name + "-SplineN11D3", data, fit<SplineN<11,3> >(data), poly_log, 11);
    plot(name + "-SplineN12D3", data, fit<SplineN<12,3> >(data), poly_log, 12);
    plot(name + "-SplineN13D3", data, fit<SplineN<13,3> >(data), poly_log, 13);
    plot(name + "-SplineN14D3", data, fit<SplineN<14,3> >(data), poly_log, 14);
    plot(name + "-SplineN15D3", data, fit<SplineN<15,3> >(data), poly_log, 15);
    plot(name + "-SplineN16D3", data, fit<SplineN<16,3> >(data), poly_log, 16);
    plot(name + "-SplineN17D3", data, fit<SplineN<17,3> >(data), poly_log, 17);
    plot(name + "-SplineN18D3", data, fit<SplineN<18,3> >(data), poly_log, 18);
    plot(name + "-SplineN19D3", data, fit<SplineN<19,3> >(data), poly_log, 19);
    plot(name + "-SplineN20D3", data, fit<SplineN<20,3> >(data), poly_log, 20);
#if 0
    plot(name + "-PolyN-17", fit<PolyN<17> >(data), poly_log, 17);
    plot(name + "-PolyN-18", fit<PolyN<18> >(data), poly_log, 18);
    plot(name + "-PolyN-19", fit<PolyN<19> >(data), poly_log, 19);
    plot(name + "-PolyN-20", fit<PolyN<20> >(data), poly_log, 20);
    plot(name + "-PolyN-21", fit<PolyN<21> >(data), poly_log, 21);
    plot(name + "-PolyN-22", fit<PolyN<22> >(data), poly_log, 22);
    plot(name + "-PolyN-23", fit<PolyN<23> >(data), poly_log, 23);
    plot(name + "-PolyN-24", fit<PolyN<24> >(data), poly_log, 24);
    plot(name + "-PolyN-25", fit<PolyN<25> >(data), poly_log, 25);
    plot(name + "-PolyN-26", fit<PolyN<26> >(data), poly_log, 26);
    plot(name + "-PolyN-27", fit<PolyN<27> >(data), poly_log, 27);
    plot(name + "-PolyN-28", fit<PolyN<28> >(data), poly_log, 28);
    plot(name + "-PolyN-29", fit<PolyN<29> >(data), poly_log, 29);
    plot(name + "-PolyN-30", fit<PolyN<30> >(data), poly_log, 30);
#endif

    gnuplotio::Gnuplot plt_max("tee " + name + "-max.gpl | gnuplot -persist");
    gnuplotio::Gnuplot plt_rmse("tee " + name + "-rmse.gpl | gnuplot -persist");

    plt_max << "set term svg enhanced background rgb 'white';\n"
<< "set output '" << name << "-max.svg';\n"
<< "set logscale y;\n"
<< "plot '" << name << "-orig.data' u 1:2 w lp title 'orig',"
<< "'" << name << "-new.data' u 1:2 w lp title 'new',"
<< "'" << name << "-poly.data' u 1:2 w lp title 'poly'";

    plt_rmse << "set term svg enhanced background rgb 'white';\n"
<< "set output '" << name << "-rmse.svg';\n"
<< "set logscale y;\n"
<< "plot '" << name << "-orig.data' u 1:3 w lp title 'orig',"
<< "'" << name << "-new.data' u 1:3 w lp title 'new',"
<< "'" << name << "-poly.data' u 1:3 w lp title 'poly'";

}

void plotSplines() {
    std::ofstream out("splines.data");
    for (double t = -3; t <= 4; t += 0.01) {
        out << t;
        for (int DEG = 0; DEG <= 5; ++DEG) {
            out << "\t" << evaluateSpline(t, 0, DEG);
        }
        for (int POS = -3; POS <= 3; ++POS) {
            out << "\t" << evaluateSpline(t, POS, 3);
        }
        out << std::endl;
    }
}

void runSim(std::string const& name) {
    std::vector<std::pair<double,double> > data;
    if ("pinhole" == name) {
        for (double x = 0; x <= 2; x += 1.0/(1024*16)) {
            data.push_back({x, x});
        }
    }
    if ("stereographic" == name) {
        for (double x = 0; x <= 2; x += 1.0/(1024*16)) {
            data.push_back({x, 2*std::tan(std::atan(x)/2)});
        }
    }
    if ("equidistance" == name) {
        for (double x = 0; x <= 2; x += 1.0/(1024*16)) {
            data.push_back({x, std::atan(x)});
        }
    }
    if ("equisolid" == name) {
        for (double x = 0; x <= 2; x += 1.0/(1024*16)) {
            data.push_back({x, 2*std::sin(std::atan(x)/2)});
        }
    }
    if ("orthographic" == name) {
        for (double x = 0; x <= 2; x += 1.0/(1024*16)) {
            data.push_back({x, std::sin(std::atan(x))});
        }
    }
    plotSplines();
    if (!data.empty()) {
        run(name, data);
    }
}

int main(int argc, char ** argv) {

    std::set<std::string> sims;
    for (size_t ii = 1; ii < size_t(argc); ++ii) {
        runSim(argv[ii]);
    }


}
