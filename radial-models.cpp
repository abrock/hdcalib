#include <iostream>
#include <vector>
#include <ceres/ceres.h>
#include "gnuplot-iostream.h"

double const resolution = 1;

template<int N>
struct PolyN {
    double const x;
    double const y;
    static size_t const num = N;

    PolyN(double const _x, double const _y) : x(_x), y(_y) {}

    template<class T>
    bool operator ()(
            T const * const data,
            T * residuals) const {
        T val(x);
        T current_x(x*x);
        for (size_t ii = 0; ii < N; ++ii) {
            val += current_x * data[ii];
            current_x *= x;
        }
        residuals[0] = y - val;
        return true;
    }
};

struct RadialOrig {
    double const x;
    double const y;
    static size_t const num = 6;

    RadialOrig(double const _x, double const _y) : x(_x), y(_y) {}

    template<class T>
    bool operator ()(
            T const * const data,
            T * residuals) const {
        T const x2 = T(x*x);
        T const x4 = x2*x2;
        T const x6 = x4*x2;
        T const a1 = data[0];
        T const a2 = data[1];
        T const a3 = data[2];
        T const b1 = data[3];
        T const b2 = data[4];
        T const b3 = data[5];
        residuals[0] = y - x*(1. + a1*x2 + a2*x4 + a3*x6) / (1. + b1*x2 + b2*x4 + b3*x6);
        return true;
    }
};

template<int N>
struct RadialOrigN {
    double const x;
    double const y;
    static size_t const num = 2*N;

    RadialOrigN(double const _x, double const _y) : x(_x), y(_y) {}

    template<class T>
    bool operator ()(
            T const * const data,
            T * residuals) const {
        T top(1);
        T bottom(1);
        T x2 = T(x*x);
        T current_x = x2;
        for (size_t ii = 0; ii < N; ++ii) {
            top += current_x * data[ii];
            bottom += current_x * data[N+ii];
            current_x *= x2;
        }
        residuals[0] = y - x*top/bottom;
        return true;
    }
};

template<int N>
struct RadialNewN {
    double const x;
    double const y;
    static size_t const num = 2*N;

    RadialNewN(double const _x, double const _y) : x(_x), y(_y) {}

    template<class T>
    bool operator ()(
            T const * const data,
            T * residuals) const {
        T top(1);
        T bottom(1);
        T current_x(x);
        for (size_t ii = 0; ii < N; ++ii) {
            top += current_x * data[ii];
            bottom += current_x * data[N+ii];
            current_x *= x;
        }
        residuals[0] = y - x*top/bottom;
        return true;
    }
};

struct Radial3_3 {
    double const x;
    double const y;
    static size_t const num = 12;

    Radial3_3(double const _x, double const _y) : x(_x), y(_y) {}

    template<class T>
    bool operator ()(
            T const * const data,
            T * residuals) const {
        T const x2 = T(x*x);
        T const x3 = x2*x;
        T const x4 = x2*x2;
        T const x5 = x4*x;
        T const x6 = x4*x2;
        T const a1 = data[0];
        T const a2 = data[1];
        T const a3 = data[2];
        T const a4 = data[3];
        T const a5 = data[4];
        T const a6 = data[5];

        T const b1 = data[6];
        T const b2 = data[7];
        T const b3 = data[8];
        T const b4 = data[9];
        T const b5 = data[10];
        T const b6 = data[11];
        residuals[0] = y - x*(1. + a1*x + a2*x2 + a3*x3 + a4*x4 + a5*x5 + a6*x6) /
                             (1. + b1*x + b2*x2 + b3*x3 + b4*x4 + b5*x5 + b6*x6);
        return true;
    }
};


template<class F>
std::vector<std::pair<double, double> > fit(std::vector<std::pair<double, double> > const& data) {
    ceres::Problem p;

    std::vector<double> params(F::num, 0.01);

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

void plot(std::string name, std::vector<std::pair<double, double> > const& residuals, std::ofstream& log, int N) {
    double min = 0;
    double max = 0;
    double squaresum = 0;
    for (auto const& it : residuals) {
        min = std::min(min, it.second);
        max = std::max(max, it.second);
        squaresum += it.second*it.second;
    }
    double const absmax = std::max(std::abs(min), std::abs(max));
    double const rmse = std::sqrt(squaresum/residuals.size());
    gnuplotio::Gnuplot plt("tee " + name + ".gpl | gnuplot -persist");

    log << N << "\t" << absmax << "\t" << rmse << "\t" << min << "\t" << max << std::endl;

    plt << "set term svg enhanced background rgb 'white';\n"
<< "set output '" << name << ".svg';\n"
<< "set title 'absmax: " << absmax << ", range: [" << min << ", " << max << "], RMSE: " << rmse << ";\n"
<< "plot " << plt.file1d(residuals, name + ".data") << " w l notitle";
}

void run(std::string const& name, std::vector<std::pair<double, double> > const& data) {
    std::ofstream orig_log(name + "-orig.data");
    orig_log << "#N absmax rmse min max " << std::endl;
    std::ofstream new_log(name + "-new.data");
    std::ofstream poly_log(name + "-poly.data");

    plot(name + "-origN-01", fit<RadialOrigN<1> >(data), orig_log, 2);
    plot(name + "-origN-02", fit<RadialOrigN<2> >(data), orig_log, 4);
    plot(name + "-origN-03", fit<RadialOrigN<3> >(data), orig_log, 6);
    plot(name + "-origN-04", fit<RadialOrigN<4> >(data), orig_log, 8);
    plot(name + "-origN-05", fit<RadialOrigN<5> >(data), orig_log, 10);
    plot(name + "-origN-06", fit<RadialOrigN<6> >(data), orig_log, 12);
    plot(name + "-origN-07", fit<RadialOrigN<7> >(data), orig_log, 14);
    plot(name + "-origN-08", fit<RadialOrigN<8> >(data), orig_log, 16);
#if 0
    plot(name + "-origN-09", fit<RadialOrigN<9> >(data), orig_log, 9);
    plot(name + "-origN-10", fit<RadialOrigN<10> >(data), orig_log, 10);
    plot(name + "-origN-11", fit<RadialOrigN<11> >(data), orig_log, 11);
    plot(name + "-origN-12", fit<RadialOrigN<12> >(data), orig_log, 12);
    plot(name + "-origN-13", fit<RadialOrigN<13> >(data), orig_log, 13);
    plot(name + "-origN-14", fit<RadialOrigN<14> >(data), orig_log, 14);
    plot(name + "-origN-15", fit<RadialOrigN<15> >(data), orig_log, 15);
    plot(name + "-origN-16", fit<RadialOrigN<16> >(data), orig_log, 16);
    plot(name + "-origN-17", fit<RadialOrigN<17> >(data), orig_log, 17);
    plot(name + "-origN-18", fit<RadialOrigN<18> >(data), orig_log, 18);
    plot(name + "-origN-19", fit<RadialOrigN<19> >(data), orig_log, 19);
    plot(name + "-origN-20", fit<RadialOrigN<20> >(data), orig_log, 20);
    plot(name + "-origN-21", fit<RadialOrigN<21> >(data), orig_log, 21);
    plot(name + "-origN-22", fit<RadialOrigN<22> >(data), orig_log, 22);
    plot(name + "-origN-23", fit<RadialOrigN<23> >(data), orig_log, 23);
    plot(name + "-origN-24", fit<RadialOrigN<24> >(data), orig_log, 24);
    plot(name + "-origN-25", fit<RadialOrigN<25> >(data), orig_log, 25);
    plot(name + "-origN-26", fit<RadialOrigN<26> >(data), orig_log, 26);
    plot(name + "-origN-27", fit<RadialOrigN<27> >(data), orig_log, 27);
    plot(name + "-origN-28", fit<RadialOrigN<28> >(data), orig_log, 28);
    plot(name + "-origN-29", fit<RadialOrigN<29> >(data), orig_log, 29);
    plot(name + "-origN-30", fit<RadialOrigN<30> >(data), orig_log, 30);
#endif

    plot(name + "-newN-01", fit<RadialNewN<1> >(data), new_log, 2);
    plot(name + "-newN-02", fit<RadialNewN<2> >(data), new_log, 4);
    plot(name + "-newN-03", fit<RadialNewN<3> >(data), new_log, 6);
    plot(name + "-newN-04", fit<RadialNewN<4> >(data), new_log, 8);
    plot(name + "-newN-05", fit<RadialNewN<5> >(data), new_log, 10);
    plot(name + "-newN-06", fit<RadialNewN<6> >(data), new_log, 12);
    plot(name + "-newN-07", fit<RadialNewN<7> >(data), new_log, 14);
    plot(name + "-newN-08", fit<RadialNewN<8> >(data), new_log, 16);

#if 0
    plot(name + "-newN-09", fit<RadialNewN<9> >(data), new_log, 9);
    plot(name + "-newN-10", fit<RadialNewN<10> >(data), new_log, 10);
    plot(name + "-newN-11", fit<RadialNewN<11> >(data), new_log, 11);
    plot(name + "-newN-12", fit<RadialNewN<12> >(data), new_log, 12);
    plot(name + "-newN-13", fit<RadialNewN<13> >(data), new_log, 13);
    plot(name + "-newN-14", fit<RadialNewN<14> >(data), new_log, 14);
    plot(name + "-newN-15", fit<RadialNewN<15> >(data), new_log, 15);
    plot(name + "-newN-16", fit<RadialNewN<16> >(data), new_log, 16);
    plot(name + "-newN-17", fit<RadialNewN<17> >(data), new_log, 17);
    plot(name + "-newN-18", fit<RadialNewN<18> >(data), new_log, 18);
    plot(name + "-newN-19", fit<RadialNewN<19> >(data), new_log, 19);
    plot(name + "-newN-20", fit<RadialNewN<20> >(data), new_log, 20);
    plot(name + "-newN-21", fit<RadialNewN<21> >(data), new_log, 21);
    plot(name + "-newN-22", fit<RadialNewN<22> >(data), new_log, 22);
    plot(name + "-newN-23", fit<RadialNewN<23> >(data), new_log, 23);
    plot(name + "-newN-24", fit<RadialNewN<24> >(data), new_log, 24);
    plot(name + "-newN-25", fit<RadialNewN<25> >(data), new_log, 25);
    plot(name + "-newN-26", fit<RadialNewN<26> >(data), new_log, 26);
    plot(name + "-newN-27", fit<RadialNewN<27> >(data), new_log, 27);
    plot(name + "-newN-28", fit<RadialNewN<28> >(data), new_log, 28);
    plot(name + "-newN-29", fit<RadialNewN<29> >(data), new_log, 29);
    plot(name + "-newN-30", fit<RadialNewN<30> >(data), new_log, 30);
#endif

    plot(name + "-PolyN-01", fit<PolyN<1> >(data), poly_log, 1);
    plot(name + "-PolyN-02", fit<PolyN<2> >(data), poly_log, 2);
    plot(name + "-PolyN-03", fit<PolyN<3> >(data), poly_log, 3);
    plot(name + "-PolyN-04", fit<PolyN<4> >(data), poly_log, 4);
    plot(name + "-PolyN-05", fit<PolyN<5> >(data), poly_log, 5);
    plot(name + "-PolyN-06", fit<PolyN<6> >(data), poly_log, 6);
    plot(name + "-PolyN-07", fit<PolyN<7> >(data), poly_log, 7);
    plot(name + "-PolyN-08", fit<PolyN<8> >(data), poly_log, 8);
    plot(name + "-PolyN-09", fit<PolyN<9> >(data), poly_log, 9);
    plot(name + "-PolyN-10", fit<PolyN<10> >(data), poly_log, 10);
    plot(name + "-PolyN-11", fit<PolyN<11> >(data), poly_log, 11);
    plot(name + "-PolyN-12", fit<PolyN<12> >(data), poly_log, 12);
    plot(name + "-PolyN-13", fit<PolyN<13> >(data), poly_log, 13);
    plot(name + "-PolyN-14", fit<PolyN<14> >(data), poly_log, 14);
    plot(name + "-PolyN-15", fit<PolyN<15> >(data), poly_log, 15);
    plot(name + "-PolyN-16", fit<PolyN<16> >(data), poly_log, 16);

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
