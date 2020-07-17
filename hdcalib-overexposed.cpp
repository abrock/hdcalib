#include <iostream>
#include <map>

#include <opencv2/highgui.hpp>

std::string replace_slash(std::string const in) {
    std::string out;
    out.reserve(in.size());
    for (const char c : in) {
        if ('/' == c) {
            out += "-";
        }
        else {
            out += c;
        }
    }
    return out;
}

int main(int argc, char ** argv) {

    for (size_t ii = 1; int(ii) < argc; ++ii) {

    }
}
