#ifndef CORNERCOLOR_H
#define CORNERCOLOR_H

#include <opencv2/imgproc.hpp>

#include "hdcalib.h"

class CornerColor
{
public:
    static CornerColor& getInstance();
private:
    CornerColor();

    /**
     * @brief data The first (leftmost) index is the page number, the second the recursion depth
     */
    std::vector<cv::Mat_<uint8_t> > data;

    std::vector<cv::Mat_<uint8_t> > subpatterns;

    size_t num_calls = 0;

public:
    // Compilers check accessibility before deleted status.
    // If the deleted methods were private the error message
    // would be "it's private", which may be misleading.
    // If they are public the error is "it's deleted"
    // which clearly tells the programmer to not do what
    // he is attempting to do.
    CornerColor(CornerColor const&)    = delete;
    void operator=(CornerColor const&) = delete;

    /**
     * @brief getColor returns the "color" of a (sub-)marker.
     * @param c
     * @param recursion
     * @return 0: black submarker.
     * 1: white submarker.
     * 2: main marker where the target is black at the top left and bottom right
     * 3: main marker where the target is white at the top right and bottom left
     */
    static size_t getColor(hdmarker::Corner const& c, int const recursion);

    static size_t getColor(cv::Point2i const id, int const page, int const recursion);

    size_t _getColor(const cv::Point2i id, const int page, int const recursion);

    static size_t getNumCalls();
    size_t _getNumCalls() const;
};

#endif // CORNERCOLOR_H
