#include "cornercolor.h"

const int subsampling = 1;
const int ss_border = 2;

void setnumber(cv::Mat &m, int n)
{
  m.at<uchar>(1, 1) = (n/1 & 0x01) * 255;
  m.at<uchar>(1, 2) = (n/2 & 0x01) * 255;
  m.at<uchar>(1, 3) = (n/4 & 0x01) * 255;
  m.at<uchar>(2, 1) = (n/8 & 0x01) * 255;
  m.at<uchar>(2, 2) = (n/16 & 0x01) * 255;
  m.at<uchar>(2, 3) = (n/32 & 0x01) * 255;
  m.at<uchar>(3, 1) = (n/64 & 0x01) * 255;
  m.at<uchar>(3, 2) = (n/128 & 0x01) * 255;
  m.at<uchar>(3, 3) = (n/256 & 0x01) * 255;
}

int smallidtomask(int id, int x, int y)
{
  int j = (id / 32) * 2 + (id % 2) + y;
  int i = (id % 32) + x;

  return (j*13 + i * 7) % 512;
}

int idtomask(int id)
{
  if ((id&2==2))
    return id ^ 170;
  else
    return id ^ 340;
}

int masktoid(int mask)
{
  if ((mask&2==2))
    return mask ^ 170;
  else
    return mask ^ 340;
}

void writemarker(cv::Mat &img, int page, int offx = 0, int offy = 0)
{
  cv::Mat marker = cv::Mat::zeros(5, 5, CV_8UC1);
  marker.at<uchar>(2, 4) = 255;

  for(int j=0;j<16;j++)
    for(int i=0;i<32;i++) {
      setnumber(marker, idtomask(j*32+i));
      marker.copyTo(img.colRange(i*5+offx,i*5+5+offx).rowRange(j*10+(i%2)*5+offy, j*10+(i%2)*5+5+offy));

      setnumber(marker, page ^ smallidtomask(j*32+i, 0, 2*((i+1)%2)-1));
      marker = 255 - marker;
      marker.copyTo(img.colRange(i*5+offx,i*5+5+offx).rowRange(j*10+((i+1)%2)*5+offy, j*10+((i+1)%2)*5+5+offy));
      marker = 255 - marker;
    }
}

void checker_recurse(cv::Mat &img, cv::Mat &checker)
{
  cv::Mat hr;
  int w = img.size().width;
  int h = img.size().height;
  int ws = subsampling+2*ss_border;
  int w_hr = w*ws;
  uint8_t *ptr_hr;

  resize(img, hr, cv::Point2i(img.size())*ws, 0, 0, cv::INTER_NEAREST);

  ptr_hr = hr.ptr<uchar>(0);

  for(int y=0;y<h;y++)
    for(int x=0;x<w;x++) {
      for(int j=ss_border;j<ws-ss_border;j++)
        for(int i=j%2+ss_border;i<ws-ss_border;i+=2)
          ptr_hr[(y*ws+j)*w_hr+x*ws+i] = 255-ptr_hr[(y*ws+j)*w_hr+x*ws+i];
    }

  checker = hr;
}

CornerColor &CornerColor::getInstance() {
    static CornerColor    instance; // Guaranteed to be destroyed.
    // Instantiated on first use.
    return instance;
}

CornerColor::CornerColor() {
    subpatterns.resize(2);
    subpatterns[0] = cv::Mat_<uint8_t>::zeros(0,0);
    subpatterns[1] = cv::Mat_<uint8_t>::zeros(5,5);
    subpatterns[1](2,2) = 255;
}

size_t CornerColor::getColor(const cv::Point2i id, const int page, const int recursion) {
    return getInstance()._getColor(id, page, recursion);
}

size_t CornerColor::getColor(hdmarker::Corner const& c, int const recursion) {
    return c.color;
    return getInstance()._getColor(c.id, c.page, recursion);
}

size_t CornerColor::_getColor(const cv::Point2i id, const int page, const int recursion) {
    num_calls++;
    if (data.size() < size_t(page)+1) {
        data.resize(size_t(page)+1);
    }
    cv::Mat_<uint8_t> & root = data[size_t(page)];
    if (subpatterns.size() < size_t(std::abs(recursion))+1) {
        subpatterns.resize(size_t(std::abs(recursion))+1);
    }
    if (root.empty()) {
        root = cv::Mat_<uint8_t>::zeros(16*10, 32*5);
        root += 255;
        writemarker(root, page);
    }
    for (size_t ii = 0; ii+1 <= size_t(recursion); ++ii) {
        if (subpatterns[ii+1].empty()) {
            checker_recurse(subpatterns[ii], subpatterns[ii+1]);
            cv::imwrite(std::to_string(ii+1) + ".png", subpatterns[ii+1]);
        }
    }

    int const factor = hdcalib::Calib::computeCornerIdFactor(recursion);

    if (id.x < 0 || id.y < 0 || id.x > factor*32 || id.y > factor*32) {
        throw std::runtime_error("ID invalid");
    }

    if (id.x % factor == 0) { // This is the case where the Corner is a main marker
        return 2 + ((id.x/factor + id.y/factor) % 2);
    }
    bool main_white = root((id*5)/factor) > 125;

    // Mapping of the ID to the actual pixel location at the correct level:
    // 1->2
    // 3->7
    // 5->12
    // 7->17
    // x->((x-1)/2)*5+2

    cv::Point2i loc(((id.x-1)/2)*5 + 2, ((id.y-1)/2)*5 + 2);

    loc = cv::Point2i{loc.x % (factor/2), loc.y % (factor/2)};
    return main_white ^ (subpatterns[size_t(recursion)](loc) > 125);
}

size_t CornerColor::getNumCalls() {
    return getInstance()._getNumCalls();
}

size_t CornerColor::_getNumCalls() const {
    return num_calls;
}
