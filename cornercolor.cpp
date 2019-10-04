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
  uint8_t *ptr_hr, *ptr_img;

  resize(img, hr, cv::Point2i(img.size())*ws, 0, 0, cv::INTER_NEAREST);

  ptr_img = img.ptr<uchar>(0);
  ptr_hr = hr.ptr<uchar>(0);

  for(int y=0;y<h;y++)
    for(int x=0;x<w;x++) {
      for(int j=ss_border;j<ws-ss_border;j++)
        for(int i=j%2+ss_border;i<ws-ss_border;i+=2)
          ptr_hr[(y*ws+j)*w_hr+x*ws+i] = 255-ptr_hr[(y*ws+j)*w_hr+x*ws+i];
    }

  checker = hr;
}

CornerColor &CornerColor::getInstance()
{
    static CornerColor    instance; // Guaranteed to be destroyed.
    // Instantiated on first use.
    return instance;
}

CornerColor::CornerColor() {}

int CornerColor::getColor(const cv::Point2i id, const int page, const int recursion) {
    return getInstance()._getColor(id, page, recursion);
}

int CornerColor::getColor(hdmarker::Corner const& c, int const recursion) {
    return getInstance()._getColor(c.id, c.page, recursion);
}

int CornerColor::_getColor(const cv::Point2i id, const int page, const int recursion) {
    if (data.size() < size_t(page)+1) {
        data.resize(size_t(page)+1);
    }
    std::vector<cv::Mat_<uint8_t> > & local = data[size_t(page)];
    if (local.size() < size_t(std::abs(recursion))+1) {
        local.resize(size_t(std::abs(recursion))+1);
    }
    if (local[0].empty()) {
        local[0] = cv::Mat_<uint8_t>::zeros(16*10, 32*5);
        local[0] += 255;
        writemarker(local[0], page);
    }
    for (size_t ii = 0; ii+1 < size_t(recursion); ++ii) {
        if (local[ii+1].empty()) {
            checker_recurse(local[ii], local[ii+1]);
        }
    }
    cv::Mat_<uint8_t> const & root = local[0];

    int const factor = hdcalib::Calib::computeCornerIdFactor(recursion);

    if (id.x < 0 || id.y < 0 || id.x > factor*32 || id.y > factor*32) {
        throw std::runtime_error("ID invalid");
    }

    cv::Point2i p = (id/factor)*5;
    if (id.x % factor == 0) {
        if (p.x >= root.cols && p.y >= root.cols) {
            return root(p - cv::Point2i(1,1)) > 125 ? 3:2;
        }
        if (p.x >= root.cols && p.y <= 0) {
            return root(p - cv::Point2i(1,0)) > 125 ? 2:3;
        }
        if (p.x <= 0 && p.y >= root.rows) {
            return root(p - cv::Point2i(0,2)) > 125 ? 2:3;
        }
        return root(p) > 125 ? 3:2;
    }

    // Mapping of the ID to the pixel location for recursion 1:
    // 1 -> 0
    // 3 -> 1
    // 5 -> 2
    // 7 -> 3
    // 9 -> 4
    // x -> (x-1)/2 (or just x/2)

    // Mapping of the ID to the pixel location for recursion 2:
    // 1-9 -> 0
    // 11-19 -> 1
    // 21-29 -> 2
    // x -> x/10

    // Note that for recursion 1 factor = 10, for recursion 2 factor = 50
    // General computation: x/(factor/5)

    p = (id / (factor/5));

    bool white = root(p) > 125;

    for (int ii = 0; ii < recursion; ++ii) {
        white = !white;
    }

    return white ? 1:0;

}
