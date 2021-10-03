/**
  This file was heavily inspired (read: mostly copied) from
  https://github.com/puzzlepaint/camera_calibration
  therefore we include their license:

Copyright 2019 ETH Zürich, Thomas Schöps

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
  may be used to endorse or promote products derived from this software
  without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

  */

#undef NDEBUG
#include <cassert>

#include "hdcalib.h"

#include <iostream>

#include <endian.h>

/* Get machine dependent optimized versions of byte swapping functions.  */
#include <bits/byteswap.h>
#include <bits/uintn-identity.h>

namespace hdcalib {

typedef size_t usize;
typedef int64_t i64;
typedef uint64_t u64;
typedef int32_t i32;
typedef uint32_t u32;
typedef int16_t i16;
typedef uint16_t u16;
typedef int8_t i8;
typedef uint8_t u8;

/* Functions to convert between host and network byte order.

   Please note that these functions normally take `unsigned long int' or
   `unsigned short int' values as arguments and also return them.  But
   this was a short-sighted decision since on different systems the types
   may have different representations but the values are always the same.  */

extern uint32_t ntohl (uint32_t __netlong) __THROW __attribute__ ((__const__));
extern uint16_t ntohs (uint16_t __netshort)
__THROW __attribute__ ((__const__));
extern uint32_t htonl (uint32_t __hostlong)
__THROW __attribute__ ((__const__));
extern uint16_t htons (uint16_t __hostshort)
__THROW __attribute__ ((__const__));

//#ifdef __OPTIMIZE__
/* We can optimize calls to the conversion functions.  Either nothing has
   to be done or we are using directly the byte-swapping functions which
   often can be inlined.  */
# if __BYTE_ORDER == __BIG_ENDIAN
/* The host byte order is the same as network byte order,
   so these functions are all just identity.  */
# define ntohl(x)	__uint32_identity (x)
# define ntohs(x)	__uint16_identity (x)
# define htonl(x)	__uint32_identity (x)
# define htons(x)	__uint16_identity (x)
# else
#  if __BYTE_ORDER == __LITTLE_ENDIAN
#   define ntohl(x)	__bswap_32 (x)
#   define ntohs(x)	__bswap_16 (x)
#   define htonl(x)	__bswap_32 (x)
#   define htons(x)	__bswap_16 (x)
#  endif
# endif
//#endif

void write_one(const u8* data, FILE* file) {
    fwrite(data, sizeof(u8), 1, file);
}

void write_one(const i8* data, FILE* file) {
    fwrite(data, sizeof(i8), 1, file);
}

void write_one(const u16* data, FILE* file) {
    u16 temp = htons(*data);
    fwrite(&temp, sizeof(u16), 1, file);
}

void write_one(const i16* data, FILE* file) {
    i16 temp = htons(*data);
    fwrite(&temp, sizeof(i16), 1, file);
}

void write_one(const u32* data, FILE* file) {
    u32 temp = htonl(*data);
    fwrite(&temp, sizeof(u32), 1, file);
}

void write_one(const i32* data, FILE* file) {
    i32 temp = htonl(*data);
    fwrite(&temp, sizeof(i32), 1, file);
}

void write_one(const float* data, FILE* file) {
    // TODO: Does this require a potential endian swap?
    fwrite(data, sizeof(float), 1, file);
}

bool Calib::saveStarsDatasetDense(const string &filename) {

    FILE* file = fopen(filename.c_str(), "wb");
    if (!file) {
        return false;
    }

    typedef size_t usize;
    typedef int64_t i64;
    typedef uint64_t u64;
    typedef int32_t i32;
    typedef uint32_t u32;
    typedef int16_t i16;
    typedef uint16_t u16;
    typedef int8_t i8;
    typedef uint8_t u8;

    // File format identifier
    u8 header[10];
    header[0] = 'c';
    header[1] = 'a';
    header[2] = 'l';
    header[3] = 'i';
    header[4] = 'b';
    header[5] = '_';
    header[6] = 'd';
    header[7] = 'a';
    header[8] = 't';
    header[9] = 'a';
    fwrite(header, 1, 10, file);

    // File format version
    const u32 version = 0;
    write_one(&version, file);

    // Cameras
    u32 num_cameras = 1;
    write_one(&num_cameras, file);
    for (int camera_index = 0; camera_index < num_cameras; ++ camera_index) {
        const cv::Size image_size = getImageSize();
        u32 width = image_size.width;
        write_one(&width, file);
        u32 height = image_size.height;
        write_one(&height, file);
    }

    // Imagesets
    u32 num_imagesets = data.size();
    write_one(&num_imagesets, file);

    std::map<i32, cv::Vec2i> target_geometry;

    runningstats::RunningStats points_per_image;

    for (int imageset_index = 0; imageset_index < num_imagesets; ++ imageset_index) {
        std::vector<Corner> main_markers = data[imageFiles[imageset_index]].getCorners();
        points_per_image.push_unsafe(main_markers.size());

        const string& filename = imageFiles[imageset_index];
        u32 filename_len = filename.size();
        write_one(&filename_len, file);
        fwrite(filename.data(), 1, filename_len, file);

        //for (int camera_index = 0; camera_index < num_cameras; ++ camera_index) {
        u32 num_features = main_markers.size();
        write_one(&num_features, file);
        for (const Corner& feature : main_markers) {
            write_one(&feature.p.x, file);
            write_one(&feature.p.y, file);
            cv::Vec2i id = feature.id;
            i32 id_32bit = id[0]*320 + id[1];
            target_geometry[id_32bit] = id;
            write_one(&id_32bit, file);
        }
        //}
    }

    // Known geometries
    u32 num_known_geometries = 1;
    write_one(&num_known_geometries, file);
    float cell_length_in_meters = 0.05;
    write_one(&cell_length_in_meters, file);

    u32 feature_id_to_position_size = target_geometry.size();
    write_one(&feature_id_to_position_size, file);
    for (const std::pair<const int, Vec2i>& item : target_geometry) {
        i32 id_32bit = item.first;
        write_one(&id_32bit, file);
        i32 x_32bit = item.second[0];
        write_one(&x_32bit, file);
        i32 y_32bit = item.second[1];
        write_one(&y_32bit, file);
    }

    clog::L("Calib::saveStarsDatasetDense", 2) << "Points per image: " << points_per_image.print();

    fclose(file);
    return true;
}

bool Calib::saveStarsDataset(const string &filename, int const offset_x, int const offset_y) {

    FILE* file = fopen(filename.c_str(), "wb");
    if (!file) {
        return false;
    }

    typedef size_t usize;
    typedef int64_t i64;
    typedef uint64_t u64;
    typedef int32_t i32;
    typedef uint32_t u32;
    typedef int16_t i16;
    typedef uint16_t u16;
    typedef int8_t i8;
    typedef uint8_t u8;

    // File format identifier
    u8 header[10];
    header[0] = 'c';
    header[1] = 'a';
    header[2] = 'l';
    header[3] = 'i';
    header[4] = 'b';
    header[5] = '_';
    header[6] = 'd';
    header[7] = 'a';
    header[8] = 't';
    header[9] = 'a';
    fwrite(header, 1, 10, file);

    // File format version
    const u32 version = 0;
    write_one(&version, file);

    // Cameras
    u32 num_cameras = 1;
    write_one(&num_cameras, file);
    for (int camera_index = 0; camera_index < num_cameras; ++ camera_index) {
        const cv::Size image_size = getImageSize();
        u32 width = image_size.width;
        write_one(&width, file);
        u32 height = image_size.height;
        write_one(&height, file);
    }

    // Imagesets
    u32 num_imagesets = data.size();
    write_one(&num_imagesets, file);

    std::map<i32, cv::Vec2i> target_geometry;

    runningstats::RunningStats points_per_image;

    std::set<cv::Scalar_<int>, cmpScalar> all_ids;
    for (auto& it : data) {
        for (Corner const& c : it.second.getCorners()) {
            all_ids.insert(c.getSimpleIdLayer());
        }
    }

    std::map<cv::Scalar_<int>, int, cmpScalar> id2index;
    {
        int counter = 0;
        for (cv::Scalar_<int> const& it : all_ids) {
            id2index[it] = counter;
            ++counter;
        }
    }

    for (int imageset_index = 0; imageset_index < num_imagesets; ++ imageset_index) {
        std::vector<Corner> main_markers = data[imageFiles[imageset_index]].getMainMarkerAdjacent(offset_x, offset_y);
        points_per_image.push_unsafe(main_markers.size());

        const string& filename = imageFiles[imageset_index];
        u32 filename_len = filename.size();
        write_one(&filename_len, file);
        fwrite(filename.data(), 1, filename_len, file);

        //for (int camera_index = 0; camera_index < num_cameras; ++ camera_index) {
        u32 num_features = main_markers.size();
        write_one(&num_features, file);
        for (const Corner& feature : main_markers) {
            write_one(&feature.p.x, file);
            write_one(&feature.p.y, file);
            if (offset_x != (feature.id.x % cornerIdFactor)) {
                clog::L("Calib::saveStarsDataset", 0) << "Fatal: id.x = " << feature.id.x
                                                      << ", offset_x = " << offset_x
                                                      << ", cornerIdFactor = " << cornerIdFactor
                                                      << ", modulo = " << (feature.id.x % cornerIdFactor);
            }
            if (offset_y != (feature.id.y % cornerIdFactor)) {
                clog::L("Calib::saveStarsDataset", 0) << "Fatal: id.y = " << feature.id.y
                                                      << ", offset_y = " << offset_y
                                                      << ", cornerIdFactor = " << cornerIdFactor
                                                      << ", modulo = " << (feature.id.y % cornerIdFactor);
            }
            assert(offset_x == (feature.id.x % cornerIdFactor));
            assert(offset_y == (feature.id.y % cornerIdFactor));
            //cv::Vec2i id((feature.id.x-offset_x)/cornerIdFactor, (feature.id.y-offset_y)/cornerIdFactor);
            cv::Vec2i id = feature.id;
            i32 id_32bit = id[0]*320 + id[1];
            //i32 id_32bit = id2index[feature.getSimpleIdLayer()];
            target_geometry[id_32bit] = id;
            write_one(&id_32bit, file);
        }
        //}
    }

    // Known geometries
    u32 num_known_geometries = 1;
    write_one(&num_known_geometries, file);
    float cell_length_in_meters = 0.05;
    write_one(&cell_length_in_meters, file);

    u32 feature_id_to_position_size = target_geometry.size();
    write_one(&feature_id_to_position_size, file);
    for (const std::pair<const int, Vec2i>& item : target_geometry) {
        i32 id_32bit = item.first;
        write_one(&id_32bit, file);
        i32 x_32bit = item.second[0];
        write_one(&x_32bit, file);
        i32 y_32bit = item.second[1];
        write_one(&y_32bit, file);
    }

    clog::L("Calib::saveStarsDataset", 2) << "Points per image: " << points_per_image.print();

    fclose(file);
    return true;
}

bool Calib::saveStarsDatasetPartial(const string &filename, int const factor, int const mod) {

    FILE* file = fopen(filename.c_str(), "wb");
    if (!file) {
        return false;
    }

    typedef size_t usize;
    typedef int64_t i64;
    typedef uint64_t u64;
    typedef int32_t i32;
    typedef uint32_t u32;
    typedef int16_t i16;
    typedef uint16_t u16;
    typedef int8_t i8;
    typedef uint8_t u8;

    // File format identifier
    u8 header[10];
    header[0] = 'c';
    header[1] = 'a';
    header[2] = 'l';
    header[3] = 'i';
    header[4] = 'b';
    header[5] = '_';
    header[6] = 'd';
    header[7] = 'a';
    header[8] = 't';
    header[9] = 'a';
    fwrite(header, 1, 10, file);

    // File format version
    const u32 version = 0;
    write_one(&version, file);

    // Cameras
    u32 num_cameras = 1;
    write_one(&num_cameras, file);
    for (int camera_index = 0; camera_index < num_cameras; ++ camera_index) {
        const cv::Size image_size = getImageSize();
        u32 width = image_size.width;
        write_one(&width, file);
        u32 height = image_size.height;
        write_one(&height, file);
    }

    // Imagesets
    u32 num_imagesets = data.size();
    write_one(&num_imagesets, file);

    std::map<i32, cv::Vec2i> target_geometry;

    runningstats::RunningStats points_per_image;

    /*
    std::set<cv::Scalar_<int>, cmpScalar> all_ids;
    for (auto& it : data) {
        for (Corner const& c : it.second.getCorners()) {
            all_ids.insert(c.getSimpleIdLayer());
        }
    }

    std::map<cv::Scalar_<int>, int, cmpScalar> id2index;
    {
        int counter = 0;
        for (cv::Scalar_<int> const& it : all_ids) {
            id2index[it] = counter;
            ++counter;
        }
    }
    */

    for (int imageset_index = 0; imageset_index < num_imagesets; ++ imageset_index) {
        std::vector<Corner> main_markers = data[imageFiles[imageset_index]].getCorners();


        const string& filename = imageFiles[imageset_index];
        u32 filename_len = filename.size();
        write_one(&filename_len, file);
        fwrite(filename.data(), 1, filename_len, file);

        //for (int camera_index = 0; camera_index < num_cameras; ++ camera_index) {
        u32 num_features = 0;
        for (const Corner& feature : main_markers) {
            if (mod != (feature.id.x % factor) || mod != (feature.id.y % factor)) {
                continue;
            }
            num_features++;
        }
        points_per_image.push_unsafe(num_features);

        write_one(&num_features, file);
        for (const Corner& feature : main_markers) {
            if (mod != (feature.id.x % factor) || mod != (feature.id.y % factor)) {
                continue;
            }
            write_one(&feature.p.x, file);
            write_one(&feature.p.y, file);
            cv::Vec2i id((feature.id.x-mod)/factor, (feature.id.y-mod)/factor);
            i32 id_32bit = id[0]*320 + id[1];
            //i32 id_32bit = id2index[feature.getSimpleIdLayer()];
            target_geometry[id_32bit] = id;
            write_one(&id_32bit, file);
        }
        //}
    }

    // Known geometries
    u32 num_known_geometries = 1;
    write_one(&num_known_geometries, file);
    float cell_length_in_meters = 0.05;
    write_one(&cell_length_in_meters, file);

    u32 feature_id_to_position_size = target_geometry.size();
    write_one(&feature_id_to_position_size, file);
    for (const std::pair<const int, Vec2i>& item : target_geometry) {
        i32 id_32bit = item.first;
        write_one(&id_32bit, file);
        i32 x_32bit = item.second[0];
        write_one(&x_32bit, file);
        i32 y_32bit = item.second[1];
        write_one(&y_32bit, file);
    }

    clog::L("Calib::saveStarsDataset", 2) << "Points per image: " << points_per_image.print();

    fclose(file);
    return true;
}

bool Calib::saveStarsDatasets(const string &prefix) {
    bool success = true;
    success &= saveStarsDatasetDense(prefix + "-dense.bin");

    success &= saveStarsDatasetPartial(prefix + "-mod-2-1.bin", 2, 1);
    success &= saveStarsDatasetPartial(prefix + "-mod-4-1.bin", 4, 1);
    success &= saveStarsDatasetPartial(prefix + "-mod-6-1.bin", 6, 1);
    success &= saveStarsDatasetPartial(prefix + "-mod-8-1.bin", 8, 1);
    for (int offset_x = 1; offset_x <= 9; offset_x += 2) {
        for (int offset_y = 1; offset_y <= 9; offset_y += 2) {
            success &= saveStarsDataset(prefix + "-" + std::to_string(offset_x) + "-" + std::to_string(offset_y) + ".bin",
                             offset_x, offset_y);
        }
    }
    return success;
}

} // namespace hdcalib
