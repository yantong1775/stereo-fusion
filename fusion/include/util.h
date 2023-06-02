#pragma once
#include <iostream>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include <string>
#include <vector>
#include <algorithm>
#include <sstream>
#include <fstream>

struct Configs
{
	float baseline, focal, doff, ratio, scale, threshold, dmin, dist_range;
	int step;
	std::string folder_path;
};

// Read config from a file.
void read_config(const std::string& filepath, Configs& config);
void read_data_list(const std::string& filepath, std::vector<std::string>& data_list);

// Given intrinsic matrix of a camera, convert depth map to point cloud
void depth2cloud(const cv::Mat& depth, const cv::Mat& K, cv::Mat& cloud);

// Given intrinsic and extrinsic matrix of a camera, convert point clouw into depth.
void cloud2depth(const cv::Mat& cloud, const cv::Mat& K, const cv::Mat& R, const cv::Mat& t, cv::Mat& depth);

// Apply geometry refinement to the sparse depth map.
void geometry_refine(cv::Mat& sparse_depth, int w_size, int row, int col);

// Extract pixel positions into a vector.
void sparsedepth2vec(const cv::Mat& sparse_depth, std::vector<double> coords);

// Metrics for evaluation.
float rmse(cv::Mat& im1, cv::Mat& im2);
float mae(cv::Mat& im1, cv::Mat& im2);
float BMP(cv::Mat& im1, cv::Mat& im2);
float avg_disp_err(cv::Mat& im1, cv::Mat& im2);

void draw_point(cv::Mat& img, cv::Mat& sparse_disp, int width, int height);
void save_color_img(const char* path, cv::Mat& input);

void Colorize(cv::Mat& input, cv::Mat& output);

struct Color
{
	int r, g, b;
};

void draw_occlusion(cv::Mat& img, std::vector<int>& occlusion, Color c);

