#include "StereoMatcher.h"
#include "delaunator.hpp"
#include <cmath>
#include <iterator>
#include <algorithm>
#include <chrono>

bool compare_edge_zdiff(const Edge& a, const Edge& b) {
	return a.getZDifference() > b.getZDifference();
}


Matcher::Matcher(int width, int height, cv::Mat& l_img, cv::Mat& r_img,
	float baseline, float focal, float doff, float ratio, float scale, float threshold)
	: width(width), height(height), baseline(baseline), focal(focal), doff(doff),
	  ratio(ratio), scale(scale), threshold(threshold),
	  l_img(l_img), r_img(r_img)
{
	l_prior_disparity = new cv::Mat(height, width, CV_32FC1, cv::Scalar(0)); // left prior disparity
	r_prior_disparity = new cv::Mat(height, width, CV_32FC1, cv::Scalar(0)); // left prior disparity
	output_disp = new cv::Mat(height, width, CV_32FC1, cv::Scalar(0)); // output disparity

	l_sparse_disparity = new cv::Mat(height, width, CV_32FC1, cv::Scalar(0)); // left prior disparity
	r_sparse_disparity = new cv::Mat(height, width, CV_32FC1, cv::Scalar(0)); // right prior disparity

	refine_r_sparse_disparity = new cv::Mat(height, width, CV_32FC1, cv::Scalar(0));
	refine_l_sparse_disparity = new cv::Mat(height, width, CV_32FC1, cv::Scalar(0));

	stereo_occlusion = new std::vector<bool>(height * width, false);
	depth_occlusion = new std::vector<bool>(height * width, false);
}

Matcher::~Matcher()
{
	delete l_prior_disparity;
	delete r_prior_disparity;
	delete output_disp;
	delete l_sparse_disparity;
	delete r_sparse_disparity;
	delete refine_r_sparse_disparity;
	delete refine_l_sparse_disparity;
	delete stereo_occlusion;
	delete depth_occlusion;
}
void Matcher::sample_disp(cv::Mat& gt_disp)
{
	/*
	This function samples disparity point on the ground truth image
	in order to simulate sparse depth point when project low resolution depth map onto high resolution RGB map.
	The sparse disparity points are stored in two empty cv::Mat with data type uint8_t.
	*/

	for (int i = 0; i < height; i = i + 10)
	{
		for (int j = 0; j < width; j = j + 10)
		{
			float disp = gt_disp.at<float>(i, j);
			l_sparse_disparity->at<float>(i, j) = disp;
			output_disp->at<float>(i, j) = disp;
			if (j - (int)disp > 0)
			{
				r_sparse_disparity->at<float>(i, j - disp) = disp;
			}
		}
	}
}

void Matcher::refine_disp(cv::Mat& sparse_disp, cv::Mat& refine_disp, cv::Mat& stereo_img, int w_size)
{
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (sparse_disp.at<float>(i, j) == 0) continue;
			geometry_refine_window(sparse_disp, w_size, i, j);
		}
	}
}

void Matcher::geometry_refine_window(cv::Mat& sparse_disp, int w_size, int row, int col)
{
	/*
		This function is to refine the sampled sparse disparity map based on geometry.
		The basic idea is as follows:
			Set a window around the center pixel,
			preserve the pixel with smallest depth/ largest disparity
			discard the other.
	*/
	int prev_row = row, prev_col = col;
	float prev_disp = sparse_disp.at<float>(row, col);
	int width = sparse_disp.cols, height = sparse_disp.rows;

	for (int i = row - w_size; i <= row + w_size; i++)
	{
		for (int j = col - w_size; j <= col + w_size; j++)
		{
			if (i < 0 || i >= height || j < 0 || j >= width) continue;
			float disp = sparse_disp.at<float>(i, j);
			if (disp == 0) continue;
			if (disp < prev_disp)
			{
				sparse_disp.at<float>(i, j) = 0;
				continue;
			}
			prev_disp = disp;
			prev_row = i;
			prev_col = j;
		}
	}
}

void Matcher::init_coords()
{
	/*
			This function initialize the coordinates of the sparse disparity points.
			The coordinates are stored in a vector of int.
	*/
	for (int i = 0; i < height; i = i + 10)
	{
		for (int j = 0; j < width; j = j + 10)
		{
			if (l_sparse_disparity->at<float>(i, j) == 0) continue;
			l_coords.push_back(j);
			l_coords.push_back(i);

		}
	}

	for (int i = 0; i < height; i = i + 10)
	{
		for (int j = 0; j < width; j = j + 10)
		{
			if (r_sparse_disparity->at<float>(i, j) == 0) continue;
			r_coords.push_back(j);
			r_coords.push_back(i);
		}
	}
}

void Matcher::compute_seed()
{
	/*
		This function initialize the correspondance seed between left and right image.
		The seeds are represented by Seed Struct.
		In This Implementation, the correspondance are calculated by the ground truth
		disparity map provided by MiddleBury dataset
	*/

	size_t size = l_coords.size();
	for (int i = 0; i < size; i+=2)
	{
		int x = l_coords[i], y = l_coords[i + 1];

		float disparity = l_sparse_disparity->at<float>(y, x);
		int cor_x = x - (int)disparity;
		if (cor_x < 0 || cor_x >= width || disparity == 0) continue;
		float r_disparity = r_sparse_disparity->at<float>(y, cor_x);
		if (r_disparity == 0) continue;
		seeds.push_back({ { x, cor_x, y }, 0 });
	}
}

void Matcher::delaunay_triangulation()
{
	/*
		This function is a 2D triangulation of the sparse tof points.
	*/

	delaunator::Delaunator left_d(l_coords);
	delaunator::Delaunator right_d(r_coords);
	for (int i = 0; i < left_d.triangles.size(); i++)
	{
		l_indices.push_back((int)(left_d.triangles[i]));
	}
	for (int i = 0; i < right_d.triangles.size(); i++)
	{
		r_indices.push_back((int)(right_d.triangles[i]));
	}
}

// Interpolate the depth of a point inside a triangle
cv::Vec3f Matcher::barycentric(cv::Point2f p, cv::Point2f p1, cv::Point2f p2, cv::Point2f p3) {
	// Calculate the barycentric coordinates of the point inside the triangle
	cv::Vec3f tmp1{ p3.x - p1.x, p2.x - p1.x, p1.x - p.x };
	cv::Vec3f tmp2{ p3.y - p1.y, p2.y - p1.y, p1.y - p.y };

	cv::Vec3f u = tmp1.cross(tmp2);

	if (std::abs(u[2]) < 1) return { -1, 1, 1 };
	return { 1.f - (u[0] + u[1]) / u[2], u[1] / u[2], u[0] / u[2] };
}

cv::Vec3f Matcher::barycentric(cv::Point2f p, cv::Point3f p1, cv::Point3f p2, cv::Point3f p3) {
	// Calculate the barycentric coordinates of the point inside the triangle
	cv::Vec3f tmp1{ p3.x - p1.x, p2.x - p1.x, p1.x - p.x };
	cv::Vec3f tmp2{ p3.y - p1.y, p2.y - p1.y, p1.y - p.y };

	cv::Vec3f u = tmp1.cross(tmp2);

	if (std::abs(u[2]) < 1) return { -1, 1, 1 };
	return { 1.f - (u[0] + u[1]) / u[2], u[1] / u[2], u[0] / u[2] };
}


void Matcher::compute_prior_bbox(std::vector<int>& indices, std::vector<double>& coords, cv::Mat& sparse_disp, cv::Mat& prior_disp)
{
	for (int i = 0; i < indices.size(); i += 3) // Iterate through the indices of triangle vertices(i, i+1, i+2).
	{
		// Now we have indices of the triangle vertcies, which are indices[i], indices[i+1], indices[i+2].
		// Find the coordinates of the vertices.
		// Since coords is a 2d position (x, y), the coordinates of ith vertex are coords[2 * indices[i]] and coords[2 * indices[i] + 1].

		cv::Point2f v1{ (float)coords[2 * indices[i]],     (float)coords[2 * indices[i] + 1] };
		cv::Point2f v2{ (float)coords[2 * indices[i + 1]], (float)coords[2 * indices[i + 1] + 1] };
		cv::Point2f v3{ (float)coords[2 * indices[i + 2]], (float)coords[2 * indices[i + 2] + 1] };

		std::vector<cv::Point2f> triangle = { v1, v2, v3 };

		// Using a bounding box to draw the triangle.
		// Find the bonding box
		float minx = std::min(std::min(v1.x, v2.x), v3.x);
		float maxx = std::max(std::max(v1.x, v2.x), v3.x);
		float miny = std::min(std::min(v1.y, v2.y), v3.y);
		float maxy = std::max(std::max(v1.y, v2.y), v3.y);

		// cast to int
		int min_x = (int)std::floor(minx);
		int max_x = (int)std::ceil(maxx);
		int min_y = (int)std::floor(miny);
		int max_y = (int)std::ceil(maxy);

		float d1 = sparse_disp.at<float>((int)v1.y, (int)v1.x);
		float d2 = sparse_disp.at<float>((int)v2.y, (int)v2.x);
		float d3 = sparse_disp.at<float>((int)v3.y, (int)v3.x);

		// Scan the bounding box and draw the triangle.

		for (int i = min_x; i <= max_x; i++)
		{
			for (int j = min_y; j <= max_y; j++)
			{
				cv::Point2f p{ (float)i + 0.5f, (float)j + 0.5f };
				cv::Vec3f b_coord = barycentric(p, triangle[0], triangle[1], triangle[2]);
				if (b_coord[0] < 0 || b_coord[1] < 0 || b_coord[2] < 0) continue;

				if (prior_disp.at<float>(j, i) != 0) continue;
				float interpo_disp = b_coord[0] * d1 + b_coord[1] * d2 + b_coord[2] * d3;
				prior_disp.at<float>(j, i) = std::round(interpo_disp);

			}
		}
	}
}

void Matcher::compute_prior_with_intensity(std::vector<int>& indices, std::vector<double>& coords, cv::Mat& sparse_disp, cv::Mat& prior_disp)
{
	float inv_gamma_c = 0.1f, epsilon = 0.2f;
	for (int i = 0; i < indices.size(); i += 3) // Iterate through the indices of triangle vertices(i, i+1, i+2).
	{
		// Now we have indices of the triangle vertcies, which are indices[i], indices[i+1], indices[i+2].
		// Find the coordinates of the vertices.
		// Since coords is a 2d position (x, y), the coordinates of ith vertex are coords[2 * indices[i]] and coords[2 * indices[i] + 1].

		cv::Point2f v1{ (float)coords[2 * indices[i]],     (float)coords[2 * indices[i] + 1] };
		cv::Point2f v2{ (float)coords[2 * indices[i + 1]], (float)coords[2 * indices[i + 1] + 1] };
		cv::Point2f v3{ (float)coords[2 * indices[i + 2]], (float)coords[2 * indices[i + 2] + 1] };

		std::vector<cv::Point2f> triangle = { v1, v2, v3 };

		//calculate the area of the triangle
		float area = 0.5f * ((v2.x - v1.x) * (v3.y - v1.y) - (v3.x - v1.x) * (v2.y - v1.y));
		if (std::abs(area) > 100) continue;

		// Get the depth of each triangle vertices.
		std::vector<float> disparity_arr;
		disparity_arr.reserve(3);

		float d1 = sparse_disp.at<float>((int)v1.y, (int)v1.x);
		float d2 = sparse_disp.at<float>((int)v2.y, (int)v2.x);
		float d3 = sparse_disp.at<float>((int)v3.y, (int)v3.x);

		float d_consistency_12 = exp(-std::abs(d1 - d2) * inv_gamma_c);
		float d_consistency_13 = exp(-std::abs(d1 - d3) * inv_gamma_c);

		bool d_consistent = (d_consistency_12 > epsilon) && (d_consistency_13 > epsilon);

		disparity_arr.push_back(d1);
		disparity_arr.push_back(d2);
		disparity_arr.push_back(d3);

		std::vector<uint8_t> intensity_arr;
		intensity_arr.reserve(3);

		uint8_t intensity1 = l_img.at<uint8_t>((int)v1.y, (int)v1.x);
		uint8_t intensity2 = l_img.at<uint8_t>((int)v2.y, (int)v2.x);
		uint8_t intensity3 = l_img.at<uint8_t>((int)v3.y, (int)v3.x);

		intensity_arr.push_back(intensity1);
		intensity_arr.push_back(intensity2);
		intensity_arr.push_back(intensity3);

		// Using a bounding box to draw the triangle.
		// Find the bonding box
		float minx = std::min(std::min(v1.x, v2.x), v3.x);
		float maxx = std::max(std::max(v1.x, v2.x), v3.x);
		float miny = std::min(std::min(v1.y, v2.y), v3.y);
		float maxy = std::max(std::max(v1.y, v2.y), v3.y);

		// cast to int
		int min_x = (int)std::floor(minx);
		int max_x = (int)std::ceil(maxx);
		int min_y = (int)std::floor(miny);
		int max_y = (int)std::ceil(maxy);

		// Scan the bounding box and draw the triangle.

		for (int i = min_x; i <= max_x; i++)
		{
			for (int j = min_y; j <= max_y; j++)
			{
				if (d_consistent)
				{
					cv::Point2f p{ (float)i + 0.5f, (float)j + 0.5f };
					cv::Vec3f b_coord = barycentric(p, triangle[0], triangle[1], triangle[2]);
					if (b_coord[0] < 0 || b_coord[1] < 0 || b_coord[2] < 0) continue;

					if (prior_disp.at<float>(j, i) != 0) continue;
					float interpo_disp = b_coord[0] * d1 + b_coord[1] * d2 + b_coord[2] * d3;
					prior_disp.at<float>(j, i) = std::round(interpo_disp);
					continue;
				}

				float cur_intensity = (float)l_img.at<uint8_t>(j, i);
				std::vector<int> valid_vertex;
				std::vector<float> consistency_arr;
				valid_vertex.reserve(3);
				for (int idx = 0; idx < 3; idx++)
				{
					float parent_intensity = (float)intensity_arr[idx];
					float consistency = exp(-std::abs(parent_intensity - cur_intensity) * inv_gamma_c);
					if (consistency > epsilon)
					{
						valid_vertex.push_back(idx);
						consistency_arr.push_back(consistency);
					}
				}
				
				// If there are 3 valid vertices, then the pixel lies in the same surface as the triangle.
				if (valid_vertex.size() == 3)
				{
					cv::Point2f p{ (float)i + 0.5f, (float)j + 0.5f };
					cv::Vec3f b_coord = barycentric(p, triangle[0], triangle[1], triangle[2]);
					if (b_coord[0] < 0 || b_coord[1] < 0 || b_coord[2] < 0) continue;

					if (prior_disp.at<float>(j, i) != 0) continue;
					float interpo_disp = b_coord[0] * d1 + b_coord[1] * d2 + b_coord[2] * d3;
					prior_disp.at<float>(j, i) = std::round(interpo_disp);
				}
				else if (valid_vertex.size() == 0) continue;
				else
				{
					// Find the min value and index of consistency_arr.

					int idx = std::distance(std::begin(consistency_arr), std::max_element(std::begin(consistency_arr), std::end(consistency_arr)));
					prior_disp.at<float>(j, i) = std::round(disparity_arr[valid_vertex[idx]]);
				}
			}
		}
	}
}

float Matcher::denominator_window(int size, const Seed& s)
{
	float ans = 0.0001;
	int u0 = s.correspondance[0], u1 = s.correspondance[1], v = s.correspondance[2];
	int range = size / 2;
	for (int i = -range; i <= range; i++)
	{
		for (int j = -range; j <= range; j++)
		{
			if (u0 + i < 0 || u0 + i >= width || u1 + i < 0 || u1 + i >= width || v + j < 0 || v + j >= height) continue;
			cv::Vec3i l_pixel = l_img.at<cv::Vec3b>(v + j, u0 + i);
			cv::Vec3i r_pixel = r_img.at<cv::Vec3b>(v + j, u1 + i);
			for (int k = 0; k < 3; k++)
			{
				ans += l_pixel[k] * l_pixel[k] + r_pixel[k] * r_pixel[k];
			}
			//ans += (tmp);
			
		}
	}
	return ans;
}

float Matcher::numerator_window_grayscale(int size, const Seed& s)
{
	float ans = 0;
	int u0 = s.correspondance[0], u1 = s.correspondance[1], v = s.correspondance[2];
	for (int i = -(size / 2); i <= size / 2; i++)
	{
		for (int j = -(size / 2); j <= size / 2; j++)
		{
			if (u0 + i < 0 || u0 + i >= width || u1 + i < 0 || u1 + i >= width || v + j < 0 || v + j >= height) continue;
			// read the pixel value from the grayscale image
			int l_pixel = l_img.at<char>(v + j, u0 + i);
			int r_pixel = r_img.at<char>(v + j, u1 + i);
			int res = l_pixel - r_pixel;
			
			ans += (float)(res * res);

		}
	}
	return ans;
}

float Matcher::denominator_window_grayscale(int size, const Seed& s)
{
	float ans = 0.0001;
	int u0 = s.correspondance[0], u1 = s.correspondance[1], v = s.correspondance[2];
	int range = size / 2;
	for (int i = -range; i <= range; i++)
	{
		for (int j = -range; j <= range; j++)
		{
			if (u0 + i < 0 || u0 + i >= width || u1 + i < 0 || u1 + i >= width || v + j < 0 || v + j >= height) continue;
			int l_pixel = l_img.at<char>(v + j, u0 + i);
			int r_pixel = r_img.at<char>(v + j, u1 + i);

			ans += (float)(l_pixel * l_pixel + r_pixel * r_pixel);
		}
	}
	return ans;
}

float Matcher::image_similarity(int size, Seed& s)
{
	float denom = 0.0001f;
	float num = 0.0f;
	int u0 = s.correspondance[0], u1 = s.correspondance[1], v = s.correspondance[2];
	int range = size / 2;
	for (int i = -range; i <= range; i++)
	{
		for (int j = -range; j <= range; j++)
		{
			if (u0 + i < 0 || u0 + i >= width || u1 + i < 0 || u1 + i >= width || v + j < 0 || v + j >= height) continue;
			cv::Vec3i l_pixel = l_img.at<cv::Vec3b>(v + j, u0 + i);
			cv::Vec3i r_pixel = r_img.at<cv::Vec3b>(v + j, u1 + i);
			cv::Vec3i res = (l_pixel - r_pixel);
			for (int k = 0; k < 3; k++)
			{
				num += res[k] * res[k];
				denom += l_pixel[k] * l_pixel[k] + r_pixel[k] * r_pixel[k];
			}
			//ans += (tmp);

		}
	}
	return num / denom;
}

float Matcher::image_similarity_grayscale(int size, Seed& s)
{
	float denom = scale;
	float num = 0.0f;
	int u0 = s.correspondance[0], u1 = s.correspondance[1], v = s.correspondance[2];
	int range = size / 2;
	char* lp;
	char* rp;
	for (int i = -range; i <= range; i++)
	{
		if (v + i < 0 || v + i >= height) continue;
		lp = l_img.ptr<char>(v + i);
		rp = r_img.ptr<char>(v + i);
		for (int j = -range; j <= range; j++)
		{
			if (u0 + j < 0 || u0 + j >= width || u1 + j < 0 || u1 + j >= width) continue;
			int l_pixel = lp[u0 + j];
			int r_pixel = rp[u1 + j];
			int res = l_pixel - r_pixel;
			
			num += res * res;
			denom += l_pixel * l_pixel + r_pixel * r_pixel;
			
		}
	}
	return num / denom;
}

float Matcher::image_similarity_cos_grayscale(int size, Seed& s)
{
	float denom = 0.0001f;
	float l_norm_square = 0.0f, r_norm_square = 0.0f;
	float num = 0.0f;
	int u0 = s.correspondance[0], u1 = s.correspondance[1], v = s.correspondance[2];
	int range = size / 2;
	for (int i = -range; i <= range; i++)
	{
		for (int j = -range; j <= range; j++)
		{
			if (u0 + i < 0 || u0 + i >= width || u1 + i < 0 || u1 + i >= width || v + j < 0 || v + j >= height) continue;
			int l_pixel = l_img.at<char>(v + j, u0 + i);
			int r_pixel = r_img.at<char>(v + j, u1 + i);
			int res = l_pixel - r_pixel;

			num = l_pixel * r_pixel;
			l_norm_square = l_pixel * l_pixel;
			r_norm_square = r_pixel * r_pixel;

		}
	}
	denom = sqrt(l_norm_square * r_norm_square);
	return num / denom;
}

float Matcher::image_similarity_NCC_grayscale(int size, Seed& s)
{
	float denom = 0.0001f;
	float l_norm_square = 0.0f, r_norm_square = 0.0f;
	float num = 0.0f;
	int u0 = s.correspondance[0], u1 = s.correspondance[1], v = s.correspondance[2];
	int range = size / 2;

	// calculate the mean of the left patch and the right patch.
	float l_mean = 0.0f, r_mean = 0.0f;
	for (int i = -range; i <= range; i++)
	{
		for (int j = -range; j <= range; j++)
		{
			if (u0 + i < 0 || u0 + i >= width || u1 + i < 0 || u1 + i >= width || v + j < 0 || v + j >= height) continue;
			int l_pixel = l_img.at<char>(v + j, u0 + i);
			int r_pixel = r_img.at<char>(v + j, u1 + i);

			l_mean += l_pixel;
			r_pixel += r_pixel;
		}
	}
	
	l_mean /= size * size;
	r_mean /= size * size;

	for (int i = -range; i <= range; i++)
	{
		for (int j = -range; j <= range; j++)
		{
			if (u0 + i < 0 || u0 + i >= width || u1 + i < 0 || u1 + i >= width || v + j < 0 || v + j >= height) continue;
			int l_pixel = l_img.at<char>(v + j, u0 + i);
			int r_pixel = r_img.at<char>(v + j, u1 + i);

			num += (l_pixel - l_mean) * (r_pixel - r_mean);
			l_norm_square += (l_pixel - l_mean) * (l_pixel - l_mean);
			r_norm_square += (r_pixel - r_mean) * (r_pixel - r_mean);

		}
	}
	denom += sqrt(l_norm_square * l_norm_square);
	return num / denom;
}

void Matcher::compute_similarity(Seed& s)
{
	// TODO
	/*
		This function compute the similarity of a pair of seed
		The inputs of this function are:
			s : seed
			l_img, r_img : left and right RGB images
			prior_disparity : prior disparity computed before
	*/

	int u0 = s.correspondance[0], u1 = s.correspondance[1], v = s.correspondance[2];
	float prior_d = l_prior_disparity->at<float>(v, u0);
	float disparity = u0 - u1;
	//s.similarity = exp(-(n / (d * 0.1)) - ((disparity - prior_d) * (disparity - prior_d) * 0.5 * inv_sigma_p));

	float sqr_disp_diff = (disparity - prior_d) * (disparity - prior_d);
	float img_similarity = image_similarity_grayscale(10, s);
	s.similarity = exp((-img_similarity - sqr_disp_diff * ratio * 0.5f));
	//s.similarity = exp(-(image_similarity(5, s) * inv_sigma_s) - ((disparity - prior_d)*(disparity - prior_d) * 0.5 * inv_sigma_p));
}

void Matcher::init_similarity()
{
	for (int i = 0; i < seeds.size(); i++) compute_similarity(seeds[i]);
}

static void find_four_neighborhood(const Seed& s, std::vector<std::vector<Seed>>& four_neighbor)
{
	/*
		This funtion defines and returns the four neighborhoods of the input seeds.
	*/
	int u0 = s.correspondance[0];
	int u1 = s.correspondance[1];
	int v = s.correspondance[2];


	four_neighbor.emplace_back(std::vector<Seed>{
		{ {u0 - 1, u1 - 1, v}, 0},
		{ {u0 - 1, u1, v}, 0 },
		{ {u0 - 1, u1 - 2, v}, 0 },
	});
	four_neighbor.emplace_back(std::vector<Seed>{
		{{u0 + 1, u1 + 1, v}, 0},
		{{u0 + 1, u1 + 2, v}, 0},
		{{u0 + 1, u1, v}, 0},
		});
	four_neighbor.emplace_back(std::vector<Seed>{
		{{u0, u1, v - 1}, 0},
		{{u0, u1 + 1, v - 1}, 0},
		{{u0, u1 - 1, v - 1}, 0},
		});
	four_neighbor.emplace_back(std::vector<Seed>{
		{{u0, u1, v + 1}, 0},
		{{u0, u1 + 1, v + 1}, 0},
		{{u0, u1 - 1, v + 1}, 0},
		});
}

void Matcher::process_seed(Seed& seed)
{
	int u0 = seed.correspondance[0], u1 = seed.correspondance[1], v = seed.correspondance[2];
	if (v < 0 || v >= height || u0 < 0 || u0 >= width || u1 < 0 || u1 >= width) return;
	if (output_disp->at<float>(v, u0) != 0) return;
	compute_similarity(seed);
}

void Matcher::process_subset(std::vector<std::vector<Seed>>& neighbor, std::priority_queue<Seed>& heap, int j)
{
	std::vector<Seed> subset = neighbor[j];
	for (int i = 0; i < subset.size(); i++)
	{
		process_seed(subset[i]);
	}

	auto best_seed = std::max_element(subset.begin(), subset.end());
	int u = best_seed->correspondance[0], u1 = best_seed->correspondance[1], v = best_seed->correspondance[2];
	
	if (best_seed->similarity >= threshold && output_disp->at<float>(v, u) == 0)
	{
		output_disp->at<float>(v, u) = u - u1;
		heap.push(*best_seed);
	}
}

void Matcher::grow()
{
	/* 
		This is the main part of the growing algorithm.
		My implementation based on heap/priority queue.
		Every time, the seed with best similarity will be pop out of the heap.
		
		We consider 4 neighborhood area of this seed.

		For each of the neighboring area of this seed, 
		we need to find the neighboring seed with the best similarity among its neighborhood.

		If the seed's similarity exceeds some threshold, and the this pixel hasn't been matched yet:
			those seeds will be add to the heap, and its disparity will be updated to the output.

		The growing algorithm will continue until the heap is empty.
	*/
	// Make a heap based on similarity
	std::priority_queue<Seed> heap(seeds.begin(), seeds.end());
	int iter_step = 0;

	while (!heap.empty())
	{
		Seed s = heap.top();
		heap.pop();
		//if (iter_step == 100000) break;
		//std::cout << iter_step << std::endl;
		std::vector<std::vector<Seed>> four_neighbor;
		find_four_neighborhood(s, four_neighbor);
		for (int i = 0; i < 4; i++)
		{
			process_subset(four_neighbor, heap, i);
		}
		iter_step++;
	}
}

void Matcher::run()
{
	refine_disp(*l_sparse_disparity, *refine_l_sparse_disparity, l_img, 5);
	refine_disp(*r_sparse_disparity, *refine_r_sparse_disparity, r_img, 5);
	init_coords();
	compute_seed();
	delaunay_triangulation();

	auto prior_start = std::chrono::steady_clock::now();

	//compute_prior_bbox(l_indices, l_coords, *l_sparse_disparity, *l_prior_disparity);
	compute_prior_with_intensity(l_indices, l_coords, *l_sparse_disparity, *l_prior_disparity);
	compute_prior_with_intensity(r_indices, r_coords, *r_sparse_disparity, *r_prior_disparity);
	auto prior_end = std::chrono::steady_clock::now();
	auto prior_dt = prior_end - prior_start;
	int64_t prior_ms = std::chrono::duration_cast<std::chrono::milliseconds>(prior_dt).count();
	std::cout << "compute prior time elapsed: " << prior_ms << std::endl;
	detect_depth_occlusion();
	detect_stereo_occlusion();
	init_similarity();
	auto t0 = std::chrono::steady_clock::now();
	grow();
	auto t1 = std::chrono::steady_clock::now();
	auto dt = t1 - t0;
	int64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(dt).count();
	std::cout << "time elapsed: " << ms << std::endl;
}

void Matcher::detect_stereo_occlusion()
{
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			float l_disp = l_prior_disparity->at<float>(i, j);
			if (l_disp == 0) continue;
			int r_j = j - (int)l_disp;
			if (r_j < 0)
			{
				(*stereo_occlusion)[i * width + j] = true;
				continue;
			}
			//float r_disp = r_prior_disparity->at<float>(i, r_j);
			//if (std::abs((int)l_disp - (int)r_disp) > 5)
			//{
			//	(*stereo_occlusion)[i * width + j] = true;
			//}
		}
	}
}

void Matcher::detect_depth_occlusion()
{
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (l_prior_disparity->at<float>(i, j) == 0)
			{
				(*depth_occlusion)[i * width + j] = true;
			}
		}
	}
}

template<typename T> 
T median(std::vector<T>& v)
{
	if (v.empty()) {
		return 0.0f;
	}
	auto n = v.size() / 2;
	nth_element(v.begin(), v.begin() + n, v.end());
	auto med = v[n];
	//if (!(v.size() & 1)) { //If the set size is even
	//	auto max_it = max_element(v.begin(), v.begin() + n);
	//	med = (T)(((int)*max_it + (int)med) / 2.0);
	//}
	return med;
}

float median_filter(cv::Mat& sparse_disp, cv::Mat& stereo_img, int row, int col)
{
	/*
		This function is used in computing the prior disparity.
		The basic idea is to find out the pixel in the window with median dispairty.
		Intensity constraints are added when selecting candidate pixels,
			only pixels whose intensity is close to the center pixel will be counted as a candidate.
	*/
	std::vector<float> v;
	v.reserve(100);
	int radius = 5;
	float inv_gamma_c = 0.1f, epsilon = 0.4f;
	int width = sparse_disp.cols, height = sparse_disp.rows;

	int i_start = std::max(0, row - radius);
	int i_end = std::min(height - 1, row + radius);
	int j_start = std::max(0, col - radius);
	int j_end = std::min(width - 1, col + radius);

	// Select valid sparse pixel under image intensity constraint.
	float parent_intensity = (float)stereo_img.at<uint8_t>(row, col);
	float* p;
	uint8_t* s;
	//int count = 0;
	for (int i = i_start; i < i_end; i++)
	{
		p = sparse_disp.ptr<float>(i);
		s = stereo_img.ptr<uint8_t>(i);
		for (int j = j_start; j < j_end; j++)
		{
			if (!p[j]) continue; // Check if this pixel is a sparse disp.
			// Check the image intensity consistency
			float cur_intensity = (float)s[j];
			float consistency = exp(-std::abs(parent_intensity - cur_intensity) * inv_gamma_c);
			if (consistency > epsilon)
			{
				v.push_back(p[j]);
			}
		}
	}
	//// Select the median disparity among valid pixels.

	float median_disp = median<float>(v);
	return median_disp;
}

cv::Mat Matcher::post_filling()
{
		/*
			This function is used to fill the holes in the disparity map.
			We use the median filter to fill the holes.
		*/
	cv::Mat output_disp_copy = output_disp->clone();

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (output_disp->at<float>(i, j) != 0) continue;
			if ((*stereo_occlusion)[i * width + j])
			{
				output_disp_copy.at<float>(i, j) = l_prior_disparity->at<float>(i, j);
				continue;
			}
			float disp = (median_filter(*output_disp, l_img, i, j));
			output_disp_copy.at<float>(i, j) = disp;
		}
	}

	//median filter the output_disp_copy
	cv::medianBlur(output_disp_copy, output_disp_copy, 5);
		
	return output_disp_copy;
}