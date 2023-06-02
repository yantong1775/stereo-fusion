#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include "Seed.h"


struct Edge {
	cv::Point3f start, end;

	// Constructor
	Edge(cv::Point3f start, cv::Point3f end) : start(start), end(end) {}

	// Calculates the difference in z-axis
	float getZDifference() const {
		return std::abs(end.z - start.z);
	}
};


bool compare_edge_zdiff(const Edge& a, const Edge& b);

struct Pixel {

	int row, col;
	float disp;

	bool operator<(const Pixel& p) const
	{
		return disp < p.disp;
	}
};

struct MetaDisparity {
	int row, col;
	int disp;
	double tp;
	double energy;
	bool visit, assignment;
};

class Matcher
{
private:
	int width, height; // image height and width
	float baseline, focal, doff; // Camera parameters
	float ratio, scale; // parameters of the similarity
	float threshold; // Threshold in growing algorithm

public:

	cv::Mat& l_img; // Left RGB image
	cv::Mat& r_img; // right RGB image

	cv::Mat* l_sparse_disparity; // left prior disparity
	cv::Mat* r_sparse_disparity; // right prior disparity

	cv::Mat* refine_r_sparse_disparity; // refined prior disparity
	cv::Mat* refine_l_sparse_disparity; // refined prior disparity

	cv::Mat* l_prior_disparity; // left prior disparity
	cv::Mat* r_prior_disparity; // right prior disparity

	std::vector<bool>* stereo_occlusion;
	std::vector<bool>* depth_occlusion;

	cv::Mat* output_disp; // output disparity

	/*
		This is the coordinates of the sampled pixels.
		The layout of coords is:
		[x1, y1, x2, y2, ...]
	*/
	std::vector<double> l_coords;
	std::vector<double> r_coords;

	/*
	Indices stores the index of the vertex of the triangles.
	*/
	std::vector<int> l_indices;
	std::vector<int> r_indices;

	/*
		This is the seed data structure, please refer to Seed.h for more information
	*/
	std::vector<Seed> seeds;



	Matcher(int width, int height, cv::Mat& l_img, cv::Mat& r_img,
		float baseline, float focal, float doff,
		float ratio, float scale, float threshold);
	~Matcher();

	void sample_disp(cv::Mat& gt_disp);
	void refine_disp(cv::Mat& sparse_disp, cv::Mat& refine_disp, cv::Mat& stereo_img, int w_size);
	void geometry_refine_window(cv::Mat& sparse_disp, int w_size, int row, int col);
	void init_coords();
	
	void compute_seed(); // Calculate the initial seeds/correspondance.
	void delaunay_triangulation(); // Using delaunay method to triangulation the sparse points.
	
	void compute_prior_bbox(std::vector<int>& indices, std::vector<double>& coords, cv::Mat& sparse_disp, cv::Mat& prior_disp); // Using bounding box method to calculate the prior disparity.
	void compute_prior_with_intensity(std::vector<int>& indices, std::vector<double>& coords, cv::Mat& sparse_disp, cv::Mat& prior_disp);

	void detect_stereo_occlusion();
	void detect_depth_occlusion();

	cv::Vec3f barycentric(cv::Point2f p, cv::Point2f p1, cv::Point2f p2, cv::Point2f p3);
	cv::Vec3f barycentric(cv::Point2f p, cv::Point3f p1, cv::Point3f p2, cv::Point3f p3);
	void compute_similarity(Seed &s); // Calculate the similarity of the seed.
	void init_similarity(); // Initilize the simlarity of the inital seeds.
	
	// numerator window and denominator window are util functino for computing the similarity.
	float numerator_window(int size, const Seed& s);
	float denominator_window(int size, const Seed& s);
	float numerator_window_grayscale (int size, const Seed& s);
	float denominator_window_grayscale(int size, const Seed& s);
	float image_similarity_grayscale(int size, Seed& s);
	float image_similarity_NCC_grayscale(int size, Seed& s);
	float image_similarity_cos_grayscale(int size, Seed& s);

	float image_similarity(int size, Seed& s);
	void process_subset(std::vector<std::vector<Seed>>& neighbor, std::priority_queue<Seed>& heap, int i);
	void process_seed(Seed& seed);
	// GROWING algorithm
	void grow();
	void region_growing();
	void run();
	void run(cv::Mat& downsample_disp_edge, cv::Mat& RGB_edge);

	// Post-filling method
	cv::Mat post_filling();

	// The following are util function for interpolating the triangle in order to compute the prior disparity.
	void DrawTriangle(const std::vector<cv::Point3f>& triangle);
	
	void DrawFlatTopTriangle(const cv::Point3f& it0,
		const cv::Point3f& it1,
		const cv::Point3f& it2);
	
	void DrawFlatBottomTriangle(const cv::Point3f& it0,
		const cv::Point3f& it1,
		const cv::Point3f& it2);

	void DrawFlatTriangle(const cv::Point3f& it0,
		const cv::Point3f& it1,
		const cv::Point3f& it2,
		const cv::Point3f& dv0,
		const cv::Point3f& dv1,
		cv::Point3f itEdge1);
};