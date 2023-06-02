#include <iostream>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include <string>
#include <algorithm>
#include <vector>
#include <cmath>
#include <sstream>
#include <fstream>

#include "PFMReadWrite.h"
#include "StereoMatcher.h"
#include "util.h"

void exec(Configs& config, std::string& data)
{
	/* Read ground truth disparity map and left, right RGB image*/
	std::cout << "Test on data: " << data << std::endl;
	std::string folder_name = "data/" + data + "/";

	cv::Mat gt_disp = cv::imread(folder_name + "input/disp1.png", CV_8UC1);
	gt_disp.convertTo(gt_disp, CV_32FC1);
	cv::Mat l_img = cv::imread(folder_name + "input/view1.png", cv::IMREAD_GRAYSCALE);
	cv::Mat r_img = cv::imread(folder_name + "input/view5.png", cv::IMREAD_GRAYSCALE);


	int height = gt_disp.rows, width = gt_disp.cols;

	Matcher matcher(
		width, height, l_img, r_img,
		config.baseline, config.focal, config.doff, config.ratio, config.scale, config.threshold);
	matcher.sample_disp(gt_disp);
	matcher.run();

	std::cout << "BMP before postfilling: " << BMP(*matcher.output_disp, gt_disp) << std::endl;
	cv::Mat post_output = matcher.post_filling();
	std::cout << "BMP after post filling: " << BMP(post_output, gt_disp) << std::endl;
	std::cout << "Average disparity error is: " << avg_disp_err(post_output, gt_disp) << std::endl;
	std::cout << "rmse disparity error is: " << rmse(post_output, gt_disp) << std::endl;

	//Find automatically the max and the min
	cv::Mat M;
	cv::Mat post_out_c;
	cv::Mat prior_disp_c;
	cv::Mat gt_disp_c;
	Colorize(*matcher.output_disp, M);
	Colorize(post_output, post_out_c);
	Colorize(*matcher.l_prior_disparity, prior_disp_c);
	Colorize(gt_disp, gt_disp_c);
	// conver the mat to eigen matrix
	//cv::imwrite("data/test/test_disparity.png", disparity_mat);
	std::ostringstream buffer;
	buffer << "disp_r" << config.ratio << "_s" << config.scale << "disparity.png";
	std::string outfile_name = buffer.str();
	std::string outfolder = folder_name + "output/";
	std::string out_path = outfolder + outfile_name;
	//std::cout << out_path;
	//cv::imwrite(out_path, M);
	cv::imwrite(outfolder + "disparity.png", *matcher.output_disp);
	cv::imwrite(out_path, post_out_c);
	cv::imwrite(outfolder + "prior.png", prior_disp_c);
	cv::imwrite(folder_name + "input/color_gt_disp.png", gt_disp_c);
	return;
}


int main()
{

	// Read configs from file

	Configs config;
	read_config("./res/Config.txt", config);
	std::vector<std::string> data_list;
	read_data_list("./res/test_data_list.txt", data_list);
	for (auto data : data_list)
	{
		exec(config, data);
	}
	
	return 0;
}