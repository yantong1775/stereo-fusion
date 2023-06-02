#include "util.h"


// Given intrinsic matrix of a camera, convert depth map to point cloud
void depth2cloud(const cv::Mat& depth, const cv::Mat& K, cv::Mat& cloud)
{
	int width = depth.cols;
	int height = depth.rows;
	cloud = cv::Mat(height, width, CV_32FC3);
	for (int i = 0; i < height; i++)
	{
		float* ptr = cloud.ptr<float>(i);
		for (int j = 0; j < width; j++)
		{
			float dep = depth.at<float>(i, j);
			if (dep == 0) continue;
			float x = (j - K.at<float>(0, 2)) * dep / K.at<float>(0, 0);
			float y = (i - K.at<float>(1, 2)) * dep / K.at<float>(1, 1);
			ptr[3 * j] = x;
			ptr[3 * j + 1] = y;
			ptr[3 * j + 2] = dep;
		}
	}
}

// Given intrinsic and extrinsic matrix of a camera, convert point clouw into depth.
void cloud2depth(const cv::Mat& cloud, const cv::Mat& K, const cv::Mat& R, const cv::Mat& t, cv::Mat& depth)
{
	int width = cloud.cols;
	int height = cloud.rows;
	for (int i = 0; i < height; i++)
	{
		const float* ptr = cloud.ptr<float>(i);
		float x = ptr[0], y = ptr[1], z = ptr[2];
		if (z == 0) continue;
		// Transform point from tof camera coordinate to rgb camera coordinate.
		float x1 = R.at<float>(0, 0) * x + R.at<float>(0, 1) * y + R.at<float>(0, 2) * z + t.at<float>(0);
		float y1 = R.at<float>(1, 0) * x + R.at<float>(1, 1) * y + R.at<float>(1, 2) * z + t.at<float>(1);
		float z1 = R.at<float>(2, 0) * x + R.at<float>(2, 1) * y + R.at<float>(2, 2) * z + t.at<float>(2);
		// Project point to rgb camera plane.
		float x2 = x1 * K.at<float>(0, 0) / z1 + K.at<float>(0, 2);
		float y2 = y1 * K.at<float>(1, 1) / z1 + K.at<float>(1, 2);
		if (x2 < 0 || x2 >= width || y2 < 0 || y2 >= height) continue;
		depth.at<float>(y2, x2) = z1;
	}	
}

// Apply geometry refinement to the sparse depth map.
void geometry_refine(cv::Mat& sparse_depth, int w_size, int row, int col)
{
	/*
		This function is to refine the sampled sparse disparity map based on geometry.
		The basic idea is as follows:
			Set a window around the center pixel,
			preserve the pixel with smallest depth/ largest disparity
			discard the other.
	*/
	int prev_row = row, prev_col = col;
	float prev_depth = sparse_depth.at<float>(row, col);
	int width = sparse_depth.cols, height = sparse_depth.rows;

	for (int i = row - w_size; i <= row + w_size; i++)
	{
		for (int j = col - w_size; j <= col + w_size; j++)
		{
			if (i < 0 || i >= height || j < 0 || j >= width) continue;
			float depth = sparse_depth.at<float>(i, j);
			if (depth == 0) continue;
			if (depth > prev_depth)
			{
				sparse_depth.at<float>(i, j) = 0;
				continue;
			}
			prev_depth = depth;
			prev_row = i;
			prev_col = j;

		}
	}
}


// Extract pixel positions into a vector.
void sparsedepth2vec(const cv::Mat& sparse_depth, std::vector<double> coords)
{
	int width = sparse_depth.cols, height = sparse_depth.rows;
	for (int i = 0; i < height; i++)
	{
		const float* ptr = sparse_depth.ptr<float>(i);
		for (int j = 0; j < width; j++)
		{
			float depth = ptr[j];
			if (depth == 0) continue;
			coords.push_back(i);
			coords.push_back(j);
		}
	}
}

void read_data_list(const std::string& filepath, std::vector<std::string>& data_list)
{
	std::ifstream stream(filepath);
	std::string line;
	while (getline(stream, line))
	{
		data_list.push_back(line);
	}
}

void read_config(const std::string& filepath, Configs& config)
{
	std::ifstream stream(filepath);
	std::string line;

	while (getline(stream, line))
	{
		if (line.find("baseline") != std::string::npos)
		{
			config.baseline = std::stof(line.substr(8 + 1, line.length()));
		}
		else if (line.find("focal") != std::string::npos)
		{
			config.focal = std::stof(line.substr(5 + 1, line.length()));
		}
		else if (line.find("doff") != std::string::npos)
		{
			config.doff = std::stof(line.substr(4 + 1, line.length()));
		}
		else if (line.find("ratio") != std::string::npos)
		{
			config.ratio = std::stof(line.substr(5 + 1, line.length()));
		}
		else if (line.find("scale") != std::string::npos)
		{
			config.scale = std::stof(line.substr(5 + 1, line.length()));
		}
		else if (line.find("threshold") != std::string::npos)
		{
			config.threshold = std::stof(line.substr(9 + 1, line.length()));
		}
		else if (line.find("step") != std::string::npos)
		{
			config.step = std::stoi(line.substr(4 + 1, line.length()));
		}
		else if (line.find("dist_range") != std::string::npos)
		{
			config.dist_range = std::stof(line.substr(10 + 1, line.length()));
		}
		else if (line.find("folder") != std::string::npos)
		{
			config.folder_path = line.substr(6 + 1, line.length());
		}
	}
}

float rmse(cv::Mat& im1, cv::Mat& im2)
{
	float res = 0.0f;
	int valid_num = 0;
	for (int i = 0; i < im1.rows; i++)
	{
		for (int j = 0; j < im1.cols; j++)
		{
			float im1_pixel = im1.at<float>(i, j);
			float im2_pixel = im2.at<float>(i, j);
			if (im1_pixel == 0 || im2_pixel == 0) continue;
			float diff = (im1.at<float>(i, j) - im2.at<float>(i, j));
			valid_num++;
			res += diff * diff;
		}
	}
	res = res / valid_num;
	return sqrt(res);
}

float mae(cv::Mat& im1, cv::Mat& im2)
{
	float res = 0.0f;
	int valid_num = 0;

	for (int i = 0; i < im1.rows; i++)
	{
		for (int j = 0; j < im1.cols; j++)
		{
			float im1_pixel = im1.at<float>(i, j);
			float im2_pixel = im2.at<float>(i, j);
			if (im1_pixel == 0 || im2_pixel == 0) continue;
			valid_num++;

			float diff = (im1.at<float>(i, j) - im2.at<float>(i, j));
			res += std::abs(diff);
		}
	}
	res = res / (im1.rows * im1.cols);
	return res;
}

float BMP(cv::Mat& im1, cv::Mat& im2)
{
	float res = 0.0f;
	float valid_num = 0;
	for (int i = 0; i < im1.rows; i++)
	{
		for (int j = 0; j < im1.cols; j++)
		{
			float im1_pixel = im1.at<float>(i, j);
			float im2_pixel = im2.at<float>(i, j);
			if (im2_pixel == 0) continue;
			if (im1_pixel == 0) continue;
			valid_num++;
			float diff = std::abs(im1.at<float>(i, j) - im2.at<float>(i, j));
			if (diff > 1) res++;
		}
	}

	res = res / valid_num;
	return res;
}

float avg_disp_err(cv::Mat& im1, cv::Mat& im2)
{
	float res = 0.0f;
	int valid_num = 0;
	for (int i = 0; i < im1.rows; i++)
	{
		for (int j = 0; j < im1.cols; j++)
		{
			float im1_pixel = im1.at<float>(i, j);
			float im2_pixel = im2.at<float>(i, j);	
			if (im2_pixel == 0) continue;
			if (im1_pixel == 0) continue;
			valid_num++;
			float diff = std::abs(im1.at<float>(i, j) - im2.at<float>(i, j));
			res += diff;
		}
	}
	return res / valid_num;
}

void draw_point(cv::Mat& img, cv::Mat& sparse_disp, int width, int height)
{
	cv::Mat bgr;
	cv::applyColorMap(sparse_disp, bgr, cv::COLORMAP_JET);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			uint8_t disp = sparse_disp.at<uint8_t>(i, j);
			if (disp > 0)
			{
				cv::Vec3b color = bgr.at<cv::Vec3b>(i, j);
				cv::circle(img, cv::Point(j, i), 2, color, -1);
			}
		}
	}
}

void Colorize(cv::Mat& input, cv::Mat& output)
{
	double Min, Max;
	cv::minMaxLoc(input, &Min, &Max);
	int max_int = ceil(Max);
	cv::Mat tmp;
	//create a window complete black
	input.convertTo(tmp, CV_8UC3, 255 / (Max - Min), -255 * Min / (Max - Min));
	tmp.convertTo(tmp, CV_8UC3);
	cv::applyColorMap(tmp, output, cv::COLORMAP_JET);
}

void save_color_img(const char* path, cv::Mat& input)
{
	cv::Mat out;
	Colorize(input, out);
	cv::imwrite(path, out);
}


void draw_occlusion(cv::Mat& img, std::vector<int>& occlusion, Color c)
{
	for (int i = 0; i < occlusion.size(); i += 2)
	{
		img.at<cv::Vec3b>(occlusion[i], occlusion[i + 1]) = cv::Vec3b(c.r, c.g, c.b);
	}
}
	