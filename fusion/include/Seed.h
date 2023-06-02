#pragma once
#include <opencv2/opencv.hpp>

struct Seed
{
	cv::Vec3i correspondance;

	float similarity;

	bool operator<(const Seed& rhs) const
	{
		return similarity < rhs.similarity;
	}
};