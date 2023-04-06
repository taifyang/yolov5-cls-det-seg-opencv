
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>


//常量
const int INPUT_WIDTH = 640;
const int INPUT_HEIGHT = 640;


//预处理
void pre_process(cv::Mat& image, cv::Mat& blob)
{
	//CenterCrop
	int crop_size = std::min(image.cols, image.rows);
	int  left = (image.cols - crop_size) / 2, top = (image.rows - crop_size) / 2;
	cv::Mat crop_image = image(cv::Rect(left, top, crop_size, crop_size));
	cv::resize(crop_image, crop_image, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));

	//Normalize
	crop_image.convertTo(crop_image, CV_32FC3, 1. / 255.);
	cv::subtract(crop_image, cv::Scalar(0.406, 0.456, 0.485), crop_image);
	cv::divide(crop_image, cv::Scalar(0.225, 0.224, 0.229), crop_image);

	cv::dnn::blobFromImage(crop_image, blob, 1, cv::Size(crop_image.cols, crop_image.rows), cv::Scalar(), true, false);
}


//网络推理
void process(cv::Mat& blob, cv::dnn::Net& net, std::vector<cv::Mat>& outputs)
{
	net.setInput(blob);
	net.forward(outputs, net.getUnconnectedOutLayersNames());
}


//后处理
std::string post_process(std::vector<cv::Mat>& detections, std::vector<std::string>& class_name)
{
	std::vector<float> values;
	for (size_t i = 0; i < detections[0].cols; i++)
	{
		values.push_back(detections[0].at<float>(0, i));
	}
	int id = std::distance(values.begin(), std::max_element(values.begin(), values.end()));

	return class_name[id];
}


int main(int argc, char** argv)
{
	std::vector<std::string> class_name;
	std::ifstream ifs("class_cls.txt");
	std::string line;

	while (getline(ifs, line))
	{
		class_name.push_back(line);
	}

	cv::Mat image = cv::imread("goldfish.jpg"), blob;
	pre_process(image, blob);

	cv::dnn::Net net = cv::dnn::readNet("yolov5n-cls.onnx");
	std::vector<cv::Mat> detections;
	process(blob, net, detections);

	std::cout << post_process(detections, class_name) << std::endl;
	return 0;
}
