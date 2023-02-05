
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>


//常量
const int INPUT_WIDTH = 640;
const int INPUT_HEIGHT = 640;


//预处理
void pre_process(cv::Mat& image, cv::Mat& blob)
{
	cv::dnn::blobFromImage(image, blob, 1. / 255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
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

	cv::dnn::Net net = cv::dnn::readNet("yolov5s-cls.onnx");
	std::vector<cv::Mat> detections;
	process(blob, net, detections);

	std::cout << post_process(detections, class_name) << std::endl;
	return 0;
}