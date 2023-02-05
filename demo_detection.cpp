#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>


//常量
const int INPUT_WIDTH = 640;
const int INPUT_HEIGHT = 640;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;


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


//可视化函数
void draw_result(cv::Mat& image, std::string label, cv::Rect box)
{
	cv::rectangle(image, box, cv::Scalar(255, 0, 0), 2);
	int baseLine;
	cv::Size label_size = cv::getTextSize(label, 0.8, 0.8, 1, &baseLine);
	cv::Point tlc = cv::Point(box.x, box.y);
	cv::Point brc = cv::Point(box.x, box.y + label_size.height + baseLine);
	cv::putText(image, label, cv::Point(box.x, box.y), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 1);
}


//后处理
cv::Mat post_process(cv::Mat& image, std::vector<cv::Mat>& outputs, std::vector<std::string>& class_name)
{
	std::vector<int> class_ids;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;

	float x_factor = (float) image.cols / INPUT_WIDTH;
	float y_factor = (float) image.rows / INPUT_HEIGHT;

	float* data = (float*)outputs[0].data;

	const int dimensions = 85;  //5+80+32
	const int rows = 25200;		//(640/8)*(640/8)*3+(640/16)*(640/16)*3+(640/32)*(640/32)*3
	for (int i = 0; i < rows; ++i)
	{
		float confidence = data[4];
		if (confidence >= CONFIDENCE_THRESHOLD)
		{
			float* classes_scores = data + 5;
			cv::Mat scores(1, class_name.size(), CV_32FC1, classes_scores);
			cv::Point class_id;
			double max_class_score;
			cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
			if (max_class_score > SCORE_THRESHOLD)
			{
				float x = data[0];
				float y = data[1];
				float w = data[2];
				float h = data[3];
				int left = int((x - 0.5 * w) * x_factor);
				int top = int((y - 0.5 * h) * y_factor);
				int width = int(w * x_factor);
				int height = int(h * y_factor);
				boxes.push_back(cv::Rect(left, top, width, height));
				confidences.push_back(confidence);
				class_ids.push_back(class_id.x);
			}
		}
		data += dimensions;
	}

	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
	for (int i = 0; i < indices.size(); i++)
	{
		int idx = indices[i];
		cv::Rect box = boxes[idx];
		std::string label = class_name[class_ids[idx]] + ":" + cv::format("%.2f", confidences[idx]);
		draw_result(image, label, box);
	}
	return image;
}


int main(int argc, char** argv)
{
	std::vector<std::string> class_name;
	std::ifstream ifs("class_det.txt");
	std::string line;

	while (getline(ifs, line))
	{
		class_name.push_back(line);
	}

	cv::Mat image = cv::imread("bus.jpg"), blob;
	pre_process(image, blob);

	cv::dnn::Net net = cv::dnn::readNet("yolov5n-det.onnx");
	std::vector<cv::Mat> detections;
	process(blob, net, detections);

	cv::Mat result = post_process(image, detections, class_name);
	cv::imshow("detection", result);
	cv::waitKey(0);
	return 0;
}