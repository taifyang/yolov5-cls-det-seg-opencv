#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>


//常量
const int INPUT_WIDTH = 640;
const int INPUT_HEIGHT = 640;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;

//LetterBox处理
void LetterBox(const cv::Mat& image, cv::Mat& outImage,
	cv::Vec4d& params, //[ratio_x,ratio_y,dw,dh]
	const cv::Size& newShape = cv::Size(640, 640),
	bool autoShape = false,
	bool scaleFill = false,
	bool scaleUp = true,
	int stride = 32,
	const cv::Scalar& color = cv::Scalar(114, 114, 114))
{
	cv::Size shape = image.size();
	float r = std::min((float)newShape.height / (float)shape.height, (float)newShape.width / (float)shape.width);
	if (!scaleUp)
	{
		r = std::min(r, 1.0f);
	}

	float ratio[2]{ r, r };
	int new_un_pad[2] = { (int)std::round((float)shape.width * r),(int)std::round((float)shape.height * r) };

	auto dw = (float)(newShape.width - new_un_pad[0]);
	auto dh = (float)(newShape.height - new_un_pad[1]);

	if (autoShape)
	{
		dw = (float)((int)dw % stride);
		dh = (float)((int)dh % stride);
	}
	else if (scaleFill)
	{
		dw = 0.0f;
		dh = 0.0f;
		new_un_pad[0] = newShape.width;
		new_un_pad[1] = newShape.height;
		ratio[0] = (float)newShape.width / (float)shape.width;
		ratio[1] = (float)newShape.height / (float)shape.height;
	}

	dw /= 2.0f;
	dh /= 2.0f;

	if (shape.width != new_un_pad[0] && shape.height != new_un_pad[1])
		cv::resize(image, outImage, cv::Size(new_un_pad[0], new_un_pad[1]));
	else
		outImage = image.clone();

	int top = int(std::round(dh - 0.1f));
	int bottom = int(std::round(dh + 0.1f));
	int left = int(std::round(dw - 0.1f));
	int right = int(std::round(dw + 0.1f));
	params[0] = ratio[0];
	params[1] = ratio[1];
	params[2] = left;
	params[3] = top;
	cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}


//预处理
void pre_process(cv::Mat& image, cv::Mat& blob)
{
	cv::Vec4d params;
	cv::Mat letterbox;
	LetterBox(image, letterbox, params, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
	cv::dnn::blobFromImage(letterbox, blob, 1. / 255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
}


//网络推理
void process(cv::Mat& blob, cv::dnn::Net& net, std::vector<cv::Mat>& outputs)
{
	net.setInput(blob);
	net.forward(outputs, net.getUnconnectedOutLayersNames());
}


//box缩放到原图尺寸
void scale_boxes(cv::Rect& box, cv::Size size)
{
	float gain = std::min(INPUT_WIDTH * 1.0 / size.width, INPUT_HEIGHT * 1.0 / size.height);
	int pad_w = (INPUT_WIDTH - size.width * gain) / 2;
	int pad_h = (INPUT_HEIGHT - size.height * gain) / 2;
	box.x -= pad_w;
	box.y -= pad_h;
	box.x /= gain;
	box.y /= gain;
	box.width /= gain;
	box.height /= gain;
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

	float* data = (float*)outputs[0].data;

	const int dimensions = 85;  //5+80
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
				int left = int(x - 0.5 * w);
				int top = int(y - 0.5 * h);
				int width = int(w);
				int height = int(h);
				cv::Rect box = cv::Rect(left, top, width, height);
				scale_boxes(box, image.size());
				boxes.push_back(box);
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

	cv::dnn::Net net = cv::dnn::readNet("yolov5s-det.onnx");
	std::vector<cv::Mat> outputs;
	process(blob, net, outputs);

	cv::Mat result = post_process(image, outputs, class_name);
	cv::imshow("detection", result);
	cv::waitKey(0);
	return 0;
}
