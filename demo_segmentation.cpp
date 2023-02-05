#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>


//常量
const int INPUT_WIDTH = 640;
const int INPUT_HEIGHT = 640;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;


//网络输出相关参数
struct OutputSeg 
{
	int id;             //结果类别id
	float confidence;   //结果置信度
	cv::Rect box;       //矩形框
	cv::Mat boxMask;    //矩形框内mask，节省内存空间和加快速度
};

//掩膜相关参数
struct MaskParams 
{
	int segChannels = 32;
	int segWidth = 160;
	int segHeight = 160;
	int netWidth = 640;
	int netHeight = 640;
	float maskThreshold = 0.5;
	cv::Size srcImgShape;
	cv::Vec4d params;
};


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
void pre_process(cv::Mat& image, cv::Mat& blob, cv::Vec4d& params)
{
	cv::Mat input_image;
	LetterBox(image, input_image, params, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
	cv::dnn::blobFromImage(input_image, blob, 1. / 255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(0, 0, 0), true, false);
}


//网络推理
void process(cv::Mat& blob, cv::dnn::Net& net, std::vector<cv::Mat>& outputs)
{
	net.setInput(blob);
	std::vector<std::string> output_layer_names{ "output0","output1" };
	net.forward(outputs, output_layer_names);
}


//取得掩膜
void GetMask(const cv::Mat& maskProposals, const cv::Mat& mask_protos, OutputSeg& output, const MaskParams& maskParams)
{
	int seg_channels = maskParams.segChannels;
	int net_width = maskParams.netWidth;
	int seg_width = maskParams.segWidth;
	int net_height = maskParams.netHeight;
	int seg_height = maskParams.segHeight;
	float mask_threshold = maskParams.maskThreshold;
	cv::Vec4f params = maskParams.params;
	cv::Size src_img_shape = maskParams.srcImgShape;
	cv::Rect temp_rect = output.box;

	//crop from mask_protos
	int rang_x = floor((temp_rect.x * params[0] + params[2]) / net_width * seg_width);
	int rang_y = floor((temp_rect.y * params[1] + params[3]) / net_height * seg_height);
	int rang_w = ceil(((temp_rect.x + temp_rect.width) * params[0] + params[2]) / net_width * seg_width) - rang_x;
	int rang_h = ceil(((temp_rect.y + temp_rect.height) * params[1] + params[3]) / net_height * seg_height) - rang_y;

	rang_w = MAX(rang_w, 1);
	rang_h = MAX(rang_h, 1);
	if (rang_x + rang_w > seg_width)
	{
		if (seg_width - rang_x > 0)
			rang_w = seg_width - rang_x;
		else
			rang_x -= 1;
	}
	if (rang_y + rang_h > seg_height)
	{
		if (seg_height - rang_y > 0)
			rang_h = seg_height - rang_y;
		else
			rang_y -= 1;
	}

	std::vector<cv::Range> roi_rangs;
	roi_rangs.push_back(cv::Range(0, 1));
	roi_rangs.push_back(cv::Range::all());
	roi_rangs.push_back(cv::Range(rang_y, rang_h + rang_y));
	roi_rangs.push_back(cv::Range(rang_x, rang_w + rang_x));

	//crop
	cv::Mat temp_mask_protos = mask_protos(roi_rangs).clone();
	cv::Mat protos = temp_mask_protos.reshape(0, { seg_channels,rang_w * rang_h });
	cv::Mat matmul_res = (maskProposals * protos).t();
	cv::Mat masks_feature = matmul_res.reshape(1, { rang_h,rang_w });
	cv::Mat dest, mask;

	//sigmoid
	cv::exp(-masks_feature, dest);
	dest = 1.0 / (1.0 + dest);

	int left = floor((net_width / seg_width * rang_x - params[2]) / params[0]);
	int top = floor((net_height / seg_height * rang_y - params[3]) / params[1]);
	int width = ceil(net_width / seg_width * rang_w / params[0]);
	int height = ceil(net_height / seg_height * rang_h / params[1]);

	cv::resize(dest, mask, cv::Size(width, height), cv::INTER_NEAREST);
	mask = mask(temp_rect - cv::Point(left, top)) > mask_threshold;
	output.boxMask = mask;
}


//可视化函数
void draw_result(cv::Mat & image, std::vector<OutputSeg> result, std::vector<std::string> class_name)
{
	std::vector<cv::Scalar> color;
	srand(time(0));
	for (int i = 0; i < class_name.size(); i++)
	{
		color.push_back(cv::Scalar(rand() % 256, rand() % 256, rand() % 256));
	}

	cv::Mat mask = image.clone();
	for (int i = 0; i < result.size(); i++)
	{
		cv::rectangle(image, result[i].box, cv::Scalar(255, 0, 0), 2);
		mask(result[i].box).setTo(color[result[i].id], result[i].boxMask);
		std::string label = class_name[result[i].id] + ":" + cv::format("%.2f", result[i].confidence);
		int baseLine;
		cv::Size label_size = cv::getTextSize(label, 0.8, 0.8, 1, &baseLine);
		cv::putText(image, label, cv::Point(result[i].box.x, result[i].box.y), cv::FONT_HERSHEY_SIMPLEX, 0.8, color[result[i].id], 1);
	}
	addWeighted(image, 0.5, mask, 0.5, 0, image);
}


//后处理
cv::Mat post_process(cv::Mat& image, std::vector<cv::Mat>& outputs, const std::vector<std::string>& class_name, cv::Vec4d& params)
{
	std::vector<int> class_ids;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;
	std::vector<std::vector<float>> picked_proposals;

	float* data = (float*)outputs[0].data;

	const int dimensions = 117;	//5+80+32
	const int rows = 25200; 	//(640/8)*(640/8)*3+(640/16)*(640/16)*3+(640/32)*(640/32)*3
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
				float x = (data[0] - params[2]) / params[0];  
				float y = (data[1] - params[3]) / params[1]; 
				float w = data[2] / params[0];
				float h = data[3] / params[1];
				int left = std::max(int(x - 0.5 * w), 0);
				int top = std::max(int(y - 0.5 * h), 0);
				int width = int(w);
				int height = int(h);
				boxes.push_back(cv::Rect(left, top, width, height));
				confidences.push_back(confidence);
				class_ids.push_back(class_id.x);

				std::vector<float> temp_proto(data + class_name.size() + 5, data + dimensions);
				picked_proposals.push_back(temp_proto);
			}
		}
		data += dimensions;
	}

	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);

	std::vector<OutputSeg> output;
	std::vector<std::vector<float>> temp_mask_proposals;
	cv::Rect holeImgRect(0, 0, image.cols, image.rows);
	for (int i = 0; i < indices.size(); ++i) 
	{
		int idx = indices[i];
		OutputSeg result;
		result.id = class_ids[idx];
		result.confidence = confidences[idx];
		result.box = boxes[idx] & holeImgRect;
		temp_mask_proposals.push_back(picked_proposals[idx]);
		output.push_back(result);
	}

	MaskParams mask_params;
	mask_params.params = params;
	mask_params.srcImgShape = image.size();
	for (int i = 0; i < temp_mask_proposals.size(); ++i)
	{
		GetMask(cv::Mat(temp_mask_proposals[i]).t(), outputs[1], output[i], mask_params);
	}

	draw_result(image, output, class_name);

	return image;
}


int main(int argc, char** argv)
{
	std::vector<std::string> class_name;
	std::ifstream ifs("class_seg.txt");
	std::string line;

	while (getline(ifs, line))
	{
		class_name.push_back(line);
	}

	cv::Mat image = cv::imread("bus.jpg"), blob;
	cv::Vec4d params;
	pre_process(image, blob, params);

	cv::dnn::Net net = cv::dnn::readNet("yolov5n-seg.onnx");
	std::vector<cv::Mat> detections;
	process(blob, net, detections);

	cv::Mat img = post_process(image, detections, class_name, params);
	cv::imshow("segmentation", img);
	cv::waitKey(0);
	return 0;
}
