#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2\nonfree\features2d.hpp>

#define NUM_FILES_TRAIN 35
#define NUM_FILES_TEST 35
#define DICTIONARY_SIZE 300
#define BAR_WIDTH 70

using namespace cv;
using namespace std;

std::map<string, int> Label_ID = {
	{ "frog", 0 },
	{ "truck", 1 },
	{ "automobile", 2 },
	{ "bird", 3 },
	{ "horse", 4 },
	{ "ship", 5 },
	{ "cat", 6 },
	{ "airplane", 7 },
	{ "bird", 8 },
	{ "dog", 9 }
};

Mat labels;
Mat train_descriptors;
Mat dictionary;
Mat trainingData;

TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.001);
int retries = 1;
int flags = KMEANS_PP_CENTERS;

void draw_progress_bar(int current, int total);

int main()
{
	//Init labels
	cout << "Initializing the labels (responses) from csv file" << endl;
	ifstream file("trainLabels.csv");
	string value;
	for (int i = 0; i < NUM_FILES_TRAIN; i++)
	{
		getline(file, value);
		string csv_label = value.substr(value.find(",") + 1);
		labels.push_back(Label_ID.find(csv_label)->second);
		draw_progress_bar(i+1, NUM_FILES_TRAIN);
	}

	cv::FileStorage fs("dictionary.yml", cv::FileStorage::READ);
	if (std::ifstream("dictionary.yml")) // dictionary already exists
	{
		cout << "Dictionary detected. Loading Vocabulary" << endl;
		fs["vocabulary"] >> dictionary;
		fs.release();
	}
	else
	{
		if (!fs.isOpened())
		{
			cout << "Dictionary not detected. Creating..." << endl;
			cout << "Extracting the Descriptors (Feature Vectors) using SIFT" << endl;
			for (int i = 0; i < NUM_FILES_TRAIN; i++)
			{
				string image_name = "train/" + to_string(i + 1) + ".png";
				Mat image = cv::imread(image_name, CV_LOAD_IMAGE_GRAYSCALE);
				if (!image.data)
				{
					cout << "[SIFT] Error reading image " << image_name << endl;
					exit(0);
				}
				cv::Ptr<cv::FeatureDetector> detector = new cv::SiftFeatureDetector();
				cv::Ptr<cv::DescriptorExtractor> extractor = new cv::SiftDescriptorExtractor();

				vector<cv::KeyPoint> keypoints;
				detector->detect(image, keypoints);

				Mat extracted_descriptor;
				extractor->compute(image, keypoints, extracted_descriptor);

				train_descriptors.push_back(extracted_descriptor);

				draw_progress_bar(i + 1, NUM_FILES_TRAIN);
			}

			cout << "Creating Bag of Words" << endl;
			//Create the Bag of Words Trainer
			cv::BOWKMeansTrainer bag_of_words_trainer(DICTIONARY_SIZE, tc, retries, flags);
			//cluster the feature vectors, dictionary
			dictionary = bag_of_words_trainer.cluster(train_descriptors);
			//store it
			cv::FileStorage fs("dictionary.yml", cv::FileStorage::WRITE);
			fs << "vocabulary" << dictionary;
			fs.release();
		}

		cout << "Creating the Training Data for KNN based on Dictionary" << endl;

		cv::Ptr<DescriptorMatcher> descr_matcher(new FlannBasedMatcher);
		cv::Ptr<cv::FeatureDetector> detector = new cv::SiftFeatureDetector();
		cv::Ptr<cv::DescriptorExtractor> extractor = new cv::SiftDescriptorExtractor();
		cv::BOWImgDescriptorExtractor bag_descr_extractor(extractor, descr_matcher);

		bag_descr_extractor.setVocabulary(dictionary);

		for (int i = 0; i < NUM_FILES_TRAIN; i++)
		{
			string image_name = "train/" + to_string(i + 1) + ".png";
			Mat image = cv::imread(image_name, CV_LOAD_IMAGE_GRAYSCALE);
			if (!image.data)
			{
				cout << "[SIFT] Error reading image " << image_name << endl;
				exit(0);
			}

			vector<cv::KeyPoint> keypoints;
			detector->detect(image, keypoints);

			Mat bowDescriptor;
			bag_descr_extractor.compute(image, keypoints, bowDescriptor);
			trainingData.push_back(bowDescriptor);

			draw_progress_bar(i + 1, NUM_FILES_TRAIN);
		}

		cout << "Training Data Created" << endl;
		cout << "Training Data Rows x Cols: " << trainingData.rows << " x " << trainingData.cols << endl;
	}



	return 0;
}


void draw_progress_bar(int current, int total)
{
	float progress = ((current * 101.0) / total) / 100.0;
	std::cout << "[";
	int pos = BAR_WIDTH * progress;
	for (int i = 0; i < BAR_WIDTH; ++i)
	{
		if (i < pos)
		{
			std::cout << "=";
		}
		else if (i == pos)
		{
			std::cout << ">";
		}
		else
		{
			std::cout << " ";
		}
	}
	std::cout << "] " << int(progress * 100.0) << " %\r";
	//std::cout.flush();
	std::cout << std::endl;
}