#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2\nonfree\features2d.hpp>

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
	cout << "Initializing the labels (responses) from csv file" << endl;
	ifstream file("trainLabels.csv");
	string value;
	{
		getline(file, value);
		string csv_label = value.substr(value.find(",") + 1);
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
				if (!image.data)
				{
					exit(0);
				}

				vector<cv::KeyPoint> keypoints;

				Mat extracted_descriptor;

				train_descriptors.push_back(extracted_descriptor);

			}

			cout << "Creating Bag of Words" << endl;
			//Create the Bag of Words Trainer
			cv::BOWKMeansTrainer bag_of_words_trainer(DICTIONARY_SIZE, tc, retries, flags);
			//cluster the feature vectors, dictionary
			dictionary = bag_of_words_trainer.cluster(train_descriptors);
			//store it
			fs << "vocabulary" << dictionary;
		}




		{
		}

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