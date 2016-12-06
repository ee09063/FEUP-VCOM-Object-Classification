#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2\nonfree\features2d.hpp>

#define NUM_FILES_TRAIN 420
#define NUM_FILES_TEST 10
#define DICTIONARY_SIZE 300
#define BAR_WIDTH 70
#define K 1
/*Pay no mind to this*/
#define GOD_CONSTANT 0

using namespace cv;
using namespace std;

std::map<string, int> Label_ID;
std::map<float, string> Label_ID_reverse;

Mat labels;
Mat train_descriptors;
Mat dictionary;
Mat trainingData;

vector<vector<cv::KeyPoint>> same_keys; 

TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.001);
int retries = 1;
int flags = KMEANS_PP_CENTERS;

void draw_progress_bar(int current, int total);

void instantiateMaps()
{
	Label_ID["frog"] = 0;
	Label_ID["truck"] = 1;
	Label_ID["automobile"] = 2;
	Label_ID["bird"] = 3;
	Label_ID["horse"] = 4;
	Label_ID["ship"] = 5;
	Label_ID["cat"] = 6;
	Label_ID["airplane"] = 7;
	Label_ID["dog"] = 8;
	Label_ID["deer"] = 9;

	Label_ID_reverse[0] = "frog";
	Label_ID_reverse[1] = "truck";
	Label_ID_reverse[2] = "automobile";
	Label_ID_reverse[3] = "bird";
	Label_ID_reverse[4] = "horse";
	Label_ID_reverse[5] = "ship";
	Label_ID_reverse[6] = "cat";
	Label_ID_reverse[7] = "airplane";
	Label_ID_reverse[8] = "dog";
	Label_ID_reverse[9] = "deer";
}

void applyKNN(cv::BOWImgDescriptorExtractor bag_descr_extractor)
{
	//KNN CLASSIFIER
	cout << "Training KNN Classifier" << endl;

	cv::Ptr<cv::KNearest> knn = new cv::KNearest;
	knn->train(trainingData, labels, cv::Mat(), false, 32, false);

	cout << "KNN Classifier Trained" << endl;

	cout << "KNN Testing" << endl;
	for (int i = 0; i < NUM_FILES_TEST; i++)
	{
		string test_image_name = "train/" + to_string(i + 1) + ".png";
		Mat image = cv::imread(test_image_name, CV_LOAD_IMAGE_GRAYSCALE);
		if (!image.data)
		{
			cout << "[SIFT 3] Error reading image " << test_image_name << endl;
			exit(0);
		}

		cv::Ptr<cv::FeatureDetector> detector_test = new cv::SiftFeatureDetector();
		//vector<cv::KeyPoint> keypoints_desc;
		//detector_test->detect(image, keypoints_desc);

		Mat bowDescriptor_test;
		//		bag_descr_extractor.compute(image, keypoints_desc, bowDescriptor_test);
		bag_descr_extractor.compute(image, same_keys.at(i), bowDescriptor_test);

		cv::Mat result;
		knn->find_nearest(bowDescriptor_test, K, result, cv::Mat(), cv::Mat());

		cout << i + 1 << " " << Label_ID_reverse.find(result.at<float>(0, 0))->second << endl;
	}
}

void applySVM(cv::BOWImgDescriptorExtractor bag_descr_extractor)
{
	cout << "Setting up SVM parameters" << endl;

	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit = tc;

	cout << "Training the SVM" << endl;

	CvSVM SVM;
	SVM.train(trainingData, labels, cv::Mat(), cv::Mat(), params);

	cout << "SVM Classifier Trained" << endl;

	cout << "SVM Testing" << endl;
	for (int i = 0; i < NUM_FILES_TEST; i++)
	{
		string test_image_name = "train/" + to_string(i + 1) + ".png";
		Mat image = cv::imread(test_image_name, CV_LOAD_IMAGE_GRAYSCALE);
		if (!image.data)
		{
			cout << "[SIFT 3] Error reading image " << test_image_name << endl;
			exit(0);
		}

		cv::Ptr<cv::FeatureDetector> detector_test = new cv::SiftFeatureDetector();
		//vector<cv::KeyPoint> keypoints_desc;
		//detector_test->detect(image, keypoints_desc);

		Mat bowDescriptor_test;
		//		bag_descr_extractor.compute(image, keypoints_desc, bowDescriptor_test);
		bag_descr_extractor.compute(image, same_keys.at(i), bowDescriptor_test);

		float result;
		result = SVM.predict(bowDescriptor_test);

		cout << i + 1 << " " << Label_ID_reverse.find(result)->second << endl;
	}
}

int main()
{
	cout << "Populating maps" << endl;

	instantiateMaps();

	cout << "Initializing the labels (responses) from csv file" << endl;

	ifstream file("trainLabels.csv");
	string value;

	int number_of_labels = 0;

	for (int i = 0; i < NUM_FILES_TRAIN - GOD_CONSTANT; i++)
	{
		getline(file, value);
		string csv_label = value.substr(value.find(",") + 1);
		int label = Label_ID.find(csv_label)->second;
		labels.push_back(label);
		number_of_labels++;
	}

	cout << "Initiated " << number_of_labels << " labels" << endl;

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
				string image_name_sift1 = "train/" + to_string(i + 1) + ".png";
				Mat image = cv::imread(image_name_sift1, CV_LOAD_IMAGE_GRAYSCALE);
				if (!image.data)
				{
					cout << "[SIFT 1] Error reading image " << image_name_sift1 << endl;
					exit(0);
				}
				cv::Ptr<cv::FeatureDetector> detector_dictionary = new cv::SiftFeatureDetector();
				cv::Ptr<cv::DescriptorExtractor> extractor_dictionary = new cv::SiftDescriptorExtractor();

				vector<cv::KeyPoint> keypoints;
				detector_dictionary->detect(image, keypoints);

				same_keys.push_back(keypoints);

				Mat extracted_descriptor;
				extractor_dictionary->compute(image, keypoints, extracted_descriptor);

				train_descriptors.push_back(extracted_descriptor);
			}

			cout << "Creating Bag of Words" << endl;

			cv::BOWKMeansTrainer bag_of_words_trainer(DICTIONARY_SIZE, tc, retries, flags);

			dictionary = bag_of_words_trainer.cluster(train_descriptors);

			//store it
			/*cv::FileStorage fs("dictionary.yml", cv::FileStorage::WRITE);
			fs << "vocabulary" << dictionary;
			fs.release();*/
		}
	}

	cout << "Creating the Training Data for KNN based on Dictionary" << endl;

	cv::Ptr<DescriptorMatcher> descr_matcher(new FlannBasedMatcher);
	cv::Ptr<cv::FeatureDetector> detector_train = new cv::SiftFeatureDetector();
	cv::Ptr<cv::DescriptorExtractor> extractor_train = new cv::SiftDescriptorExtractor();
	cv::BOWImgDescriptorExtractor bag_descr_extractor(extractor_train, descr_matcher);

	bag_descr_extractor.setVocabulary(dictionary);

	for (int i = 0; i < NUM_FILES_TRAIN; i++)
	{
		string image_name_sift2 = "train/" + to_string(i + 1) + ".png";
		Mat image = cv::imread(image_name_sift2, CV_LOAD_IMAGE_GRAYSCALE);
		if (!image.data)
		{
			cout << "[SIFT 2] Error reading image " << image_name_sift2 << endl;
			exit(0);
		}

		//vector<cv::KeyPoint> keypoints;
		//detector_train->detect(image, keypoints);

		Mat bowDescriptor;
		//bag_descr_extractor.compute(image, keypoints, bowDescriptor);

		bag_descr_extractor.compute(image, same_keys.at(i), bowDescriptor);

		trainingData.push_back(bowDescriptor);

		//draw_progress_bar(i + 1, NUM_FILES_TRAIN);
	}

	cout << "Training Data Created" << endl;
	cout << "Training Data Size: " << trainingData.size() << endl;
	cout << "Number of labels: " << labels.size() << endl;

	//applyKNN(bag_descr_extractor);

	//applySVM(bag_descr_extractor);

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