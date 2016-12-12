#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2\nonfree\features2d.hpp>

#define NUM_FILES_TRAIN 10000
#define NUM_FILES_TEST 50
#define DICTIONARY_SIZE 100
#define K 5

using namespace cv;
using namespace std;

std::map<string, int> Label_ID;
std::map<float, string> Label_ID_reverse;

string TEST_DIR = "test/";

Mat labels;
vector<int> label_vec;
vector<int> lost_images;
Mat train_descriptors;
Mat trainingData;

vector<vector<cv::KeyPoint>> same_keys; 

TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.001);
int retries = 1;
int flags = KMEANS_PP_CENTERS;

void draw_progress_bar(int current, int total);

bool contains(vector<int> vec, int elem)
{
	for (int i = 0; i < vec.size(); i++)
	{
		if (vec.at(i) == elem)
		{
			return true;
		}
	}
	return false;
}

void correctLabels()
{
	cout << "Correcting labels" << endl;
	cout << "Number of labels to remove " << lost_images.size() << endl;
	for (int i = 0; i < label_vec.size(); i++)
	{
		int elem = i;
		if (!contains(lost_images, elem))
		{
			cout << "From vector to matrix: " << i + 1 << " / " << label_vec.size() << '\r';
			Mat row = (Mat_<int>(1, 1) << elem);
			labels.push_back(row);
		}
		else
		{
			cout << endl;
			string removed_image_name = "train/" + to_string(i + 1) + ".png";
			cout << "Removing label " << i << " --> " << removed_image_name << endl;
		}
	}
}

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
	std::cout << "Training KNN Classifier" << endl;

	cv::Ptr<cv::KNearest> knn = new cv::KNearest;
	knn->train(trainingData, labels, cv::Mat(), false, 32, false);

	std::cout << "KNN Classifier Trained" << endl;

	std::cout << "KNN Testing" << endl;
	for (int i = 0; i < NUM_FILES_TEST; i++)
	{
		string test_image_name = TEST_DIR + to_string(i + 1) + ".png";
		Mat image = cv::imread(test_image_name, CV_LOAD_IMAGE_GRAYSCALE);
		if (!image.data)
		{
			std::cout << "[SIFT 3] Error reading image " << test_image_name << endl;
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

		std::cout << i + 1 << " " << Label_ID_reverse.find(result.at<float>(0, 0))->second << endl;
	}
}

void applySVM(cv::BOWImgDescriptorExtractor bag_descr_extractor)
{
	std::cout << "Setting up SVM parameters" << endl;

	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit = tc;

	std::cout << "Training the SVM" << endl;

	CvSVM SVM;
	SVM.train(trainingData, labels, cv::Mat(), cv::Mat(), params);

	std::cout << "SVM Classifier Trained" << endl;

	std::cout << "SVM Testing" << endl;
	for (int i = 0; i < NUM_FILES_TEST; i++)
	{
		string test_image_name = TEST_DIR + to_string(i + 1) + ".png";
		Mat image = cv::imread(test_image_name, CV_LOAD_IMAGE_GRAYSCALE);
		if (!image.data)
		{
			std::cout << "[SIFT 3] Error reading image " << test_image_name << endl;
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

		std::cout << i + 1 << " " << Label_ID_reverse.find(result)->second << endl;
	}
}

int main()
{
	std::cout << "Populating maps" << endl;

	instantiateMaps();

	std::cout << "Initializing the labels (responses) from csv file" << endl;

	ifstream file("trainLabels.csv");
	string value;

	int number_of_labels = 0;

	for (int i = 0; i < NUM_FILES_TRAIN; i++)
	{
		getline(file, value);
		string csv_label = value.substr(value.find(",") + 1);
		int label = Label_ID.find(csv_label)->second;
		label_vec.push_back(label);
		//labels.push_back(label);
		number_of_labels++;
		std::cout << i+1 << " / " << NUM_FILES_TRAIN << '\r';
	}

	std::cout << endl;
	std::cout << "Initiated " << number_of_labels << " labels" << endl;

	Mat dictionary;
	cv::FileStorage fs("vocabulary.xml", cv::FileStorage::READ);

	if (fs.isOpened()) // dictionary already exists
	{
		std::cout << "Dictionary detected. Loading Vocabulary" << endl;
		fs[ "vocabulary" ] >> dictionary;
		std::cout << "Loaded Vocabulary info: " << dictionary.size() << endl;
	}
	else
	{
		fs.release();
		std::cout << "Vocabulary not detected. Creating..." << endl;
		std::cout << "Extracting the Descriptors (Feature Vectors) using SIFT" << endl;

		string expected_name;
		int num_of_images = 1;

		for (int i = 0; i < NUM_FILES_TRAIN; i++)
		{
			expected_name = "train/" + to_string(num_of_images) + ".png";
			string image_name_sift1 = "train/" + to_string(i + 1) + ".png";
			if (expected_name.compare(image_name_sift1) != 0)
			{
				std::cout << "Expected name was " << expected_name << endl;
				std::cout << "Image name was " << image_name_sift1 << endl;
				exit(-1);
			}
			Mat image = cv::imread(image_name_sift1, CV_LOAD_IMAGE_GRAYSCALE);
			if (!image.data)
			{
				std::cout << "[SIFT 1] Error reading image " << image_name_sift1 << endl;
				exit(-1);
			}
			cv::Ptr<cv::FeatureDetector> detector_dictionary = new cv::SiftFeatureDetector();
			cv::Ptr<cv::DescriptorExtractor> extractor_dictionary = new cv::SiftDescriptorExtractor();

			vector<cv::KeyPoint> keypoints;
			detector_dictionary->detect(image, keypoints);

			same_keys.push_back(keypoints);

			Mat extracted_descriptor;
			extractor_dictionary->compute(image, keypoints, extracted_descriptor);

			train_descriptors.push_back(extracted_descriptor);

			std::cout << i+1 << " / " << NUM_FILES_TRAIN << " -- " << image_name_sift1 <<'\r';
			num_of_images++;
		}

		std::cout << endl;
		std::cout << "Creating Bag of Words" << endl;

		cv::BOWKMeansTrainer bag_of_words_trainer(DICTIONARY_SIZE, tc, retries, flags);

		dictionary = bag_of_words_trainer.cluster(train_descriptors);

		std::cout << "Created Dictionary info: " << dictionary.size() << endl;

		//store it
		/*fs.open("vocabulary.xml", cv::FileStorage::WRITE);
		fs << "vocabulary" << dictionary;
		fs.release();*/
	}

	std::cout << "Creating the Training Data for KNN and SVM based on Dictionary" << endl;

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
			std::cout << "[SIFT 2] Error reading image " << image_name_sift2 << endl;
			exit(0);
		}

		//vector<cv::KeyPoint> keypoints;
		//detector_train->detect(image, keypoints);

		Mat bowDescriptor;
		//bag_descr_extractor.compute(image, keypoints, bowDescriptor);

		bag_descr_extractor.compute(image, same_keys.at(i), bowDescriptor);

		if (bowDescriptor.cols != 100)
		{
			cout << endl;
			cout << "Error at image " << image_name_sift2 << " Descriptor Size: " << bowDescriptor.size();
			lost_images.push_back(i);
			cout << endl;
		}

		trainingData.push_back(bowDescriptor);

		std::cout << i + 1 << " / " << NUM_FILES_TRAIN << " -- " << image_name_sift2 << '\r';
	}

	cout << endl;
	correctLabels();

	std::cout << endl;
	std::cout << "Training Data Created" << endl;
	std::cout << "Training Data Size: " << trainingData.size() << endl;
	std::cout << "Number of labels: " << labels.size() << endl;

	//applyKNN(bag_descr_extractor);

	//applySVM(bag_descr_extractor);
	
	return 0;
}