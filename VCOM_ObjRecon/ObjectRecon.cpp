#include <iostream>
#include <fstream>
#include <string>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\ml\ml.hpp>
#include <opencv2\nonfree\features2d.hpp>

using namespace cv;
using namespace std;

int NUM_FILES_TRAIN;
int NUM_FILES_TEST;
int DICTIONARY_SIZE;
int K;

std::map<string, int> Label_ID;
std::map<float, string> Label_ID_reverse;

string TEST_DIR = "test/";

Mat labels;
vector<int> label_vec;
Mat train_descriptors;
Mat trainingData;

//vector<vector<cv::KeyPoint>> same_keys;

vector<float> KNNResult;
vector<float> SVMResult;

TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.001);
int retries = 1;
int flags = KMEANS_PP_CENTERS;

void initConfigVars()
{
	ifstream file("vars.config");
	string str;
	string delimiter = "=";

	for (int i = 0; i < 4; i++)
	{
		getline(file, str);
		size_t pos = 0;
		string var;
		string var_value;
		while ((pos = str.find(delimiter)) != string::npos)
		{
			var = str.substr(0, pos);
			var_value = str.erase(0, pos + delimiter.length());
		}

		int value = std::stoi(var_value);

		if (i == 0)
		{
			NUM_FILES_TRAIN = value;
		}
		else if (i == 1)
		{
			NUM_FILES_TEST = value;
		}
		else if (i == 2)
		{
			DICTIONARY_SIZE = value;
		} 
		else if (i == 3)
		{
			K = value;
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

void initializeLabels(int number_of_images)
{
	cout << endl << "Initializing the labels (responses) from csv file" << endl;

	ifstream file("trainLabels.csv");
	string value;

	int number_of_labels = 0;

	for (int i = 0; i < number_of_images; i++)
	{
		getline(file, value);	
		string csv_label = value.substr(value.find(",") + 1);
		int label = Label_ID.find(csv_label)->second;
		labels.push_back(label);
		number_of_labels++;
		cout << i + 1 << " / " << number_of_images << '\r';

	}

	cout << endl;
	cout << "Initiated " << number_of_labels << " labels" << endl;
}

bool resultIsExpected(int pos, float result)
{
	return result == (pos / 10);
}

void printResults()
{
	cout << endl << "Results: " << endl;
	//process KNN results
	int KNNSuccess = 0;
	ofstream knn_sub("knn_submission.csv");
	knn_sub << "id,label\n";

	cout << "Comparing " << KNNResult.size() << " KNN Results" << endl;

	for (int i = 0; i < KNNResult.size(); i++)
	{
		if (resultIsExpected(i, KNNResult.at(i)))
		{
			KNNSuccess++;
		}
		knn_sub << i + 1 << "," << KNNResult.at(i) << "\n";
	}

	cout << "KNN Success Rate: " << KNNSuccess << " / " << KNNResult.size() << endl;

	//process SVM results

	int SVMSuccess = 0;
	ofstream svm_sub("svm_submission.csv");
	svm_sub << "id,label\n";

	cout << "Comparing " << SVMResult.size() << " SVM Results" << endl;

	for (int i = 0; i < SVMResult.size(); i++)
	{
		if (resultIsExpected(i, SVMResult.at(i)))
		{
			SVMSuccess++;
		}
		svm_sub << i + 1 << "," << SVMResult.at(i) << "\n";
	}

	cout << "SVM Success Rate: " << SVMSuccess << " / " << SVMResult.size() << endl;
}

void applyKNN(cv::BOWImgDescriptorExtractor bag_descr_extractor)
{
	//KNN CLASSIFIER
	cout << endl << "Training KNN Classifier" << endl;

	cv::Ptr<cv::KNearest> knn = new cv::KNearest;
	knn->train(trainingData, labels, cv::Mat(), false, 32, false);

	cout << "KNN Classifier Trained" << endl;

	cout << "KNN Testing" << endl;

	for (int i = 0; i < NUM_FILES_TEST; i++)
	{
		string test_image_name = TEST_DIR + to_string(i + 1) + ".png";
		Mat image = cv::imread(test_image_name, CV_LOAD_IMAGE_GRAYSCALE);
		
		if (!image.data)
		{
			std::cout << "[SIFT 3] Error reading image " << test_image_name << endl;
			exit(0);
		}

		cout << (i + 1) << " / " << NUM_FILES_TEST << '\r';

		cv::Ptr<cv::FeatureDetector> detector_test = new cv::SiftFeatureDetector();
		vector<cv::KeyPoint> keypoints;
		detector_test->detect(image, keypoints);

		Mat bowDescriptor_test;
		bag_descr_extractor.compute(image, keypoints, bowDescriptor_test);

		if (bowDescriptor_test.cols != DICTIONARY_SIZE)
		{
			cout << endl;
			cout << "Error at image " << test_image_name << " Descriptor Size: " << bowDescriptor_test.size();
			cout << endl;
		}
		else
		{
			cv::Mat result;
			knn->find_nearest(bowDescriptor_test, K, result, cv::Mat(), cv::Mat());
			KNNResult.push_back(result.at<float>(0, 0));
		}
	}
	cout << endl;
}

void applySVM(cv::BOWImgDescriptorExtractor bag_descr_extractor)
{
	cout << endl << "Setting up SVM parameters" << endl;

	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::RBF;
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

		cout << (i + 1) << " / " << NUM_FILES_TEST << '\r';

		cv::Ptr<cv::FeatureDetector> detector_test = new cv::SiftFeatureDetector();		
		vector<cv::KeyPoint> keypoints;
		detector_test->detect(image, keypoints);

		Mat bowDescriptor_test;
		bag_descr_extractor.compute(image, keypoints, bowDescriptor_test);

		if (bowDescriptor_test.cols != DICTIONARY_SIZE)
		{
			cout << endl;
			cout << "Error at image " << test_image_name << " Descriptor Size: " << bowDescriptor_test.size();
			cout << endl;
		}
		else
		{
			float result;
			result = SVM.predict(bowDescriptor_test);
			SVMResult.push_back(result);
		}
	}
	cout << endl;
}

int main()
{
	cout << endl << "Populating maps" << endl << endl;

	initConfigVars();
	
	instantiateMaps();

	Mat dictionary;
	cv::FileStorage fs("vocabulary.xml", cv::FileStorage::READ);

	fs.release();
	cout << "Vocabulary not detected. Creating..." << endl;
	cout << "Extracting the Descriptors (Feature Vectors) using SIFT" << endl;

	for (int i = 0; i < NUM_FILES_TRAIN; i++)
	{
		string image_name_sift1 = "train/" + to_string(i + 1) + ".png";
		Mat image = cv::imread(image_name_sift1, CV_LOAD_IMAGE_GRAYSCALE);
		if (image.data)
		{
			cv::Ptr<cv::FeatureDetector> detector_dictionary = new cv::SiftFeatureDetector();
			cv::Ptr<cv::DescriptorExtractor> extractor_dictionary = new cv::SiftDescriptorExtractor();

			vector<cv::KeyPoint> keypoints;
			detector_dictionary->detect(image, keypoints);

			Mat extracted_descriptor;
			extractor_dictionary->compute(image, keypoints, extracted_descriptor);

			train_descriptors.push_back(extracted_descriptor);

			std::cout << i + 1 << " / " << NUM_FILES_TRAIN << " -- " << image_name_sift1 << '\r';
		}
	}

	cout << endl;
	cout << endl << "Creating Bag of Words" << endl;

	cv::BOWKMeansTrainer bag_of_words_trainer(DICTIONARY_SIZE, tc, retries, flags);
	dictionary = bag_of_words_trainer.cluster(train_descriptors);

	cout << "Bag of Words info: " << dictionary.size() << endl << endl;

	cout << "Creating the Training Data for KNN and SVM based on Bag of Words" << endl;

	cv::Ptr<DescriptorMatcher> descr_matcher(new FlannBasedMatcher);
	cv::Ptr<cv::FeatureDetector> detector_train = new cv::SiftFeatureDetector();
	cv::Ptr<cv::DescriptorExtractor> extractor_train = new cv::SiftDescriptorExtractor();
	cv::BOWImgDescriptorExtractor bag_descr_extractor(extractor_train, descr_matcher);
	
	bag_descr_extractor.setVocabulary(dictionary);

	int number_of_images = 0;

	for (int i = 0; i < NUM_FILES_TRAIN; i++)
	{
		string image_name_sift2 = "train/" + to_string(i + 1) + ".png";
		Mat image = cv::imread(image_name_sift2, CV_LOAD_IMAGE_GRAYSCALE);
		
		if (image.data)
		{
			number_of_images++;
			cv::Ptr<cv::FeatureDetector> detector_train = new cv::SiftFeatureDetector();
			vector<cv::KeyPoint> keypoints;
			detector_train->detect(image, keypoints);

			Mat bowDescriptor;
			bag_descr_extractor.compute(image, keypoints, bowDescriptor);

			if (bowDescriptor.cols != DICTIONARY_SIZE)
			{
				std::cout << endl;
				std::cout << "Error in train image " << image_name_sift2 << " Descriptor Size: " << bowDescriptor.size();
				std::cout << endl;
			}

			trainingData.push_back(bowDescriptor);

			std::cout << i + 1 << " / " << NUM_FILES_TRAIN << " -- " << image_name_sift2 << '\r';
		}
	}

	cout << endl;

	initializeLabels(number_of_images);

	cout << endl << "Training Data Created" << endl;
	cout << "Training Data Size: " << trainingData.size() << endl;
	cout << "Number of labels: " << labels.size() << endl;

	applyKNN(bag_descr_extractor);

	applySVM(bag_descr_extractor);

	printResults();
	
	return 0;
}