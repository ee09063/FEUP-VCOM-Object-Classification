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

// vars.config parameters
int NUM_FILES_TRAIN;
int NUM_FILES_TEST;
int DICTIONARY_SIZE;
int K;
int USE_KNN;
int USE_SVM;
int USE_BAYES;

std::map<string, int> Label_ID;
std::map<float, string> Label_ID_reverse;

string TEST_DIR = "test/";

Mat labels;
Mat train_descriptors;
Mat trainingData;

TermCriteria tc(CV_TERMCRIT_ITER, 1000, 0.001);
int retries = 1;
int flags = KMEANS_PP_CENTERS;

// Read the configuration file
void initConfigVars()
{
	ifstream file("vars.config");
	string str;
	string delimiter = "=";

	std::cout << "Reading config file" << endl;

	for (int i = 0; i < 7; i++)
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
		else if (i == 4)
		{
			USE_KNN = value;
		}
		else if (i == 5)
		{
			USE_SVM = value;
		}
		else if (i == 6)
		{
			USE_BAYES = value;
		}

	}

	std::cout << "Number of files for training: " << NUM_FILES_TRAIN << endl;
	std::cout << "Number of files for testing: " << NUM_FILES_TEST << endl;
	std::cout << "Vocabulary size: " << DICTIONARY_SIZE << endl;
	std::cout << "KNN K " << K << endl;
	std::cout << "USE KNN " << USE_KNN << endl;
	std::cout << "USE SVM " << USE_SVM << endl;
	std::cout << "USE BAYES " << USE_BAYES << endl << endl;
}

// Initialize the maps
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

// Initialize the labels from CIFAR's file
void initializeLabels(int number_of_images)
{
	std::cout << endl << "Initializing the labels (responses) from csv file" << endl;

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
		std::cout << i + 1 << " / " << number_of_images << '\r';

	}

	std::cout << endl;
	std::cout << "Initiated " << number_of_labels << " labels" << endl;
}

// Apply kNN, SVM, BAYES
void applyMethods(cv::BOWImgDescriptorExtractor bag_descr_extractor)
{
	if (USE_KNN == 0 && USE_SVM == 0 && USE_BAYES == 0)
	{
		return;
	}

	// Prepare the submission files
	ofstream knn_sub("knn_submission.csv");
	
	ofstream svm_linear_sub("svm_linear_sub.csv");
	ofstream svm_rbf_sub("svm_rbf_sub.csv");
	ofstream svm_sig_sub("svm_sig_sub.csv");

	ofstream bayes_sub("bayes_submission.csv");
	
	//OpenCV's implementation
	cv::Ptr<cv::KNearest> knn = new cv::KNearest;
	
	cv::Ptr<cv::SVM> svm_linear = new cv::SVM;
	cv::Ptr<cv::SVM> svm_rbf = new cv::SVM;
	cv::Ptr<cv::SVM> svm_sig = new cv::SVM;
	
	cv::Ptr<cv::NormalBayesClassifier> bayes = new cv::NormalBayesClassifier;

	//Train the classifiers
	if (USE_KNN == 1)
	{
		std::cout << endl << "Training KNN Classifier" << endl;
		
		knn->train(trainingData, labels, cv::Mat(), false, 32, false);
		std::cout << "KNN Classifier Trained" << endl;
		knn_sub << "id,label\n";
	}	
	
	if (USE_SVM == 1)
	{
		std::cout << endl << "Setting up SVM parameters" << endl;
		
		CvSVMParams params_lin;
		params_lin.svm_type = CvSVM::C_SVC;
		params_lin.kernel_type = CvSVM::LINEAR;
		params_lin.term_crit = tc;
		
		CvSVMParams params_rbf;
		params_rbf.svm_type = CvSVM::C_SVC;
		params_rbf.kernel_type = CvSVM::RBF;
		params_rbf.term_crit = tc;

		CvSVMParams params_sig;
		params_sig.svm_type = CvSVM::C_SVC;
		params_sig.kernel_type = CvSVM::SIGMOID;
		params_sig.term_crit = tc;

		std::cout << "Training the SVM Linear" << endl;	
		svm_linear->train(trainingData, labels, cv::Mat(), cv::Mat(), params_lin);
		svm_linear_sub << "id,label\n";
		std::cout << "SVM Linear Trained" << endl;

		std::cout << "Training SVM RBF" << endl;
		svm_rbf->train(trainingData, labels, cv::Mat(), cv::Mat(), params_rbf);
		svm_rbf_sub << "id,label\n";
		std::cout << "SVM RBF Trained" << endl;

		std::cout << "Training the SVM Sigmoid" << endl;
		svm_sig->train(trainingData, labels, cv::Mat(), cv::Mat(), params_sig);
		svm_sig_sub << "id,label\n";
		std::cout << "SVM Sigmoid Trained" << endl;
	}

	if (USE_BAYES == 1)
	{
		std::cout << "Training Normal Bayes Classifier" << endl;
		bayes->train(trainingData, labels);
		bayes_sub << "id,label\n";
		std::cout << "Bayes Classifier Trained" << endl;
	}

	std::cout << endl << "Testing" << endl;

	for (int i = 0; i < NUM_FILES_TEST; i++)
	{
		string test_image_name = TEST_DIR + to_string(i + 1) + ".png";
		Mat image = cv::imread(test_image_name, CV_LOAD_IMAGE_GRAYSCALE);

		if (!image.data)
		{
			std::cout << "[SIFT Testing] Error reading image " << test_image_name << endl;
			exit(0);
		}

		std::cout << (i + 1) << " / " << NUM_FILES_TEST << '\r';

		cv::Ptr<cv::FeatureDetector> detector_test = new cv::SiftFeatureDetector();
		vector<cv::KeyPoint> keypoints;
		detector_test->detect(image, keypoints);

		Mat bowDescriptor_test;
		bag_descr_extractor.compute(image, keypoints, bowDescriptor_test);

		// If SIFT cannot extract descriptors this will have size [0x0]
		// So we just assign a random label, the same to every submission
		if (bowDescriptor_test.cols != DICTIONARY_SIZE)
		{	
			string rand_label = Label_ID_reverse.find(rand() % 10)->second;
			if (USE_KNN) knn_sub << i + 1 << "," << rand_label << "\n";
			if (USE_SVM)
			{
				svm_linear_sub << i + 1 << "," << rand_label << "\n";
				svm_rbf_sub << i + 1 << "," << rand_label << "\n";
				svm_sig_sub << i + 1 << "," << rand_label << "\n";
			}
			if (USE_BAYES) bayes_sub << i + 1 << "," << rand_label << "\n";
		}
		else
		{
			if (USE_KNN)
			{
				cv::Mat resultKNN;
				knn->find_nearest(bowDescriptor_test, K, resultKNN, cv::Mat(), cv::Mat());
				knn_sub << i + 1 << "," << Label_ID_reverse.find(resultKNN.at<float>(0, 0))->second << "\n";
			}
			
			if (USE_SVM)
			{
				float resultSVMLin;
				float resultSVMRBF;
				float resultSVMSig;
				
				resultSVMLin = svm_linear->predict(bowDescriptor_test);
				svm_linear_sub << i + 1 << "," << Label_ID_reverse.find(resultSVMLin)->second << "\n";

				resultSVMRBF = svm_rbf->predict(bowDescriptor_test);
				svm_rbf_sub << i + 1 << "," << Label_ID_reverse.find(resultSVMRBF)->second << "\n";

				resultSVMSig = svm_sig->predict(bowDescriptor_test);
				svm_sig_sub << i + 1 << "," << Label_ID_reverse.find(resultSVMSig)->second << "\n";
			}

			if (USE_BAYES)
			{
				float resultBayes;
				resultBayes = bayes->predict(bowDescriptor_test);
				bayes_sub << i + 1 << "," << Label_ID_reverse.find(resultBayes)->second << "\n";
			}
		}
	}
	std::cout << endl;
}

int main()
{
	std::cout << endl << "Populating maps" << endl << endl;

	initConfigVars();
	
	instantiateMaps();

	// The dictionary to be created by the Bag of Words 
	Mat dictionary;

	std::cout << "Creating Vocabulary" << endl;
	std::cout << "Extracting the Descriptors using SIFT" << endl;

	//Extract the descriptors from the test images using SIFT to build the vocabulary
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

	std::cout << endl;
	std::cout << endl << "Creating Bag of Words" << endl;

	cv::BOWKMeansTrainer bag_of_words_trainer(DICTIONARY_SIZE, tc, retries, flags);
	dictionary = bag_of_words_trainer.cluster(train_descriptors);

	std::cout << "Bag of Words info: " << dictionary.size() << endl << endl;

	std::cout << "Creating the Training Data for KNN and SVM based on Bag of Words" << endl;

	cv::Ptr<DescriptorMatcher> descr_matcher(new FlannBasedMatcher);
	cv::Ptr<cv::FeatureDetector> detector_train = new cv::SiftFeatureDetector();
	cv::Ptr<cv::DescriptorExtractor> extractor_train = new cv::SiftDescriptorExtractor();
	cv::BOWImgDescriptorExtractor bag_descr_extractor(extractor_train, descr_matcher);
	
	bag_descr_extractor.setVocabulary(dictionary);

	int number_of_images = 0;

	//After the vocabulary is built, we use it + the bag of words to compute the descriptors to create the feature vectors
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

			//If SIFT cannot extract descriptors, the size will be [0x0]
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

	std::cout << endl;

	initializeLabels(number_of_images);

	std::cout << endl << "Training Data Created" << endl;
	std::cout << "Training Data Size: " << trainingData.size() << endl;
	std::cout << "Number of labels: " << labels.size() << endl;

	applyMethods(bag_descr_extractor);
	
	return 0;
}