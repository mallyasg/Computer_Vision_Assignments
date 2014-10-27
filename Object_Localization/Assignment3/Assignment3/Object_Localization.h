# include "opencv2\core\core.hpp"
# include "opencv2\imgproc\imgproc.hpp"
# include "opencv2\highgui\highgui.hpp"
# include "opencv2\features2d\features2d.hpp"
# include "opencv2\nonfree\nonfree.hpp"
# include "opencv2\calib3d\calib3d.hpp"
# include "opencv2\opencv.hpp"
# include <iostream>
# include <vector>
# include <limits>

using namespace cv;
using namespace std;

class Object_Localization {

private:

	cv::Mat Input_Image;
	cv::Mat Input_Image_Gray;
	std::vector<cv::Mat> Template_Images;
	std::vector<cv::Mat> Template_Images_Gray;
	std::vector<cv::KeyPoint> Image_Key_Points;
	cv::Mat Image_Features;
	std::vector< std::vector<cv::KeyPoint> > Template_Key_Points;
	std::vector<cv::Mat> Template_Features;
	std::vector< std::vector< DMatch > > Matching_Features;
	std::vector< cv::Mat > Image_Matches;
	std::vector< cv::Mat > Homography_Matrices;
	std::vector< std::vector<uchar> >  Inlier_Mask;
	std::vector< int > Num_Inliers;
	std::vector< cv::Mat > Final_Image;

public:

	Object_Localization();
	Object_Localization(char **File_Names, int Num_Files);
	/*
	 * Detects the key points in the input image.
	 */
	void Detect_Image_Key_Points();
	/*
	 * Extracts the descriptor for the key points 
	 * in the input image.
	 */
	void Detect_Image_Features();
	/*
	 * Detects the key points in the Templage image(s).
	 */
	void Detect_Template_Image_Key_Points();
	/*
	 * Extracts the descriptor for the key points
	 * in the Template image(s).
	 */
	void Detect_Template_Image_Features();
	/*
	 * Matches the keypoints of the Input Image and the 
	 * template image(s) using the L2 Normalized Brute 
	 * Force Matcher.
	 */
	void Match_Features(float scale);
	/*
	 * Draws the keypoints on the composite image - i.e.
	 * image containing input image and template image 
	 * placed side by side - and connects the crudely
	 * matched points with lines.
	 */
	void Draw_Image_Matches();
	/*
	 * Displays the composite images.
	 */
	void Display_Image_Matches();
	/*
	 * Creates a pair of vector of matching key points in homogeneous co-ordinates. 
	 */
	void create_3D_Points(std::vector< cv::Point3f > *points1, std::vector< cv::Point3f > *points2, int template_Index);
	
	void combinationUtil(std::vector < int > index_Array, int Size_Of_Index_Array, int Num_Points_Required, int index, 
		std::vector < int > data, int current_Ele_Index, std::vector< std::vector < int > > *current_Index);
	std::vector< std::vector < int > > create_Possible_Combinations(int Num_Matches);
	/*
	 * Checks if the randomly selected key points of the input image are collinear. If collinear returns true else
	 * false.
	 */
	bool check_Collinear_Points(std::vector<cv::Point3f>current_Points);
	/* 
	 * Computes the Homography matrix using a set of 4 matching key points
	 */
	cv::Mat computeH(std::vector<cv::Point3f>current_Points1, std::vector<cv::Point3f>current_Points2);
	/*
	 * Finds the number of inliers for a given homography matrix.
	 */
	int Find_Number_Of_Inliers(std::vector<cv::Point3f>points1, std::vector<cv::Point3f>points2, cv::Mat Homography_Matrix, std::vector<uchar> *inliers,
		double *Distance_Standard_Dev);
	/*
	 * Finds the Homography matrix assosciated with a set of corresponding
	 * key points using RANSAC algorithm.
	 */
	void Find_Homography();
	/*
	 * Save the final images that contain the inliers and their matches onto a file.
	 */
	void Save_Final_Images();
	/*
	 * Saves the Homography matrix and the location of the inlier
	 * key points onto a file.
	 */
	void Save_Data_To_File();

};

# define DISTANCE_THRESHOLD 30
# define NUM_POINTS_REQUIRED 3