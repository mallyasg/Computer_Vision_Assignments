# include "opencv2\core\core.hpp"
# include "opencv2\imgproc\imgproc.hpp"
# include "opencv2\highgui\highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

class Image_Segmentation {
private:
	cv::Mat Input_Image;
	cv::Mat Segmented_Image;
public:
	int spatialRad;
	int colorRad;
	int maxPyrLevel;
	Image_Segmentation();
	Image_Segmentation(char *Input_File_Name, int Spatial_Rad,
		int Color_Rad, int Max_Pyr_Level = 0);
	void Mean_Shift_Segmentation(int Flood_Fill_Required);
	void Display_Input_Image(string Window_Name);
	void Display_Segmented_Image(string Window_Name);
	cv::Mat Get_Input_Image();
	cv::Mat Get_Segmented_Image();
	void Set_Input_Image(cv::Mat Input_Image);
	void Set_Segmented_Image(cv::Mat Segmented_Image);
	void Clear_Segmented_Image();
	void Clear_Input_Image();
	cv::Mat Convert_RGB_2_CIELAB();
	cv::Mat Convert_CIELAB_2_RGB();
	cv::Mat Edge_Detection(int Low_Threshold, int ratio, int Kernel_Size);
	cv::Mat Find_Image_Contours(cv::Mat Edge_Image, vector<Vec3b> &colorTab);
	void Watershed_Segmentation();
	void Save_Segmented_Image(string File_Name);
	void floodFillPostprocess(Mat& img, const Scalar& colorDiff);
};
