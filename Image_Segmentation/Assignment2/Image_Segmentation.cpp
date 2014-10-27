# include "Image_Segmentation.h"

Image_Segmentation::Image_Segmentation() {
	
	/* 
	 * Initialize the Input Image, Segmented Image and the Meanshift filtering Parameters to zero
	 */
	Input_Image.create(cv::Size(0, 0), CV_8UC3);
	Segmented_Image.create(cv::Size(0, 0), CV_8UC3);
	spatialRad = 0;
	colorRad = 0;
	maxPyrLevel = 0;

}

Image_Segmentation::Image_Segmentation(char *Input_File_Name, int Spatial_Rad,
	int Color_Rad, int Max_Pyr_Level) {

	/*
	 * This constructor reads the image from a file and assigns the values passed to the Mean shift filtering
	 * parameters.
	 */
	Input_Image = cv::imread(Input_File_Name);
	Segmented_Image.create(cv::Size(Input_Image.cols, Input_Image.rows), Input_Image.type());
	spatialRad = Spatial_Rad;
	colorRad = Color_Rad;
	maxPyrLevel = Max_Pyr_Level;

}

void Image_Segmentation::Mean_Shift_Segmentation(int Flood_Fill_Required) {

	/*
	 * Perform the mean shift filering by calling the function pyrMeanShiftFiltering. 
	 * The Input image to be segmented is converted to the CIELAB space using the 
	 * custom function Convert_RGB_2_CIELAB. The output of the Mean Shift Filtering is
	 * converted from CIELAB back to RGB Color space using the custom function 
	 * Convert_CIELAB_2_RGBand assigned to the Segmented_Image Image container of the 
	 * class.
	*/

	cv::Mat Segmented_Image;
	cv::pyrMeanShiftFiltering(this->Convert_RGB_2_CIELAB(), Segmented_Image, this->spatialRad, this->colorRad, this->maxPyrLevel);
	
	this->Set_Segmented_Image(Segmented_Image);

	Segmented_Image = this->Convert_CIELAB_2_RGB();
	if (Flood_Fill_Required) {
		floodFillPostprocess(Segmented_Image, cv::Scalar(2));
	}

	this->Set_Segmented_Image(Segmented_Image);

}

void Image_Segmentation::floodFillPostprocess(Mat& img, const Scalar& colorDiff)
{
	CV_Assert(!img.empty());
	RNG rng = theRNG();
	Mat mask(img.rows + 2, img.cols + 2, CV_8UC1, Scalar::all(0));
	for (int y = 0; y < img.rows; y++)
	{
		for (int x = 0; x < img.cols; x++)
		{
			if (mask.at<uchar>(y + 1, x + 1) == 0)
			{
				// Fill in the segmented region starting from the seed point defined at (x, y) with randomly generated color
				Scalar newVal(rng(256), rng(256), rng(256));
				//floodFill(img, mask, Point(x, y), newVal, 0, colorDiff, colorDiff);
				floodFill(img, mask, Point(x, y), newVal, 0, cv::Scalar::all(0), colorDiff);
			}
		}
	}
}

void Image_Segmentation::Display_Input_Image(string Window_Name) {

	/*
	 * This function is used to create a window for viewing the image. The function
	 * calls the OpenCV namedWindow to create a window with the Window_Name. Then
	 * the function imshow is called to display the Input_Image in the above created
	 * window. The image is displayed untill any key is pressed.
	 */
	cv::namedWindow(Window_Name, CV_WINDOW_AUTOSIZE);
	cv::imshow(Window_Name, Input_Image);
	cv::waitKey(0);

}

void Image_Segmentation::Display_Segmented_Image(string Window_Name) {
	/*
	 * This function is used to create a window for viewing the image. The function
	 * calls the OpenCV namedWindow to create a window with the Window_Name. Then
	 * the function imshow is called to display the Segmented_Image in the above 
	 * created window. The image is displayed untill any key is pressed.
	 */
	cv::namedWindow(Window_Name, CV_WINDOW_AUTOSIZE);
	cv::imshow(Window_Name, Segmented_Image);
	cv::waitKey(0);
}

cv::Mat Image_Segmentation::Get_Input_Image() {
	// Return the Input Image
	return Input_Image;
}

cv::Mat Image_Segmentation::Get_Segmented_Image() {
	// Return the Segmented Image
	return Segmented_Image;
}
void Image_Segmentation::Set_Input_Image(cv::Mat Input_Image) {
	// Set the value of the Input Image
	this->Input_Image = Input_Image;

}
void Image_Segmentation::Set_Segmented_Image(cv::Mat Segmented_Image) {
	// Set the value of the Segmented Image
	this->Segmented_Image = Segmented_Image;

}

void Image_Segmentation::Clear_Segmented_Image() {
	// Clear the contents of the Segmented Image
	this->Segmented_Image = Mat(Size(this->Segmented_Image.cols, this->Segmented_Image.rows), CV_8UC3, cv::Scalar(0, 0, 0));
}

void Image_Segmentation::Clear_Input_Image() {
	// Clear the contents of the Input Image
	this->Input_Image = Mat(Size(this->Input_Image.cols, this->Input_Image.rows), CV_8UC3, cv::Scalar(0, 0, 0));
}

cv::Mat Image_Segmentation::Convert_CIELAB_2_RGB() {
	// Conver the Segmented Image i.e CIELAB image back to RGB image
	Mat RGB_Image;
	cv::cvtColor(this->Segmented_Image, RGB_Image, cv::COLOR_Lab2BGR);
	return RGB_Image;
}

cv::Mat Image_Segmentation::Convert_RGB_2_CIELAB() {

	// Covert the RGB image i.e. the input to CIE_LAB image
	Mat CIELAB_Image;
	cv::cvtColor(this->Input_Image, CIELAB_Image, cv::COLOR_BGR2Lab);
	return CIELAB_Image; 
}

cv::Mat Image_Segmentation::Edge_Detection(int Low_Threshold, int ratio, int Kernel_Size) {

	// Detect the Edges in the image using Canny Edge detector.
	cv::Mat Gray_Image, Edge_Image;

	cv::cvtColor(this->Input_Image, Gray_Image, CV_BGR2GRAY);
	cv::blur(Gray_Image, Edge_Image, cv::Size(3, 3));


	cv::Canny(Edge_Image, Edge_Image, Low_Threshold, Low_Threshold * ratio, Kernel_Size);
	
	return Edge_Image;

}

cv::Mat Image_Segmentation::Find_Image_Contours(cv::Mat Edge_Image, vector<Vec3b> &colorTab) {

	// Find the contours in the image
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	cv::findContours(Edge_Image, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	Mat marker(Edge_Image.size(), CV_32S);
	int idx = 0;
	int compCount = 0;
	
	/* 
	 * Iterate through the top level contours. The area of the contours is compared against
	 * a threshold in order to remove contours which do not contribute in the segmentation
	 * of the image. 
	 */
	for (; idx >= 0; idx = hierarchy[idx][0], compCount++) {
		if (fabs(contourArea(contours[idx])) < 25)
			continue;
		drawContours(marker, contours, idx, Scalar::all(compCount + 1), 1, 8, hierarchy, INT_MAX);
	}
	
	/*
	 * Load the vector of colors with random color values for coloring the segmenation
	 */
	for (int i = 0; i < compCount; i++)
	{
		int b = theRNG().uniform(0, 255);
		int g = theRNG().uniform(0, 255);
		int r = theRNG().uniform(0, 255);

		colorTab.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
	}

	return marker;
}

void Image_Segmentation::Watershed_Segmentation() {

	/*
	* Perform Edge Detection on the Image to find the contours.
	* Use these markers as input to the watershed algorithm.
	*/
	vector<Vec3b> colorTab;
	int compCount = 0;
	cv::Mat Gray_Image;
	cv::cvtColor(this->Input_Image, Gray_Image, CV_BGR2GRAY);
	cv::Mat Edge_Image = this->Edge_Detection(75, 3, 3);
	cv::Mat Markers = this->Find_Image_Contours(Edge_Image, colorTab);
	
	cv::watershed(this->Input_Image, Markers);
	
	Mat Watershed_Segmented_Image(Markers.size(), CV_8UC3);

	/* 
	 * Paint the watershed image using the color vector to clearly partition
	 * the image
     */
	for (int i = 0; i < Markers.rows; i++)
		for (int j = 0; j < Markers.cols; j++)
		{
			int index = Markers.at<int>(i, j);
			if (index == -1)
				Watershed_Segmented_Image.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
			else if (index <= 0 || index > colorTab.size())
				Watershed_Segmented_Image.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
			else
				Watershed_Segmented_Image.at<Vec3b>(i, j) = colorTab[index - 1];
		}

	/*
	 * Convert the Single channel Gray scale image to 3 channel gray scale image. This is 
	 * done in order to blend the gray scale image with the color partitioned segmented image
	 */
	cv::cvtColor(Gray_Image, Gray_Image, CV_GRAY2RGB);
	cv::addWeighted(Watershed_Segmented_Image, 0.5, Gray_Image, 0.5, 0.0, Watershed_Segmented_Image);
	this->Set_Segmented_Image(Watershed_Segmented_Image);
}

void Image_Segmentation::Save_Segmented_Image(string File_Name) {

	/*
	 * Write the Segmented image to a file
	 */
	cv::imwrite(File_Name, this->Segmented_Image);

}