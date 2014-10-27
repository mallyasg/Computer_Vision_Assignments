# include "Object_Localization.h"

Object_Localization::Object_Localization() {

	// Create an image of size 0
	Input_Image.create(cv::Size(0, 0),     // Size of the image
		               CV_8UC3);           // Number of channels and number of bits per channel

}

Object_Localization::Object_Localization(char **File_Names, int Num_Files) {

	cv::initModule_nonfree();
	// Read the input image
	Input_Image = imread(File_Names[0]);     
	// Convert the color of Input Image to gray scale to be used in SIFT
	cv::cvtColor(Input_Image,                
		         Input_Image_Gray, 
				 CV_RGB2GRAY);
	
	for (int i = 0; i < Num_Files - 1; i++) {

		// Read the template images onto a temporary matrix object
		cv::Mat Temp_Image = imread(File_Names[i + 1]);  
		// Push the temporary matrix object onto the vector
		Template_Images.push_back(Temp_Image);           
		cv::Mat Temp_Image_Gray;
		// Convert the color of Template Images to gray scale to be used in SIFT
		cv::cvtColor(Template_Images[i],                 
			         Temp_Image_Gray, 
					 CV_RGB2GRAY);
		Template_Images_Gray.push_back(Temp_Image_Gray);
	}


}

void Object_Localization::Detect_Image_Key_Points() {

	// Create a SIFT feature detector handle
	cv::Ptr<cv::FeatureDetector> Detector = FeatureDetector::create("SIFT");  
	// Detect the key points in the image using SIFT
	Detector->detect(this->Input_Image_Gray,  // Input Image 
		             this->Image_Key_Points); // Vector to store extracted keypoints


}

void Object_Localization::Detect_Image_Features() {

	// Create a SIFT feature descriptor handle
	cv::Ptr<cv::DescriptorExtractor> Extractor = DescriptorExtractor::create("SIFT");     
	// Extract the feature descriptors
	Extractor->compute(this->Input_Image_Gray, // Input Image
		               this->Image_Key_Points, // Extracted Keypoints
					   this->Image_Features);  // Mat to store the Descriptor of Keypoints

}

void Object_Localization::Detect_Template_Image_Key_Points() {

	cv::Ptr<cv::FeatureDetector> Detector = FeatureDetector::create("SIFT");

	for (int i = 0; i < Template_Images.size(); i++) {
		// Create a vector to hold the keypoints of template images
		std::vector<cv::KeyPoint> Temp_Key_Points;                          
		// Detect the key point in the template images
		Detector->detect(this->Template_Images_Gray[i], 
			             Temp_Key_Points);        
		// Push the keypoints onto the vector    
		Template_Key_Points.push_back(Temp_Key_Points);

	}

}

void Object_Localization::Detect_Template_Image_Features() {

	cv::Ptr<cv::DescriptorExtractor> Extractor = DescriptorExtractor::create("SIFT");

	for (int i = 0; i < Template_Images.size(); i++) {

		cv::Mat Temp_Descriptor;

		// Extract the descriptors of the keypoint 
		Extractor->compute(this->Template_Images_Gray[i],
			               this->Template_Key_Points[i],
			               Temp_Descriptor);

		// Push the descriptor onto the vector
		this->Template_Features.push_back(Temp_Descriptor);

	}


}

void Object_Localization::Match_Features(float scale) {


	// Create an object of the Brute Force Matcher class, that uses L2 
	// normalization to compute the distance between the two descriptors
	cv::BFMatcher matcher(NORM_L2);

	for (int i = 0; i < this->Template_Features.size(); i++) {

		// Create a vector that holds the matches of the keypoints
		std::vector< std::vector< DMatch > >matches;
		
		// Use K-Nearest Neighbour match to match the keypoints. Restrict the
		// number of matches to 2.
		matcher.knnMatch(this->Image_Features,
			             this->Template_Features[i],
			             matches,
			             2);

		std::vector< DMatch > good_matches;

		for (int j = 0; j < matches.size(); j++)
		{
			// If the ratio of the first match to the second match using KNN
			// match is less than a particular scale then use the first match
			// as a good match for further processing.
			if (matches[j][0].distance < scale * matches[j][1].distance) {
				
				good_matches.push_back(matches[j][0]);
			
			}
		}

		// Push the good matches onto the vector for each template images.
		this->Matching_Features.push_back(good_matches);
	}


}

void Object_Localization::Draw_Image_Matches() {

	cv::Mat img_matches;

	for (int i = 0; i < Template_Images.size(); i++) {

		// Add the cols of the Input image and the Template image to obtain cols 
		// and take the max of the rows of Input image and template image as rows. 
		// These cols and rows are used to allocate memory for the composite image.

		int cols = this->Input_Image.cols + this->Template_Images[i].cols;
		int rows = (this->Template_Images[i].rows > this->Input_Image.rows) ?
			this->Template_Images[i].rows : this->Input_Image.rows;

		// Create Mat objects for Input Image and Template Images with Keypoints
		// marked on them.
		cv::Mat Input_Image_Key_Point, Template_Image_Key_Point;

		this->Input_Image.copyTo(Input_Image_Key_Point);
		this->Template_Images[i].copyTo(Template_Image_Key_Point);

		// Draw key points
		cv::drawKeypoints(this->Input_Image, 
			              this->Image_Key_Points, 
						  Input_Image_Key_Point);

		cv::drawKeypoints(this->Template_Images[i], 
			              this->Template_Key_Points[i], 
						  Template_Image_Key_Point);

		// Create the composite image using the cols and rows.
		cv::Mat Composite_Images(cv::Size(cols, rows), 
			                     CV_8UC3);
		// Create region of interest in the composite image, such that the left part of
		// the image has the Input Image and the right part the template image.
		cv::Mat Left_Image(Composite_Images, 
			              cv::Rect(0, 
						           0, 
								   this->Input_Image.cols, 
								   this->Input_Image.rows));
		cv::Mat Right_Image(Composite_Images, 
			                cv::Rect(this->Input_Image.cols, 
							         0, 
									 this->Template_Images[i].cols, 
									 this->Template_Images[i].rows));
		// Copy the keypoint contained Input image and the template image onto the left 
		// and right part of the composite image.
		Input_Image_Key_Point.copyTo(Left_Image);
		Template_Image_Key_Point.copyTo(Right_Image);

		// Draw lines using the line function of OpenCV for the matching keypoints
		for (int j = 0; j < this->Matching_Features[i].size(); j++) {

			cv::Point2f trainPoint, queryPoint;

			trainPoint.x = this->Template_Key_Points[i][this->Matching_Features[i][j].trainIdx].pt.x + 
				           this->Input_Image.cols;
			
			trainPoint.y = this->Template_Key_Points[i][this->Matching_Features[i][j].trainIdx].pt.y;

			queryPoint = this->Image_Key_Points[this->Matching_Features[i][j].queryIdx].pt;

			cv::line(Composite_Images,
				     queryPoint,
				     trainPoint,
				     cv::Scalar(0, 0, 255),
				     1,
				     8);

		}

		// Push the composite images onto the vector
		this->Image_Matches.push_back(Composite_Images);

	}

}

void Object_Localization::Display_Image_Matches() {

	// Display and save the composite images
	for (int i = 0; i < this->Image_Matches.size(); i++) {

		cv::imshow("Matches", 
			       this->Image_Matches[i]);

		string Save_Image_File_Name = "Matched_Image_";

		Save_Image_File_Name = Save_Image_File_Name + 
			                   std::to_string(i) + 
							   ".jpg";

		cv::imwrite(Save_Image_File_Name, 
			        this->Image_Matches[i]);

		cv::waitKey(0);

	}

}

void Object_Localization::create_3D_Points(std::vector< cv::Point3f > *points1, 
	                                       std::vector< cv::Point3f > *points2, 
										   int template_Index) {

	// For the matching keypoints obtained using the BFMatcher we create 3D 
	// homogeneous points for the keypoints in the input image and template
	// image.
	for (int i = 0; i < this->Matching_Features[template_Index].size(); i++) {
		// Get the position of left keypoints
		float x = this->Image_Key_Points[this->Matching_Features[template_Index][i].queryIdx].pt.x;
		float y = this->Image_Key_Points[this->Matching_Features[template_Index][i].queryIdx].pt.y;
		
		// Scale them in-order to reduce the errors in calculation of the 
		// Homography matrix. The scaling is done to the center of the 
		// image.
		x = x - (float)(this->Input_Image.cols / 2.0);
		y = y - (float)(this->Input_Image.rows / 2.0);

		points1->push_back(cv::Point3f(x, y, 1));

		// Get the position of right keypoints
		x = this->Template_Key_Points[template_Index][this->Matching_Features[template_Index][i].trainIdx].pt.x;
		y = this->Template_Key_Points[template_Index][this->Matching_Features[template_Index][i].trainIdx].pt.y;

		x = x - (float)(this->Input_Image.cols / 2.0);
		y = y - (float)(this->Input_Image.rows / 2.0);

		points2->push_back(cv::Point3f(x, y, 1));
	}

}

bool Object_Localization::check_Collinear_Points(std::vector<cv::Point3f>current_Points) {

	cv::Mat point1(cv::Size(1, 3), CV_64FC1);
	cv::Mat point2(cv::Size(1, 3), CV_64FC1);
	cv::Mat point3(cv::Size(1, 3), CV_64FC1);
	cv::Mat line(cv::Size(1, 3), CV_64FC1);
	double dot_Product = 0;
	bool is_Collinear = false;

	// For the four points selected at random, we take three points
	// and compute the scalar triple product. The scalar triple
	// product gives the volume of the parallelopiped formed by the 
	// the three vectors. If the three vectors are coplanar we get 
	// the volume as zero. If the volume of one combination of the 
	// point is zero we return true.
	for (int i = 0; i < current_Points.size() - 2; i++) {

		point1 = cv::Mat(current_Points[i]);
		for (int j = i + 1; j < current_Points.size() - 1; j++) {

			point2 = cv::Mat(current_Points[j]);
			line = point1.cross(point2);

			for (int k = j + 1; k < current_Points.size(); k++) {

				point3 = cv::Mat(current_Points[k]);
				dot_Product = point3.dot(line);
				if (dot_Product < 10e-2) {
					is_Collinear = true;
					break;
				}

			}

			if (is_Collinear == true) {

				break;
			}

		}

		if (is_Collinear == true) {

			break;

		}

	}

	return is_Collinear;

}

cv::Mat Object_Localization::computeH(std::vector<cv::Point3f>current_Points1, 
	                                  std::vector<cv::Point3f>current_Points2) {

	int Num_Points = current_Points1.size();
	cv::Mat A, b;
	cv::Mat Homography_Matrix;//

	A.create(cv::Size(9, Num_Points * 2), CV_64F);
	
	// Load the data into the A matrix and vector b
	A.setTo(cv::Scalar(0));
	
	// For each point create the A matrix, which is derived
	// using x2 = Hx1 equation.
	for (int i = 0; i < Num_Points; i = i + 1) {

		A.at<double>(2 * i, 0) = current_Points1[i].x;
		A.at<double>(2 * i, 1) = current_Points1[i].y;
		A.at<double>(2 * i, 2) = 1.0;
		A.at<double>(2 * i, 3) = 0.0;
		A.at<double>(2 * i, 4) = 0.0;
		A.at<double>(2 * i, 5) = 0.0;
		A.at<double>(2 * i, 6) = -current_Points1[i].x * current_Points2[i].x;
		A.at<double>(2 * i, 7) = -current_Points1[i].y * current_Points2[i].x;
		A.at<double>(2 * i, 8) = -current_Points2[i].x;

		A.at<double>(2 * i + 1, 0) = 0.0;
		A.at<double>(2 * i + 1, 1) = 0.0;
		A.at<double>(2 * i + 1, 2) = 0.0;
		A.at<double>(2 * i + 1, 3) = current_Points1[i].x;
		A.at<double>(2 * i + 1, 4) = current_Points1[i].y;
		A.at<double>(2 * i + 1, 5) = 1.0;
		A.at<double>(2 * i + 1, 6) = -current_Points1[i].x * current_Points2[i].y;
		A.at<double>(2 * i + 1, 7) = -current_Points1[i].y * current_Points2[i].y;
		A.at<double>(2 * i + 1, 8) = -current_Points2[i].y;
	}

	// Use the solve equation of SVD to solve for the 
	// homogeneous equation Ah = 0
	cv::SVD::solveZ(A, Homography_Matrix);

	return Homography_Matrix;
}

int Object_Localization::Find_Number_Of_Inliers(std::vector<cv::Point3f>points1, 
	                                            std::vector<cv::Point3f>points2, 
												cv::Mat Homography_Matrix, 
												std::vector<uchar> *inliers, 
												double *Distance_Standard_Dev) {

	int Num_Of_Points = points1.size();
	int i, j, num_inlier;
	double Current_Distance = 0, Sum_Distance = 0, Mean_Distance = 0;
	cv::Mat H(cv::Size(3, 3), CV_64FC1);
	cv::Point2f tmp_pt;
	std::vector<uchar> inliers_temp;
	
	std::vector<double> Distance;
	cv::Mat x1(cv::Size(1, 3), CV_64FC1);
	cv::Mat x2(cv::Size(1, 3), CV_64FC1);
	cv::Mat pt;
	cv::Mat Inverse_H(cv::Size(3, 3), CV_64FC1);

	// Load the homography matrix data from a vector to a 3x3
	// matrix.
	for (i = 0; i < 3; i++) {
		
		for (j = 0; j < 3; j++) {
			H.at<double>(i, j) = Homography_Matrix.at<double>((i * 3) + j, 0);
		}

	}

	// Invert the Homography Matrix
	cv::invert(H, Inverse_H);

	Sum_Distance = 0;
	num_inlier = 0;

	// For each of the good matches that we obtained, use the 
	// Homography matrix to map the keypoints on the input image.
	// Find the distance between the warped keypoints and the 
	// keypoints on the template image. Similarly warp the 
	// keypoints on the template image using the inverse of the
	// homography matrix and find its distance from the keypoints
	// in the input image. Add both the distances and if its less
	// than DISTANCE_THRESHOLD we take that point as inlier. 
	
	for (i = 0; i< Num_Of_Points; i++){
		
		x1.at<double>(0, 0) = points1[i].x;
		x1.at<double>(1, 0) = points1[i].y;
		x1.at<double>(2, 0) = points1[i].z;

		x2.at<double>(0, 0) = points2[i].x;
		x2.at<double>(1, 0) = points2[i].y;
		x2.at<double>(2, 0) = points2[i].z;
		

		pt = H * x1;

		tmp_pt.x = pt.at<double>(0, 0) / pt.at<double>(2, 0);
		tmp_pt.y = pt.at<double>(1, 0) / pt.at<double>(2, 0);

		Current_Distance = (pow(tmp_pt.x - points2[i].x, 2.0) + 
			                pow(tmp_pt.y - points2[i].y, 2.0));
		
		pt = Inverse_H * x2;
		
		tmp_pt.x = pt.at<double>(0, 0) / pt.at<double>(2, 0);
		tmp_pt.y = pt.at<double>(1, 0) / pt.at<double>(2, 0);
		
		Current_Distance += (pow(tmp_pt.x - points1[i].x, 2.0) + 
			                 pow(tmp_pt.y - points1[i].y, 2.0));

		if (Current_Distance < DISTANCE_THRESHOLD) {
			
			// The current point is an inlier, load the data
			// 255 onto the inlers vector. This is used as an
			// inlier mask.
			num_inlier++;
			inliers->push_back(255);
			inliers_temp.push_back(255);
			Distance.push_back(Current_Distance);
			Sum_Distance += Current_Distance;

		}
		else {

			// Since this point is not an inlier load the data
			// 0 into the inliers vector.
			inliers->push_back(0);
			inliers_temp.push_back(0);
			Distance.push_back(Current_Distance);
		}
	}
	// Compute the standard deviation of the distance  mean_dist = sum_dist/(double)num_inlier; 
	
	for (i = 0; i < inliers_temp.size(); i++) {
		// If a particular point is an inlier, compute the standard deviation.
		if (inliers_temp[i] == 255) {

			(*Distance_Standard_Dev) += pow(Distance[i] - Mean_Distance, 2.0);

		}
			
	}
	
	(*Distance_Standard_Dev) /= (double)(num_inlier - 1);
	
	return num_inlier;

}

void Object_Localization::combinationUtil(std::vector < int > index_Array, 
	                                      int Size_Of_Index_Array, 
										  int Num_Points_Required, 
										  int index, std::vector < int > data, 
										  int current_Ele_Index, 
										  std::vector< std::vector < int > > *current_Index) {

	if (index == Num_Points_Required) {
		current_Index->push_back(data);
		return;
	}

	if (current_Ele_Index >= Size_Of_Index_Array) {

		return;

	}
	
	data[index] = index_Array[current_Ele_Index];
	combinationUtil(index_Array, Size_Of_Index_Array, Num_Points_Required, index + 1, data, current_Ele_Index + 1, current_Index);
		

	combinationUtil(index_Array, Size_Of_Index_Array, Num_Points_Required, index, data, current_Ele_Index + 1, current_Index);

}

std::vector< std::vector < int > > Object_Localization::create_Possible_Combinations(int Num_Matches) {

	std::vector< std::vector < int > > current_Index;
	std::vector < int > data;

	data.reserve(NUM_POINTS_REQUIRED);

	for (int i = 0; i < NUM_POINTS_REQUIRED; i++) {
		data.push_back(0);
	}

	std::vector < int > index_Array;

	for (int i = 0; i < Num_Matches; i++) {
		index_Array.push_back(i);
	}

	combinationUtil(index_Array, index_Array.size(), NUM_POINTS_REQUIRED, 0, data, 0, &current_Index);

	return current_Index;

}

void Object_Localization::Find_Homography() {

	std::vector<cv::DMatch> outMatches;
	
	for (int i = 0; i < this->Matching_Features.size(); i++) {

		bool is_Collinear = true;
		std::vector<int> current_Index;
		std::vector<cv::Point3f>current_Points1, current_Points2;
		int Num_Inlier = 0;
		double Distance_Standard_Dev = 0;
		double Best_Distance_Standard_Dev = 0;
		int Num_Inlier_Max = 0;
		int iterations = 0;
		int Num_Iterations = 1000;
		int Num_Matches = 0;
		double p = 0.99;
		int Num_Points_Required = 4;

		Num_Matches = this->Matching_Features[i].size();   // Gives you the total number of matches
		std::vector<cv::Point3f> points1, points2;         // Stores the matching keypoints

		this->create_3D_Points(&points1, &points2, i);

		std::vector<uchar> inliers;// (points1.size(), 0);
		std::vector<uchar> Best_Inliers;// (points1.size(), 0);
		
		cv::Mat Homography_Matrix;
		cv::Mat Best_Homography_Matrix;
		double e = 0;

		// Create a set of all possible combinations of the points
		//std::vector< std::vector < int > > current_Index;
		//current_Index = create_Possible_Combinations(Num_Matches);

		while (Num_Iterations > iterations) {

			is_Collinear = true;
			inliers.clear();
			
			while (is_Collinear) {

				current_Index.clear();
				current_Points1.clear();
				current_Points2.clear();
				is_Collinear = false;
				// To make sure we do not pick the same points for index, we iterate 
				// throught the loop to exclude the possibilities when we have the same
				// index more than once.
				for (int j = 0; j < Num_Points_Required; j++) {

					current_Index.push_back(rand() % Num_Matches);

					for (int k = 0; k < j; k++) {
						if (current_Index[j] == current_Index[k]) {

							is_Collinear = true;
							break;

						}
					}

					if (is_Collinear == true) {
						break;
					}

					// Load the Points into the Matrix to solve for the Homography Matrix
					current_Points1.push_back(points1[current_Index[j]]);
					current_Points2.push_back(points2[current_Index[j]]);

				}

				// Check if the points loaded are collinear.
				if (is_Collinear == false) {

					is_Collinear = check_Collinear_Points(current_Points1);

				}
				

			}

			// Compute the Homography Matrix using the randomly sampled 4 
			// keypoint correspondances.
			Homography_Matrix = computeH(current_Points1, 
				                         current_Points2);

			// Find the number of inliers for the Homography Matrix that we computed.
			Num_Inlier = Find_Number_Of_Inliers(points1, 
				                                points2, 
												Homography_Matrix, 
												&inliers, 
												&Distance_Standard_Dev);

			// If the Number of Inliers is more than the inliers for the previous best
			// Homography matrix, then replace the inlier mask, the Homography matrix 
			// and the number of inliers
			if (Num_Inlier > Num_Inlier_Max || 
				(Num_Inlier == Num_Inlier_Max && 
				Best_Distance_Standard_Dev < Distance_Standard_Dev)) {

				Best_Inliers.clear();
				Num_Inlier_Max = Num_Inlier;
				Homography_Matrix.copyTo(Best_Homography_Matrix);
				for (int vec_iter = 0; vec_iter < inliers.size(); vec_iter++) {

					Best_Inliers.push_back(inliers[vec_iter]);
				
				}
				
				inliers.clear();
				Best_Distance_Standard_Dev = Distance_Standard_Dev;

			}

			// Compute the error.
			e = 1 - (double)Num_Inlier / (double)Num_Matches;
			double Nr = log(1 - p);
			double Dr = log(1 - pow(1 - e, Num_Points_Required));
			// Recompute the Number of Iterations required to converge
			Num_Iterations = (int) (Nr / Dr);
			iterations++;
		}

		std::vector< Point2f > Left;
		std::vector< Point2f > Right;

		// Push the best homography matrix, best inlier mask and the 
		// number of inliers for the best homography matrix.
		this->Homography_Matrices.push_back(Best_Homography_Matrix);
		this->Inlier_Mask.push_back(Best_Inliers);
		this->Num_Inliers.push_back(Num_Inlier_Max);

		int cols = this->Input_Image.cols + this->Template_Images[i].cols;
		int rows = (this->Template_Images[i].rows > this->Input_Image.rows) ?
			this->Template_Images[i].rows : this->Input_Image.rows;

		// Create the composite image using the region of interest as
		// above. Draw the inlier keypoints on the input and the template
		// images.
		cv::Mat Composite_Images(cv::Size(cols, rows), CV_8UC3);
		cv::Mat Left_Image(Composite_Images, 
			               cv::Rect(0, 
						            0, 
									this->Input_Image.cols, 
									this->Input_Image.rows));
		cv::Mat Right_Image(Composite_Images, 
			                cv::Rect(this->Input_Image.cols, 
							         0, 
									 this->Template_Images[i].cols, 
									 this->Template_Images[i].rows));

		cv::Mat Input_Image_With_Key_Point, Template_Image_With_Key_Point;

		this->Input_Image.copyTo(Input_Image_With_Key_Point);
		this->Template_Images[i].copyTo(Template_Image_With_Key_Point);
		
		std::vector<cv::KeyPoint> Image_Inlier_Key_Point;
		std::vector<cv::KeyPoint> Template_Inlier_Key_Point;

		// Choose the inliers as keypoints to mark on the image
		for (int ii = 0; ii < this->Matching_Features[i].size(); ii++) {

			if (Best_Inliers[ii] == 255) {

				Image_Inlier_Key_Point.push_back(this->Image_Key_Points[this->Matching_Features[i][ii].queryIdx]);
				Template_Inlier_Key_Point.push_back(this->Template_Key_Points[i][this->Matching_Features[i][ii].trainIdx]);

			}

		}

		cv::drawKeypoints(this->Input_Image, 
			              Image_Inlier_Key_Point, 
						  Input_Image_With_Key_Point);

		cv::drawKeypoints(this->Template_Images[i], 
			              Template_Inlier_Key_Point, 
						  Template_Image_With_Key_Point);

		Input_Image_With_Key_Point.copyTo(Left_Image);
		Template_Image_With_Key_Point.copyTo(Right_Image);

		for (int j = 0; j < Image_Inlier_Key_Point.size(); j++) {

			cv::Point2f trainPoint;
			trainPoint.x = Template_Inlier_Key_Point[j].pt.x + this->Input_Image.cols;
			trainPoint.y = Template_Inlier_Key_Point[j].pt.y;

			cv::line(Composite_Images,
				Image_Inlier_Key_Point[j].pt,
				trainPoint,
				cv::Scalar(0, 0, 255),
				1,
				8);

		}

		this->Final_Image.push_back(Composite_Images);

	}
	
}

void Object_Localization::Save_Final_Images() {

	// Save the final image, where the object in the template is
	// localized in the input image.
	for (int i = 0; i < this->Final_Image.size(); i++) {

		string Save_Image_File_Name = "Final_Image_";

		Save_Image_File_Name = Save_Image_File_Name + std::to_string(i) + ".jpg";

		cv::imwrite(Save_Image_File_Name, this->Final_Image[i]);


	}
}

void Object_Localization::Save_Data_To_File() {

	FileStorage fs("myFile.yml", FileStorage::WRITE);
	string Identifier;
	std::vector<cv::KeyPoint> Image_Inlier_Key_Point;
	std::vector<cv::KeyPoint> Template_Inlier_Key_Point;
	cv::Mat H(cv::Size(3, 3), CV_64F);

	for (int i = 0; i < this->Homography_Matrices.size(); i++) {

		Image_Inlier_Key_Point.clear();
		Template_Inlier_Key_Point.clear();
		
		// Convert the vector to a 3x3 matrix
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < 3; k++) {
				H.at<double>(j, k) = this->Homography_Matrices[i].at<double>(j * 3 + k, 0) / 
					this->Homography_Matrices[i].at<double>(8, 0);
			}
		}

		// Write the matrix to yml file
		Identifier = "Homography_Matrix_" + to_string(i);

		fs << Identifier << H;

		// Load the inlier key points into vectors
		
		for (int ii = 0; ii < this->Matching_Features[i].size(); ii++) {

			if (this->Inlier_Mask[i][ii] == 255) {

				Image_Inlier_Key_Point.push_back(this->Image_Key_Points[this->Matching_Features[i][ii].queryIdx]);
				Template_Inlier_Key_Point.push_back(this->Template_Key_Points[i][this->Matching_Features[i][ii].trainIdx]);

			}

		}

		// Write the template key points to yml file
		Identifier = "Template_Image_Key_Points_" + to_string(i);
		fs << Identifier << Template_Inlier_Key_Point;

	}

	// Write the image key points to yml file
	Identifier = "Image_Key_Points";
	fs << Identifier << Image_Inlier_Key_Point;

	fs.release();

}