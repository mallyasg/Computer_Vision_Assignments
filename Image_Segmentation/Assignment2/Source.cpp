# include "Image_Segmentation.h"
# define NORMAL

int main(int argc, char **argv) {

	// Load the images
	Image_Segmentation Image1(argv[1], atoi(argv[4]), atoi(argv[5]), atoi(argv[6]));
	Image_Segmentation Image2(argv[2], atoi(argv[4]), atoi(argv[5]), atoi(argv[6]));
	Image_Segmentation Image3(argv[3], atoi(argv[4]), atoi(argv[5]), atoi(argv[6]));

	Image_Segmentation Image1_Watershed(argv[1], atoi(argv[4]), atoi(argv[5]), atoi(argv[6]));
	Image_Segmentation Image2_Watershed(argv[2], atoi(argv[4]), atoi(argv[5]), atoi(argv[6]));
	Image_Segmentation Image3_Watershed(argv[3], atoi(argv[4]), atoi(argv[5]), atoi(argv[6]));


	Image1.Mean_Shift_Segmentation(1);
	Image2.Mean_Shift_Segmentation(1);
	Image3.Mean_Shift_Segmentation(1);

	// Save the Mean Shift Segmented image
	string File_Name1 = "image1_meanshift_" + std::to_string(atoi(argv[4])) + "_" + std::to_string(atoi(argv[5])) + "_" + std::to_string(atoi(argv[6])) + ".jpg";
	string File_Name2 = "image2_meanshift_" + std::to_string(atoi(argv[4])) + "_" + std::to_string(atoi(argv[5])) + "_" + std::to_string(atoi(argv[6])) + ".jpg";
	string File_Name3 = "image3_meanshift_" + std::to_string(atoi(argv[4])) + "_" + std::to_string(atoi(argv[5])) + "_" + std::to_string(atoi(argv[6])) + ".jpg";
	
	/*Image1.Display_Segmented_Image("Mean Shift Segmented Image1");
	Image2.Display_Segmented_Image("Mean Shift Segmented Image2");
	Image3.Display_Segmented_Image("Mean Shift Segmented Image3");*/
	
	Image1.Save_Segmented_Image(File_Name1);
	Image2.Save_Segmented_Image(File_Name2);
	Image3.Save_Segmented_Image(File_Name3);

	// Perform Watershed segmentation
	//Image1_Watershed.Watershed_Segmentation();
	//Image2_Watershed.Watershed_Segmentation();
	//Image3_Watershed.Watershed_Segmentation();

	//File_Name1 = "image1_watershed_Canny_" + std::to_string(75) + "Contour_Length_" + std::to_string(25) + ".jpg";
	//File_Name2 = "image2_watershed_Canny_" + std::to_string(75) + "Contour_Length_" + std::to_string(25) + ".jpg";
	//File_Name3 = "image3_watershed_Canny_" + std::to_string(75) + "Contour_Length_" + std::to_string(25) + ".jpg";
	
	//Image1_Watershed.Save_Segmented_Image(File_Name1);
	//Image2_Watershed.Save_Segmented_Image(File_Name2);
	//Image3_Watershed.Save_Segmented_Image(File_Name3);
	
	return 1;
}