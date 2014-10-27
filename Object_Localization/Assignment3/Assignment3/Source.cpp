# include "Object_Localization.h"

int main(int argc, char **argv) {

	if (argc < 2) {
		cout << "----------------------Invalid Usage!!! -----------------------\n";
		cout << "Please use the following command\n";
		cout << "Object_Localization.exe <Input_Image> <Template_Image1> ";
		cout << "[<Templage_Image2>....\n";
	}

	Object_Localization Image(&argv[1], argc - 1);
	
	Image.Detect_Image_Key_Points();
	Image.Detect_Image_Features();
	Image.Detect_Template_Image_Key_Points();
	Image.Detect_Template_Image_Features();
	Image.Match_Features(0.75);
	Image.Draw_Image_Matches();
	//Image.Display_Image_Matches();
	Image.Find_Homography();
	Image.Save_Final_Images();
	Image.Save_Data_To_File();
	return 0;

}