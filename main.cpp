/****************************************************************************
*
* Copyright (c) 2008 by Yao Wei, all rights reserved.
*
* Author:      	Yao Wei
* Contact:     	njustyw@gmail.com
*
* This software is partly based on the following open source:
*
*		- OpenCV
*
****************************************************************************/

#include <cstdio>
#include <string>
#include <opencv/highgui.h>
#include <string>
#include <stdlib.h>
#include <iostream>

#include "AAM_IC.h"
#include "AAM_Basic.h"
#include "AAM_VJFaceDetect.h"
#include "FacePredict.h"
#include "AAM_Util.h"
#include "AgeEstimation.h"

using namespace std;

enum{ TYPE_AAM_BASIC = 0, TYPE_AAM_IC = 1};

string resultDir = "./ans/";
string trainDir = string("./trainingSets/");
string ansDir = "./ans/";

string Int2String(int i) {
	char c[10];

	if (!i) return string("0");

	int index = 0;
	int left = i;
	while (left) {
		left /= 10;
		index++;
	}

	c[index] = '\0';
	while(--index) {
		c[index] = i % 10 + '0';
		i /= 10;
	}
	c[index] = i + '0';

	return string(c);
}

int String2Int(string str) {
	int i = 0;
	for (int j = 0; j < str.length(); j++) {
		i = i * 10 + str[j] - '0';
	}
	return i;
}

int String2Int(char* str) {
	int i = 0;
	int j = 0;
	while (*(str + j)) {
		i = i * 10 + *(str + j) - '0';
		j++;
	}
	return i;
}

AAM_Shape makeShape(double points[], int sizes) {
    AAM_Shape Shape(points, sizes);
    return Shape;
}

extern "C"
{
	int age2Group(int age) {
		int group = -1;
		for (int i = 0; i < NGROUPS; i++) {
			if (age >= AGE_GROUPS[i][0] && age <= AGE_GROUPS[i][1]) {
				group = i;
				break;
			}
		}

		return group;
	}

	AAM_Shape getMyShape(char* originalImageFileName, int age = -1) {
		string groupFile;
		int group = age2Group(age);
		if (group == -1) {
			groupFile = string("mean");
		} else {
			groupFile = "Group" + Int2String(age);
		}

		//load image
		IplImage* originalImage = cvLoadImage(originalImageFileName, 1);
		if (originalImage == 0) {
			AgingException agingException(1);
			throw agingException;
		}

		IplImage *image = cvCreateImage(cvGetSize(originalImage), originalImage->depth, originalImage->nChannels);
		cvCopy(originalImage, image);
		AAM_Shape Shape;

		AAM *aam = NULL;
		int type;
		std::string aamFileName = resultDir + groupFile + ".aam_ic";//"mean" +".aam_ic";

		std::ifstream fs(aamFileName.c_str());
		if(fs == 0) {
			AgingException agingException(3);
			throw agingException;
		}
		fs >> type;

		//aam-basic
		if(type == 0)		aam = new AAM_Basic;
		else if(type == 1)  aam = new AAM_IC;

		//read model from file
		aam->Read(fs);
		fs.close();

		//intial face detector
		AAM_VJFaceDetect fjdetect;
		fjdetect.LoadCascade("haarcascade_frontalface_alt2.xml");

		//detect face for intialization
		Shape = fjdetect.Detect(image, aam->GetMeanShape());
		//printf("wuxuef2\n");

		//do image alignment
		aam->Fit(image, Shape, 30, false);  //if true, show process

		ofstream outfile;
		string wuxuefTmpResultDir = resultDir + "aam_result.txt";
		outfile.open(wuxuefTmpResultDir.c_str());
		Shape.Write(outfile);
		outfile.close();

		//resize the current image
		cvSetImageROI(originalImage, cvRect(Shape.MinX(), Shape.MinY(), Shape.GetWidth(), Shape.GetHeight()));
		IplImage *facialImage = cvCreateImage(cvGetSize(originalImage), originalImage->depth, originalImage->nChannels);
		cvCopy(originalImage, facialImage, NULL);
		cvResetImageROI(originalImage);

		CvSize stdsize;
		stdsize.width = stdwidth;
		stdsize.height = stdwidth / facialImage->width * facialImage->height;
		IplImage *stdImage = cvCreateImage(stdsize, originalImage->depth, originalImage->nChannels);
		cvResize(facialImage, stdImage, CV_INTER_LINEAR);

		//draw the shape
		CvSize ssize;
		ssize.width = 130;
		ssize.height = 130;
		IplImage *shapeImg = cvCreateImage(ssize, originalImage->depth, originalImage->nChannels);
		cvSet(shapeImg, CV_RGB(0,0,0));
		AAM_Shape temShape = Shape;
		double orgwid = temShape.MaxX() - temShape.MinX();
		double orghei = temShape.MaxY() - temShape.MinY();
		for (int i = 0; i < 68; i++) {
			temShape[i].x = (temShape[i].x - Shape.MinX()) * stdwidth / orgwid;
			temShape[i].y = (temShape[i].y - Shape.MinY()) * stdsize.height / orghei;
		}
		temShape.Sketch(shapeImg);
		//cvShowImage("shape", shapeImg);
		cvReleaseImage(&shapeImg);

		return Shape;
	}

    int fit(char* originalImageFileName, char* curAge, char* predictAge, char* pointsStr, int sizes) {
        int processState = 0;
        char* pch;
        double* points = new double[sizes];

        int index = 0;
        pch = strtok(pointsStr, " ");
        while (pch != NULL) {
            points[index++] = atof(pch);
            pch = strtok(NULL, " ");
        }

        try {
            IplImage* originalImage = cvLoadImage(originalImageFileName, 1);
            IplImage *image = cvCreateImage(cvGetSize(originalImage), originalImage->depth, originalImage->nChannels);

            AAM_Shape Shape = makeShape(points, sizes);

            //Facial Prediction
            FacePredict face_predict;
            std::string mfile = resultDir + "facial.predict_model";
            std::ifstream model(mfile.c_str());
            face_predict.Read(model);
            model.close();

            IplImage* newImage = face_predict.predict(Shape, *originalImage, atoi(curAge), atoi(predictAge), false);
            std::string newfile = std::string(originalImageFileName);
            newfile = newfile.insert(newfile.find_last_of('/') + 1, "result_");
            //newfile = newfile.insert(newfile.find_last_of('.'), std::string("_G" + string(predictAge)));
            cvSaveImage(newfile.c_str(), newImage);

            /*
            cvNamedWindow("PredictedFacialImage");
            cvShowImage("PredictedFacialImage", newImage);
            cvWaitKey(0);*/

            cvReleaseImage(&image);
        }
        catch (AgingException ex) {
            processState = ex.getStateCode();
        }

        return processState;
    }

    int fitWithArray(char* originalImageFileName, char* curAge, char* predictAge, double* points, int sizes) {
        int processState = 0;

        try {
            IplImage* originalImage = cvLoadImage(originalImageFileName, 1);
            IplImage *image = cvCreateImage(cvGetSize(originalImage), originalImage->depth, originalImage->nChannels);

            AAM_Shape Shape = makeShape(points, sizes);

            //Facial Prediction
            FacePredict face_predict;
            std::string mfile = resultDir + "facial.predict_model";
            std::ifstream model(mfile.c_str());
            face_predict.Read(model);
            model.close();

            IplImage* newImage = face_predict.predict(Shape, *originalImage, atoi(curAge), atoi(predictAge), false);
            std::string newfile = std::string(originalImageFileName);
            newfile = newfile.insert(newfile.find_last_of('/') + 1, "result_");
            //newfile = newfile.insert(newfile.find_last_of('.'), std::string("_G" + string(predictAge)));
            cvSaveImage(newfile.c_str(), newImage);


            cvNamedWindow("PredictedFacialImage");
            cvShowImage("PredictedFacialImage", newImage);
            cvWaitKey(0);

            cvReleaseImage(&image);
        }
        catch (AgingException ex) {
            processState = ex.getStateCode();
        }

        return processState;
    }

    int ageEsti(char* image, AAM_Shape& curShape) {
    	AgeEstimation aes;
    	std::string mfile = resultDir + "facial.predict_model";
    	std::ifstream model(mfile.c_str());
    	aes.Read(model);
    	model.close();

    	int guess = -1;
    	int newGuess = -1;
    	IplImage* originalImage = cvLoadImage(image, 1);

    	do {
    		guess = newGuess;
    		curShape = getMyShape(image, guess);
    		newGuess = (int)aes.predict(curShape, *originalImage);
    		cout << newGuess << endl;
    	} while (guess != newGuess);

    	return guess;
    }

    double* getShape(char* originalImageFileName) {
        AAM_Shape Shape;
        double* points;
        int processState = 0;

        try {
        	int age = ageEsti(originalImageFileName, Shape);
            int sizes = Shape.NPoints() * 2;
            points = new double[sizes + 1];
            int index = 0;
            for (int i = 0; i < sizes; i += 2) {
                index = i / 2;
                points[i] = Shape[index].x;
                points[i + 1] = Shape[index].y;
            }
            points[sizes] = age;
        }
        catch (AgingException ex) {
            processState = ex.getStateCode();
            points = new double[68 * 2 + 1];
            points[68 * 2] = 0 - processState;
        }

        return points;
    }

}


int train()
{
	//==================== Read in the images and points data====================
	std::vector<std::string> trainPaths = ScanNSortAllDirectorys(trainDir);
	std::vector<std::string> m_vimgFiles, m_vptsFiles;
	std::vector<std::string> t_imgFiles, t_ptsFiles;
	int nG_Samples[AGE_AREA] = {0};

	for (int i = 0; i < trainPaths.size(); i++) {
		t_imgFiles = ScanNSortDirectory(trainPaths[i], "jpg");
		t_ptsFiles = ScanNSortDirectory(trainPaths[i], "pts");

		if(t_imgFiles.size() != t_ptsFiles.size())
		{
			fprintf(stderr, "ERROR(%s, %d): #Shapes != #Images\n",
				__FILE__, __LINE__);
			exit(0);
		}
		//int age_group = getAgeGroup(trainPaths[i]);
		int age_group = i;
		nG_Samples[age_group] = t_imgFiles.size();
		m_vimgFiles.insert(m_vimgFiles.end(), t_imgFiles.begin(), t_imgFiles.end());
		m_vptsFiles.insert(m_vptsFiles.end(), t_ptsFiles.begin(), t_ptsFiles.end());
	}

	std::vector<AAM_Shape> AllShapes;
	AAM_Shape referenceShape;
	std::vector<IplImage*> AllImages;

	for(int i = 0; i < m_vimgFiles.size(); i++)
	{
		AllImages.push_back(cvLoadImage(m_vimgFiles[i].c_str(), 1));
		referenceShape.ReadPTS(m_vptsFiles[i]);
		AllShapes.push_back(referenceShape);
	}

	//============================== train AAM ===============================
	int group_size = 0;
	std::vector<AAM_Shape> GroupShapes;
	std::vector<AAM_Shape>::iterator itr_shape = AllShapes.begin();
	std::vector<IplImage*> GroupImages;
	std::vector<IplImage*>::iterator itr_image = AllImages.begin();

	AAM_IC aam_ic; aam_ic.Train(AllShapes, AllImages);
	std::string aamfile = ansDir + "mean" +".aam_ic";
	std::ofstream fs(aamfile.c_str());
	fs << TYPE_AAM_IC << std::endl;
	aam_ic.Write(fs);
	fs.close();

	for (int i = 0; i < NGROUPS; i++) {

		//get the samples in a designated age group
		group_size = 0;
		for (int j = AGE_GROUPS[i][0]; j <= AGE_GROUPS[i][1]; j++)
			group_size += nG_Samples[j];

		for (int j = 0; j < group_size; j++, itr_shape++, itr_image++) {
			GroupShapes.push_back(*itr_shape);
			GroupImages.push_back(*itr_image);
		}


		AAM_IC aam_ic; aam_ic.Train(GroupShapes, GroupImages);

		std::string aamfile = ansDir + "Group" + Int2String(i) +".aam_ic";
		std::ofstream fs(aamfile.c_str());
		fs << TYPE_AAM_IC << std::endl;
		aam_ic.Write(fs);
		fs.close();

		GroupShapes.clear();
		GroupImages.clear();
	}

	//==========================train Face Predict model===================
	FacePredict face_predict;
	face_predict.Train(AllShapes, AllImages, nG_Samples, /*AllTextures,*/ 0.95, 0.95);

	string tmpFilePath = ansDir + string("facial.predict_model");
	std::ofstream file(tmpFilePath.c_str());

	face_predict.Write(file);
	file.close();

	//=======================Age Estimation==================================
	AgeEstimation aes;
	aes.train(AllShapes, AllImages, nG_Samples);

//	AAM_Shape curShape = getMyShape("input.jpg");
//	IplImage* originalImage = cvLoadImage("input.jpg", 1);
//	printf("%lf", aes.predict(curShape, *originalImage));

	for(int j = 0; j < AllImages.size(); j++) {
		cvReleaseImage(&AllImages[j]);
		AllShapes[j].clear();
	}

	return 0;
}


//int main() {
//    char* originalImageFileName = "input.jpg";
//	char* curAge = "1";
//	char* predictAge = "3";
//	char* ResultsSavePath = "ResultsSavePath";
//
//    try {
//        double* points = getShape(originalImageFileName, curAge);
//        int sizes = 68 * 2;
//
//        int state = fitWithArray(originalImageFileName, curAge, predictAge, points, sizes);
//        cout << state << endl;
//    }
//    catch (AgingException ex) {
//        cout << ex.getStateCode() << endl;
//    }

//	double* points = getShape("input.jpg");
//
//	for (int i = 0; i < 68 * 2; i++) {
//		cout << points[i];
//	}
//
//	cout << endl;
//	cout << points[68 * 2] << endl;
//    return 0;
//}

