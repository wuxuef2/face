///****************************************************************************
//*
//* Copyright (c) 2015 by wuxuefeng, all rights reserved.
//*
//* Author:      	wuxuefeng
//*
//* This software is partly based on the following open source:
//*
//*		- OpenCV , AAMLibrary
//*
//****************************************************************************/
//
#ifndef AGEESTIMATION_H
#define AGEESTIMATION_H

#include <opencv2/ml/ml.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "AAM_Util.h"
#include "AAM_PDM.h"
#include "AAM_TDM.h"

//children
//const int AGE_GROUPS[][2] = {{0, 0}, {1, 1}, {2, 2}, {3, 5}, {6, 9}, {10, 18}};  //age group

class AgeEstimation
{
public:
	AgeEstimation();
    ~AgeEstimation();

    //build the shape and texture space seperately & get the mean mapping parameters of each age group
    void train(const std::vector<AAM_Shape> &AllShapes,
               const std::vector<IplImage*> &AllImages,
               const int g_samples[],
               //CvMat* AllTextures,
               double shape_percentage = 0.95,
               double texture_percentage = 0.95);

    //predict the new face in other age group
    float predict(const AAM_Shape& Shape, const IplImage& curImage);

    // Read data from stream
    void Read(std::ifstream& is);

private:
    AAM_PDM		__shape;		/*shape distribution model*/
    AAM_TDM		__texture;		/*texture distribution model*/
    AAM_PAW		__paw;			/*piecewise affine warp*/
    CvSVM 		SVM;

    int			__nShapeModes;
    int			__nTextureModes;
    CvMat*		__MeanS;
    CvMat*		__MeanT;
    AAM_Shape	__AAMRefShape;

    int		__nGSamples[AGE_AREA];	//the number of samples in each group
    CvMat*	__ShapeParamGroups;
    CvMat*	__TextureParamGroups;

    std::string SVMParam;
};

#endif
