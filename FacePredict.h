///****************************************************************************
//*
//* Copyright (c) 2012 by Li Ying, all rights reserved.
//*
//* Author:      	Li Ying
//* Contact:     	liyingchocolate@gmail.com
//*
//* This software is partly based on the following open source:
//*
//*		- OpenCV , AAMLibrary
//*
//****************************************************************************/
//
#ifndef FACEPREDICT_H
#define FACEPREDICT_H

#include "AAM_Util.h"
#include "AAM_PDM.h"
#include "AAM_TDM.h"

//FGNET
//#define AGE_AREA 9
//#define NGROUPS 9
//const int AGE_GROUPS[][2] = {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}, {7, 7}, {8, 8}};

class FacePredict
{
public:
    FacePredict();
    ~FacePredict();

    //build the shape and texture space seperately & get the mean mapping parameters of each age group
    void Train(const std::vector<AAM_Shape> &AllShapes,
               const std::vector<IplImage*> &AllImages,
               const int g_samples[],
               //CvMat* AllTextures,
               double shape_percentage = 0.95,
               double texture_percentage = 0.95);

    //predict the new face in other age group
    IplImage* predict(const AAM_Shape& Shape, const IplImage& curImage,
                      int curAgeG, int newAgeG, bool save = false);

    //predict the new face in other age group with iamges of father and mother
    IplImage* predict(const AAM_Shape& Shape, const IplImage& curImage,
                      const AAM_Shape& ShapeF, const IplImage& ImageF, double RatioF,
                      const AAM_Shape& ShapeM, const IplImage& ImageM, double RatioM,
                      int curAgeG, int newAgeG, bool save = false);

    //combine the shape and texture to predict the fical image
    void FaceSynthesis(AAM_Shape &shape, CvMat* texture, IplImage* newImage);

    //calculate the mean parameters of all shapes and textures in each age group
    void CalcClassicParams(const std::vector<AAM_Shape> &AllShapes, const CvMat* AllTextures);

    //calculate the mean parameters of all shapes in a certain age group
    void CalcMeanShapeParams(const std::vector<AAM_Shape> &GroupShapes, int group);

    //calculate the mean parameters of all textures in a certain age group
    void CalcMeanTextureParams(const CvMat* GroupTextures, int group);

    //caculate predicted parameters of shape
    void CalcNewShapeParams(CvMat* curParam, CvMat* newParam, int curAgeG, int newAgeG);

    //caculate predicted parameters of texture
    void CalcNewTextureParams(CvMat* curParam, CvMat* newParam, int curAgeG, int newAgeG);

    //caculate parameters by ratios of father and mother
    void CalcParamsByRatio(CvMat* curParam, CvMat* ParamF, double RatioF, CvMat* ParamM, double RatioM, CvMat* newParam);

    //assign the certain age to the corresponding age groups
    int AgeGroup(int age);

    // Read data from stream
    void Read(std::ifstream& is);

    // write data to stream
    void Write(std::ofstream& os);

    //get the number of samples in each group
    inline const int* NG_Samples() const
    {
        return __nGSamples;
    }

private:
    AAM_PDM		__shape;		/*shape distribution model*/
    AAM_TDM		__texture;		/*texture distribution model*/
    AAM_PAW		__paw;			/*piecewise affine warp*/

    int			__nShapeModes;
    int			__nTextureModes;
    CvMat*		__MeanS;
    CvMat*		__MeanT;
    AAM_Shape	__AAMRefShape;

    int		__nGSamples[AGE_AREA];	//the number of samples in each group
    CvMat*	__ShapeParamGroups;
    CvMat*	__TextureParamGroups;
};

#endif
