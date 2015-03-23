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

#ifndef AAM_CAM_H
#define AAM_CAM_H

#include "AAM_TDM.h"
#include "AAM_PDM.h"
#ifdef _WIN32
#include <windows.h>
#include <stdio.h>
#include <tchar.h>
#else
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#endif

class AAM_Basic;

//combined appearance model
class AAM_CAM
{
    friend class AAM_Basic;
public:
    AAM_CAM();
    ~AAM_CAM();

    //build combined appearance model
    void Train(const std::vector<AAM_Shape>& AllShapes,
               const std::vector<IplImage*>& AllImages,
               double shape_percentage = 0.95,
               double texture_percentage = 0.95,
               double appearance_percentage = 0.95);

    // Get dimension of combined appearance vector
    inline const int nParameters()const
    {
        return __AppearanceEigenVectors->cols;
    }

    // Get number of modes of combined appearance variation
    inline const int nModes()const
    {
        return __AppearanceEigenVectors->rows;
    }

    // Get variance of i'th mode of combined appearance variation
    inline double Var(int i)const
    {
        return cvmGet(__AppearanceEigenValues, 0, i);
    }

    //Get mean combined appearance
    inline const CvMat* GetMean()const
    {
        return __MeanAppearance;
    }

    //Get combined appearance eigen-vectors of PCA (appearance modes)
    inline const CvMat* GetBases()const
    {
        return __AppearanceEigenVectors;
    }

    // Show Model Variation according to various of parameters
    void ShowVariation();
    //function used in ShowVariation
    friend void ontrackcam(int pos);

    //draw the image according the searching result
    void DrawAppearance(IplImage* image, const AAM_Shape& Shape, CvMat* Texture);

    //calculate shape according to appearance parameters
    inline void CalcShape(CvMat* s, const CvMat* c, const CvMat* pose)
    {
        CalcLocalShape(s, c);
        CalcGlobalShape(s, pose);
    }
    void CalcLocalShape(CvMat* s, const CvMat* c);
    void CalcGlobalShape(CvMat* s, const CvMat* pose);

    //calculate texture according to appearance parameters
    void CalcTexture(CvMat* t, const CvMat* c);

    //Calculate combined appearance parameters from shape and texture params.
    void CalcParams(CvMat* c, const CvMat* bs, const CvMat* bg);

    //Limit appearance parameters.
    void Clamp(CvMat* c, double s_d = 3.0);

    // Read data from stream
    void Read(std::ifstream& is);

    // write data to stream
    void Write(std::ofstream& os);

    //do PCA of appearance datas
    void DoPCA(const CvMat* AllAppearances, double percentage);

    //convert shape and texture instance to appearance parameters
    void ShapeTexture2Combined(const CvMat* Shape, const CvMat* Texture,
                               CvMat* Appearance);

private:
    AAM_PDM		__shape;		/*shape distribution model*/
    AAM_TDM		__texture;		/*texture distribution model*/
    AAM_PAW		__paw;			/*piecewise affine warp*/
    double      __WeightsS2T;   /*ratio between shape and texture model*/

    CvMat* __MeanAppearance;
    CvMat* __AppearanceEigenValues;
    CvMat* __AppearanceEigenVectors;

    CvMat* __Qs;
    CvMat* __Qg;
    CvMat* __MeanS;
    CvMat* __MeanG;

private:
    //these cached variables are used for speed up
    CvMat*			__Points;
    CvMemStorage*	__Storage;
    CvMat*			__pq;
};

#endif // !AAM_CAM_H
