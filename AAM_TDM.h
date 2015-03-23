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

#ifndef AAM_TDM_H
#define AAM_TDM_H

#include "AAM_Util.h"
#include "AAM_Shape.h"
#include "AAM_PAW.h"

class AAM_CAM;

//Texture distribution model.
class AAM_TDM
{
    friend class AAM_CAM;
public:
    AAM_TDM();
    ~AAM_TDM();

    //build texture distribution model
    void Train(const std::vector<AAM_Shape> &AllShapes,
               const AAM_PAW& m_warp,
               const std::vector<IplImage*> &AllImages,
               double percentage = 0.95,
               bool registration  = true);

    //additonal, overloading function, to get AllTextures
    void Train(const std::vector<AAM_Shape> &AllShapes,
               const AAM_PAW& m_warp,
               const std::vector<IplImage*> &AllImages,
               CvMat* AllTextures,
               double percentage = 0.95,
               bool registration  = true);

    // Read data from stream
    void Read(std::ifstream& is);

    // write data to stream
    void Write(std::ofstream& os);

    // do pca of texture data
    void DoPCA(const CvMat* AllTextures, double percentage);

    //calculate texture according to parameters lamda
    void CalcTexture(const CvMat* lamda, CvMat* texture);

    //calculate parameters lamda according to texture sample
    void CalcParams(const CvMat* texture, CvMat* lamda);

    //Limit texture parameters.
    void Clamp(CvMat* lamda, double s_d = 3.0);

    // align texture to lossen the affect of light variations
    static void AlignTextures(CvMat* AllTextures);

    //calculate mean texture
    static void CalcMeanTexture(const CvMat* AllTextures, CvMat* meanTexture);

    //normailize texture to mean texture
    static void AlignTextureToRef(const CvMat* refTextrure, CvMat* Texture);

    //normalize texture make sure: sum of element is o and variance is 1
    static void ZeroMeanUnitLength(CvMat* Texture);

    //Get number of color-pixels in texture model
    inline const int nPixels()const
    {
        return __MeanTexture->cols;
    }

    //Get number of modes of texture variation
    inline const int nModes()const
    {
        return __TextureEigenVectors->rows;
    }

    //Get mean texture
    inline const CvMat* GetMean()const
    {
        return __MeanTexture;
    }

    //Get texture eigen-vectors of PCA (modes modes)
    inline const CvMat* GetBases()const
    {
        return __TextureEigenVectors;
    }

    inline const double Var(int i)const
    {
        return cvmGet(__TextureEigenValues,0,i);
    }

private:
    // Save the raw texture
    void SaveSeriesTemplate(const CvMat* AllTextures, const AAM_PAW& m_warp);

    CvMat*  __MeanTexture;
    CvMat*  __TextureEigenVectors;
    CvMat*  __TextureEigenValues;
};

#endif //
