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

#ifndef AAM_PDM_H
#define AAM_PDM_H

#include "AAM_Util.h"
#include "AAM_Shape.h"

const double stdwidth = 128;

class AAM_CAM;

//2D point distribution model
class AAM_PDM
{
    friend class AAM_CAM;
public:
    AAM_PDM();
    ~AAM_PDM();

    //build shape distribution model
    void Train(const std::vector<AAM_Shape> &AllShapes, double percentage = 0.95);

    //additional
    //build shape distribution model and get the alined shapes
    void Train(const std::vector<AAM_Shape> &AllShapes, std::vector<AAM_Shape> &AlignedShapes, double percentage = 0.95);

    // Read data from stream
    void Read(std::ifstream& is);

    // write data to stream
    void Write(std::ofstream& os);

    // align shapes using procrustes analysis
    static void AlignShapes(std::vector<AAM_Shape> &AllShapes);

    //calculate mean shape of all shapes
    static void CalcMeanShape(AAM_Shape &MeanShape,
                              const std::vector<AAM_Shape> &AllShapes);

    //do PCA of shape data
    void DoPCA(const CvMat* AllShapes, double percentage);

    //calculate shape according to parameters p and q
    void CalcLocalShape(const CvMat* p, CvMat* s);
    void CalcGlobalShape(const CvMat* q, CvMat* s);
    void CalcShape(const CvMat* p, const CvMat* q, CvMat* s);
    void CalcShape(const CvMat* pq, CvMat* s);
    void CalcShape(const CvMat* pq, AAM_Shape& shape);

    //calculate parameters p and q according to shape
    void CalcParams(const CvMat* s, CvMat* p, CvMat* q);
    void CalcParams(const CvMat* s, CvMat* pq);
    void CalcParams(const AAM_Shape& shape, CvMat* pq);

    //Limit shape parameters.
    void Clamp(CvMat* p, double s_d = 3.0);

    //Get number of points in shape model
    inline const int nPoints()const
    {
        return __MeanShape->cols / 2;
    }

    //Get number of modes of shape variation
    inline const int nModes()const
    {
        return __ShapesEigenVectors->rows;
    }

    //Get mean shape
    inline const CvMat* GetMean()const
    {
        return __MeanShape;
    }

    //Get shape eigen-vectors of PCA (shape modes)
    inline const CvMat* GetBases()const
    {
        return __ShapesEigenVectors;
    }

    inline const double Var(int i)const
    {
        return cvmGet(__ShapesEigenValues,0,i);
    }

    //Get AAM reference shape (Maybe NOT Central)
    inline const AAM_Shape GetAAMReferenceShape()const
    {
        return __AAMRefShape;
    }

private:
    CvMat*		__MeanShape;
    CvMat*		__ShapesEigenVectors;
    CvMat*		__ShapesEigenValues;
    AAM_Shape	__AAMRefShape;

private:
    CvMat*		__matshape;
    AAM_Shape   __x;
    AAM_Shape   __y;
};

#endif //
