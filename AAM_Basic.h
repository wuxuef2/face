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

#ifndef AAM_BASIC_H
#define AAM_BASIC_H

#include "AAM_Util.h"
#include "AAM_TDM.h"
#include "AAM_CAM.h"

/**
   The basic Active appearace model building and fitting method.
   Refer to: T.F. Cootes, G.J. Edwards and C.J. Taylor. "Active Appearance Models". ECCV 1998
*/
class AAM_Basic:public AAM
{
public:
    AAM_Basic();
    ~AAM_Basic();

    //build cootes's basis aam
    void Train(const std::vector<AAM_Shape>& AllShapes,
               const std::vector<IplImage*>& AllImages,
               double shape_percentage = 0.95,
               double texture_percentage = 0.95,
               double appearance_percentage = 0.98);

    //Fit the image using aam basic.
    virtual int Fit(const IplImage* image, AAM_Shape& Shape,
                    int max_iter = 30, bool showprocess = false);

    //draw the image according the searching result(0:point, 1:triangle, 2:appearance)
    virtual void Draw(IplImage* image, int type);

    virtual inline const AAM_Shape GetMeanShape()const
    {
        return __cam.__shape.GetAAMReferenceShape();
    }

    // Read data from stream
    virtual void Read(std::ifstream& is);

    // write data to stream
    virtual void Write(std::ofstream& os);

    //init search parameters
    void InitParams(const IplImage* image, const CvMat* s, CvMat* c);

private:
    //Calculates the pixel difference from a model instance and an image
    double EstResidual(const IplImage* image, const CvMat* c,
                       CvMat* est_s, CvMat* diff);

    //Draw image with different type
    void DrawPoint(IplImage* image);
    void DrawTriangle(IplImage* image);
    void DrawAppearance(IplImage* image);

    //Calculate combined appearance parameters
    void CalcCVectors(const std::vector<AAM_Shape>& AllShapes,
                      const std::vector<IplImage*>& AllImages,
                      CvMat* CParams);

    //Build displacement sets for C parameters
    CvMat* CalcCParamDisplacementVectors(const std::vector<double>& vStdDisp);

    //Build displacement sets for Pose parameters
    CvMat* CalcPoseDisplacementVectors(const std::vector<double> &vScaleDisp,
                                       const std::vector<double> &vRotDisp, const std::vector<double> &vXDisp,
                                       const std::vector<double> &vYDisp);

    //Build gradient matrices
    void CalcGradientMatrix(const CvMat* CParams,
                            const CvMat* vCDisps,
                            const CvMat* vPoseDisps,
                            const std::vector<AAM_Shape>& AllShapes,
                            const std::vector<IplImage*>& AllImages);

    //Build gradient matrices in terms of C parameters */
    void EstCParamGradientMatrix(CvMat* GParam,
                                 const CvMat* CParams,
                                 const std::vector<AAM_Shape>& AllShapes,
                                 const std::vector<IplImage*>& AllImages,
                                 const CvMat* vCDisps);

    //Build gradient matrices in terms of pose */
    void EstPoseGradientMatrix(CvMat* GPose,
                               const CvMat* CParams,
                               const std::vector<AAM_Shape>& AllShapes,
                               const std::vector<IplImage*>& AllImages,
                               const CvMat* vPoseDisps);

    //is the current shape within the image boundary?
    static bool IsShapeWithinImage(const CvMat* s, int w, int h);

private:
    AAM_CAM __cam;
    CvMat*	__Rc;
    CvMat*	__Rq;

private:
    //speed up for on-line alignment
    CvMat*	__current_c;	//current appearance parameters
    CvMat*  __update_c;		//update appearance parameters after certain iteration
    CvMat*	__delta_c;		//difference between successive c
    CvMat*	__p;			//current shape parameters
    CvMat*	__current_q;	//current pose parameters
    CvMat*	__update_q;		//update pose parameters after certain iteration
    CvMat*	__delta_q;		//defference between two successive q
    CvMat*	__current_lamda;//current pose parameters
    CvMat*	__current_s;	//current shape
    CvMat*	__t_m;			//model texture instance
    CvMat*	__t_s;			//warped texture at current shape instance
    CvMat*	__delta_t;		//differnce between __ts and __tm
};

#endif //
