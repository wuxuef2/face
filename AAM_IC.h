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


#ifndef AAM_IC_H
#define AAM_IC_H

//#include <windows.h>
#include "AAM_Util.h"
#include "AAM_Shape.h"
#include "AAM_PDM.h"
#include "AAM_TDM.h"
#include "AAM_PAW.h"


/**
   Active appearace model using project-out inverse compositional fitting method.
   Refer to: I. Matthews and S. Baker. "Active Appearance Models Revisited". 2004
*/
class AAM_IC  :public AAM
{
public:
    AAM_IC();
    ~AAM_IC();

    //build aam inverse compositional model
    void Train(const std::vector<AAM_Shape>& AllShapes,
               const std::vector<IplImage*>& AllImages,
               double shape_percentage = 0.95, double texture_percentage = 0.95);

    //Fit the image using Inverse Compositional.
    virtual int Fit(const IplImage* image, AAM_Shape& Shape,
                    int max_iter = 30, bool showprocess = false);

    //Set all search parameters to zero
    void SetAllParamsZero();

    //draw the image according the searching result(0:point, 1:triangle, 2:appearance)
    virtual void Draw(IplImage* image, int type);

    // Read data from stream
    virtual void Read(std::ifstream& is);

    // write data to stream
    virtual void Write(std::ofstream& os);

    //Get Mean Shape of IC model
    inline const AAM_Shape GetMeanShape()const
    {
        return __sMean;
    }

private:
    //Draw image with different type
    void DrawPoint(IplImage* image);
    void DrawTriangle(IplImage* image);
    void DrawAppearance(IplImage* image);

    //calclulate the texture parameters project to linear subspace span(A)
    void CalcAppearanceVariation(const CvMat* error_t, CvMat* lamda);


    //Evaluate the Jacobians dN_dq and dW_dp of piecewise affine warp at(x;0)
    void CalcWarpJacobian(CvMat* Jx, CvMat* Jy);

    //Calculate index of gradients for every point in texture.
    //If point is outside texture, set to -1.
    CvMat* CalcGradIdx();

    //Calculate the gradient of texture template A0.
    void CalcTexGrad(const CvMat* texture, CvMat* dTx, CvMat* dTy);

    //Calculating the modified steepest descent image.
    void CalcModifiedSD(CvMat* SD, const CvMat* dTx, const CvMat* dTy,
                        const CvMat* Jx, const CvMat* Jy);

    //Inverse compose current warp with shape parameter update.
    //Update warp N.W(x;p,q)<-- N.W(x;p,q) . N.W(x;delta_p,delta_q)^-1.
    void InverseCompose(const CvMat* dpq, const CvMat* s, CvMat* NewS);

    //Compute the Hessian matrix using modified steepest descent image.
    void CalcHessian(CvMat* H, const CvMat* SD);

private:

    //these variables are used for train PAW
    CvMat*			__Points;
    CvMemStorage*	__Storage;

private:
    //is the current shape within the image boundary?
    static bool IsShapeWithinImage(const CvMat* s, int w, int h);

    AAM_PDM		__shape;		/*shape distribution model*/
    AAM_TDM		__texture;		/*shape distribution model*/
    AAM_PAW		__paw;			/*piecewise affine warp*/
    AAM_Shape	__sMean;		/*mean shape of model*/
    AAM_Shape	__sStar1, __sStar2, __sStar3, __sStar4;/*global shape transform bases*/
    CvMat*		__G;			/*Update matrix*/
    /*product of inverse Hessian with steepest descent image*/

private:
    //pre-allocated stuff for online alignment
    CvMat*		__update_s0;	/*shape change at the base mesh */
    CvMat*		__inv_pq;		/*inverse parameters at the base mesh*/

    CvMat*		__warp_t;		/*warp image to base mesh*/
    CvMat*		__error_t;		/*error between warp image and template image A0*/
    CvMat*		__search_pq;	/*search parameters */
    CvMat*		__delta_pq;		/*parameters change to be updated*/
    CvMat*		__current_s;		/*current search shape*/
    CvMat*		__update_s;		/*shape after composing the warp*/
    CvMat*		__delta_s;		/*shape change between two successive iteration*/
    CvMat*		__lamda;		/*appearance parameters*/
};

#endif
