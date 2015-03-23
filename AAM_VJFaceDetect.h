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


#ifndef AAM_VJFACEDETECT_H
#define AAM_VJFACEDETECT_H

#include "AAM_Shape.h"

// Viola and Jones's AdaBoost Haar-like Face Detector
class AAM_VJFaceDetect
{
public:
    AAM_VJFaceDetect();
    ~AAM_VJFaceDetect();

    //detect most central face in  image
    AAM_Shape Detect(const IplImage* image, const AAM_Shape& MeanShape);

    //load adaboost cascade file for detect face
    void LoadCascade(const char* cascade_name);

private:
    /**************************************************************************/
    /* The following two functions are borrowed from Stephen Milborrow's stasm*/
    /**************************************************************************/
    // Make the ones face box smaller and move it down a bit.
    void AdjustViolaJonesShape (AAM_Shape &Shape);

    // align MeanShape to the Viola Jones global detector face box
    void AlignToViolaJones(AAM_Shape &StartShape, const AAM_Shape &DetShape,
                           const AAM_Shape& MeanShape);

    CvMemStorage* __storage;
    CvHaarClassifierCascade* __cascade;

};

#endif // AAM_VJFACEDETECT_H
