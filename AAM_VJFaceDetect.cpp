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

#include "AAM_VJFaceDetect.h"


AAM_VJFaceDetect::AAM_VJFaceDetect()
{
    __cascade = NULL;
    __storage = NULL;
}

AAM_VJFaceDetect::~AAM_VJFaceDetect()
{
    cvReleaseMemStorage(&__storage);
    cvReleaseHaarClassifierCascade(&__cascade);
}

void AAM_VJFaceDetect::LoadCascade(const char* cascade_name)
{
    __cascade = (CvHaarClassifierCascade*)cvLoad(cascade_name, 0, 0, 0);
    if(__cascade == 0)
    {
        printf("ERROR(%s, %d): Can't load cascade file %s\n",
               __FILE__, __LINE__, cascade_name);
        exit(0);
    }
    __storage = cvCreateMemStorage(0);

}
/*************************************************************************************************************/
AAM_Shape AAM_VJFaceDetect::Detect(const IplImage* image, const AAM_Shape& MeanShape)
{
    IplImage* pWork = cvCreateImage
                      (cvSize(image->width/2, image->height/2), image->depth, image->nChannels);

    cvPyrDown(image, pWork, CV_GAUSSIAN_5x5 );

    CvSeq* pFaces = cvHaarDetectObjects(pWork, __cascade, __storage,
                                        1.1, 3, CV_HAAR_DO_CANNY_PRUNING );

    cvReleaseImage(&pWork);

    if(0 == pFaces->total)//can't find a face
    {
        //fprintf(stderr, "ERROR(%s, %d): Can NOT find any face!\n", __FILE__, __LINE__);
        //exit(0);
        AgingException agingException(2);
        throw agingException;
    }

    int iSelectedFace = 0;
    // get most central face
    double MaxOffset = 1e307;
    // max abs dist from center of face to center of image
    for (int iFace = 0; iFace < pFaces->total; iFace++)
    {
        CvRect* r = (CvRect*)cvGetSeqElem(pFaces, iFace);
        double Offset = fabs(r->x*2.0 + r->width - image->width/2.0);
        if (Offset < MaxOffset)
        {
            MaxOffset = Offset;
            iSelectedFace = iFace;
        }
    }

    // Explained by YAO Wei, 2008-1-28.
    // Write the global detector shape into Shape. We must convert the Viola Jones
    // shape coordinates to our internal shape coordinates.
    CvRect* r = (CvRect*)cvGetSeqElem(pFaces, iSelectedFace);

    int scale = 2;
    CvPoint pt1, pt2;
    pt1.x = r->x*scale;
    pt2.x = (r->x+r->width)*scale;
    pt1.y = r->y*scale;
    pt2.y = (r->y+r->height)*scale;

    //show the face detection
    /*
    IplImage*img = cvCreateImage(cvSize(image->width,image->height),IPL_DEPTH_8U,3);
    cvCopyImage(image, img);
    cvRectangle(img, pt1, pt2, CV_RGB(0,255,0), 2, 8, 0);
    cvNamedWindow("face detection", CV_WINDOW_AUTOSIZE);
    cvShowImage("face detection", img);
    cvReleaseImage(&img);
    cvWaitKey(0);*/

    AAM_Shape DetShape;
    DetShape.resize(2);
    DetShape[0].x = r->x*2.0;
    DetShape[0].y = r->y*2.0;
    DetShape[1].x = DetShape[0].x + 2.0*r->width;
    DetShape[1].y = DetShape[0].y + 2.0*r->height;

    AAM_Shape StartShape;
//	AdjustViolaJonesShape(DetShape);
    AlignToViolaJones(StartShape, DetShape, MeanShape);

    return StartShape;
}

void AAM_VJFaceDetect::AdjustViolaJonesShape(AAM_Shape &Shape)
{
    // following are for aligning base shape to Viola Jones detector shape
    static const double CONF_VjHeightShift = 0.15;  // shift height down by 15%
    static const double CONF_VjShrink      = 0.85;   // shrink size of VJ box by 20%

    double xMin = Shape[0].x;
    double yMin = Shape[0].y;
    double xMax = Shape[1].x;
    double yMax = Shape[1].y;

    double NewHeight = CONF_VjShrink * (yMax - yMin);
    double yMean = (yMin + yMax) / 2;

    //yMean += CONF_VjHeightShift * (yMax - yMin);    // move face down
    double NewWidth = CONF_VjShrink * (xMax - xMin);
    double xMean = (xMin + xMax) / 2;

    Shape[0].x = xMean - 0.5 * NewWidth;
    Shape[0].y = yMean - 0.5 * NewHeight;
    Shape[1].x = xMean + 0.5 * NewWidth;
    Shape[1].y = yMean + 0.5 * NewHeight;
}

void AAM_VJFaceDetect::AlignToViolaJones(AAM_Shape &StartShape,
        const AAM_Shape &DetShape, const AAM_Shape& MeanShape)
{
    AAM_Shape Base, VjAlign;
    Base.resize(2);
    VjAlign.resize(2);

    double meanCenter = (MeanShape.MinY() + MeanShape.MaxY()) / 2;
    Base[0].x = MeanShape.MinX();
    Base[0].y = meanCenter;
    Base[1].x = MeanShape.MaxX();
    Base[1].y = meanCenter;

    double yMean = (DetShape[1].y + DetShape[0].y)/2;
    VjAlign[0].x = DetShape[0].x;
    VjAlign[0].y = yMean;
    VjAlign[1].x = DetShape[1].x;
    VjAlign[1].y = yMean;

    double a, b, tx, ty;
    Base.AlignTransformation(VjAlign, a, b, tx, ty);
    StartShape = MeanShape;
    StartShape.TransformPose(a, b, tx, ty);

}
