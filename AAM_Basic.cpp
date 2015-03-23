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

#include "AAM_Basic.h"
//#include <direct.h>
//#include <Windows.h>

#define MAG(a,b) sqrt((a)*(a)+(b)*(b))
#define THETA(a,b) atan((b)/(a))
#define curtime cvGetTickCount() / (cvGetTickFrequency()*1000.)

AAM_Basic::AAM_Basic()
{
    __Rc = 0;
    __Rq = 0;

    __current_c = 0;
    __update_c = 0;
    __delta_c = 0;
    __p = 0;
    __current_q = 0;
    __update_q = 0;
    __delta_q = 0;
    __current_lamda = 0;
    __current_s = 0;
    __t_m = 0;
    __t_s = 0;
    __delta_t = 0;
}

AAM_Basic::~AAM_Basic()
{
    cvReleaseMat(&__Rc);
    cvReleaseMat(&__Rq);

    cvReleaseMat(&__current_c);
    cvReleaseMat(&__update_c);
    cvReleaseMat(&__delta_c);
    cvReleaseMat(&__p);
    cvReleaseMat(&__current_q);
    cvReleaseMat(&__update_q);
    cvReleaseMat(&__delta_q);
    cvReleaseMat(&__current_lamda);
    cvReleaseMat(&__current_s);
    cvReleaseMat(&__t_s);
    cvReleaseMat(&__t_m);
    cvReleaseMat(&__delta_t);
}

//============================================================================
void AAM_Basic::Train(const std::vector<AAM_Shape>& AllShapes,
                      const std::vector<IplImage*>& AllImages,
                      double shape_percentage /* = 0.95 */,
                      double texture_percentage /* = 0.95 */,
                      double appearance_percentage /* = 0.95 */)
{
    __cam.Train(AllShapes, AllImages, shape_percentage, texture_percentage,
                appearance_percentage);

    printf("################################################\n");
    printf("Build Prediction(Jacobian) Matrix...\n");

    printf("Calculating appearance parameters...\n");
    CvMat* CParams = cvCreateMat(AllShapes.size(), __cam.nModes(), CV_64FC1);
    CalcCVectors(AllShapes, AllImages, CParams);

    printf("Perturbing appearance and pose parameters...\n");
    // generate c and pose displacement sets
    std::vector<double> vStdDisp;
    vStdDisp.resize(4);
    std::vector<double> vXYDisp;
    vXYDisp.resize(4);
    std::vector<double> vScaleDisp;
    vScaleDisp.resize(4);
    std::vector<double> vRotDisp;
    vRotDisp.resize(4);

    vStdDisp[0] = -0.5;
    vStdDisp[1] = +0.5;
    vStdDisp[2] = -0.25;
    vStdDisp[3] = +0.25;
    vScaleDisp[0] = +0.95;
    vScaleDisp[1] = +1.05;
    vScaleDisp[2] = +0.85;
    vScaleDisp[3] = +1.15;
    vRotDisp[0] = (-5.0 / 180.0 * CV_PI);
    vRotDisp[1] = (+5.0 / 180.0 * CV_PI);
    vRotDisp[2] = (-15.0 / 180.0 * CV_PI);
    vRotDisp[3] = (+15.0 / 180.0 * CV_PI);
    vXYDisp[0] = -0.025;
    vXYDisp[1] = +0.025;
    vXYDisp[2] = -0.075;
    vXYDisp[3] = +0.075;

    CvMat* vCDisps = CalcCParamDisplacementVectors(vStdDisp);
    CvMat* vPoseDisps = CalcPoseDisplacementVectors(vScaleDisp, vRotDisp, vXYDisp, vXYDisp);

    CalcGradientMatrix(CParams, vCDisps, vPoseDisps, AllShapes, AllImages);

    cvReleaseMat(&CParams);
    cvReleaseMat(&vCDisps);
    cvReleaseMat(&vPoseDisps);

    //allocate memory for on-line fitting
    __current_c = cvCreateMat(1, __cam.nModes(), CV_64FC1);
    __update_c = cvCreateMat(1, __cam.nModes(), CV_64FC1);
    __delta_c = cvCreateMat(1, __cam.nModes(), CV_64FC1);
    __p = cvCreateMat(1, __cam.__shape.nModes(), CV_64FC1);
    __current_q = cvCreateMat(1, 4, CV_64FC1);
    __update_q = cvCreateMat(1, 4, CV_64FC1);
    __delta_q = cvCreateMat(1, 4, CV_64FC1);
    __current_lamda = cvCreateMat(1, __cam.__texture.nModes(), CV_64FC1);
    __current_s = cvCreateMat(1, __cam.__shape.nPoints()*2, CV_64FC1);
    __t_s = cvCreateMat(1, __cam.__texture.nPixels(), CV_64FC1);
    __t_m = cvCreateMat(1, __cam.__texture.nPixels(), CV_64FC1);
    __delta_t = cvCreateMat(1, __cam.__texture.nPixels(), CV_64FC1);

    printf("################################################\n\n");
}

//============================================================================
void AAM_Basic::InitParams(const IplImage* image, const CvMat* s, CvMat* c)
{
    //shape parameter
    __cam.__shape.CalcParams(s, __p, __current_q);

    //texture parameter
    __cam.__paw.FasterGetWarpTextureFromMatShape(s, image, __t_s, true);
    __cam.__texture.AlignTextureToRef(__cam.__MeanG, __t_s);
    __cam.__texture.CalcParams(__t_s, __current_lamda);

    //combined appearance parameter
    __cam.CalcParams(c, __p, __current_lamda);
}

//============================================================================
void AAM_Basic::CalcCVectors(const std::vector<AAM_Shape>& AllShapes,
                             const std::vector<IplImage*>& AllImages,
                             CvMat* CParams)
{
    int npixels = __cam.__texture.nPixels();
    int npointsby2 = __cam.__shape.nPoints()*2;
    int nfeatures = __cam.nParameters();
    CvMat* a = cvCreateMat(1, nfeatures, CV_64FC1);//appearance vector
    CvMat* s = cvCreateMat(1, npointsby2, CV_64FC1);//shape vector
    CvMat* t = cvCreateMat(1, npixels, CV_64FC1);//texture vector

    for(int i = 0; i < AllShapes.size(); i++)
    {
        //calculate current shape and texture vector
        AllShapes[i].Point2Mat(s);
        __cam.__paw.FasterGetWarpTextureFromMatShape(s, AllImages[i], t, true);
        __cam.__texture.AlignTextureToRef(__cam.__MeanG, t);

        //convert shape and texture vector to appearance vector
        __cam.ShapeTexture2Combined(s, t, a);

        //calculate appearance parameters by project to appearance spaces
        CvMat c;
        cvGetRow(CParams, &c, i);
        cvProjectPCA(a, __cam.__MeanAppearance, __cam.__AppearanceEigenVectors, &c);
    }

    cvReleaseMat(&s);
    cvReleaseMat(&t);
    cvReleaseMat(&a);
}

//============================================================================
int AAM_Basic::Fit(const IplImage* image, AAM_Shape& Shape,
                   int max_iter /* = 30 */,bool showprocess /* = false */)
{
    //intial some stuff
    double t = curtime;
    double e1, e2, e3;
    double k_v[6] = {-1,-1.15,-0.7,-0.5,-0.2,-0.0625};
    Shape.Point2Mat(__current_s);

//	InitParams(image, __current_s, __current_c);
    __cam.__shape.CalcParams(__current_s, __p, __current_q);
    cvZero(__current_c);
    IplImage* Drawimg =
        cvCreateImage(cvGetSize(image), image->depth, image->nChannels);
    MKDIR("result");
    char filename[100];
    //calculate error
    e3 = EstResidual(image, __current_c, __current_s, __delta_t);
    if(e3 == -1) return 0;

    int iter;

    //do a number of iteration until convergence
    for( iter = 0; iter <max_iter; iter++)
    {
        if(showprocess)
        {
            cvCopy(image, Drawimg);
            __cam.CalcShape(__current_s, __update_c, __current_q);
            Draw(Drawimg, 2);
            sprintf(filename, "result/Iter-%02d.jpg", iter);
            cvSaveImage(filename, Drawimg);
        }

        // predict pose and parameter update
        cvGEMM(__delta_t, __Rq, 1, NULL, 0, __delta_q, CV_GEMM_B_T);
        cvGEMM(__delta_t, __Rc, 1, NULL, 0, __delta_c, CV_GEMM_B_T);

        // if the prediction above didn't improve th fit,
        // try amplify and later damp the prediction
        for(int k = 0; k < 6; k++)
        {
            cvScaleAdd(__delta_q, cvScalar(k_v[k]), __current_q,  __update_q);
            cvScaleAdd(__delta_c, cvScalar(k_v[k]), __current_c,  __update_c);
            __cam.Clamp(__update_c);//constrain parameters
            e2 = EstResidual(image, __update_c, __current_s, __delta_t);
            if(k==0) e1 = e2;
            else if(e2 != -1 && e2 < e1)break;
        }

        //check for convergence
        if((iter>max_iter/3&&fabs(e2-e3)<0.01*e3) || e2<0.001 ) break;
        else if (cvNorm(__delta_c)<0.001 && cvNorm(__delta_q)<0.001) break;
        else
        {
            cvCopy(__update_q, __current_q);
            cvCopy(__update_c, __current_c);
            e3 = e2;
        }
    }
    __cam.CalcShape(__current_s, __current_c, __current_q);
    Shape.Mat2Point(__current_s);
    t = curtime - t;
    printf("AAM-Basic Fitting time cost: %.3f\n", t);
    return iter;
}

//============================================================================
double AAM_Basic::EstResidual(const IplImage* image, const CvMat* c,
                              CvMat* est_s, CvMat* diff)
{
    // generate model texture
    __cam.CalcTexture(__t_m, c);

    // generate model shape
    __cam.CalcShape(est_s, c, __current_q);

    //calculate warped texture
    if(!AAM_Basic::IsShapeWithinImage(est_s, image->width, image->height))
        return -1;
    __cam.__paw.FasterGetWarpTextureFromMatShape(est_s, image, __t_s, true);
    __cam.__texture.AlignTextureToRef(__cam.__MeanG, __t_s);

    //calc pixel difference: g_s - g_m
    cvSub(__t_s, __t_m, diff);

    return cvNorm(diff);


}


//============================================================================
CvMat* AAM_Basic::
CalcCParamDisplacementVectors(const std::vector<double>& vStdDisp)
{
    int np =__cam.nModes();
    CvMat*	cDisplacements = cvCreateMat(vStdDisp.size(), np, CV_64FC1);
    cvZero(cDisplacements);

    // calculate displacements, for each parameter
    for(int j = 0; j < vStdDisp.size(); j++)
    {
        for (int i = 0; i < np; i++)
        {
            // for each displacement
            cvmSet(cDisplacements, j, i, vStdDisp[j]*sqrt(__cam.Var(i)));
        }
    }

    return cDisplacements;
}

//============================================================================
CvMat* AAM_Basic::CalcPoseDisplacementVectors(const std::vector<double> &vScaleDisp,
        const std::vector<double> &vRotDisp,
        const std::vector<double> &vXDisp,
        const std::vector<double> &vYDisp)
{
    CvMat* poseDisplacements = cvCreateMat(vXDisp.size(), 4, CV_64FC1);
    cvZero(poseDisplacements);

    for(int i = 0; i < vScaleDisp.size(); i++)
    {
        //add scale displacements
        cvmSet(poseDisplacements, i, 0, vScaleDisp[i]*cos(vRotDisp[i])-1);

        //add rotation displacements
        cvmSet(poseDisplacements, i, 1, vScaleDisp[i]*sin(vRotDisp[i]));

        //add x displacements
        cvmSet(poseDisplacements, i, 2, vXDisp[i]);

        //add y displacements
        cvmSet(poseDisplacements, i, 3, vYDisp[i]);
    }

    return poseDisplacements;
}

//============================================================================
void AAM_Basic::CalcGradientMatrix(const CvMat* CParams,
                                   const CvMat* vCDisps,
                                   const CvMat* vPoseDisps,
                                   const std::vector<AAM_Shape>& AllShapes,
                                   const std::vector<IplImage*>& AllImages)
{
    int npixels = __cam.__texture.nPixels();
    int np = __cam.nModes();

    // do model parameter experiments
    {
        printf("Calculating parameter gradient matrix...\n");
        CvMat* GParam = cvCreateMat(np, npixels, CV_64FC1);
        cvZero(GParam);
        CvMat* GtG = cvCreateMat(np, np, CV_64FC1);
        CvMat* GtGInv = cvCreateMat(np, np, CV_64FC1);

        // estimate Rc
        EstCParamGradientMatrix(GParam, CParams, AllShapes, AllImages, vCDisps);
        __Rc = cvCreateMat(np, npixels, CV_64FC1);
        cvGEMM(GParam, GParam, 1, NULL, 0, GtG, CV_GEMM_B_T);
        cvInvert(GtG, GtGInv, CV_SVD );
        cvMatMul(GtGInv, GParam, __Rc);

        cvReleaseMat(&GtG);
        cvReleaseMat(&GtGInv);
        cvReleaseMat(&GParam);
    }

    // do pose experiments, this is for global shape normalization
    {
        printf("Calculating pose gradient matrix...\n");
        CvMat* GtG = cvCreateMat(4, 4, CV_64FC1);
        CvMat* GtGInv = cvCreateMat(4, 4, CV_64FC1);
        CvMat* GPose = cvCreateMat(4, npixels, CV_64FC1);
        cvZero(GPose);

        // estimate Rt
        EstPoseGradientMatrix(GPose, CParams, AllShapes, AllImages, vPoseDisps);
        __Rq = cvCreateMat(4, npixels, CV_64FC1);
        cvGEMM(GPose, GPose, 1, NULL, 0, GtG, CV_GEMM_B_T);
        cvInvert(GtG, GtGInv, CV_SVD);
        cvMatMul(GtGInv, GPose, __Rq);

        cvReleaseMat(&GtG);
        cvReleaseMat(&GtGInv);
        cvReleaseMat(&GPose);
    }
}


//============================================================================
void AAM_Basic::EstCParamGradientMatrix(CvMat* GParam,
                                        const CvMat* CParams,
                                        const std::vector<AAM_Shape>& AllShapes,
                                        const std::vector<IplImage*>& AllImages,
                                        const CvMat* vCDisps)
{
    int nExperiment = 0;
    int ntotalExp = AllShapes.size()*vCDisps->rows/2*__cam.nModes();
    int np = __cam.nModes();
    int npixels = __cam.__texture.nPixels();
    int npointsby2 = __cam.__shape.nPoints()*2;
    CvMat c;											//appearance parameters
    CvMat* c1 = cvCreateMat(1, np, CV_64FC1);
    CvMat* c2 = cvCreateMat(1, np, CV_64FC1);
    CvMat* s1 = cvCreateMat(1, npointsby2, CV_64FC1);	//shape vector
    CvMat* s2 = cvCreateMat(1, npointsby2, CV_64FC1);
    CvMat* t1 = cvCreateMat(1, npixels, CV_64FC1);		//texture vector
    CvMat* t2 = cvCreateMat(1, npixels, CV_64FC1);
    CvMat* delta_g1 = cvCreateMat(1, npixels, CV_64FC1);
    CvMat* delta_g2 = cvCreateMat(1, npixels, CV_64FC1);
    CvMat* cDiff = cvCreateMat(1, npixels, CV_64FC1);
    AAM_Shape aam_s1, aam_s2;
    std::vector<double> normFactors;
    normFactors.resize(np);

    // for each training example in the training set
    for(int i = 0; i < AllShapes.size(); i++)
    {
        cvGetRow(CParams, &c, i);
        for(int j = 0; j < vCDisps->rows; j+=2)
        {
            for(int k = 0; k < vCDisps->cols; k++)
            {
                printf("Performing (%d/%d)\r", nExperiment++, ntotalExp);

                //shift current appearance parameters
                cvCopy(&c, c1);
                cvCopy(&c, c2);
                cvmSet(c1, 0, k, cvmGet(&c,0,k)+cvmGet(vCDisps,j,k));
                cvmSet(c2, 0, k, cvmGet(&c,0,k)+cvmGet(vCDisps,j+1,k));

                //generate model texture
                __cam.CalcTexture(t1, c1);
                __cam.CalcTexture(t2, c2);

                //generate model shape
                __cam.CalcLocalShape(s1, c1);
                aam_s1.Mat2Point(s1);
                __cam.CalcLocalShape(s2, c2);
                aam_s2.Mat2Point(s2);

                //align model shape to annotated shape
                aam_s1.AlignTo(AllShapes[i]);
                aam_s1.Point2Mat(s1);
                aam_s2.AlignTo(AllShapes[i]);
                aam_s2.Point2Mat(s2);

                //sample the shape to get warped texture
                __cam.__paw.FasterGetWarpTextureFromMatShape(s1, AllImages[i], delta_g1, true);
                __cam.__texture.AlignTextureToRef(__cam.__MeanG, delta_g1);

                __cam.__paw.FasterGetWarpTextureFromMatShape(s2, AllImages[i], delta_g2, true);
                __cam.__texture.AlignTextureToRef(__cam.__MeanG, delta_g2);

                //calculate pixel difference(g_s - g_m)
                cvSub(delta_g1, t1, delta_g1);
                cvSub(delta_g2, t2, delta_g2);

                //form central displacement
                cvSub(delta_g2, delta_g1, cDiff);
                cvConvertScale(cDiff, cDiff, 1.0/(cvmGet(vCDisps,j+1,k)-cvmGet(vCDisps,j,k)));

                //accumulate into k-th row
                CvMat Gk;
                cvGetRow(GParam, &Gk, k);
                cvAdd(&Gk, cDiff, &Gk);

                //increment normalisation factor
                normFactors[k] = normFactors[k]+1;
            }
        }
    }

    //normalize
    for(int j = 0; j < np; j++)
    {
        CvMat Gj;
        cvGetRow(GParam, &Gj, j);
        cvConvertScale(&Gj, &Gj, 1.0/normFactors[j]);
    }

    cvReleaseMat(&c1);
    cvReleaseMat(&c2);
    cvReleaseMat(&s1);
    cvReleaseMat(&s2);
    cvReleaseMat(&t1);
    cvReleaseMat(&t2);
    cvReleaseMat(&delta_g1);
    cvReleaseMat(&delta_g2);
    cvReleaseMat(&cDiff);
}

//============================================================================
void AAM_Basic::EstPoseGradientMatrix(CvMat* GPose,
                                      const CvMat* CParams,
                                      const std::vector<AAM_Shape>& AllShapes,
                                      const std::vector<IplImage*>& AllImages,
                                      const CvMat* vPoseDisps)
{
    int nExperiment = 0;
    int ntotalExp = AllShapes.size()*vPoseDisps->rows/2*vPoseDisps->cols;
    int npixels = __cam.__texture.nPixels();
    int npointsby2 = __cam.__shape.nPoints()*2;
    int smodes = __cam.__shape.nModes();
    CvMat c;									//appearance parameters
    CvMat* q = cvCreateMat(1, 4, CV_64FC1);		//pose parameters
    CvMat* q1 = cvCreateMat(1, 4, CV_64FC1);
    CvMat* q2 = cvCreateMat(1, 4, CV_64FC1);
    CvMat* p = cvCreateMat(1, smodes, CV_64FC1);//shape parameters
    CvMat* s1 = cvCreateMat(1, npointsby2, CV_64FC1);//shape vector
    CvMat* s2 = cvCreateMat(1, npointsby2, CV_64FC1);
    CvMat* t1 = cvCreateMat(1, npixels, CV_64FC1);//texture vector
    CvMat* t2 = cvCreateMat(1, npixels, CV_64FC1);
    CvMat* delta_g1 = cvCreateMat(1, npixels, CV_64FC1);
    CvMat* delta_g2 = cvCreateMat(1, npixels, CV_64FC1);
    CvMat* cDiff = cvCreateMat(1, npixels, CV_64FC1);
    std::vector<double> normFactors;
    normFactors.resize(4);
    CvMat* AbsPoseDisps = cvCreateMat(vPoseDisps->rows, vPoseDisps->cols, CV_64FC1);

    // for each training example in the training set
    for(int i = 0; i < AllShapes.size(); i++)
    {
        cvGetRow(CParams, &c, i);

        //calculate pose parameters
        AllShapes[i].Point2Mat(s1);
        __cam.__shape.CalcParams(s1, p, q);

        cvCopy(vPoseDisps, AbsPoseDisps);
        int w = AllShapes[i].GetWidth();
        int h = AllShapes[i].GetHeight();
        int W = AllImages[i]->width;
        int H = AllImages[i]->height;

        for(int j = 0; j < vPoseDisps->rows; j+=2)
        {
            //translate relative translation to abs translation about x & y
            CV_MAT_ELEM(*AbsPoseDisps, double, j, 2) *= w;
            CV_MAT_ELEM(*AbsPoseDisps, double, j, 3) *= h;
            CV_MAT_ELEM(*AbsPoseDisps, double, j+1, 2) *= w;
            CV_MAT_ELEM(*AbsPoseDisps, double, j+1, 3) *= h;

            for(int k = 0; k < vPoseDisps->cols; k++)
            {
                printf("Performing (%d/%d)\r", nExperiment++, ntotalExp);

                //shift current pose parameters
                cvCopy(q, q1);
                cvCopy(q, q2);
                if(k == 0 || k == 1)
                {
                    double scale, theta, dscale1, dtheta1, dscale2, dtheta2;
                    scale = MAG(cvmGet(q,0,0)+1,cvmGet(q,0,1));
                    theta = THETA(cvmGet(q,0,0)+1,cvmGet(q,0,1));
                    dscale1 = MAG(cvmGet(AbsPoseDisps,j,0)+1,cvmGet(AbsPoseDisps,j,1));
                    dtheta1 = THETA(cvmGet(AbsPoseDisps,j,0)+1,cvmGet(AbsPoseDisps,j,1));
                    dscale2 = MAG(cvmGet(AbsPoseDisps,j+1,0)+1,cvmGet(AbsPoseDisps,j+1,1));
                    dtheta2 = THETA(cvmGet(AbsPoseDisps,j+1,0)+1,cvmGet(AbsPoseDisps,j+1,1));

                    dscale1 = dscale1*scale;
                    dtheta1 += theta;
                    dscale2 = dscale2*scale;
                    dtheta2 += theta;

                    if(k == 0)
                    {
                        cvmSet(q1,0,0,dscale1*cos(theta)-1);
                        cvmSet(q1,0,1,dscale1*sin(theta));
                        cvmSet(q2,0,0,dscale2*cos(theta)-1);
                        cvmSet(q2,0,1,dscale2*sin(theta));
                    }

                    else
                    {
                        cvmSet(q1,0,0,scale*cos(dtheta1)-1);
                        cvmSet(q1,0,1,scale*sin(dtheta1));
                        cvmSet(q2,0,0,scale*cos(dtheta2)-1);
                        cvmSet(q2,0,1,scale*sin(dtheta2));
                    }
                }

                else
                {
                    cvmSet(q1,0,k,cvmGet(q,0,k)+cvmGet(AbsPoseDisps,j,k));
                    cvmSet(q2,0,k,cvmGet(q,0,k)+cvmGet(AbsPoseDisps,j+1,k));
                }

                //generate model texture
                __cam.CalcTexture(t1, &c);
                __cam.CalcTexture(t2, &c);
//				{
//					char filename[100];
//					sprintf(filename, "a/%d.jpg", nExperiment);
//					__cam.__paw.SaveWarpImageFromVector(filename, t1);
//				}

                //generate model shape
                //__cam.__shape.CalcShape(p, q1, s1);
                //__cam.__shape.CalcShape(p, q2, s2);
                __cam.CalcLocalShape(s1, &c);
                __cam.__shape.CalcGlobalShape(q1,s1);
                __cam.CalcLocalShape(s2, &c);
                __cam.__shape.CalcGlobalShape(q2,s2);

                //sample the shape to get warped texture
                if(!AAM_Basic::IsShapeWithinImage(s1, W, H)) cvZero(delta_g1);
                else __cam.__paw.FasterGetWarpTextureFromMatShape(s1, AllImages[i],	delta_g1, true);
                __cam.__texture.AlignTextureToRef(__cam.__MeanG, delta_g1);

                if(!AAM_Basic::IsShapeWithinImage(s2, W, H)) cvZero(delta_g2);
                else __cam.__paw.FasterGetWarpTextureFromMatShape(s2, AllImages[i], delta_g2, true);
                __cam.__texture.AlignTextureToRef(__cam.__MeanG, delta_g2);

//				{
//					char filename[100];
//					sprintf(filename, "a/%d-.jpg", nExperiment);
//					__cam.__paw.SaveWarpImageFromVector(filename, delta_g1);
//					sprintf(filename, "a/%d+.jpg", nExperiment);
//					__cam.__paw.SaveWarpImageFromVector(filename, delta_g2);
//				}

                //calculate pixel difference(g_s - g_m)
                cvSub(delta_g1, t1, delta_g1);
                cvSub(delta_g2, t2, delta_g2);

                //form central displacement
                cvSub(delta_g2, delta_g1, cDiff);
                cvConvertScale(cDiff, cDiff, 1.0/(cvmGet(AbsPoseDisps,j+1,k)-cvmGet(AbsPoseDisps,j,k)));

                //accumulate into k-th row
                CvMat Gk;
                cvGetRow(GPose, &Gk, k);
                cvAdd(&Gk, cDiff, &Gk);

                //increment normalisation factor
                normFactors[k] = normFactors[k]+1;
            }
        }
    }

    //normalize
    for(int j = 0; j < vPoseDisps->cols; j++)
    {
        CvMat Gj;
        cvGetRow(GPose, &Gj, j);
        cvConvertScale(&Gj, &Gj, 1.0/normFactors[j]);
    }

    cvReleaseMat(&s1);
    cvReleaseMat(&s2);
    cvReleaseMat(&t1);
    cvReleaseMat(&t2);
    cvReleaseMat(&delta_g1);
    cvReleaseMat(&delta_g2);
    cvReleaseMat(&cDiff);
    cvReleaseMat(&AbsPoseDisps);
}

//===========================================================================
bool AAM_Basic::IsShapeWithinImage(const CvMat* s, int w, int h)
{
    double* fasts = s->data.db;
    int npoints = s->cols / 2;

    for(int i = 0; i < npoints; i++)
    {
        if(fasts[2*i] > w-1 || fasts[2*i] < 0)
            return false;
        if(fasts[2*i+1] > h-1 || fasts[2*i+1] < 0)
            return false;
    }
    return true;
}

//===========================================================================
void AAM_Basic::Draw(IplImage* image, int type)
{
    if(type == 0) DrawPoint(image);
    else if(type == 1) DrawTriangle(image);
    else if(type == 2)	DrawAppearance(image);
}

//============================================================================
void AAM_Basic::DrawPoint(IplImage* image)
{
    double* p = __current_s->data.db;
    for(int i = 0; i < __cam.__shape.nPoints(); i++)
    {
        cvCircle(image, cvPoint(p[2*i], p[2*i+1]), 3, CV_RGB(255, 0, 0));
    }
}

//============================================================================
void AAM_Basic::DrawTriangle(IplImage* image)
{
    double* p = __current_s->data.db;
    int idx1, idx2, idx3;
    for(int i = 0; i < __cam.__paw.nTri(); i++)
    {
        idx1 = __cam.__paw.__tri[i][0];
        idx2 = __cam.__paw.__tri[i][1];
        idx3 = __cam.__paw.__tri[i][2];
        cvLine(image, cvPoint(p[2*idx1], p[2*idx1+1]), cvPoint(p[2*idx2], p[2*idx2+1]),
               CV_RGB(128,255,0));
        cvLine(image, cvPoint(p[2*idx2], p[2*idx2+1]), cvPoint(p[2*idx3], p[2*idx3+1]),
               CV_RGB(128,255,0));
        cvLine(image, cvPoint(p[2*idx3], p[2*idx3+1]), cvPoint(p[2*idx1], p[2*idx1+1]),
               CV_RGB(128,255,0));
    }
}


//============================================================================
void AAM_Basic::DrawAppearance(IplImage* image)
{
    AAM_Shape Shape;
    Shape.Mat2Point(__current_s);
    AAM_PAW paw;
    paw.Train(Shape, __cam.__Points, __cam.__Storage, __cam.__paw.GetTri(), false);
    int x1, x2, y1, y2, idx1, idx2;
    int tri_idx, v1, v2, v3;
    int xby3, idxby3;
    int minx, miny, maxx, maxy;
    AAM_Shape refShape;
    refShape.Mat2Point(__cam.__MeanS);
    refShape.Translate(-refShape.MinX(), -refShape.MinY());
    double minV, maxV;
    cvMinMaxLoc(__t_m, &minV, &maxV);
    cvConvertScale(__t_m, __t_m, 255/(maxV-minV), -minV*255/(maxV-minV));
    byte* pimg;
    double* fastt = __t_m->data.db;

    minx = Shape.MinX();
    miny = Shape.MinY();
    maxx = Shape.MaxX();
    maxy = Shape.MaxY();
    for(int y = miny; y < maxy; y++)
    {
        y1 = y-miny;
        pimg = (byte*)(image->imageData + image->widthStep*y);
        for(int x = minx; x < maxx; x++)
        {
            x1 = x-minx;
            idx1 = paw.__rect[y1][x1];
            if(idx1 >= 0)
            {
                tri_idx = paw.PixTri(idx1);
                v1 = paw.Tri(tri_idx, 0);
                v2 = paw.Tri(tri_idx, 1);
                v3 = paw.Tri(tri_idx, 2);

                x2 = paw.__alpha[idx1]*refShape[v1].x + paw.__belta[idx1]*refShape[v2].x +
                     paw.__gamma[idx1]*refShape[v3].x;
                y2 = paw.__alpha[idx1]*refShape[v1].y + paw.__belta[idx1]*refShape[v2].y +
                     paw.__gamma[idx1]*refShape[v3].y;

                xby3 = 3*x;
                idx2 = __cam.__paw.__rect[y2][x2];
                idxby3 = 3*idx2;
                pimg[xby3] = fastt[idxby3];
                pimg[xby3+1] = fastt[idxby3+1];
                pimg[xby3+2] = fastt[idxby3+2];
            }
        }
    }
}

//===========================================================================
void AAM_Basic::Write(std::ofstream& os)
{
    printf("Saving the Basic AAM Model to file...");

    __cam.Write(os);
    os << __Rc;
    os << __Rq;

    printf("OK\n");
}

//===========================================================================
void AAM_Basic::Read(std::ifstream& is)
{
    printf("Reading the Basic AAM Model from file...");

    __cam.Read(is);
    __Rc = cvCreateMat(__cam.nModes(), __cam.__texture.nPixels(), CV_64FC1);
    __Rq = cvCreateMat(4, __cam.__texture.nPixels(), CV_64FC1);
    is >> __Rc;
    is >> __Rq;

    //allocate memory for on-line fitting
    __current_c = cvCreateMat(1, __cam.nModes(), CV_64FC1);
    __update_c = cvCreateMat(1, __cam.nModes(), CV_64FC1);
    __delta_c = cvCreateMat(1, __cam.nModes(), CV_64FC1);
    __p = cvCreateMat(1, __cam.__shape.nModes(), CV_64FC1);
    __current_q = cvCreateMat(1, 4, CV_64FC1);
    __update_q = cvCreateMat(1, 4, CV_64FC1);
    __delta_q = cvCreateMat(1, 4, CV_64FC1);
    __current_lamda = cvCreateMat(1, __cam.__texture.nModes(), CV_64FC1);
    __current_s = cvCreateMat(1, __cam.__shape.nPoints()*2, CV_64FC1);
    __t_s = cvCreateMat(1, __cam.__texture.nPixels(), CV_64FC1);
    __t_m = cvCreateMat(1, __cam.__texture.nPixels(), CV_64FC1);
    __delta_t = cvCreateMat(1, __cam.__texture.nPixels(), CV_64FC1);

    printf("OK\n");

}
