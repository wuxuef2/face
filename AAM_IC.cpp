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
#include "AAM_IC.h"

#define MAG(x, y) sqrt((x)*(x)+(y)*(y))

static int gi = 0;

AAM_IC::AAM_IC()
{
    __Points = 0;
    __Storage = 0;

    __update_s0 = 0;
    __delta_s = 0;
    __warp_t = 0;
    __error_t = 0;
    __search_pq = 0;
    __delta_pq = 0;
    __current_s = 0;
    __update_s = 0;
}

AAM_IC::~AAM_IC()
{
    cvReleaseMat(&__Points);
    cvReleaseMemStorage(&__Storage);

    cvReleaseMat(&__update_s0);
    cvReleaseMat(&__delta_s);
    cvReleaseMat(&__warp_t);
    cvReleaseMat(&__error_t);
    cvReleaseMat(&__search_pq);
    cvReleaseMat(&__delta_pq);
    cvReleaseMat(&__current_s);
    cvReleaseMat(&__update_s);
}

//============================================================================
CvMat* AAM_IC::CalcGradIdx()
{
    CvMat* pos= cvCreateMat(__paw.nPix(), 4, CV_32SC1);

    int i = 0;
    int width = __paw.Width(), height = __paw.Height();
    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++)
        {
            if(__paw.Rect(y, x) >= 0)
            {
                int *ppos = (int*)(pos->data.ptr + 	pos->step*i);
                ppos[0] = (x-1<0)		?-1:__paw.Rect(y, x-1);  // left
                ppos[1] = (x+1>=width)	?-1:__paw.Rect(y, x+1);  // right
                ppos[2] = (y-1<0)		?-1:__paw.Rect(y-1, x);  // top
                ppos[3] = (y+1>=height)	?-1:__paw.Rect(y+1, x);  // bottom
                i++;
            }
        }
    }

    return pos;
}

//============================================================================
void AAM_IC::CalcTexGrad(const CvMat* texture, CvMat* dTx, CvMat* dTy)
{
    printf("Calculating texture gradient...");

    double* _x = dTx->data.db;
    double* _y = dTy->data.db;
    double* t = texture->data.db;
    CvMat *p = CalcGradIdx();

    for(int i = 0; i < __paw.nPix(); i++)
    {
        int *fastp = (int*)(p->data.ptr + p->step*i);

        // x direction
        if(fastp[0] >= 0 && fastp[1] >= 0)
        {
            _x[3*i+0] = (t[3*fastp[1]+0] - t[3*fastp[0]+0])/2;
            _x[3*i+1] = (t[3*fastp[1]+1] - t[3*fastp[0]+1])/2;
            _x[3*i+2] = (t[3*fastp[1]+2] - t[3*fastp[0]+2])/2;
        }

        else if(fastp[0] >= 0 && fastp[1] < 0)
        {
            _x[3*i+0] = t[3*i+0] - t[3*fastp[0]+0];
            _x[3*i+1] = t[3*i+1] - t[3*fastp[0]+1];
            _x[3*i+2] = t[3*i+2] - t[3*fastp[0]+2];
        }

        else if(fastp[0] < 0 && fastp[1] >= 0)
        {
            _x[3*i+0] = t[3*fastp[1]+0] - t[3*i+0];
            _x[3*i+1] = t[3*fastp[1]+1] - t[3*i+1];
            _x[3*i+2] = t[3*fastp[1]+2] - t[3*i+2];
        }
        else
        {
            _x[3*i+0] = 0;
            _x[3*i+1] = 0;
            _x[3*i+2] = 0;
        }

        // y direction
        if(fastp[2] >= 0 && fastp[3] >= 0)
        {
            _y[3*i+0] = (t[3*fastp[3]+0] - t[3*fastp[2]+0])/2;
            _y[3*i+1] = (t[3*fastp[3]+1] - t[3*fastp[2]+1])/2;
            _y[3*i+2] = (t[3*fastp[3]+2] - t[3*fastp[2]+2])/2;
        }

        else if(fastp[2] >= 0 && fastp[3] < 0)
        {
            _y[3*i+0] = t[3*i+0] - t[3*fastp[2]+0];
            _y[3*i+1] = t[3*i+1] - t[3*fastp[2]+1];
            _y[3*i+2] = t[3*i+2] - t[3*fastp[2]+2];
        }

        else if(fastp[2] < 0 && fastp[3] >= 0)
        {
            _y[3*i+0] = t[3*fastp[3]+0] - t[3*i+0];
            _y[3*i+1] = t[3*fastp[3]+1] - t[3*i+1];
            _y[3*i+2] = t[3*fastp[3]+2] - t[3*i+2];
        }

        else
        {
            _y[3*i+0] = 0;
            _y[3*i+1] = 0;
            _y[3*i+2] = 0;
        }
    }
    cvReleaseMat(&p);
    printf("Done\n");
}

//============================================================================
void AAM_IC::CalcWarpJacobian(CvMat* Jx, CvMat* Jy)
{
    printf("Calculating warp Jacobian...");

    int nPoints = __shape.nPoints();
    __sMean.Mat2Point(__shape.GetMean());
    __sStar1.resize(nPoints);
    __sStar2.resize(nPoints);
    __sStar3.resize(nPoints);
    __sStar4.resize(nPoints);
    for(int n = 0; n < nPoints; n++) // Equation (43)
    {
        __sStar1[n].x = __sMean[n].x;
        __sStar1[n].y = __sMean[n].y;
        __sStar2[n].x = -__sMean[n].y;
        __sStar2[n].y = __sMean[n].x;
        __sStar3[n].x = 1;
        __sStar3[n].y = 0;
        __sStar4[n].x = 0;
        __sStar4[n].y = 1;
    }

    const CvMat* B = __shape.GetBases();
    const CvMat* mean = __shape.GetMean();
    cvZero(Jx);
    cvZero(Jy);
    for(int i = 0; i < __paw.nPix(); i++)
    {
        int tri_idx = __paw.PixTri(i);
        int v1 = __paw.Tri(tri_idx, 0);
        int v2 = __paw.Tri(tri_idx, 1);
        int v3 = __paw.Tri(tri_idx, 2);
        double *fastJx = (double*)(Jx->data.ptr + Jx->step*i);
        double *fastJy = (double*)(Jy->data.ptr + Jy->step*i);

        // Equation (50) dN_dq
        fastJx[0] = __paw.Alpha(i)*__sStar1[v1].x +
                    __paw.Belta(i)*__sStar1[v2].x +  __paw.Gamma(i)*__sStar1[v3].x;
        fastJy[0] = __paw.Alpha(i)*__sStar1[v1].y +
                    __paw.Belta(i)*__sStar1[v2].y +  __paw.Gamma(i)*__sStar1[v3].y;

        fastJx[1] = __paw.Alpha(i)*__sStar2[v1].x +
                    __paw.Belta(i)*__sStar2[v2].x +  __paw.Gamma(i)*__sStar2[v3].x;
        fastJy[1] = __paw.Alpha(i)*__sStar2[v1].y +
                    __paw.Belta(i)*__sStar2[v2].y +  __paw.Gamma(i)*__sStar2[v3].y;

        fastJx[2] = __paw.Alpha(i)*__sStar3[v1].x +
                    __paw.Belta(i)*__sStar3[v2].x +  __paw.Gamma(i)*__sStar3[v3].x;
        fastJy[2] = __paw.Alpha(i)*__sStar3[v1].y +
                    __paw.Belta(i)*__sStar3[v2].y +  __paw.Gamma(i)*__sStar3[v3].y;

        fastJx[3] = __paw.Alpha(i)*__sStar4[v1].x +
                    __paw.Belta(i)*__sStar4[v2].x +  __paw.Gamma(i)*__sStar4[v3].x;
        fastJy[3] = __paw.Alpha(i)*__sStar4[v1].y +
                    __paw.Belta(i)*__sStar4[v2].y +  __paw.Gamma(i)*__sStar4[v3].y;

        // Equation (51) dW_dp
        for(int j = 0; j < __shape.nModes(); j++)
        {
            fastJx[j+4] = __paw.Alpha(i)*cvmGet(B,j,2*v1) +
                          __paw.Belta(i)*cvmGet(B,j,2*v2) + __paw.Gamma(i)*cvmGet(B,j,2*v3);

            fastJy[j+4] = __paw.Alpha(i)*cvmGet(B,j,2*v1+1) +
                          __paw.Belta(i)*cvmGet(B,j,2*v2+1) + __paw.Gamma(i)*cvmGet(B,j,2*v3+1);
        }
    }

    printf("Done\n");
}

//============================================================================
void AAM_IC::CalcModifiedSD(CvMat* SD, const CvMat* dTx, const CvMat* dTy,
                            const CvMat* Jx, const CvMat* Jy)
{
    printf("Calculating steepest descent images...");

    int i, j;

    //create steepest descent images
    double* _x = dTx->data.db;
    double* _y = dTy->data.db;
    double temp;
    for(i = 0; i < __shape.nModes()+4; i++)
    {
        for(j = 0; j < __paw.nPix(); j++)
        {
            temp = _x[3*j  ]*cvmGet(Jx,j,i) +_y[3*j  ]*cvmGet(Jy,j,i);
            cvmSet(SD,i,3*j,temp);

            temp = _x[3*j+1]*cvmGet(Jx,j,i) +_y[3*j+1]*cvmGet(Jy,j,i);
            cvmSet(SD,i,3*j+1,temp);

            temp = _x[3*j+2]*cvmGet(Jx,j,i) +_y[3*j+2]*cvmGet(Jy,j,i);
            cvmSet(SD,i,3*j+2,temp);
        }
    }

    //project out appearance variation i.e. modify the steepest descent image
    const CvMat* B = __texture.GetBases();
    CvMat* V = cvCreateMat(4+__shape.nModes(), __texture.nModes(), CV_64FC1);
    CvMat SDMat, BMat;
    cvGEMM(SD, B, 1., NULL, 1., V, CV_GEMM_B_T);
    // Equation (63),(64)
    for(i = 0; i < __shape.nModes()+4; i++)
    {
        for(j = 0; j < __texture.nModes(); j++)
        {
            cvGetRow(SD, &SDMat, i);
            cvGetRow(B, &BMat, j);
            cvScaleAdd(&BMat, cvScalar(-cvmGet(V,i,j)), &SDMat, &SDMat);
        }
    }

    printf("Done\n");
}

//============================================================================
void AAM_IC::CalcHessian(CvMat* H, const CvMat* SD)
{
    printf("Calculating Hessian inverse matrix...");

    CvMat* HH = cvCreateMat(H->rows, H->cols, CV_64FC1);
    cvMulTransposed(SD, HH, 0);// Equation (65)
    cvInvert(HH, H, CV_SVD);
    cvReleaseMat(&HH);

    printf("Done\n");
}

//============================================================================
void AAM_IC::Train(const std::vector<AAM_Shape>& AllShapes, const std::vector<IplImage*>& AllImages,
                   double shape_percentage /* = 0.95 */, double texture_percentage /* = 0.95 */)
{
    if(AllShapes.size() != AllImages.size())
    {
        fprintf(stderr, "ERROE(%s, %d): #Shapes != #Images\n",
                __FILE__, __LINE__);
        exit(0);
    }

    //building shape and texture distribution model
    __shape.Train(AllShapes, shape_percentage);
    __Points = cvCreateMat (1, __shape.nPoints(), CV_32FC2);
    __Storage = cvCreateMemStorage(0);
    __paw.Train(__shape.GetAAMReferenceShape(), __Points, __Storage);
    __texture.Train(AllShapes, __paw, AllImages, texture_percentage, false);  //if true, save the image to file

    printf("################################################\n");
    printf("Build Inverse Compositional Image Alignmennt Model...\n");

    //calculate gradient of texture
    CvMat* dTx = cvCreateMat(1, __texture.nPixels(), CV_64FC1);
    CvMat* dTy = cvCreateMat(1, __texture.nPixels(), CV_64FC1);
    CalcTexGrad(__texture.GetMean(), dTx, dTy);
    // save gradient image
    MKDIR("Modes");
    __paw.SaveWarpImageFromVector("Modes/dTx.jpg", dTx);
    __paw.SaveWarpImageFromVector("Modes/dTy.jpg", dTy);

    //draw the mean face
    char gid[10];
    //_itoa(gi, gid,10);
    snprintf(gid, 10, "%d", gi);
    std::string name = std::string(gid);
    std::string resultDir = "./test/";
    //std::string resultDir = "../test2/";
    std::string dir = resultDir + "meanFace/meanG" + name  + ".jpg";
    __paw.SaveWarpImageFromVector(dir.c_str(), __texture.GetMean());
    gi++;

    //calculate warp Jacobian at base shape
    CvMat* Jx = cvCreateMat(__paw.nPix(), __shape.nModes()+4, CV_64FC1);
    CvMat* Jy = cvCreateMat(__paw.nPix(), __shape.nModes()+4, CV_64FC1);
    CalcWarpJacobian(Jx,Jy);

    //calculate modified steepest descent image
    CvMat* SD = cvCreateMat(__shape.nModes()+4, __texture.nPixels(), CV_64FC1);
    CalcModifiedSD(SD, dTx, dTy, Jx, Jy);

    //calculate inverse Hessian matrix
    CvMat* H = cvCreateMat(__shape.nModes()+4, __shape.nModes()+4, CV_64FC1);
    CalcHessian(H, SD);

    //calculate update matrix (multiply inverse Hessian by modified steepest descent image)
    __G = cvCreateMat(__shape.nModes()+4, __texture.nPixels(), CV_64FC1);
    cvMatMul(H, SD, __G);

    //release
    cvReleaseMat(&Jx);
    cvReleaseMat(&Jy);
    cvReleaseMat(&dTx);
    cvReleaseMat(&dTy);
    cvReleaseMat(&SD);
    cvReleaseMat(&H);

    //alocate memory for on-line fitting stuff
    __update_s0 = cvCreateMat(1, __shape.nPoints()*2, CV_64FC1);
    __delta_s = cvCreateMat(1, __shape.nPoints()*2, CV_64FC1);
    __inv_pq = cvCreateMat(1, __shape.nModes()+4, CV_64FC1);
    __warp_t = cvCreateMat(1, __texture.nPixels(), CV_64FC1);
    __error_t = cvCreateMat(1, __texture.nPixels(), CV_64FC1);
    __search_pq = cvCreateMat(1, __shape.nModes()+4, CV_64FC1);
    __delta_pq = cvCreateMat(1, __shape.nModes()+4, CV_64FC1);
    __current_s = cvCreateMat(1, __shape.nPoints()*2, CV_64FC1);
    __update_s = cvCreateMat(1, __shape.nPoints()*2, CV_64FC1);
    __lamda  = cvCreateMat(1, __texture.nModes(), CV_64FC1);

    printf("################################################\n\n");
}

//============================================================================
int AAM_IC::Fit(const IplImage* image, 		AAM_Shape& Shape,
                int max_iter /* = 30 */, 	bool showprocess /* = false */)
{
    //initialize some stuff
    double t = (double)cvGetTickCount();
    CvMat p;
    cvGetCols(__search_pq, &p, 4, 4+__shape.nModes());
    double e1(1e100), e2;
    Shape.Point2Mat(__current_s);
    const CvMat* A0 = __texture.GetMean();
    SetAllParamsZero();
    __shape.CalcParams(__current_s, __search_pq);

    IplImage* Drawimg = 0;
    char filename[100];

    if(showprocess)
    {
        Drawimg = cvCreateImage(cvGetSize(image), image->depth, image->nChannels);
        MKDIR("result");
        cvCopy(image, Drawimg);
        Draw(Drawimg, 2);
        sprintf(filename, "result/Init.jpg");
        cvSaveImage(filename, Drawimg);
    }

    int iter;

    for( iter = 0; iter < max_iter; iter++)
    {
        printf("%d ", iter);
        //check the current shape
        if(!AAM_IC::IsShapeWithinImage(__current_s, image->width, image->height))
        {
            fprintf(stderr, "ERROR(%s, %d): Shape out of image\n",
                    __FILE__, __LINE__);
            return iter;
        }

        //warp image to template image A0
        __paw.FasterGetWarpTextureFromMatShape(__current_s, image, __warp_t, true);
        AAM_TDM::AlignTextureToRef(A0, __warp_t);

        //calculate error image
        cvSub(__warp_t, A0, __error_t);

        if(showprocess)
        {
            cvCopy(image, Drawimg);
            Draw(Drawimg, 2);
            sprintf(filename, "result/Iter-%02d.jpg", iter+1);
            cvSaveImage(filename, Drawimg);
        }

        //check for texture divergence
        e2 = cvNorm(__error_t);
        if(e2 < 0.01 || (iter>max_iter/3&&fabs(e2-e1)<0.01*e1)) break;
        e1 = e2;

        //1. calculate dot product of modified steepest descent images
        //   with error image
        //2. calculate delta q and delta p by multiplying by inverse Hessian.
        //In summary: we calculate parameters update
        cvGEMM(__error_t, __G, 1, NULL, 1, __delta_pq, CV_GEMM_B_T);

        //apply inverse compositional algorithm to update parameters
        InverseCompose(__delta_pq, __current_s, __update_s);
        //smooth shape
        cvAddWeighted(__current_s, 0.4, __update_s, 0.6, 0, __update_s);
        //update parameters
        __shape.CalcParams(__update_s, __search_pq);
        //calculate constrained new shape
        __shape.CalcShape(__search_pq, __update_s);

        //check for shape convergence
        cvSub(__current_s, __update_s, __delta_s);
        if(/*cvNorm(__delta_s)<0.01*/cvNorm(__delta_s, 0, CV_C) < 0.25)	break;
        else cvCopy(__update_s, __current_s);
    }

    Shape.Mat2Point(__current_s);

    t = ((double)cvGetTickCount()-t)/(cvGetTickFrequency()*1000.);
    printf("AAM IC Fitting time cost %.3f millisec\n", t);

    cvReleaseImage(&Drawimg);

    return iter;
}

//============================================================================
void AAM_IC::SetAllParamsZero()
{
    cvZero(__warp_t);
    cvZero(__error_t);
    cvZero(__search_pq);
    cvZero(__delta_pq);
    cvZero(__lamda);
}

//============================================================================
void AAM_IC::InverseCompose(const CvMat* dpq, const CvMat* s, CvMat* NewS)
{
    // Firstly: Estimate the corresponding changes to the base mesh
    cvConvertScale(dpq, __inv_pq, -1);
    __shape.CalcShape(__inv_pq, __update_s0);	// __update_s0 = N.W(s0, -delta_p, -delta_q)

    //Secondly: Composing the Incremental Warp with the Current Warp Estimate.
    double *S0 = __update_s0->data.db;
    double *S = s->data.db;
    double *SEst = NewS->data.db;
    double x, y, xw, yw;
    int k;
    double alpha, belta, gamma;
    int v1, v2, v3;
    for(int i = 0; i < __shape.nPoints(); i++)
    {
        x = 0.0;
        y = 0.0;
        k = 0;
        //The only problem with this approach is which triangle do we use?
        //In general there will be several triangles that share the ith vertex.
        for(int j = 0; j < __paw.nTri(); j++)
        {
            if(__paw.vTri(i, j) > 0)
            {
                // see Figure (11)
                v1 = __paw.Tri(j, 0);
                v2 = __paw.Tri(j, 1);
                v3 = __paw.Tri(j, 2);
                AAM_PAW::CalcWarpParameters(S0[2*i],S0[2*i+1], __sMean[v1].x, __sMean[v1].y,
                                            __sMean[v2].x, __sMean[v2].y, __sMean[v3].x, __sMean[v3].y,
                                            alpha, belta, gamma);

                xw = alpha*S[2*v1] + belta*S[2*v2] + gamma*S[2*v3];
                yw = alpha*S[2*v1+1] + belta*S[2*v2+1] + gamma*S[2*v3+1];
                x += xw;
                y += yw;
                k++;
            }
        }
        // average the result so as to smooth the warp at each vertex
        SEst[2*i] = x/k;
        SEst[2*i+1] = y/k;
    }
}

//============================================================================
void AAM_IC::CalcAppearanceVariation(const CvMat* error_t, CvMat* lamda)
{
    cvGEMM(error_t, __texture.GetBases(), 1, NULL, 1, lamda, CV_GEMM_B_T);
}

void AAM_IC::Draw(IplImage* image, int type)
{
    if(type == 0) DrawPoint(image);
    else if(type == 1) DrawTriangle(image);
    else if(type == 2)
    {
        CalcAppearanceVariation(__error_t, __lamda);
        __texture.CalcTexture(__lamda, __warp_t);
        DrawAppearance(image);
    }
    else ;
}

//============================================================================
void AAM_IC::DrawPoint(IplImage* image)
{
    double* p = __current_s->data.db;
    for(int i = 0; i < __shape.nPoints(); i++)
    {
        cvCircle(image, cvPoint(p[2*i], p[2*i+1]), 1, CV_RGB(0, 255, 0),1);
    }
}

//============================================================================
void AAM_IC::DrawTriangle(IplImage* image)
{
    double* p = __current_s->data.db;
    int idx1, idx2, idx3;
    for(int i = 0; i < __paw.nTri(); i++)
    {
        idx1 = __paw.__tri[i][0];
        idx2 = __paw.__tri[i][1];
        idx3 = __paw.__tri[i][2];
        cvLine(image, cvPoint(p[2*idx1], p[2*idx1+1]), cvPoint(p[2*idx2], p[2*idx2+1]),
               CV_RGB(128,255,0),2);
        cvLine(image, cvPoint(p[2*idx2], p[2*idx2+1]), cvPoint(p[2*idx3], p[2*idx3+1]),
               CV_RGB(128,255,0),2);
        cvLine(image, cvPoint(p[2*idx3], p[2*idx3+1]), cvPoint(p[2*idx1], p[2*idx1+1]),
               CV_RGB(128,255,0),2);
    }
}

//============================================================================
void AAM_IC::DrawAppearance(IplImage* image)
{
    AAM_Shape Shape;
    Shape.Mat2Point(__current_s);
    AAM_PAW paw;
    paw.Train(Shape, __Points, __Storage, __paw.GetTri(), false);
    int x1, x2, y1, y2, idx1, idx2;
    int xby3, idxby3;
    int minx, miny, maxx, maxy;
    int tri_idx, v1, v2, v3;
    AAM_Shape refShape = __sMean;
    refShape.Translate(-refShape.MinX(), -refShape.MinY());
    double minV, maxV;
    cvMinMaxLoc(__warp_t, &minV, &maxV);
    cvConvertScale(__warp_t, __warp_t, 255/(maxV-minV), -minV*255/(maxV-minV));
    byte* pimg;
    double* fastt = __warp_t->data.db;

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
                tri_idx = paw.__pixTri[idx1];
                v1 = paw.__tri[tri_idx][0];
                v2 = paw.__tri[tri_idx][1];
                v3 = paw.__tri[tri_idx][2];

                x2 = paw.__alpha[idx1]*refShape[v1].x + paw.__belta[idx1]*refShape[v2].x +
                     paw.__gamma[idx1]*refShape[v3].x;
                y2 = paw.__alpha[idx1]*refShape[v1].y + paw.__belta[idx1]*refShape[v2].y +
                     paw.__gamma[idx1]*refShape[v3].y;

                xby3 = 3*x;
                idx2 = __paw.__rect[y2][x2];
                idxby3 = 3*idx2;
                pimg[xby3] = fastt[idxby3];
                pimg[xby3+1] = fastt[idxby3+1];
                pimg[xby3+2] = fastt[idxby3+2];
            }
        }
    }
}


//============================================================================
bool AAM_IC::IsShapeWithinImage(const CvMat* s, int w, int h)
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

//============================================================================
void AAM_IC::Write(std::ofstream& os)
{
    printf("Writing the AAM-IC Model to file...");

    __shape.Write(os);
    __texture.Write(os);
    __paw.Write(os);

    __sMean.Write(os);
    __sStar1.Write(os);
    __sStar2.Write(os);
    __sStar3.Write(os);
    __sStar4.Write(os);

    os << __G << std::endl;

    printf("Done\n\n");
}

//============================================================================
void AAM_IC::Read(std::ifstream& is)
{
    printf("Reading the AAM-IC Model from file...");

    __shape.Read(is);
    __texture.Read(is);
    __paw.Read(is);

    int nPoints = __shape.nPoints();
    __sMean.resize(nPoints);
    __sStar1.resize(nPoints);
    __sStar2.resize(nPoints);
    __sStar3.resize(nPoints);
    __sStar4.resize(nPoints);
    __sMean.Read(is);
    __sStar1.Read(is);
    __sStar2.Read(is);
    __sStar3.Read(is);
    __sStar4.Read(is);

    __G = cvCreateMat(__shape.nModes()+4, __texture.nPixels(), CV_64FC1);
    is >> __G;

    //alocate memory for on-line fitting stuff
    __Points = cvCreateMat (1, __shape.nPoints(), CV_32FC2);
    __Storage = cvCreateMemStorage(0);

    __update_s0 = cvCreateMat(1, __shape.nPoints()*2, CV_64FC1);
    __delta_s = cvCreateMat(1, __shape.nPoints()*2, CV_64FC1);
    __inv_pq = cvCreateMat(1, __shape.nModes()+4, CV_64FC1);
    __warp_t = cvCreateMat(1, __texture.nPixels(), CV_64FC1);
    __error_t = cvCreateMat(1, __texture.nPixels(), CV_64FC1);
    __search_pq = cvCreateMat(1, __shape.nModes()+4, CV_64FC1);
    __delta_pq = cvCreateMat(1, __shape.nModes()+4, CV_64FC1);
    __current_s = cvCreateMat(1, __shape.nPoints()*2, CV_64FC1);
    __update_s = cvCreateMat(1, __shape.nPoints()*2, CV_64FC1);
    __lamda  = cvCreateMat(1, __texture.nModes(), CV_64FC1);

    printf("Done\n\n");
}
