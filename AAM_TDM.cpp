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

#include "AAM_TDM.h"
#include "AAM_PAW.h"
//#include <direct.h>
//#include <io.h>


AAM_TDM::AAM_TDM()
{
    __MeanTexture = 0;
    __TextureEigenVectors = 0;
    __TextureEigenValues = 0;
}

AAM_TDM::~AAM_TDM()
{
    cvReleaseMat(&__MeanTexture);
    cvReleaseMat(&__TextureEigenVectors);
    cvReleaseMat(&__TextureEigenValues);
}

//============================================================================
void AAM_TDM::Train(const std::vector<AAM_Shape> &AllShapes,
                    const AAM_PAW& m_warp,
                    const std::vector<IplImage*> &AllImages,
                    double percentage,
                    bool registration)
{
    if(AllShapes.size() != AllImages.size())
    {
        fprintf(stderr, "ERROR(%s, %d): #Shapes != #Images\n",
                __FILE__, __LINE__);
        exit(0);
    }

    printf("################################################\n");
    printf("Build Texture Distribution Model...\n");

    int nPoints = m_warp.nPoints();
    int nPixels = m_warp.nPix()*3;
    int nSamples = AllShapes.size();

    CvMat *AllTextures = cvCreateMat(nSamples, nPixels, CV_64FC1);

    printf("Calclating texture vectors...\n");
    for(int i = 0; i < nSamples; i++)
    {
        CvMat oneTexture;
        cvGetRow(AllTextures, &oneTexture, i);
        m_warp.FasterGetWarpTextureFromShape(AllShapes[i], AllImages[i],
                                             &oneTexture, true);
    }

    // align texture so as to minimize the lighting variation
    printf("Align textures to minimize the lighting variation ...\n");
    AAM_TDM::AlignTextures(AllTextures);

    //now do pca
    DoPCA(AllTextures, percentage);

    if(registration) SaveSeriesTemplate(AllTextures, m_warp);

    cvReleaseMat(&AllTextures);
    printf("################################################\n\n");
}

//additonal, overloading function
void AAM_TDM::Train(const std::vector<AAM_Shape> &AllShapes,
                    const AAM_PAW& m_warp,
                    const std::vector<IplImage*> &AllImages,
                    CvMat* AllTextures,
                    double percentage,
                    bool registration)
{
    if(AllShapes.size() != AllImages.size())
    {
        fprintf(stderr, "ERROR(%s, %d): #Shapes != #Images\n",
                __FILE__, __LINE__);
        exit(0);
    }

    printf("################################################\n");
    printf("Build Texture Distribution Model...\n");

    int nPoints = m_warp.nPoints();
    int nPixels = m_warp.nPix()*3;
    int nSamples = AllShapes.size();

    printf("Calclating texture vectors...\n");
    for(int i = 0; i < nSamples; i++)
    {
        CvMat oneTexture;
        cvGetRow(AllTextures, &oneTexture, i);
        m_warp.FasterGetWarpTextureFromShape(AllShapes[i], AllImages[i],
                                             &oneTexture, true);  //true, means normalization
    }

    // align texture so as to minimize the lighting variation
    printf("Align textures to minimize the lighting variation ...\n");
    AAM_TDM::AlignTextures(AllTextures);

    //now do pca
    DoPCA(AllTextures, percentage);

    if(registration) SaveSeriesTemplate(AllTextures, m_warp);

    printf("################################################\n\n");
}

//============================================================================
void AAM_TDM::DoPCA(const CvMat* AllTextures, double percentage)
{
    printf("Doing PCA of textures datas...");

    int nSamples = AllTextures->rows;
    int nPixels = AllTextures->cols;
    int nEigenAtMost = MIN(nSamples, nPixels);

    CvMat* tmpEigenValues = cvCreateMat(1, nEigenAtMost, CV_64FC1);
    CvMat* tmpEigenVectors = cvCreateMat(nEigenAtMost, nPixels, CV_64FC1);
    __MeanTexture = cvCreateMat(1, nPixels, CV_64FC1 );

    cvCalcPCA(AllTextures, __MeanTexture,
              tmpEigenValues, tmpEigenVectors, CV_PCA_DATA_AS_ROW);
    double allSum = cvSum(tmpEigenValues).val[0];
    double partSum = 0.0;
    int nTruncated = 0;
    double largesteigval = cvmGet(tmpEigenValues, 0, 0);
    for(int i = 0; i < nEigenAtMost; i++)
    {
        double thiseigval = cvmGet(tmpEigenValues, 0, i);
        if(thiseigval / largesteigval < 0.0001) break; // firstly check(remove small values)
        partSum += thiseigval;
        ++ nTruncated;
        if(partSum/allSum >= percentage)	break;    //secondly check
    }

    __TextureEigenValues = cvCreateMat(1, nTruncated, CV_64FC1);
    __TextureEigenVectors = cvCreateMat(nTruncated, nPixels, CV_64FC1);

    CvMat G;
    cvGetCols(tmpEigenValues, &G, 0, nTruncated);
    cvCopy(&G, __TextureEigenValues);

    cvGetRows(tmpEigenVectors, &G, 0, nTruncated);
    cvCopy(&G, __TextureEigenVectors);

    cvReleaseMat(&tmpEigenVectors);
    cvReleaseMat(&tmpEigenValues);

    printf("Done (%d/%d)\n", nTruncated, nEigenAtMost);
}

//============================================================================
void AAM_TDM::CalcTexture(const CvMat* lamda, CvMat* texture)
{
    cvBackProjectPCA(lamda, __MeanTexture, __TextureEigenVectors, texture);  //texture, the inverse-projected result
}

//============================================================================
void AAM_TDM::CalcParams(const CvMat* texture, CvMat* lamda)
{
    cvProjectPCA(texture, __MeanTexture, __TextureEigenVectors, lamda);  //lamda, the projected result
}

void AAM_TDM::Clamp(CvMat* lamda, double s_d /* = 3.0 */)
{
    double* fastp = lamda->data.db;
    double* fastv = __TextureEigenValues->data.db;
    int nmodes = nModes();
    double limit;

    for(int i = 0; i < nmodes; i++)
    {
        limit = s_d*sqrt(fastv[i]);
        if(fastp[i] > limit) fastp[i] = limit;
        else if(fastp[i] < -limit) fastp[i] = -limit;
    }
}

//============================================================================
void AAM_TDM::AlignTextures(CvMat* AllTextures)
{
    int nsamples = AllTextures->rows;
    int npixels = AllTextures->cols;
    CvMat* meanTexture = cvCreateMat(1, npixels, CV_64FC1);
    CvMat* newmeanTexture = cvCreateMat(1, npixels, CV_64FC1);
    CvMat* refTexture = cvCreateMat(1, npixels, CV_64FC1);
    CvMat ti;

    // calculate the mean texture
    AAM_TDM::CalcMeanTexture(AllTextures, meanTexture);
    AAM_TDM::ZeroMeanUnitLength(meanTexture);
    //cvNormalize(meanTexture, meanTexture);

    // We choose an initial estimate
    cvCopy(meanTexture, refTexture);

    // do a number of alignment iterations until convergence
    double diff, diff_max = 0.001;
    const int max_iter = 30;
    for(int iter = 0; iter < max_iter; iter++)
    {
        //align all textures to the mean texture estimate
        for(int i = 0; i < nsamples; i++)
        {
            cvGetRow(AllTextures, &ti, i);
            AAM_TDM::AlignTextureToRef(refTexture, &ti);
        }

        //estimate new mean texture
        AAM_TDM::CalcMeanTexture(AllTextures, newmeanTexture);
        AAM_TDM::ZeroMeanUnitLength(newmeanTexture);
        //cvNormalize(newmeanTexture, newmeanTexture);

        diff = cvNorm(refTexture, newmeanTexture, CV_RELATIVE_L2);

        printf("Alignment iteration #%i, mean texture est. diff. = %g\n", iter, diff );

        if(diff <= diff_max) break; //converged

        //if not converged, come on iterations
        cvCopy(newmeanTexture, refTexture);
    }

    cvReleaseMat(&meanTexture);
    cvReleaseMat(&newmeanTexture);
    cvReleaseMat(&refTexture);
}

//============================================================================
void AAM_TDM::CalcMeanTexture(const CvMat* AllTextures, CvMat* meanTexture)
{
    CvMat submat;
    for(int i = 0; i < meanTexture->cols; i++)
    {
        cvGetCol(AllTextures, &submat, i);
        cvmSet(meanTexture, 0, i, cvAvg(&submat).val[0]);
    }
}

//============================================================================
void AAM_TDM::AlignTextureToRef(const CvMat* refTextrure, CvMat* Texture)
{
    double alpha, belta;

    alpha = cvDotProduct(refTextrure, Texture);
    belta = cvSum(Texture).val[0] / Texture->cols;

    cvConvertScale(Texture, Texture, 1.0/alpha, -belta/alpha);
}

//============================================================================
void AAM_TDM::ZeroMeanUnitLength(CvMat* Texture)
{
    CvScalar mean =  cvAvg(Texture);
    cvSubS(Texture, mean, Texture);
    double norm = cvNorm(Texture);
    cvConvertScale(Texture, Texture, 1.0/norm);
}

//============================================================================
void AAM_TDM::SaveSeriesTemplate(const CvMat* AllTextures, const AAM_PAW& m_warp)
{
    printf("Saving the face template image...");
    if(ACCESS("registration", 0))	MKDIR("registration");
    if(ACCESS("Modes", 0))	MKDIR("Modes");
    if(ACCESS("Tri", 0))	MKDIR("Tri");
    char filename[100];

    for(int i = 0; i < AllTextures->rows; i++)
    {
        CvMat oneTexture;
        cvGetRow(AllTextures, &oneTexture, i);
        sprintf(filename, "registration/%03i.jpg", i);
        m_warp.SaveWarpImageFromVector(filename, &oneTexture);
    }

    for(int nmodes = 0; nmodes < nModes(); nmodes++)
    {
        CvMat oneVar;
        cvGetRow(__TextureEigenVectors, &oneVar, nmodes);

        sprintf(filename, "Modes/A%03i.jpg", nmodes+1);
        m_warp.SaveWarpImageFromVector(filename, &oneVar);
    }

    IplImage* templateimg = cvCreateImage
                            (cvSize(m_warp.Width(), m_warp.Height()), IPL_DEPTH_8U, 3);
    IplImage* convexImage = cvCreateImage
                            (cvSize(m_warp.Width(), m_warp.Height()), IPL_DEPTH_8U, 3);
    IplImage* TriImage = cvCreateImage
                         (cvSize(m_warp.Width(), m_warp.Height()), IPL_DEPTH_8U, 3);

    m_warp.GetWarpImageFromVector(templateimg, __MeanTexture);
    cvSaveImage("Modes/Template.jpg", templateimg);
    m_warp.SaveWarpImageFromVector("Modes/A00.jpg", __MeanTexture);

    cvZero(convexImage);

    for( int i = 0; i < m_warp.nTri(); i++)
    {
        CvPoint p, q;
        int ind1, ind2;

        cvCopy(templateimg, TriImage);

        ind1 = m_warp.Tri(i, 0);
        ind2 = m_warp.Tri(i, 1);
        p = cvPointFrom32f(m_warp.Vertex(ind1));
        q = cvPointFrom32f(m_warp.Vertex(ind2));
        cvLine(TriImage, p, q, CV_RGB(255, 255, 255));
        cvLine(convexImage, p, q, CV_RGB(255, 255, 255));

        ind1 = m_warp.Tri(i, 1);
        ind2 = m_warp.Tri(i, 2);
        p = cvPointFrom32f(m_warp.Vertex(ind1));
        q = cvPointFrom32f(m_warp.Vertex(ind2));
        cvLine(TriImage, p, q, CV_RGB(255, 255, 255));
        cvLine(convexImage, p, q, CV_RGB(255, 255, 255));

        ind1 = m_warp.Tri(i, 2);
        ind2 = m_warp.Tri(i, 0);
        p = cvPointFrom32f(m_warp.Vertex(ind1));
        q = cvPointFrom32f(m_warp.Vertex(ind2));
        cvLine(TriImage, p, q, CV_RGB(255, 255, 255));
        cvLine(convexImage, p, q, CV_RGB(255, 255, 255));

        sprintf(filename, "Tri/%03i.jpg", i);
        cvSaveImage(filename, TriImage);
    }
    cvSaveImage("Tri/convex.jpg", convexImage);

    cvReleaseImage(&templateimg);
    cvReleaseImage(&convexImage);
    cvReleaseImage(&TriImage);
    printf("Done\n");
}

//============================================================================
void AAM_TDM::Write(std::ofstream& os)
{
    os << nPixels() << " " << nModes() << std::endl;
    os << __MeanTexture << std::endl;
    os << __TextureEigenValues << std::endl;
    os << __TextureEigenVectors << std::endl;
    os << std::endl;
}

//============================================================================
void AAM_TDM::Read(std::ifstream& is)
{
    int _npixels, _nModes;
    is >> _npixels >> _nModes;

    __MeanTexture = cvCreateMat(1, _npixels, CV_64FC1);
    __TextureEigenValues = cvCreateMat(1, _nModes, CV_64FC1);
    __TextureEigenVectors = cvCreateMat(_nModes, _npixels, CV_64FC1);

    is >> __MeanTexture;
    is >> __TextureEigenValues;
    is >> __TextureEigenVectors;
}
