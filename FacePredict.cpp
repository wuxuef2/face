///****************************************************************************
//*
//* Copyright (c) 2012 by Li Ying, all rights reserved.
//*
//* Author:      	Li Ying
//* Contact:     	liyingchocolate@gmail.com
//*
//* This software is partly based on the following open source:
//*
//*		- OpenCV , AAMLibrary
//*
//****************************************************************************/

#include "FacePredict.h"
//#include <direct.h>


#define free2dvector(vec)										\
{																\
	for(int i = 0; i < vec.size(); i++) vec[i].clear();			\
	vec.clear();												\
}

/* get reference to pixel at (col,row),
   for multi-channel images (col) should be multiplied by number of channels */
#define CV_IMAGE_ELEM( image, elemtype, row, col )       \
    (((elemtype*)((image)->imageData + (image)->widthStep*(row)))[(col)])

FacePredict::FacePredict()
{
    __MeanS = 0;
    __MeanT = 0;
    __ShapeParamGroups = 0;
    __TextureParamGroups = 0;
}

FacePredict::~FacePredict()
{
    cvReleaseMat(&__MeanS);
    cvReleaseMat(&__MeanT);
    cvReleaseMat(&__ShapeParamGroups);
    cvReleaseMat(&__TextureParamGroups);
}

void FacePredict::Train(const std::vector<AAM_Shape> &AllShapes,
                        const std::vector<IplImage*> &AllImages,
                        const int ng_samples[],
                        //CvMat* AllTextures,
                        double shape_percentage, /* = 0.95 */
                        double texture_percentage /* = 0.95 */)
{
    if(AllShapes.size() != AllImages.size())
    {
        //fprintf(stderr, "ERROE(%s, %d): #Shapes != #Images\n", __FILE__, __LINE__);
        //exit(0);
        AgingException ex(3);
        throw ex;
    }

    CvMat* Points;
    CvMemStorage* Storage;

    //construct the pivot space of shape and texutre seperately
    std::vector<AAM_Shape> AllAlignedShapes;
    __shape.Train(AllShapes, AllAlignedShapes, 0.95);
    Points = cvCreateMat (1, __shape.nPoints(), CV_32FC2);
    Storage = cvCreateMemStorage(0);
    __paw.Train(__shape.GetAAMReferenceShape(), Points, Storage);

    int nSamples = AllShapes.size();
    int nPointsby2 = __shape.nPoints() * 2;
    __nShapeModes = __shape.nModes();
    int nPixelsby3 = __paw.nPix() * 3;

    CvMat* AllTextures = cvCreateMat(nSamples, nPixelsby3, CV_64FC1);
    __texture.Train(AllShapes, __paw, AllImages, AllTextures, 0.95, false);  //if true, save the images

    __nTextureModes = __texture.nModes();


    __MeanS = cvCreateMat(1, nPointsby2, CV_64FC1);
    __MeanT = cvCreateMat(1, nPixelsby3, CV_64FC1);
    cvCopy(__shape.GetMean(), __MeanS);
    cvCopy(__texture.GetMean(), __MeanT);
    __AAMRefShape.Mat2Point(__MeanS);  //center at (0, 0)

    //calculate the mean parameters of all shapes and textures in each age group
    for (int i = 0; i < AGE_AREA; i++)
    {
        __nGSamples[i] = ng_samples[i];
    }

    FacePredict::CalcClassicParams(AllAlignedShapes, AllTextures);
}

IplImage* FacePredict::predict(const AAM_Shape& shape, const IplImage& curImage,
                               int curAgeG, int newAgeG, bool save)

{
    if (newAgeG > NGROUPS || curAgeG > NGROUPS)
    {
        //fprintf(stderr, "ERROE(%s, %d): Age group larger than %d\n", __FILE__, __LINE__, NGROUPS);
        //exit(0);
        AgingException ex(6);
        throw ex;
    }

    if(curImage.nChannels != 3 || curImage.depth != 8)
    {
        //fprintf(stderr, "ERROR(%s: %d): The image channels must be 3, and the depth must be 8!\n", __FILE__, __LINE__);
        //exit(0);
        AgingException ex(6);
        throw ex;
    }

    //get the current shape parameters
    AAM_Shape curShape = shape;
    curShape.Centralize();
    double thisfacewidth = curShape.GetWidth();
    if(stdwidth < thisfacewidth)
        curShape.Scale(stdwidth / thisfacewidth);
    curShape.AlignTo(__AAMRefShape);

    CvMat* p = cvCreateMat(1, __nShapeModes, CV_64FC1);
    CvMat* pq = cvCreateMat(1, 4+__nShapeModes, CV_64FC1);
    __shape.CalcParams(curShape, pq);
    cvGetCols(pq, p, 4, 4+__nShapeModes);

    //get the current texture parameters
    CvMat* curTexture = cvCreateMat(1, __paw.nPix() * 3, CV_64FC1);
    __paw.FasterGetWarpTextureFromShape(shape, &curImage, curTexture, false);
    /*IplImage *meanImg = cvCreateImage(cvGetSize(&curImage), curImage.depth, curImage.nChannels);
    __paw.GetWarpImageFromVector(meanImg,curTexture);
    cvShowImage("org Texture", meanImg);*/
    __texture.AlignTextureToRef(__MeanT, curTexture);
    CvMat* lamda = cvCreateMat(1, __nTextureModes, CV_64FC1);
    __texture.CalcParams(curTexture, lamda);

    //caculate new shape and texture parameters
    CvMat newShapeParams;
    CvMat* newpq = cvCreateMat(1, 4+__nShapeModes, CV_64FC1);
    cvmSet(newpq, 0, 0, cvmGet(pq, 0, 0));
    cvmSet(newpq, 0, 1, cvmGet(pq, 0, 1));
    cvmSet(newpq, 0, 2, cvmGet(pq, 0, 2));
    cvmSet(newpq, 0, 3, cvmGet(pq, 0, 3));
    cvGetCols(newpq, &newShapeParams, 4, 4+__nShapeModes);
    FacePredict::CalcNewShapeParams(p, &newShapeParams, curAgeG, newAgeG);

    CvMat* newTextureParams = cvCreateMat(1, __nTextureModes, CV_64FC1);
    FacePredict::CalcNewTextureParams(lamda, newTextureParams, curAgeG, newAgeG) ;

    //calculate the new shape and texture
    AAM_Shape newShape;
    __shape.CalcShape(newpq, newShape);
    /*CvSize newsize;
    AAM_Shape temNS = newShape;
    temNS.Translate(-temNS.MinX(), -temNS.MinY());
    newsize.width = 128;
    newsize.height = 128;
    IplImage *shapeNImg = cvCreateImage(newsize, curImage.depth, curImage.nChannels);
    cvSet(shapeNImg, CV_RGB(0,0,0));
    temNS.Sketch(shapeNImg);
    cvShowImage("new shape", shapeNImg);
    cvReleaseImage(&shapeNImg);*/

    CvMat* newTexture = cvCreateMat(1, __paw.nPix() * 3, CV_64FC1);
    __texture.CalcTexture(newTextureParams, newTexture);
    /*IplImage *meanNImg = cvCreateImage(cvGetSize(&curImage), curImage.depth, curImage.nChannels);
    __paw.GetWarpImageFromVector(meanNImg,newTexture);
    cvShowImage("Texture", meanNImg);*/

    //systhetize the shape and texture

    IplImage* newImage = cvCreateImage(cvSize(stdwidth, stdwidth / newShape.GetWidth() * newShape.GetHeight()), IPL_DEPTH_8U, 3);
    FacePredict::FaceSynthesis(newShape, newTexture, newImage);

    if(save)
        cvSaveImage("facial prediction.jpg", newImage);

    cvReleaseMat(&p);
    cvReleaseMat(&pq);
    cvReleaseMat(&curTexture);
    cvReleaseMat(&lamda);
    cvReleaseMat(&newTextureParams);
    cvReleaseMat(&newpq);
    cvReleaseMat(&newTexture);

    return newImage;
}

IplImage* FacePredict::predict(const AAM_Shape& Shape, const IplImage& curImage,
                               const AAM_Shape& ShapeF, const IplImage& ImageF, double RatioF,
                               const AAM_Shape& ShapeM, const IplImage& ImageM, double RatioM,
                               int curAgeG, int newAgeG, bool save)
{
    if (newAgeG > NGROUPS || curAgeG > NGROUPS)
    {
        //fprintf(stderr, "ERROE(%s, %d): Age group larger than %d\n", __FILE__, __LINE__, NGROUPS);
        //exit(0);
        AgingException ex(6);
        throw ex;
    }

    if(curImage.nChannels != 3 || curImage.depth != 8)
    {
        //fprintf(stderr, "ERROR(%s: %d): The image channels must be 3, and the depth must be 8!\n", __FILE__, __LINE__);
        //exit(0);
        AgingException ex(7);
        throw ex;
    }


    /*get the current shape parameters*/
    AAM_Shape curShape = Shape;
    curShape.Centralize();
    double thisfacewidth = curShape.GetWidth();
    if(stdwidth < thisfacewidth)
        curShape.Scale(stdwidth / thisfacewidth);
    curShape.AlignTo(__AAMRefShape);

    CvMat* p = cvCreateMat(1, __nShapeModes, CV_64FC1);
    CvMat* pq = cvCreateMat(1, 4+__nShapeModes, CV_64FC1);
    __shape.CalcParams(curShape, pq);
    cvGetCols(pq, p, 4, 4+__nShapeModes);

    /*get the current texture parameters*/
    CvMat* curTexture = cvCreateMat(1, __paw.nPix() * 3, CV_64FC1);
    __paw.FasterGetWarpTextureFromShape(Shape, &curImage, curTexture, false);
    __texture.AlignTextureToRef(__MeanT, curTexture);
    CvMat* lamda = cvCreateMat(1, __nTextureModes, CV_64FC1);
    __texture.CalcParams(curTexture, lamda);


    //father
    CvMat* pF = cvCreateMat(1, __nShapeModes, CV_64FC1);
    CvMat* lamdaF = cvCreateMat(1, __nTextureModes, CV_64FC1);
    if (RatioF == 0)
    {
        cvZero(pF);
        cvZero(lamdaF);
    }
    else
    {
        AAM_Shape shapeF = ShapeF;
        shapeF.Centralize();
        thisfacewidth = ShapeF.GetWidth();
        if(stdwidth < thisfacewidth)
            shapeF.Scale(stdwidth / thisfacewidth);
        shapeF.AlignTo(__AAMRefShape);
        CvMat* pqF = cvCreateMat(1, 4+__nShapeModes, CV_64FC1);
        __shape.CalcParams(shapeF, pqF);
        cvGetCols(pqF, pF, 4, 4+__nShapeModes);

        CvMat* TextureF = cvCreateMat(1, __paw.nPix() * 3, CV_64FC1);
        __paw.FasterGetWarpTextureFromShape(ShapeF, &ImageF, TextureF, false);
        __texture.AlignTextureToRef(__MeanT, TextureF);
        __texture.CalcParams(TextureF, lamdaF);
    }


    //mother
    CvMat* pM = cvCreateMat(1, __nShapeModes, CV_64FC1);
    CvMat* lamdaM = cvCreateMat(1, __nTextureModes, CV_64FC1);
    if (RatioM == 0)
    {
        cvZero(pM);
        cvZero(lamdaM);
    }
    else
    {
        AAM_Shape shapeM = ShapeM;
        shapeM.Centralize();
        thisfacewidth = ShapeM.GetWidth();
        if(stdwidth < thisfacewidth)
            shapeM.Scale(stdwidth / thisfacewidth);
        shapeM.AlignTo(__AAMRefShape);
        CvMat* pqM = cvCreateMat(1, 4+__nShapeModes, CV_64FC1);
        __shape.CalcParams(shapeM, pqM);
        cvGetCols(pqM, pM, 4, 4+__nShapeModes);

        CvMat* TextureM = cvCreateMat(1, __paw.nPix() * 3, CV_64FC1);
        __paw.FasterGetWarpTextureFromShape(ShapeM, &ImageM, TextureM, false);
        __texture.AlignTextureToRef(__MeanT, TextureM);
        __texture.CalcParams(TextureM, lamdaM);
    }

    /*caculate new shape and texture parameters*/
    CvMat newShapeParams;
    CvMat* newpq = cvCreateMat(1, 4+__nShapeModes, CV_64FC1);
    cvmSet(newpq, 0, 0, cvmGet(pq, 0, 0));
    cvmSet(newpq, 0, 1, cvmGet(pq, 0, 1));
    cvmSet(newpq, 0, 2, cvmGet(pq, 0, 2));
    cvmSet(newpq, 0, 3, cvmGet(pq, 0, 3));
    cvGetCols(newpq, &newShapeParams, 4, 4+__nShapeModes);
    CvMat* newSP = cvCreateMat(1, __nShapeModes, CV_64FC1);
    FacePredict::CalcNewShapeParams(p, newSP, curAgeG, newAgeG);
    FacePredict::CalcParamsByRatio(newSP, pF, RatioF, pM, RatioM, &newShapeParams);

    CvMat* newTP = cvCreateMat(1, __nTextureModes, CV_64FC1);
    FacePredict::CalcNewTextureParams(lamda, newTP, curAgeG, newAgeG);
    CvMat* newTextureParams = cvCreateMat(1, __nTextureModes, CV_64FC1);
    FacePredict::CalcParamsByRatio(newTP, lamdaF, RatioF, lamdaM, RatioM, newTextureParams);

    /*calculate the new shape and texture*/
    AAM_Shape newShape;
    __shape.CalcShape(newpq, newShape);

    CvMat* newTexture = cvCreateMat(1, __paw.nPix() * 3, CV_64FC1);
    __texture.CalcTexture(newTextureParams, newTexture);

    /*systhetize the shape and texture*/

    IplImage* newImage = cvCreateImage(cvSize(stdwidth, stdwidth / newShape.GetWidth() * newShape.GetHeight()), IPL_DEPTH_8U, 3);
    FacePredict::FaceSynthesis(newShape, newTexture, newImage);

    if(save)
        cvSaveImage("facial prediction.jpg", newImage);

    cvReleaseMat(&p);
    cvReleaseMat(&pq);
    cvReleaseMat(&curTexture);
    cvReleaseMat(&lamda);
    cvReleaseMat(&newTextureParams);
    cvReleaseMat(&newpq);
    cvReleaseMat(&newTexture);

    return newImage;
}

void FacePredict::FaceSynthesis(AAM_Shape &shape, CvMat* texture, IplImage* newImage)
{
    double thisfacewidth = shape.GetWidth();
    shape.Scale(stdwidth / thisfacewidth);
    shape.Translate(-shape.MinX(), -shape.MinY());

    AAM_PAW paw;
    CvMat* points = cvCreateMat (1, __shape.nPoints(), CV_32FC2);
    CvMemStorage* storage = cvCreateMemStorage(0);
    paw.Train(shape, points, storage, __paw.GetTri(), false);  //the actual shape

    __AAMRefShape.Translate(-__AAMRefShape.MinX(), -__AAMRefShape.MinY());  //refShape, central point is at (0,0);translate the min to (0,0)
    double minV, maxV;
    cvMinMaxLoc(texture, &minV, &maxV);
    cvConvertScale(texture, texture, 1/(maxV-minV)*255, -minV*255/(maxV-minV));

    cvZero(newImage);

    int x1, x2, y1, y2, idx1 = 0, idx2 = 0;
    int tri_idx, v1, v2, v3;
    int minx, miny, maxx, maxy;
    minx = shape.MinX();
    miny = shape.MinY();
    maxx = shape.MaxX();
    maxy = shape.MaxY();
    for(int y = miny; y < maxy; y++)
    {
        y1 = y-miny;
        for(int x = minx; x < maxx; x++)
        {
            x1 = x-minx;
            idx1 = paw.Rect(y1, x1);
            if(idx1 >= 0)
            {
                tri_idx = paw.PixTri(idx1);
                v1 = paw.Tri(tri_idx, 0);
                v2 = paw.Tri(tri_idx, 1);
                v3 = paw.Tri(tri_idx, 2);

                x2 = paw.Alpha(idx1)*__AAMRefShape[v1].x + paw.Belta(idx1)*__AAMRefShape[v2].x +
                     paw.Gamma(idx1)*__AAMRefShape[v3].x;
                y2 = paw.Alpha(idx1)*__AAMRefShape[v1].y + paw.Belta(idx1)*__AAMRefShape[v2].y +
                     paw.Gamma(idx1)*__AAMRefShape[v3].y;

                idx2 = __paw.Rect(y2, x2);
                if(idx2 < 0) continue;

                CV_IMAGE_ELEM(newImage, byte, y, 3*x) = cvmGet(texture, 0, 3*idx2);
                CV_IMAGE_ELEM(newImage, byte, y, 3*x+1) = cvmGet(texture, 0, 3*idx2+1);
                CV_IMAGE_ELEM(newImage, byte, y, 3*x+2) = cvmGet(texture, 0, 3*idx2+2);
            }
        }
    }
    cvReleaseMat(&points);
    cvReleaseMemStorage(&storage);
}

void FacePredict::CalcClassicParams(const std::vector<AAM_Shape> &AllAlignedShapes, const CvMat* AllTextures)
{
    int begin = 0, end = 0;
    std::vector<AAM_Shape> GroupShapes;
    CvMat GroupTextures;
    __ShapeParamGroups = cvCreateMat(NGROUPS, __nShapeModes, CV_64FC1);
    __TextureParamGroups = cvCreateMat(NGROUPS, __nTextureModes, CV_64FC1);

    for (int i = 0; i < NGROUPS; i++)
    {
        for (int j = AGE_GROUPS[i][0]; j <= AGE_GROUPS[i][1]; j++)
            end += __nGSamples[j];

        AAM_Shape oneShape;
        for (int j = begin; j < end; j++)
        {
            oneShape = AllAlignedShapes[j];
            GroupShapes.push_back(oneShape);
        }
        cvGetRows(AllTextures, &GroupTextures, begin, end);
        FacePredict::CalcMeanShapeParams(GroupShapes, i);
        FacePredict::CalcMeanTextureParams(&GroupTextures, i);



        //std::string s;
        //char c[10];
        //itoa(i,c,10);
        //std::ofstream alignft;
        //alignft.open("txs_"+ std::string(c) +".txt");
        //alignt++;

        //for(int i = 0; i < end-begin; i++)
        //{
        //	for(int j = 0; j < AllTextures->cols; j++)
        //		alignft << cvmGet(&GroupTextures, i ,j) << " ";
        //	alignft << std::endl;
        //}
        //alignft.close();



        begin = end;
    }
}

void FacePredict::CalcMeanShapeParams(const std::vector<AAM_Shape> &GroupShapes, int group)
{
    int nSamples = GroupShapes.size();

    CvMat mParams;
    cvGetRow(__ShapeParamGroups, &mParams, group);

    CvMat* p = cvCreateMat(1, __nShapeModes, CV_64FC1);
    CvMat* pq = cvCreateMat(1, 4+__nShapeModes, CV_64FC1);
    for (int i = 0; i < nSamples; i++)
    {
        __shape.CalcParams(GroupShapes[i], pq);
        cvGetCols(pq, p, 4, 4+__nShapeModes);
        cvAdd(&mParams, p, &mParams);
    }

    CvMat * size = cvCreateMat(1, __nShapeModes, CV_64FC1);
    for (int i = 0; i < __nShapeModes; i++)
        cvmSet(size, 0, i, nSamples);
    cvDiv(&mParams, size, &mParams);

    cvReleaseMat(&p);
    cvReleaseMat(&pq);
    cvReleaseMat(&size);
}

void FacePredict::CalcMeanTextureParams(const CvMat* GroupTextures, int group)
{
    int nSamples = GroupTextures->rows;

    CvMat mParams;
    cvGetRow(__TextureParamGroups, &mParams, group);  //resize the mParams

    CvMat* lamda = cvCreateMat(1, __nTextureModes, CV_64FC1);
    CvMat* oneTexture = cvCreateMat(1, GroupTextures->cols, CV_64FC1);

    for (int i = 0; i < nSamples; i++)
    {
        cvGetRow(GroupTextures, oneTexture, i);
        __texture.CalcParams(oneTexture, lamda);
        cvAdd(&mParams, lamda, &mParams);
    }
    CvMat * size = cvCreateMat(1, __nTextureModes, CV_64FC1);
    for (int i = 0; i < __nTextureModes; i++)
        cvmSet(size, 0, i, nSamples);
    cvDiv(&mParams, size, &mParams);

    cvReleaseMat(&lamda);
    cvReleaseMat(&size);
    cvReleaseMat(&oneTexture);
}

void FacePredict::CalcNewShapeParams(CvMat* curParam, CvMat* newParam, int curAgeG, int newAgeG)
{
    CvMat* diff = cvCreateMat(1, __nShapeModes, CV_64FC1);
    CvMat* curClassicP = cvCreateMat(1, __nShapeModes, CV_64FC1);
    CvMat* newClassicP = cvCreateMat(1, __nShapeModes, CV_64FC1);
    cvGetRow(__ShapeParamGroups, curClassicP, curAgeG);
    cvGetRow(__ShapeParamGroups, newClassicP, newAgeG);
    cvSub(newClassicP, curClassicP, diff);
    cvAdd(curParam, diff, newParam);

    cvReleaseMat(&diff);
    cvReleaseMat(&curClassicP);
    cvReleaseMat(&newClassicP);
}

void FacePredict::CalcNewTextureParams(CvMat* curParam, CvMat* newParam, int curAgeG, int newAgeG)
{
    CvMat* diff = cvCreateMat(1, __nTextureModes, CV_64FC1);
    CvMat* curClassicP = cvCreateMat(1, __nTextureModes, CV_64FC1);
    CvMat* newClassicP = cvCreateMat(1, __nTextureModes, CV_64FC1);
    cvGetRow(__TextureParamGroups, curClassicP, curAgeG);
    cvGetRow(__TextureParamGroups, newClassicP, newAgeG);
    cvSub(newClassicP, curClassicP, diff);
    cvAdd(curParam, diff, newParam);

    cvReleaseMat(&diff);
    cvReleaseMat(&curClassicP);
    cvReleaseMat(&newClassicP);
}

void FacePredict::CalcParamsByRatio(CvMat* curParam, CvMat* ParamF, double RatioF, CvMat* ParamM, double RatioM, CvMat* newParam)
{
    cvAddWeighted(ParamF, RatioF, ParamM, RatioM, 0, newParam);
    cvAddWeighted(newParam, 1, curParam, 1 - RatioF - RatioM, 0, newParam);
}

int FacePredict::AgeGroup(int age)
{
    int group = 0;
    for (int i = 0; i < NGROUPS; i++, group++)
    {
        if (age <= AGE_GROUPS[i][1])
            break;
    }
    return group;
}

void FacePredict::Read(std::ifstream& is)
{
    __shape.Read(is);
    __ShapeParamGroups = cvCreateMat(NGROUPS, __shape.nModes(), CV_64FC1);
    is >> __ShapeParamGroups;
    __texture.Read(is);
    __TextureParamGroups = cvCreateMat(NGROUPS, __texture.nModes(), CV_64FC1);
    is >> __TextureParamGroups;
    __paw.Read(is);

    __nShapeModes = __shape.nModes();
    __nTextureModes = __texture.nModes();
    __MeanS = cvCreateMat(1, __shape.nPoints() * 2, CV_64FC1);
    __MeanT = cvCreateMat(1, __paw.nPix() * 3, CV_64FC1);
    cvCopy(__shape.GetMean(), __MeanS);
    cvCopy(__texture.GetMean(), __MeanT);
    __AAMRefShape.Mat2Point(__MeanS);
}


void FacePredict::Write(std::ofstream& os)
{
    __shape.Write(os);
    os << __ShapeParamGroups << std::endl;
    __texture.Write(os);
    os << __TextureParamGroups << std::endl;
    __paw.Write(os);
}
