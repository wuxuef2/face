/*
 * AgeEstimation.cpp
 *
 *  Created on: Mar 22, 2015
 *      Author: wuxuef
 */


#include "AgeEstimation.h"
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

AgeEstimation::AgeEstimation()
{
    __MeanS = 0;
    __MeanT = 0;
    __ShapeParamGroups = 0;
    __TextureParamGroups = 0;
    SVMParam = std::string("SVM_AAM.xml");
}

AgeEstimation::~AgeEstimation()
{
    cvReleaseMat(&__MeanS);
    cvReleaseMat(&__MeanT);
    cvReleaseMat(&__ShapeParamGroups);
    cvReleaseMat(&__TextureParamGroups);
}


void printMat(CvMat mat) {
	for (int i = 0; i < mat.rows; i++) {
		for (int j = 0; j < mat.cols; j++) {
			printf("%lf,", cvmGet(&mat, i, j));
		}
		printf("\n___________________________________________________________\n");
	}
}

void AgeEstimation::train(const std::vector<AAM_Shape> &AllShapes,
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

    int sampleNumbers = 0;

    for (int i = 0; i < AGE_AREA; i++) {
    	sampleNumbers += ng_samples[i];
    }

	AAM_Shape oneShape;
    int pointsNumber = 68;
    float* points = new float[sampleNumbers * 2 * pointsNumber];
    float* labs = new float[sampleNumbers];
    float* begin = points;
    int counter = 0;
    for (int i = 0; i < AGE_AREA; i++) {
    	for (int j = 0; j < ng_samples[i]; j++) {
			AAM_Shape oneShape = AllAlignedShapes[counter];
			oneShape.shap2Array(begin);
			begin += 2 * pointsNumber;
			labs[counter] = i;
			counter++;
    	}
    }
    CvMat trainingDataMat = cvMat(sampleNumbers, 2 * pointsNumber, CV_32FC1, points);
    CvMat labelsMat = cvMat(sampleNumbers, 1, CV_32FC1, labs);
    printMat(trainingDataMat);
    //printMat(labelsMat);

    //训练参数设定
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;				 //SVM类型
	params.kernel_type = CvSVM::LINEAR;			 //核函数的类型

	//SVM训练过程的终止条件, max_iter:最大迭代次数  epsilon:结果的精确性
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 10000, FLT_EPSILON);

    SVM.train(&trainingDataMat, &labelsMat, NULL, NULL, params);
    SVM.save(SVMParam.c_str());
    delete[] points;
    delete[] labs;
}

float AgeEstimation::predict(const AAM_Shape& shape, const IplImage& curImage)
{
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

    int pointsNumber = curShape.NPoints();
    float* points = new float[2 * pointsNumber];
    curShape.shap2Array(points);
    CvMat sample = cvMat(1, 2 * pointsNumber, CV_32FC1, points);

    if (SVM.get_support_vector_count() <= 0) {
    	SVM.load(SVMParam.c_str());
    }

    float ans = SVM.predict(&sample);

    printMat(sample);
    cvReleaseMat(&p);
    cvReleaseMat(&pq);
    cvReleaseMat(&curTexture);
    cvReleaseMat(&lamda);
    delete[] points;

    return ans;
}


void AgeEstimation::Read(std::ifstream& is)
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




