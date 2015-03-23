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

#include <set>
#include <cstdio>
//#include <windows.h>
//#include <cvaux.h>

#include "AAM_PAW.h"
#include "AAM_TDM.h"

using namespace std;

#define curtime cvGetTickCount() / (cvGetTickFrequency()*1000.)

#define free2dvector(vec)										\
{																\
	for(int i = 0; i < vec.size(); i++) vec[i].clear();			\
	vec.clear();												\
}

AAM_PAW::AAM_PAW()
{
    __nTriangles = 0;
    __nPixels = 0;
    __n = 0;
    __width = 0;
    __height = 0;
}

AAM_PAW::~AAM_PAW()
{
    __pixTri.clear();
    __alpha.clear();
    __belta.clear();
    __gamma.clear();

    free2dvector(__rect);
    free2dvector(__vtri);
    free2dvector(__tri);
}

//============================================================================
void AAM_PAW::Train(const AAM_Shape& ReferenceShape,
                    CvMat* Points,
                    CvMemStorage* Storage,
                    const std::vector<std::vector<int> >* tri,
                    bool buildVtri)
{
//	double t = curtime;

    __referenceshape = ReferenceShape;
    __n = __referenceshape.NPoints();// get the number of vertex point

    for(int i = 0; i < __n; i++)
        CV_MAT_ELEM(*Points, CvPoint2D32f, 0, i) = __referenceshape[i];

    CvMat* ConvexHull = cvCreateMat (1, __n, CV_32FC2);
    cvConvexHull2(Points, ConvexHull, CV_CLOCKWISE, 0);

    CvRect rect = cvBoundingRect(ConvexHull, 0);
    CvSubdiv2D* Subdiv = cvCreateSubdivDelaunay2D(rect, Storage);
    for(int ii = 0; ii < __n; ii++)
        cvSubdivDelaunay2DInsert(Subdiv, __referenceshape[ii]);

    //firstly: build triangle
    if(tri == 0)	Delaunay(Subdiv, ConvexHull);
    else	 __tri = *tri;
    __nTriangles = __tri.size();// get the number of triangles

    //secondly: build correspondence of Vertex-Triangle
    if(buildVtri)	FindVTri();

    //Thirdly: build pixel point in all triangles
    CalcPixelPoint(rect, ConvexHull);
    __nPixels = __pixTri.size();// get the number of pixels

    cvReleaseMat(&ConvexHull);

//	t = curtime - t;
//	printf("Delaunay: %.2f\n", t);
}

void AAM_PAW::GetDomainBounds(int& w,int& h,int& xmin,int& ymin)
{
    xmin = __xmin;
    ymin = __ymin;
    w = __width;
    h = __height;
}

//============================================================================
void AAM_PAW::Delaunay(const CvSubdiv2D* Subdiv, const CvMat* ConvexHull)
{
    // firstly we build edges
    int i;
    CvSeqReader  reader;
    CvQuadEdge2D* edge;
    CvPoint2D32f org, dst;
    CvSubdiv2DPoint* org_pt, * dst_pt;
    std::vector<std::vector<int> > edges;
    std::vector<int> one_edge;
    one_edge.resize(2);
    std::vector<int> one_tri;
    one_tri.resize(3);
    int ind1, ind2;

    cvStartReadSeq( (CvSeq*)(Subdiv->edges), &reader, 0 );
    for(i = 0; i < Subdiv->edges->total; i++)
    {
        edge = (CvQuadEdge2D*)(reader.ptr);
        if( CV_IS_SET_ELEM(edge))
        {
            org_pt = cvSubdiv2DEdgeOrg((CvSubdiv2DEdge)edge);
            dst_pt = cvSubdiv2DEdgeDst((CvSubdiv2DEdge)edge);

            if( org_pt && dst_pt )
            {
                org = org_pt->pt;
                dst = dst_pt->pt;
                if (cvPointPolygonTest(ConvexHull, org, 0) >= 0 &&
                        cvPointPolygonTest( ConvexHull, dst, 0) >= 0)
                {
                    for (int j = 0; j < __n; j++)
                    {
                        if (fabs(org.x-__referenceshape[j].x)<1e-6 &&
                                fabs(org.y-__referenceshape[j].y)<1e-6)
                        {
                            for (int k = 0; k < __n; k++)
                            {
                                if (fabs(dst.x-__referenceshape[k].x)<1e-6
                                        &&fabs(dst.y-__referenceshape[k].y)<1e-6)
                                {
                                    one_edge[0] = j;
                                    one_edge[1] = k;
                                    edges.push_back (one_edge);
                                }
                            }
                        }
                    }
                }
            }
        }

        CV_NEXT_SEQ_ELEM( Subdiv->edges->elem_size, reader );
    }

    // secondly we start to build triangles
    for (i = 0; i < edges.size(); i++)
    {
        ind1 = edges[i][0];
        ind2 = edges[i][1];

        for (int j = 0; j < __n; j++)
        {
            // At most, there are only 2 triangles that can be added
            if(AAM_PAW::IsCurrentEdgeAlreadyIn(edges, ind1, j) &&
                    AAM_PAW::IsCurrentEdgeAlreadyIn(edges, ind2, j) )
            {
                one_tri[0] = ind1;
                one_tri[1] = ind2;
                one_tri[2] = j;
                if (AAM_PAW::VectorIsNotInFormerTriangles(one_tri, __tri) )
                {
                    __tri.push_back(one_tri);
                }
            }
        }
    }

    //OK, up to now, we have already builded the triangles!
}

//============================================================================
bool AAM_PAW::IsCurrentEdgeAlreadyIn(const vector<vector<int> > &edges,
                                     int ind1, int ind2)
{
    for (int i = 0; i < edges.size (); i++)
    {

        if ((edges[i][0] == ind1 && edges[i][1] == ind2) ||
                (edges[i][0] == ind2 && edges[i][1] == ind1) )
            return true;
    }

    return false;
}

//============================================================================
bool AAM_PAW::VectorIsNotInFormerTriangles(const vector<int>& one_tri,
        const vector<vector<int> > &tris)
{
    set<int> tTriangle;
    set<int> sTriangle;

    for (int i = 0; i < tris.size (); i ++)
    {
        tTriangle.clear();
        sTriangle.clear();
        for (int j = 0; j < 3; j++ )
        {
            tTriangle.insert(tris[i][j]);
            sTriangle.insert(one_tri[j]);
        }
        if (tTriangle == sTriangle)    return false;
    }

    return true;
}

//============================================================================
void AAM_PAW::CalcPixelPoint(const CvRect rect, CvMat* ConvexHull)
{
    CvPoint2D32f point[3];
    CvMat tempVert = cvMat(1, 3, CV_32FC2, point);
    int ll = 0;
    double alpha, belta, gamma;
    CvPoint2D32f pt;
    int ind1, ind2, ind3;
    int ii, jj;
//	double x, y, x1, y1, x2, y2, x3, y3, c;

    __xmin = rect.x;
    __ymin = rect.y;
    __width = rect.width;
    __height = rect.height;
    int left = rect.x, right = left + __width;
    int top = rect.y, bottom = top + __height;

    __rect.resize(__height);
    for (int i = top; i < bottom; i++)
    {
        ii = i - top;
        __rect[ii].resize(__width);
        for (int j = left; j < right; j++)
        {
            jj = j - left;
            pt = cvPoint2D32f(j, i);
            __rect[ii][jj] = -1;

            // firstly: the point (j, i) is located in the ConvexHull
            if(cvPointPolygonTest(ConvexHull, pt, 0) >= 0 )
            {
                // then we find the triangle that the point lies in
                for (int k = 0; k < __nTriangles; k++)
                {
                    ind1 = __tri[k][0];
                    ind2 = __tri[k][1];
                    ind3 = __tri[k][2];
                    point[0] = __referenceshape[ind1];
                    point[1] = __referenceshape[ind2];
                    point[2] = __referenceshape[ind3];

                    // secondly: the point(j,i) is located in the k-th triangle
                    if(cvPointPolygonTest(&tempVert, pt, 0) >= 0)
                    {
                        __rect[ii][jj] = ll++;
                        __pixTri.push_back(k);

                        // calculate alpha and belta for warp
                        AAM_PAW::CalcWarpParameters(j, i, point[0].x, point[0].y,
                                                    point[1].x, point[1].y, point[2].x, point[2].y, alpha, belta, gamma);

//						x = pt.x;		 y = pt.y;
//						x1 = point[0].x; y1 = point[0].y;
//						x2 = point[1].x; y2 = point[1].y;
//						x3 = point[2].x; y3 = point[2].y,
//						c = (+x2*y3-x2*y1-x1*y3-x3*y2+x3*y1+x1*y2);
//
//						alpha = (y*x3-y3*x+x*y2-x2*y+x2*y3-x3*y2) / c;
//						belta  = (-y*x3+x1*y+x3*y1+y3*x-x1*y3-x*y1) / c;
//						gamma = 1 - alpha - belta;

                        __alpha.push_back(alpha);
                        __belta.push_back(belta);
                        __gamma.push_back(gamma);

                        // make sure each point only located in only one triangle
                        break;
                    }

                }
            }
        }
    }
}

//============================================================================
void AAM_PAW::FindVTri()
{
    __vtri.resize(__n);
    for(int i = 0; i < __n; i++)
    {
        __vtri[i].resize(__nTriangles);
        for(int j = 0; j < __nTriangles; j++)
        {
            if(__tri[j][0] == i || __tri[j][1] == i || __tri[j][2] == i)
                __vtri[i][j] = 1;
        }
    }
}

//============================================================================
void AAM_PAW::CalcWarpParameters(double x, double y, double x1, double y1,
                                 double x2, double y2, double x3, double y3,
                                 double &alpha, double &beta, double &gamma)
{
    double c = (+x2*y3-x2*y1-x1*y3-x3*y2+x3*y1+x1*y2);
    alpha = (y*x3-y3*x+x*y2-x2*y+x2*y3-x3*y2) / c;
    beta  = (-y*x3+x1*y+x3*y1+y3*x-x1*y3-x*y1) / c;
    gamma = 1 - alpha - beta;
}

//============================================================================
inline static void GetColorPixel(byte& pB, byte& pG, byte& pR,
                                 const IplImage* image, int x, int y)
{
    int ixB = x, ixG = x, ixR =x;
    if(image->nChannels == 3)
    {
        ixB = 3*x;
        ixG = ixB+1;
        ixR = ixB+2;
    }
    byte* p = (byte*)(image->imageData + image->widthStep*y);
    pB = p[ixB];
    pG = p[ixG];
    pR = p[ixR];
}

void AAM_PAW::GetPixel(double& pB, double& pG, double& pR,
                       const IplImage *image, double x, double y)
{
    int X = (int)(x+0.5);
    int Y = (int)(y+0.5);

    byte pB1, pG1, pR1;
    GetColorPixel(pB1, pG1, pR1, image, X, Y);
    pB = pB1;
    pG = pG1;
    pR = pR1;
}

void AAM_PAW::GetBilinearPixel(double& pB, double& pG, double& pR,
                               const IplImage *image, double x, double y)
{
    int X = (int)x;
    int Y = (int)y;
    int X1 = X+1;
    int Y1 = Y+1;
    double s = x-X;
    double t = y-Y;
    double s1 = 1-s;
    double t1 = 1-t;

    byte ltb, ltg, ltr;
    byte lbb, lbg, lbr;
    byte rtb, rtg, rtr;
    byte rbb, rbg, rbr;
    GetColorPixel(ltb, ltg, ltr, image, X, Y);
    GetColorPixel(lbb, lbg, lbr, image, X, Y1);
    GetColorPixel(rtb, rtg, rtr, image, X1, Y);
    GetColorPixel(rbb, rbg, rbr, image, X1, Y1);

    double b1 = t1 * ltb + t * lbb;
    double b2 = t1 * rtb + t * rbb;
    double g1 = t1 * ltg + t * lbg;
    double g2 = t1 * rtg + t * rbg;
    double r1 = t1 * ltr + t * lbr;
    double r2 = t1 * rtr + t * rbr;

    pB = b1 * s1 + b2 * s;
    pG = g1 * s1 + g2 * s;
    pR = r1 * s1 + r2 * s;
}
//============================================================================

void AAM_PAW::GetWarpTextureFromShape(const AAM_Shape& Shape, const IplImage* image,
                                      CvMat* Texture, bool normalize)const
{
    double *data = Texture->data.db;
    int v1, v2, v3, tri_idx;
    double x, y;

    for(int i = 0, k = 0; i < __nPixels; i++, k+=3) //all pixels inside the shape template
    {
        tri_idx = __pixTri[i];
        v1 = __tri[tri_idx][0];
        v2 = __tri[tri_idx][1];
        v3 = __tri[tri_idx][2];

        x = __alpha[i]*Shape[v1].x + __belta[i]*Shape[v2].x + __gamma[i]*Shape[v3].x;
        y = __alpha[i]*Shape[v1].y + __belta[i]*Shape[v2].y + __gamma[i]*Shape[v3].y;

        // I commend using this call, because it runs faster
        AAM_PAW::GetBilinearPixel(data[k], data[k+1], data[k+2], image, x, y); //get texture from the image to shape template

        // also you can use a simple interpolation
        //AAM_PAW::GetPixel(data[k], data[k+1], data[k+2], image, x, y);
    }
    if(normalize) AAM_TDM::ZeroMeanUnitLength(Texture);/*cvNormalize(Texture, Texture);*/
}

//==========================================================================
void AAM_PAW::GetWarpImageFromVector(IplImage* image, const CvMat* Texture)const
{
    if(image->nChannels != 3 || image->depth != 8)
    {
        fprintf(stderr, "ERROR(%s: %d): The image channels must be 3, "
                "and the depth must be 8!\n", __FILE__, __LINE__);
        exit(0);
    }

    CvMat* tempMat = cvCreateMat(1, Texture->cols, CV_64FC1);
    cvCopy(Texture, tempMat);
    double minV, maxV;
    cvMinMaxLoc(tempMat, &minV, &maxV);
    cvConvertScale(tempMat, tempMat, 255/(maxV-minV), -minV*255/(maxV-minV));

    cvZero(image);
    int k = 0, x3;
    double *texturedb = tempMat->data.db;
    byte* p;

    for(int y = 0; y < __height; y++)
    {
        p = (byte*)(image->imageData + image->widthStep*y);
        for(int x = 0; x < __width; x++)
        {
            if(__rect[y][x] >= 0)
            {
                x3 = 3*x;
                p[x3] = texturedb[k];
                p[x3+1] = texturedb[k+1];
                p[x3+2] = texturedb[k+2];
                /*
                CV_IMAGE_ELEM(image, byte, y, x * 3 + 0) = texturedb[k];
                CV_IMAGE_ELEM(image, byte, y, x * 3 + 1) = texturedb[k+1];
                CV_IMAGE_ELEM(image, byte, y, x * 3 + 2) = texturedb[k+2];
                */
                k+=3;
            }
        }
    }
    cvReleaseMat(&tempMat);
}

void AAM_PAW::SaveWarpImageFromVector(const char* filename, const CvMat* Texture)const
{
    IplImage* Warpimg = cvCreateImage(cvSize(__width, __height), IPL_DEPTH_8U, 3);
    this->GetWarpImageFromVector(Warpimg, Texture);
    cvSaveImage(filename, Warpimg);
    cvReleaseImage(&Warpimg);
}


void AAM_PAW::FasterGetWarpTextureFromShape(const AAM_Shape& Shape,
        const IplImage* image,	CvMat* Texture,
        bool normalize)const
{
    double *data = Texture->data.db;
    int v1, v2, v3, tri_idx;
    double x, y;
    int X, Y , X1, Y1;
    double s , t, s1, t1;
    int ixB1, ixG1, ixR1, ixB2, ixG2, ixR2;
    byte* p1, * p2;
    byte ltb, ltg, ltr, lbb, lbg, lbr, rtb, rtg, rtr, rbb, rbg, rbr;
    double b1 , b2, g1, g2 , r1, r2;
    int nchannel = image->nChannels;
    int off_g = (image->nChannels == 3) ? 1 : 0;
    int off_r = (image->nChannels == 3) ? 2 : 0;

    for(int i = 0, k = 0; i < __nPixels; i++, k+=3)
    {
        tri_idx = __pixTri[i];
        v1 = __tri[tri_idx][0];
        v2 = __tri[tri_idx][1];
        v3 = __tri[tri_idx][2];

        x = __alpha[i]*Shape[v1].x + __belta[i]*Shape[v2].x + __gamma[i]*Shape[v3].x;
        y = __alpha[i]*Shape[v1].y + __belta[i]*Shape[v2].y + __gamma[i]*Shape[v3].y;

        X = (int)x;
        Y = (int)y;
        X1 = X+1;
        Y1 = Y+1;
        s = x-X;
        t = y-Y;
        s1 = 1-s;
        t1 = 1-t;

        ixB1 = nchannel*X;
        ixG1= ixB1+off_g;
        ixR1 = ixB1+off_r;
        ixB2 = nchannel*X1;
        ixG2= ixB2+off_g;
        ixR2 = ixB2+off_r;

        p1 = (byte*)(image->imageData + image->widthStep*Y);
        p2 = (byte*)(image->imageData + image->widthStep*Y1);

        ltb = p1[ixB1];
        ltg = p1[ixG1];
        ltr = p1[ixR1];
        lbb = p2[ixB1];
        lbg = p2[ixG1];
        lbr = p2[ixR1];
        rtb = p1[ixB2];
        rtg = p1[ixG2];
        rtr = p1[ixR2];
        rbb = p2[ixB2];
        rbg = p2[ixG2];
        rbr = p2[ixR2];

        b1 = t1 * ltb + t * lbb;
        b2 = t1 * rtb + t * rbb;
        g1 = t1 * ltg + t * lbg;
        g2 = t1 * rtg + t * rbg;
        r1 = t1 * ltr + t * lbr;
        r2 = t1 * rtr + t * rbr;

        data[k] = b1 * s1 + b2 * s;
        data[k+1] = g1 * s1 + g2 * s;
        data[k+2] = r1 * s1 + r2 * s;
    }
    if(normalize) AAM_TDM::ZeroMeanUnitLength(Texture);
}

void AAM_PAW::FasterGetWarpTextureFromMatShape
(const CvMat* Shape, const IplImage* image,
 CvMat* Texture, bool normalize)const
{
    AAM_Shape aamshape;
    aamshape.Mat2Point(Shape);
    FasterGetWarpTextureFromShape(aamshape, image, Texture, normalize);
}

void AAM_PAW::Write(std::ofstream& os)
{
    int i, j;

    os << __n << " " << __nTriangles << " " << __nPixels << " "
       << __xmin << " " << __ymin << " " << __width << " " << __height
       << std::endl;

    for(i = 0; i < __nTriangles; i++)
        os << __tri[i][0] << " " << __tri[i][1] << " " << __tri[i][2] << std::endl;
    os << std::endl;

    for(i = 0; i < __vtri.size(); i++)
    {
        for( j = 0; j < __nTriangles; j++)
        {
            os << __vtri[i][j] << " ";
        }
        os << std::endl;
    }
    os << std::endl;

    for(i = 0;  i < __nPixels; i++)	os << __pixTri[i] << " ";
    os << std::endl;

    for(i = 0; i < __nPixels; i++)	os << __alpha[i] << " ";
    os << std::endl;

    for(i = 0; i < __nPixels; i++)	os << __belta[i] << " ";
    os << std::endl;

    for(i = 0; i < __nPixels; i++)	os << __gamma[i] << " ";
    os << std::endl;

    for(i = 0; i < __height; i++)
    {
        for(j = 0; j < __width; j++)
            os << __rect[i][j]<< " ";
        os << std::endl;
    }
    os << std::endl;

    __referenceshape.Write(os);

    os << std::endl;
}

void AAM_PAW::Read(std::ifstream& is)
{
    int i, j;
    is >> __n >> __nTriangles >> __nPixels >> __xmin >> __ymin >> __width >> __height;

    __tri.resize(__nTriangles);
    for(i = 0; i < __nTriangles; i++)
    {
        __tri[i].resize(3);
        is >> __tri[i][0] >> __tri[i][1] >> __tri[i][2];
    }

    __vtri.resize(__n);
    for(i = 0; i < __n; i++)
    {
        __vtri[i].resize(__nTriangles);
        for( j = 0; j < __nTriangles; j++)	is >> __vtri[i][j];
    }

    __pixTri.resize(__nPixels);
    for(i = 0;  i < __nPixels; i++)	is >> __pixTri[i];

    __alpha.resize(__nPixels);
    for(i = 0; i < __nPixels; i++)	is >> __alpha[i];

    __belta.resize(__nPixels);
    for(i = 0; i < __nPixels; i++)	is >> __belta[i];

    __gamma.resize(__nPixels);
    for(i = 0; i < __nPixels; i++)	is >> __gamma[i];

    __rect.resize(__height);
    for(i = 0; i < __height; i++)
    {
        __rect[i].resize(__width);
        for(j = 0; j < __width; j++) is >> __rect[i][j];
    }

    __referenceshape.resize(__n);
    __referenceshape.Read(is);
}
