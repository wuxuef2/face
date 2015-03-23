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

#ifndef AAM_PAW_H
#define AAM_PAW_H

#include "AAM_Util.h"
#include "AAM_Shape.h"

class AAM_IC;
class AAM_Basic;

// 2D piecewise affine warp
class AAM_PAW
{
    friend class AAM_IC;
    friend class AAM_Basic;
public:
    AAM_PAW();
    ~AAM_PAW();

    //build a piecewise affine warp
    void Train(const AAM_Shape& ReferenceShape,
               CvMat* Points,
               CvMemStorage* Storage,
               const std::vector<std::vector<int> >* tri = 0,
               bool buildVtri = true);

    // Read data from stream
    void Read(std::ifstream& is);

    // write data to stream
    void Write(std::ofstream& os);

    inline const int nPoints()const
    {
        return __n;
    }
    inline const int nPix()const
    {
        return __nPixels;
    }
    inline const int nTri()const
    {
        return __nTriangles;
    }

    //Get size and bounds of warping domain.
    void GetDomainBounds(int& w,int& h,int& xmin,int& ymin);

    inline const CvPoint2D32f Vertex(int i)const
    {
        return __referenceshape[i];
    }

    inline const std::vector<std::vector<int> >* GetTri()const
    {
        return &__tri;
    }

    //index of point for j-th vertex of i-th triangle
    inline const int Tri(int i, int j)const
    {
        return __tri[i][j];
    }

    // does vertex @i share @j-th triangle
    inline const int vTri(int i, int j)const
    {
        return __vtri[i][j];
    }

    // index of triangle the pixel lies in
    inline const int PixTri(int i)const
    {
        return __pixTri[i];
    }

    //coeffiects of affine warp
    inline const double Alpha(int i)const
    {
        return __alpha[i];
    }
    inline const double Belta(int i)const
    {
        return __belta[i];
    }
    inline const double Gamma(int i)const
    {
        return __gamma[i];
    }

    //width and height boundary
    inline const int Width()const
    {
        return __width;
    }
    inline const int Height()const
    {
        return __height;
    }
    //is point(j,i) in boundary: not(-1), yes(index of pixel)
    inline const int Rect(int i, int j)const
    {
        return __rect[i][j];
    }

    // Warp the image to the face template
    void GetWarpTextureFromShape(const AAM_Shape& Shape, const IplImage* image,
                                 CvMat* Texture, bool normalize = false)const;
    // the same as above, but a bit of faster
    void FasterGetWarpTextureFromShape(const AAM_Shape& Shape,
                                       const IplImage* image,	CvMat* Texture, bool normalize = false)const;
    void FasterGetWarpTextureFromMatShape(const CvMat* Shape,
                                          const IplImage* image,	CvMat* Texture, bool normalize = false)const;

    // for display, translate the texture vector to the visual-able image format
    void GetWarpImageFromVector(IplImage* image, const CvMat* Texture)const;
    //save the texture image to file
    void SaveWarpImageFromVector(const char* filename, const CvMat* Texture)const;


    // bilinear interpolate
    static void GetBilinearPixel(double& pB, double& pG, double& pR,
                                 const IplImage *image,	double x, double y);

    // apply no interpolation algorithm to get pixel
    static void GetPixel(double& pB, double& pG, double& pR,
                         const IplImage *image, double x, double y);

    //Get the warp parameters for a specific point according the piecewise
    static void CalcWarpParameters(double x, double y, double x1, double y1,
                                   double x2, double y2, double x3, double y3,
                                   double &alpha, double &beta, double &gamma);



private:
    // build triangles
    void Delaunay(const CvSubdiv2D* Subdiv, const CvMat *ConvexHull);

    // calculate all pixels in the triangles
    void CalcPixelPoint(const CvRect rect, CvMat *ConvexHull);

    // Is the current edge (ind1, ind2) already in the AAM model edges?
    static bool IsCurrentEdgeAlreadyIn(const std::vector<std::vector<int> > &edges,
                                       int ind1, int ind2);

    // Help to build up triangles in the mesh
    static bool VectorIsNotInFormerTriangles(const std::vector<int>& one_tri,
            const std::vector<std::vector<int> > &tris);

    //Find triangles containing each landmark.
    void FindVTri();

private:
    int __n;						/*number of landmarks*/
    int __nPixels;					/*number of pixels*/
    int __nTriangles;				/*number of triangles*/
    int __width, __height, __xmin, __ymin; /*Domain of warp region*/

    std::vector<std::vector<int> > __tri;	/*triangle vertexes index*/
    std::vector<std::vector<int> > __vtri;	/*vertex vs triangle*/
    std::vector<int>			   __pixTri;
    std::vector<double>			   __alpha, __belta,  __gamma;
    std::vector<std::vector<int> > __rect; /*height by width*/

    AAM_Shape __referenceshape;



};

#endif //
