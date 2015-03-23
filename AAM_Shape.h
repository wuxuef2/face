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

#ifndef AAM_SHAPE_H
#define AAM_SHAPE_H

#include "AAM_Util.h"

class AAM_Shape
{
private:
    // point data.
    std::vector<CvPoint2D32f> m_vPoint;

public:
    // Constructors and Destructor
    AAM_Shape() {
        resize(0);
    }

    AAM_Shape(const AAM_Shape &v);
    ~AAM_Shape() {
        clear();
    }

    AAM_Shape(double points[], int len) {
        CvPoint2D32f pnt;
        m_vPoint.clear();
        for (int i = 0; i < len; i += 2) {
            pnt.x = points[i];
            pnt.y = points[i + 1];
            m_vPoint.push_back(pnt);
        }
    }

    //access elements
    const CvPoint2D32f operator[] (int i)const {
        return m_vPoint[i];
    }

    CvPoint2D32f& operator[] (int i) {
        return m_vPoint[i];
    }

    inline const int NPoints()const {
        return	m_vPoint.size();
    }

    void shap2Array(float* &array) {
    	int j = 0;
    	for (int i = 0; i < m_vPoint.size(); i++) {
			array[j] = m_vPoint[i].x;
			j++;
			*(array + j) = m_vPoint[i].y;
			j++;
		}
    }

    // operators
    AAM_Shape& operator=(const AAM_Shape &s);
    AAM_Shape& operator=(double value);
    AAM_Shape operator+(const AAM_Shape &s);
    AAM_Shape& operator+=(const AAM_Shape &s);
    AAM_Shape operator-(const AAM_Shape &s);
    AAM_Shape& operator-=(const AAM_Shape &s);
    AAM_Shape operator*(double value);
    AAM_Shape& operator*=(double value);
    double operator*(const AAM_Shape &s);
    AAM_Shape operator/(double value);
    AAM_Shape& operator/=(double value);
    bool operator==(double value);

    void clear() {
        resize(0);
    }

    void resize(int length) {
        m_vPoint.resize(length);
    }

    void print() const {
        int i, size = NPoints();
        printf("%d\n", size);
        for(i = 0 ; i < size; i++)
            printf("%3.1f ", m_vPoint[i].x);
        printf("\n");

        for(i = 0 ; i < size; i++)
            printf("%3.1f ", m_vPoint[i].y);
        printf("\n");
    }

    void Read(std::ifstream& is);
    void Write(std::ofstream& os);
    bool ReadAnnotations(const std::string &filename);
    void ReadASF(const std::string &filename);
    void ReadPTS(const std::string &filename);

    const double  MinX()const;
    const double  MinY()const;
    const double  MaxX()const;
    const double  MaxY()const;
    inline const double  GetWidth() const {
        return MaxX()-MinX();
    }
    inline const double  GetHeight() const {
        return MaxY()-MinY();
    }

    // Transformations
    void COG(double &x, double &y)const;
    void Centralize();
    void Translate(double x, double y);
    void Scale(double s);
    void Rotate(double theta);
    void ScaleXY(double sx, double sy);
    double	Normalize();

    // Align the shapes to reference shape
    //													[a -b Tx]
    // returns the similarity transform: T(a,b,tx,ty) = [b  a Ty]
    //													[0  0  1]
    void AlignTransformation(const AAM_Shape &ref,
                                double &a, double &b, double &tx, double &ty)const;

    // Align the shapes to reference shape as above, but no returns
    void AlignTo(const AAM_Shape &ref);

    // Transform Shape using similarity transform T(a,b,tx,ty)
    void TransformPose(double a, double b, double tx, double ty);

    // Euclidean norm
    double GetNorm2()const;

    // conversion between CvMat and AAM_Shape
    void Mat2Point(const CvMat* res);
    void Point2Mat(CvMat* res)const;

    void Sketch(IplImage* image);

private:
    void CopyData(const AAM_Shape &s);
    void Transform(double c00, double c01, double c10, double c11);

};


#endif  // asmutil_h

