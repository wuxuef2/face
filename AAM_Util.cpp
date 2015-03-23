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

#include "AAM_Util.h"

std::ostream& operator<<(std::ostream &os, const CvMat* mat)
{
    for(int i = 0; i < mat->rows; i++)
    {
        for(int j = 0; j < mat->cols; j++)
        {
            os << CV_MAT_ELEM(*mat, double, i, j) << " ";
        }
        os << std::endl;
    }
    return os;
}

std::istream& operator>>(std::istream &is, CvMat* mat)
{
    for(int i = 0; i < mat->rows; i++)
    {
        for(int j = 0; j < mat->cols; j++)
        {
            is >> CV_MAT_ELEM(*mat, double, i, j);
        }
    }
    return is;
}

// compare function for the qsort() call below
static int str_compare(const void *arg1, const void *arg2)
{
    return strcmp((*(std::string*)arg1).c_str(), (*(std::string*)arg2).c_str());
}

#ifdef WIN32
std::vector<std::string> ScanNSortDirectory(const std::string &path,
        const std::string &extension)
{
    WIN32_FIND_DATA wfd;
    HANDLE hHandle;
    std::string searchPath, searchFile;
    std::vector<std::string> vFilenames;
    int nbFiles = 0;

    searchPath = path + "/*" + extension;
    hHandle = FindFirstFile(searchPath.c_str(), &wfd);
    if (INVALID_HANDLE_VALUE == hHandle)
    {
        fprintf(stderr, "ERROR(%s, %d): Can not find (*.%s)files in directory %s\n",
                __FILE__, __LINE__, extension.c_str(), path.c_str());
        //exit(0);
        return vFilenames;
    }
    do
    {
        //. or ..
        if (wfd.cFileName[0] == '.')
        {
            continue;
        }
        // if exists sub-directory
        if (wfd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
        {
            continue;
        }
        else//if file
        {
            searchFile = path + "/" + wfd.cFileName;
            vFilenames.push_back(searchFile);
            nbFiles++;
        }
    }
    while (FindNextFile(hHandle, &wfd));

    FindClose(hHandle);

    // sort the filenames
    qsort((void *)&(vFilenames[0]), (size_t)nbFiles, sizeof(std::string), str_compare);

    return vFilenames;
}
//additional
std::vector<std::string> ScanNSortAllDirectorys(const std::string &path)
{
    WIN32_FIND_DATA wfd;
    HANDLE hHandle;
    std::string searchPath, searchDir;
    std::vector<std::string> vDirnames;
    int nbDirs = 0;

    searchPath = path + "/*";
    hHandle = FindFirstFile(searchPath.c_str(), &wfd);
    if (INVALID_HANDLE_VALUE == hHandle)
    {
        fprintf(stderr, "ERROR(%s, %d): Can not find sub-directories in directory %s\n",
                __FILE__, __LINE__, path.c_str());
        //exit(0);
        return vDirnames;
    }
    do
    {
        //. or ..
        if (wfd.cFileName[0] == '.')
        {
            continue;
        }
        // if exists sub-directory
        if (wfd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
        {
            searchDir = path + "/" + wfd.cFileName;
            vDirnames.push_back(searchDir);
            nbDirs++;
        }
        else//if file
        {
            continue;
        }
    }
    while (FindNextFile(hHandle, &wfd));

    FindClose(hHandle);

    // sort the filenames
    qsort((void *)&(vDirnames[0]), (size_t)nbDirs, sizeof(std::string), str_compare);

    return vDirnames;
}
#else

#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>

#define MAX_PATH 1024

static int match(const char* s1, const char* s2)
{
    int diff = strlen(s1) - strlen(s2);
    if(diff >= 0 && !strcmp(s1+diff, s2))
        return 1;
    return 0;
}

std::vector<std::string> ScanNSortDirectory(const std::string &path, const std::string &extension)
{
    struct dirent *d;
    DIR* dir;
    struct stat s;
    char fullpath[MAX_PATH];
    std::vector<std::string> allfiles;
    int num = 0;

    dir = opendir(path.c_str());
    if(dir == NULL)
    {
        fprintf(stderr, "Can not open directory %s\n", path.c_str());
        exit(0);
    }


    std::string ext = std::string(extension);
    transform(ext.begin(), ext.end(), ext.begin(), ::toupper);


    while(d = readdir(dir))
    {
        sprintf(fullpath, "%s/%s", path.c_str(), d->d_name);
        if(stat(fullpath, &s) != -1)
        {
            if(S_ISDIR(s.st_mode))
                continue;


            if(match(d->d_name, extension.c_str()) || match(d->d_name, ext.c_str()))
            {
                allfiles.push_back(std::string(fullpath));
                num++;
            }


        }

    }
    closedir(dir);
    qsort((void*)&(allfiles[0]), size_t(num), sizeof(std::string), str_compare);
    return allfiles;
}

std::vector<std::string> ScanNSortAllDirectorys(const std::string &path)
{
    struct dirent *entry;
	DIR* dir;
	struct stat s;
	char fullpath[MAX_PATH];
    std::string searchPath, searchDir;
    std::vector<std::string> vDirnames;
    int nbDirs = 0;

    dir = opendir(path.c_str());

    searchPath = path + "/*";
    if(dir == NULL)
	{
		fprintf(stderr, "Can not open directory %s\n", path.c_str());
		exit(0);
	}

    while((entry = readdir(dir)) != NULL)
	{
		sprintf(fullpath, "%s/%s", path.c_str(), entry->d_name);
		if(strncmp(entry->d_name, "..", 2) != 0 &&
		       strncmp(entry->d_name, ".", 1) != 0) {
			if (entry->d_type == DT_DIR) {
				vDirnames.push_back(std::string(fullpath));
				nbDirs++;
			}
		}

	}
	closedir(dir);

    // sort the filenames
    qsort((void *)&(vDirnames[0]), (size_t)nbDirs, sizeof(std::string), str_compare);

    return vDirnames;
}
#endif // WIN32

int getAgeGroup(const std::string &dir)
{
    int age_group;
    int length = dir.length();
    age_group = atoi(dir.substr(length-2, length).c_str());

    return age_group;
}

AAM::AAM()
{
}

AAM::~AAM()
{
}
