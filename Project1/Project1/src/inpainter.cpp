/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2008-2012, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "inpainter.h"
#include <opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/types_c.h>
#include <vector>

using namespace cv;
using namespace std;

Inpainter::Inpainter(cv::Mat inputImage,cv::Mat mask,int halfPatchWidth,int mode)
{
    this->inputImage=inputImage.clone();
    this->mask=mask.clone();
    this->workImage=inputImage.clone();
    this->result.create(inputImage.size(),inputImage.type());
    this->halfPatchWidth=halfPatchWidth;
}

int Inpainter::checkValidInputs(){
    if(this->inputImage.type()!=CV_8UC3)
        return ERROR_INPUT_MAT_INVALID_TYPE;
    if(this->mask.type()!=CV_8UC1)
        return ERROR_INPUT_MASK_INVALID_TYPE;
    if(!CV_ARE_SIZES_EQ(&mask,&inputImage))
        return ERROR_MASK_INPUT_SIZE_MISMATCH;
    if(halfPatchWidth==0)
        return ERROR_HALF_PATCH_WIDTH_ZERO;
    return CHECK_VALID;
}


void PatchMatch(const Mat & SourceImage, const Mat & TargetImage, const Mat & Mask, int nPatchSize, Mat & NearestNeighbor);
Vec3f MeanShift(vector<Vec3b> vecVoteColor, vector<float> vecVoteWeight, int sigma);

void Inpainter::inpaint(cv::VideoWriter & video)
{
	Mat Weight = Mat(workImage.size(), CV_32F);
	distanceTransform(mask, Weight, CV_DIST_L2, 3);

	// �Ƚ�mask��������Ϊ�������������������Ϣ
	for (int i =0; i < workImage.rows; i++)
	{
		for (int j = 0; j < workImage.cols; j++)
		{
			float dist = Weight.at<float>(i, j);
			Weight.at<float>(i, j) = (float)pow(1.3,  -dist);

			if (mask.at<uchar>(i,j))
			{
				inputImage.at<Vec3b>(i,j)[0] = rand() % 255;
				inputImage.at<Vec3b>(i, j)[1] = rand() % 255;
				inputImage.at<Vec3b>(i, j)[2] = rand() % 255;
			}
		}
	}




	// �����ͬ�ĳ߶�
	int nPyrmidNum = 3;
	Mat CurWork = Mat();
	Mat CurMask;
	Mat CurWeight;
	int PatchSize = 2 * halfPatchWidth + 1;
	while (nPyrmidNum >= 0)
	{
		// ��ײ��
		float scale = 1.0 / (1 << nPyrmidNum);
		resize(inputImage, workImage, Size(inputImage.cols * scale, scale * inputImage.rows));
		resize(mask, CurMask, Size(mask.cols * scale, scale * mask.rows));
		resize(Weight, CurWeight, Size(Weight.cols * scale, scale * Weight.rows));

		if (CurWork.cols > 0) // ����Ѿ���ͼƬ��
		{
			resize(CurWork, CurWork, Size(inputImage.cols * scale, scale * inputImage.rows));
			// mask�����������һ�ε��������
			for (int i = 0; i < CurWork.rows; i++)
			{
				for (int j = 0; j< CurWork.cols; j++)
				{
					if (CurMask.at<uchar>(i,j) == 0)
					{
						CurWork.at<Vec3b>(i,j) = workImage.at<Vec3b>(i,j);
					}
				}
			}
		}
		else
		{
			CurWork = workImage.clone();
		}
		
		int nIterMaxNum = 30;

		int minLen = min(CurWork.rows, CurWork.cols);

		if (minLen < 2 * PatchSize)
		{
			nPyrmidNum--;
			continue;
		}
		
		// ѭ��ֱ����������
		while (true)
		{
			Mat LastImage = CurWork.clone();

			Mat OutPutFrame;
			resize(CurWork, OutPutFrame, Size(inputImage.cols, inputImage.rows));


			video << OutPutFrame;

			// patchMatch �������patch�������
			Mat NNF;
			PatchMatch(CurWork, CurWork, CurMask, PatchSize, NNF);

			// ѭ��ͼƬ
			for (int i = 0; i < CurWork.rows; i++)
			{
				for (int j =0; j < CurWork.cols; j++)
				{
					//��Ҫ��������
					if (CurMask.at<uchar>(i,j))
					{

						vector<Vec3b> vecVoteColor;
						vector<float> vecVoteWeight;
						vector<float> vecDist;

						// ���о����õ��patch��
						for (int k = -PatchSize + 1; k <= 0; k++)
						{
							for (int m = -PatchSize + 1; m <= 0; m++)
							{
								int nPosY = i + k;
								int nPosX = j + m;

								// patch�컹��ͼ��Χ��
								if (nPosY < 0 
									|| nPosX < 0
									|| nPosX + PatchSize >= CurWork.cols -1
									|| nPosY + PatchSize >= CurWork.rows - 1)
									continue;

								Rect sTCurPatchRect(nPosX, nPosY, PatchSize, PatchSize);  // ��ǰ���
								Mat CurPatch = CurWork(sTCurPatchRect);

								int NNF_X = NNF.at<Vec3i>(nPosY, nPosX)[0];
								int NNF_Y = NNF.at<Vec3i>(nPosY, nPosX)[1];

								Rect NearPatchRect(NNF_X, NNF_Y, PatchSize, PatchSize);  // ��ǰ���
								Mat NearestPatch = CurWork(NearPatchRect);


								// ����patch֮��ľ���
								int dist = norm(CurPatch, NearestPatch);

								// ��ĳ����ɫͶƱ
								Vec3b VoteColor = NearestPatch.at<Vec3b>(-k, -m);

								// Ȩ��
								float fWeight = Weight.at<float>(nPosY + PatchSize / 2, nPosX + PatchSize / 2);

								vecDist.push_back(dist * dist);
								vecVoteColor.push_back(VoteColor);
								vecVoteWeight.push_back(fWeight);
							}
						}

						if (vecVoteWeight.size() < 3)
						{
							continue;
						}
						// ����һ��
						vector<float> vecDistCopy;
						vecDistCopy.assign(vecDist.begin(), vecDist.end());//��v2��ֵ��v1

						// ����
						sort(vecDistCopy.begin(), vecDistCopy.end());

						int nId = vecDistCopy.size() * 3 / 4;
						float nSigma = vecDistCopy[nId];

						// ����Ȩ��
						for (int i =0 ; i < vecVoteWeight.size(); i++)
						{
							// ���sigma��Ϊ0
							if(nSigma != 0)
								vecVoteWeight[i] = vecVoteWeight[i] * exp(-(vecDist[i]) / (2 * nSigma));
						}

						float nMax = 0;
						for (int i = 0; i < vecVoteWeight.size(); i++)
						{
							if (nMax < vecVoteWeight[i])
							{
								nMax = vecVoteWeight[i];
							}
						}
						// ��һ��,����Ȩ�ع�С
						for (int i = 0; i < vecVoteWeight.size(); i++)
						{
							vecVoteWeight[i] = vecVoteWeight[i] / nMax;
						}


						CurWork.at<Vec3b>(i,j)  = MeanShift(vecVoteColor, vecVoteWeight, 50);
						
					}
				}
			}
			nIterMaxNum--;


		

			if (nIterMaxNum <= 0)
			{
				break;
			}

			float diff = 0;
			int Num = 0;
			for (int i = 0; i < LastImage.rows; i++)
			{
				for (int j = 0; j < LastImage.cols; j++)
				{
					if (CurMask.at<uchar>(i, j))
					{
						Vec3f a1 = LastImage.at<Vec3b>(i, j);
						Vec3f a2 = CurWork.at<Vec3b>(i, j);
						Vec3f a3 = a1 - a2;
						
						diff += a3[0] * a3[0] + a3[1] * a3[1] + a3[2] * a3[2];

						Num++;
					}

				}
			}

			diff = diff / Num;

			imshow("CurWork", CurWork);
			cvWaitKey(100);

			printf("scale: %f, dff: %f\n", scale, diff);
		
			if (diff < 100)
			{
				break;
			}

		}

		// ����һ�㣬���һ��
		nPyrmidNum--;
	}

	result = CurWork;
}