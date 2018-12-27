

#include <opencv.hpp>

using namespace  cv;


// PatchMatch: 快速寻找patch块的最近邻


float DistPatch(const Mat & PatchA, const Mat & PatchB, int PatchSize)
{
	return PatchSize * PatchSize * norm(PatchA, PatchB) ;
}


void GuessAndImprove(const Mat & SourceImage, const Mat & TargetImage, const Mat & mask,
	int x , int y, int Guees_x, int guess_y, int PatchSize, Mat &NearestNeighbor)
{
	// 当前的patch块
	if (x == Guees_x && y == guess_y)
	{
		return;
	}

	Rect RE2(Guees_x, guess_y, PatchSize, PatchSize);

	Mat CurMask = mask(RE2);
	int unValidNum = 0;
	for (int i = 0; i < CurMask.rows; i++)
	{
		for (int j = 0; j < CurMask.cols; j++)
		{
			if (CurMask.at<uchar>(i, j))
			{
				unValidNum++;
			}
		}
	}

	if (unValidNum  * 10 > CurMask.rows * CurMask.cols)
	{
		return;
	}

	Rect RE(x, y, PatchSize, PatchSize);

	Mat patchA = SourceImage(RE);
	Mat patchB = TargetImage(RE2);

	int CurDist = DistPatch(patchA, patchB, PatchSize);

	int CurBestDist = NearestNeighbor.at<Vec3i>(y, x)[2];

	if (CurDist < CurBestDist)
	{
		NearestNeighbor.at<Vec3i>(y, x)[0] = Guees_x;
		NearestNeighbor.at<Vec3i>(y, x)[1] = guess_y;
		NearestNeighbor.at<Vec3i>(y, x)[2] = CurDist;
	}




}

void PatchMatch(const Mat & SourceImage,const Mat & TargetImage, const Mat & Mask,  int nPatchSize, Mat & NearestNeighbor)
{
	// 最近邻数据
	NearestNeighbor = Mat::zeros(SourceImage.size(), CV_32SC3);

	int nIterNum = 0;
	int nIterMaxNum = 5;
	int32_t nMaxCols = TargetImage.cols - nPatchSize - 1;
	int32_t nMaxRows = TargetImage.rows - nPatchSize - 1;

	// 先设置随机位置
	for (int i = 0; i < SourceImage.rows - nPatchSize; i++)
	{
		for (int j = 0; j < SourceImage.cols - nPatchSize; j++)
		{
			int nRandX = rand() % (nMaxCols);   // x 坐标
			int nRandY = rand() % (nMaxRows);   // y 坐标

			NearestNeighbor.at<Vec3i>(i, j)[0] = nRandX;
			NearestNeighbor.at<Vec3i>(i, j)[1] = nRandY;

			Rect RE(j, i, nPatchSize, nPatchSize);
			Mat patchA = SourceImage(RE);

			Rect RE2(nRandX, nRandY, nPatchSize, nPatchSize);
			Mat patchB = TargetImage(RE2);

			NearestNeighbor.at<Vec3i>(i,j)[2] = DistPatch(patchA, patchB, nPatchSize);

		}
	}

	while (nIterNum < nIterMaxNum)
	{
		int nColStart = 1;
		int nColEnd = SourceImage.cols - nPatchSize ;
		int nRowStart = 1;
		int nRowEnd = SourceImage.rows - nPatchSize ;
		int nStep = 1;

		if (nIterNum % 2)
		{
			nColStart = SourceImage.cols - nPatchSize - 2;
			nColEnd = -1;
			nRowStart = SourceImage.rows - nPatchSize - 2;
			nRowEnd = -1;
			nStep = -1;
		}


		for (int i = nRowStart; i != nRowEnd; i += nStep)
		{
			for (int j = nColStart; j != nColEnd; j += nStep)
			{
				// 有效范围内
				if (j - nStep < SourceImage.cols - nPatchSize)
				{
					int nGuessX = NearestNeighbor.at<Vec3i>(i, j - nStep)[0] + nStep;
					int nGuessY = NearestNeighbor.at<Vec3i>(i, j - nStep)[1];

					if (nGuessX < TargetImage.cols - nPatchSize && nGuessX >= 0)
					{
						// propagation 
						GuessAndImprove(SourceImage, TargetImage, Mask, j, i, nGuessX, nGuessY, nPatchSize, NearestNeighbor);
					}

				}
				// 有效范围内
				if (i - nStep < SourceImage.rows - nPatchSize)
				{
					int nGuessX = NearestNeighbor.at<Vec3i>(i - nStep, j)[0];
					int nGuessY = NearestNeighbor.at<Vec3i>(i - nStep, j)[1] + nStep;

					if (nGuessY < TargetImage.rows - nPatchSize && nGuessY >= 0)
					{
						// propagation 
						GuessAndImprove(SourceImage, TargetImage, Mask, j, i, nGuessX, nGuessY, nPatchSize, NearestNeighbor);
					}
				}

				// random guess

				int rs_start = max(TargetImage.rows,  TargetImage.cols);

				int nBestX = NearestNeighbor.at<Vec3i>(i, j)[0];
				int nBestY = NearestNeighbor.at<Vec3i>(i, j)[1];

				for (int mag = rs_start; mag >= 1; mag /= 2) 
				{
					/* Sampling window */
					int xmin = max(nBestX - mag, 0), xmax = min(nBestX + mag + 1, TargetImage.cols - nPatchSize - 1);
					int ymin = max(nBestY - mag, 0), ymax = min(nBestY + mag + 1, TargetImage.rows - nPatchSize - 1);

					int xp = xmin + rand() % (xmax - xmin);
					int yp = ymin + rand() % (ymax - ymin);

					GuessAndImprove(SourceImage, TargetImage, Mask, j, i, xp, yp,  nPatchSize, NearestNeighbor);

				}
			}
		}

		nIterNum++;
	}
}