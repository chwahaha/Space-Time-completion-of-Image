


#include <opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;


Vec3f MeanShift(vector<Vec3b> vecVoteColor, vector<float> vecVoteWeight, int sigma)
{
	Vec3f vecMean3f = Vec3f{0,0,0};

	float fTotalWeight = 0;
	for (int i = 0; i < vecVoteColor.size(); i++)
	{
		vecMean3f += (Vec3f)(vecVoteColor[i] * vecVoteWeight[i]);
		fTotalWeight += vecVoteWeight[i];
	}

	vecMean3f = vecMean3f / fTotalWeight;

	for (int t = 0; t < 5; t++)
	{
		float scale = 3 / (1 << t);// ·ù¶ÈÓÉ3sigmaµ½0.2sigma

		float Thresh = (scale * sigma) * (scale * sigma);

		int nIterNum = 0;
		while (1)
		{
			Vec3f vecCurMean3f = Vec3f{ 0,0,0 };
			int GroupNum = 0;

			fTotalWeight = 0;
			for (int i = 0; i < vecVoteColor.size(); i++)
			{
				Vec3f Diff = (Vec3f)vecVoteColor[i] - vecMean3f;

				if (Diff[0] * Diff[0] + Diff[1] * Diff[1] + Diff[2] * Diff[2] < Thresh)
				{
					vecCurMean3f += vecVoteColor[i] * vecVoteWeight[i];

					fTotalWeight += vecVoteWeight[i];

					GroupNum++;
				}
			}


			if (GroupNum == 0)
			{
				break;
			}
			vecCurMean3f = vecCurMean3f / fTotalWeight;

			Vec3f Diff = vecCurMean3f - vecMean3f;

			if (Diff[0] * Diff[0] + Diff[1] * Diff[1] + Diff[2] * Diff[2] < 10)
			{
				break;
			}

			vecMean3f = vecCurMean3f;

			nIterNum++;
			if (nIterNum > 10)
				break;
		}
	}

	return vecMean3f;

}