#include <stdio.h>
#include <iostream>
#include <vector>
#include <queue>
#include <opencv2/opencv.hpp>

#define MP make_pair
#define F first
#define S second

using namespace cv;
using namespace std;

bool equalize = 1;
int imgcnt = 0;

Mat pic[100];
int cum_yshift[100];

int main(int argc, char *argv[])
{
	if(argc != 2)
	{
		fprintf(stderr, "Usage: ./blending input_directory/\n");
		exit(-1);
	}

	string dir = string(argv[1]);
	if(dir.back() != '/') dir.push_back('/');

	string metadata_nme = dir + "metadata.txt";
	FILE *metadata = fopen(metadata_nme.c_str(), "r");

	imgcnt = 0;	

	char fname[100];
	while(~fscanf(metadata, "%s%d", fname, &cum_yshift[imgcnt]))
	{
		pic[imgcnt] = imread(fname);

		if(equalize && imgcnt > 0)
		{
			int overlap = cum_yshift[imgcnt-1] + pic[imgcnt-1].cols - cum_yshift[imgcnt];

			float inten_del = 0, cnt = 0;
			for(int x = 0; x < pic[imgcnt].rows; x++)
			{
				if(x < 2 || x >= pic[imgcnt].rows - 2)
				{
					for(int y = 0; y < overlap; y++)
					{
						Vec3b my_clr = pic[imgcnt].at<Vec3b>(x, y);
						Vec3b fin_clr = pic[imgcnt-1].at<Vec3b>(x, y + cum_yshift[imgcnt] - cum_yshift[imgcnt-1]);

						float my_inten = 0.114 * my_clr[0] + 0.587 * my_clr[1] + 0.299 * my_clr[2];
						float fin_inten = 0.114 * fin_clr[0] + 0.587 * fin_clr[1] + 0.299 * fin_clr[2];

						inten_del += my_inten - fin_inten;
						cnt++;
					}
				}
			}
			inten_del /= cnt;
			pic[imgcnt].convertTo(pic[imgcnt], -1, 1, -inten_del);
		}

		imgcnt ++;
	}

	Mat fin(pic[0].rows, cum_yshift[imgcnt-1] + pic[0].cols, CV_8UC3);

	for(int x = 0; x < pic[0].rows; x++)
		for(int y = 0; y < pic[0].cols; y++)
			fin.at<Vec3b>(x, y) = pic[0].at<Vec3b>(x, y);

	for(int i = 1; i < imgcnt; i++)
	{
		// Alpha blending
		int overlap = cum_yshift[i-1] + pic[i-1].cols - cum_yshift[i];
		for(int x = 0; x < pic[i].rows; x++)
		{
			for(int y = 0; y < overlap; y++)
			{
				float alpha = 1.0 * y / overlap;
				fin.at<Vec3b>(x, y + cum_yshift[i]) = alpha * pic[i].at<Vec3b>(x, y)
												+ (1-alpha) * fin.at<Vec3b>(x, y + cum_yshift[i]);
			}

			for(int y = overlap; y < pic[i].cols; y++)
				fin.at<Vec3b>(x, y + cum_yshift[i]) = pic[i].at<Vec3b>(x, y);
		}
	}

	imwrite("panorama.jpg", fin);
}
