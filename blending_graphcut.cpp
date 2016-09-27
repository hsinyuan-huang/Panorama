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
		Mat diff(pic[i].rows, cum_yshift[i-1] + pic[i-1].cols - cum_yshift[i], CV_32SC1);
		for(int x = 0; x < pic[i].rows; x++)
			for(int y = 0; y < diff.cols; y++)
			{
				Vec3b past = pic[i-1].at<Vec3b>(x, y + cum_yshift[i] - cum_yshift[i-1]);
				Vec3b  now = pic[i].at<Vec3b>(x, y);
				int val = max(max(abs((int)past[0] - (int)now[0]), abs((int)past[1] - (int)now[1])),
						abs((int)past[2] - (int)now[2]));
				diff.at<int>(x, y) = val;
			}

		priority_queue<pair<int, pair<int, int> > > Q;

		// "choice" matrix is the decision of which picture to use:
		// -1 to use the (i-1)th picture,
		//  1 to use the ith picture.
		//
		// This matrix only contains the precise intersection of the two image,
		// as you can see from the size of this mat.

		Mat choice = Mat::zeros(pic[i].rows, cum_yshift[i-1] + pic[i-1].cols - cum_yshift[i], CV_32SC1);
		for(int x = 0; x < pic[i].rows; x++)
		{
			Q.push(MP(1000, MP(-1, x * choice.cols + 0)));
			Q.push(MP(1000, MP(-1, x * choice.cols + 1)));
			Q.push(MP(1000, MP(-1, x * choice.cols + 2)));
			Q.push(MP(1000, MP(-1, x * choice.cols + 3)));
			Q.push(MP(1000, MP(-1, x * choice.cols + 4)));
			Q.push(MP(1000, MP(1, x * choice.cols + choice.cols - 1)));
			Q.push(MP(1000, MP(1, x * choice.cols + choice.cols - 2)));
			Q.push(MP(1000, MP(1, x * choice.cols + choice.cols - 3)));
			Q.push(MP(1000, MP(1, x * choice.cols + choice.cols - 4)));
			Q.push(MP(1000, MP(1, x * choice.cols + choice.cols - 5)));
			Q.push(MP(1000, MP(1, x * choice.cols + choice.cols - 6)));
			Q.push(MP(1000, MP(1, x * choice.cols + choice.cols - 7)));
		}

		int dx[4] = {0, 0, 1, -1};
		int dy[4] = {1, -1, 0, 0};

		while(!Q.empty())
		{
			pair<int, pair<int, int> > one_pix = Q.top(); Q.pop();
			int clr = one_pix.S.F;
			int x = one_pix.S.S / choice.cols;
			int y = one_pix.S.S % choice.cols;

			if(choice.at<int>(x, y) != 0) continue;
			choice.at<int>(x, y) = clr;

			for(int k = 0; k < 4; k++)
			{
				int nx = x + dx[k];
				int ny = y + dy[k];
				if(nx < 0 || nx >= choice.rows || ny < 0 || ny >= choice.cols)
					continue;

				if(choice.at<int>(nx, ny) == 0)
				{
					Q.push(MP(diff.at<int>(nx, ny), MP(clr, nx * choice.cols + ny)));
				}
			}
		}

		// Graph cut
		for(int x = 0; x < pic[i].rows; x++)
		{
			for(int y = 0; y < choice.cols; y++)
			{
				if(choice.at<int>(x, y) == 1)
				{
					fin.at<Vec3b>(x, y + cum_yshift[i]) = pic[i].at<Vec3b>(x, y);

					int boundary = 0;
					for(int k = 0; k < 4; k++)
					{
						int nx = x + dx[k];
						int ny = y + dy[k];
						if(nx < 0 || nx >= choice.rows || ny < 0 || ny >= choice.cols)
							continue;

						if(choice.at<int>(nx, ny) == -1)
							boundary = 1;
					}

					//For the boundary line
					//if(boundary)
					//	fin.at<Vec3b>(x, y + cum_yshift[i]) = Vec3b(0, 0, 255);
				}
			}

			for(int y = choice.cols; y < pic[i].cols; y++)
				fin.at<Vec3b>(x, y + cum_yshift[i]) = pic[i].at<Vec3b>(x, y);
		}
	}

	imwrite("panorama.jpg", fin);
}
