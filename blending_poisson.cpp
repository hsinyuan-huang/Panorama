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

int mv_cnt = 0;

int dx[4] = {0, 0, 1, -1};
int dy[4] = {1, -1, 0, 0};

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

			string newfname = string(fname);
			newfname.resize(newfname.size() - 4);
			newfname += "_shift.jpg";
			imwrite(newfname.c_str(), pic[imgcnt]);
		}

		imgcnt ++;
	}

	for(int i = 0; i < imgcnt; i++)
		pic[i].convertTo(pic[i], CV_32FC3);

	Mat fin = Mat::zeros(pic[0].rows, cum_yshift[imgcnt-1] + pic[0].cols, CV_32FC3);
	for(int x = 0; x < pic[0].rows; x++)
		for(int y = 0; y < pic[0].cols; y++)
			fin.at<Vec3f>(x, y) = pic[0].at<Vec3f>(x, y);

	for(int i = 1; i < imgcnt; i++)
	{
		int overlap = cum_yshift[i-1] + pic[i-1].cols - cum_yshift[i];

		for(int y = overlap; y < pic[i].cols; y++)
			fin.at<Vec3f>(0, y + cum_yshift[i]) = pic[i].at<Vec3f>(0, y);
		for(int y = overlap; y < pic[i].cols; y++)
			fin.at<Vec3f>(pic[i].rows - 1, y + cum_yshift[i]) = pic[i].at<Vec3f>(pic[i].rows - 1, y);

		for(int x = 0; x < pic[i].rows; x++)
			fin.at<Vec3f>(x, pic[i].cols - 1 + cum_yshift[i]) = pic[i].at<Vec3f>(x, pic[i].cols - 1);

		int T = 10000;

		Mat res(pic[i].rows, pic[i].cols, CV_32FC3);

		for(int x = 1; x < pic[i].rows - 1; x++)
		{
			for(int y = overlap; y < pic[i].cols - 1; y++)
			{
				res.at<Vec3f>(x, y) = 0;
				for(int k = 0; k < 4; k++)
				{
					int nx = x + dx[k];
					int ny = y + dy[k];
					res.at<Vec3f>(x, y) += pic[i].at<Vec3f>(x, y) - pic[i].at<Vec3f>(nx, ny);
					res.at<Vec3f>(x, y) += fin.at<Vec3f>(nx, ny + cum_yshift[i]) - fin.at<Vec3f>(x, y + cum_yshift[i]);
				}
			}
		}

		Mat search_p;
		search_p = res.clone();

		float init_error = 0;

		for(int t = 0; t < T; t++)
		{
			Mat vec(pic[i].rows, pic[i].cols, CV_32FC3);
			for(int x = 1; x < pic[i].rows - 1; x++)
			{
				for(int y = overlap; y < pic[i].cols - 1; y++)
				{
					vec.at<Vec3f>(x, y) = 0;
					for(int k = 0; k < 4; k++)
					{
						int nx = x + dx[k];
						int ny = y + dy[k];

						if(nx >= 1 && nx < pic[i].rows - 1 && ny >= overlap && ny < pic[i].cols - 1)
							vec.at<Vec3f>(x, y) += search_p.at<Vec3f>(x, y) - search_p.at<Vec3f>(nx, ny);
						else
							vec.at<Vec3f>(x, y) += search_p.at<Vec3f>(x, y);
					}
				}
			}

			Vec3f alpha1, alpha2;
			alpha1 = alpha2 = Vec3f(0, 0, 0);
			for(int x = 1; x < pic[i].rows - 1; x++)
			{
				for(int y = overlap; y < pic[i].cols - 1; y++)
				{
					alpha1 += res.at<Vec3f>(x, y).mul(res.at<Vec3f>(x, y));
					alpha2 += search_p.at<Vec3f>(x, y).mul(vec.at<Vec3f>(x, y));
				}
			}
			Vec3f alpha;
			divide(alpha1, alpha2, alpha);

			Vec3f gamma1, gamma2;
			gamma1 = gamma2 = Vec3f(0, 0, 0);
			gamma2 = alpha1;
			for(int x = 1; x < pic[i].rows - 1; x++)
			{
				for(int y = overlap; y < pic[i].cols - 1; y++)
				{
					fin.at<Vec3f>(x, y + cum_yshift[i]) += alpha.mul(search_p.at<Vec3f>(x, y));
					res.at<Vec3f>(x, y) -= alpha.mul(vec.at<Vec3f>(x, y));

					gamma1 += res.at<Vec3f>(x, y).mul(res.at<Vec3f>(x, y));
				}
			}
			Vec3f gamma;
			divide(gamma1, gamma2, gamma);

			for(int x = 1; x < pic[i].rows - 1; x++)
			{
				for(int y = overlap; y < pic[i].cols - 1; y++)
				{
					search_p.at<Vec3f>(x, y) = res.at<Vec3f>(x, y) + gamma.mul(search_p.at<Vec3f>(x, y));
				}
			}

			printf("Iter %d: Error = %f %f %f\n", t, gamma1[0], gamma1[1], gamma1[2]);
			if(t == 0) init_error = max(max(gamma1[0], gamma1[1]), gamma1[2]);

			if(max(max(gamma1[0], gamma1[1]), gamma1[2]) < 0.01 * init_error) break;

			// Gaussian Seidal + Jacobi
			/*
			Mat diff(pic[i].rows, pic[i].cols, CV_32FC3);

			for(int x = 1; x < pic[i].rows - 1; x++)
			{
				for(int y = overlap; y < pic[i].cols - 1; y++)
				{
					diff.at<Vec3f>(x, y) = 0;
					for(int k = 0; k < 4; k++)
					{
						int nx = x + dx[k];
						int ny = y + dy[k];
						diff.at<Vec3f>(x, y) += pic[i].at<Vec3f>(x, y) - pic[i].at<Vec3f>(nx, ny);
						diff.at<Vec3f>(x, y) += fin.at<Vec3f>(nx, ny + cum_yshift[i]);
					}

					fin.at<Vec3f>(x, y + cum_yshift[i]) = diff.at<Vec3f>(x, y) / 4.f;
				}
			}
			*/

			// Jacobi
			/*
			for(int x = 1; x < pic[i].rows - 1; x++)
			{
				for(int y = overlap; y < pic[i].cols - 1; y++)
				{
					fin.at<Vec3f>(x, y + cum_yshift[i]) = diff.at<Vec3f>(x, y) / 4.f;
				}
			}
			*/

			if(t % 10 == 0)
			{
				Mat out;
				fin.convertTo(out, CV_8UC3);

				string foutname = dir + "panorama" + to_string(mv_cnt) + ".jpg";
				imwrite(foutname.c_str(), out);
				mv_cnt ++;
			}
		}
	}

	Mat out;
	fin.convertTo(out, CV_8UC3);
	imwrite("panorama.jpg", out);
}
