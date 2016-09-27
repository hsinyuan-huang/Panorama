#include <stdio.h>
#include <iostream>
#include <sys/stat.h>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat GaussianFilter(Mat ori, float sigma)
{
	// A good size for kernel support
	int ksz = (int)((sigma - 0.35) / 0.15);
	if(ksz < 0) ksz = 1;
	if(ksz % 2 == 0) ksz ++; // Must be odd

	Size kernel_size(ksz, ksz);

	Mat fin;
	GaussianBlur(ori, fin, kernel_size, sigma, sigma, BORDER_REFLECT_101);

	return fin;
}

Mat SmoothGrad(Mat ori, float sigma, int dx, int dy)
{
	Mat fin;
	Sobel(ori, fin, -1, dx, dy, 3, 1, 0, BORDER_REFLECT_101);

	return GaussianFilter(fin, sigma);
}

Mat fHM;
bool comp_fHM(Vec2i A, Vec2i B)
{
	return fHM.at<float>(A[0], A[1]) > fHM.at<float>(B[0], B[1]);
}

vector<float> suppr;
bool comp_suppr(int A, int B)
{
	return suppr[A] > suppr[B];
}

float dist2(Vec2i A, Vec2i B)
{
	return 1.f * (A[0] - B[0]) * (A[0] - B[0]) + 1.f * (A[1] - B[1]) * (A[1] - B[1]);
}

void solve2x2(float a, float b, float c, float d, float e, float f, float &x, float &y)
{
	float det = a * d - b * c;
	
	x = (d * e - b * f) / det;
	y = (-c * e + a * f) / det;
}

float subpix(Mat &gray_pic, float x, float y)
{
	int intx = (int)x;
	int inty = (int)y;

	if(x < 0 || intx >= gray_pic.rows - 1) return -1;
	if(y < 0 || inty >= gray_pic.cols - 1) return -1;

	float a = gray_pic.at<float>(intx, inty);
	float b = gray_pic.at<float>(intx+1, inty);
	float c = gray_pic.at<float>(intx, inty+1);
	float d = gray_pic.at<float>(intx+1, inty+1);

	float u = x - intx;
	float v = y - inty;
	
	return ((1-u) * a + u * b) * (1-v) + ((1-u) * c + u * d) * v;
}

Vec3f bilinear(Mat &color_pic, float x, float y)
{
	int intx = (int)x;
	int inty = (int)y;

	if(x < 0 || intx >= color_pic.rows - 1) return Vec3f(-1, -1, -1);
	if(y < 0 || inty >= color_pic.cols - 1) return Vec3f(-1, -1, -1);

	Vec3f a = color_pic.at<Vec3f>(intx, inty);
	Vec3f b = color_pic.at<Vec3f>(intx+1, inty);
	Vec3f c = color_pic.at<Vec3f>(intx, inty+1);
	Vec3f d = color_pic.at<Vec3f>(intx+1, inty+1);

	float u = x - intx;
	float v = y - inty;
	
	return ((1-u) * a + u * b) * (1-v) + ((1-u) * c + u * d) * v;
}

class descriptor
{
public:
	float x, y, theta;
	float inten[64];
	//vector<float> vec_inten(inten, inten + 64) to transform
};

class image_feats
{
public:
	vector<descriptor> feats;
};

// -1 for no match, i.e. the closest point is not too close
int find_match(flann::Index &kdtree, descriptor A)
{
	flann::SearchParams knn_param;
	vector<int> knn_idx(2);
	vector<float> knn_dis(2);

	vector<float> vec_inten(A.inten, A.inten + 64);
	kdtree.knnSearch(vec_inten, knn_idx, knn_dis, 2, knn_param);

	if(knn_dis[0] < 0.6 * knn_dis[1])
		return knn_idx[0];
	return -1;
}

// level start from 0
vector<Vec3f> interest_point(Mat pic, int num_IP, int level, vector<descriptor> &vec_desc)
{
	Mat gray_pic;

	cvtColor(pic, gray_pic, CV_BGR2GRAY);
	gray_pic.convertTo(gray_pic, CV_32F);

	for(int l = 0; l < level; l++)
	{
		gray_pic = GaussianFilter(gray_pic, 1.0);
		Mat new_gray = Mat::zeros(gray_pic.rows / 2, gray_pic.cols / 2, CV_32F);
		Mat new_pic = Mat::zeros(gray_pic.rows / 2, gray_pic.cols / 2, CV_8UC3);

		for(int r = 0; r < new_gray.rows; r++)
			for(int c = 0; c < new_gray.cols; c++)
			{
				new_gray.at<float>(r, c) = gray_pic.at<float>(2 * r, 2 * c);
				new_pic.at<Vec3b>(r, c) = pic.at<Vec3b>(2 * r, 2 * c);
			}
		
		gray_pic = new_gray;
		pic = new_pic;
	}

	string lvl = to_string(level);
	imwrite("gray" + lvl + ".jpg", gray_pic);

	Mat clone_pic1 = pic.clone();
	Mat clone_pic2 = pic.clone();

	Mat Gx = SmoothGrad(gray_pic, 1.0, 1, 0);
	Mat Gy = SmoothGrad(gray_pic, 1.0, 0, 1);
	
	Mat Sx2 = GaussianFilter(Gx.mul(Gx), 1.5);
	Mat Sy2 = GaussianFilter(Gy.mul(Gy), 1.5);
	Mat Sxy = GaussianFilter(Gx.mul(Gy), 1.5);

	fHM = (Sx2.mul(Sy2) - Sxy.mul(Sxy)) / (Sx2 + Sy2);

	vector<Vec2i> corners;
	for(int i = 1; i < fHM.rows - 1; i++) // Exclude boundary
	{
		for(int j = 1; j < fHM.cols - 1; j++) // Exclude boundary
		{
			bool local_max = 1;
			for(int di = -1; di <= 1; di++)
				for(int dj = -1; dj <= 1; dj++)
					if(!(di == 0 && dj == 0)
					&& fHM.at<float>(i, j) < fHM.at<float>(i+di, j+dj))
						local_max = 0;

			if(local_max && fHM.at<float>(i, j) > 10.f)
			{
				corners.push_back(Vec2i(i, j));

				// Color the original picture w/ detected corners
				for(int di = -1; di <= 1; di++)
					for(int dj = -1; dj <= 1; dj++)
						clone_pic1.at<Vec3b>(i+di, j+dj) = Vec3b(0, 0, 255);
			}
		}
	}

	imwrite("all_feat" + lvl + ".jpg", clone_pic1);

	sort(corners.begin(), corners.end(), comp_fHM);
	corners.resize(min((int)corners.size(), num_IP * 20));
	
	suppr.clear();
	
	vector<int> perm;
	for(int i = 0; i < (int)corners.size(); i++)
	{
		float rad = 1e50;
		for(int j = 0; j < i; j++)
		{
			if(fHM.at<float>(corners[i][0], corners[i][1])
			 < 0.9 * fHM.at<float>(corners[j][0], corners[j][1]))
				rad = min(dist2(corners[i], corners[j]), rad);
			else break;
		}

		suppr.push_back(rad);
		perm.push_back(i);
	}

	sort(perm.begin(), perm.end(), comp_suppr);

	vector<Vec3f> intrst;
	Mat SGx = SmoothGrad(gray_pic, 4.5, 1, 0);
	Mat SGy = SmoothGrad(gray_pic, 4.5, 0, 1);

	Mat Blur_gray = GaussianFilter(gray_pic, 2.f);

	for(int i = 0; i < num_IP; i++)
	{
		int x = corners[perm[i]][0];
		int y = corners[perm[i]][1];

		// Sub-pixel correction
		Vec3f xytheta;

		float a = fHM.at<float>(x+1, y) + fHM.at<float>(x-1, y) - 2 * fHM.at<float>(x, y);
		float b = (fHM.at<float>(x-1, y-1) - fHM.at<float>(x-1, y+1)
				-  fHM.at<float>(x+1, y-1) - fHM.at<float>(x+1, y+1)) / 4.f;
		float c = b;
		float d = fHM.at<float>(x, y+1) + fHM.at<float>(x, y-1) - 2 * fHM.at<float>(x, y);
		float e = (fHM.at<float>(x+1, y) - fHM.at<float>(x-1, y)) / 2.f;
		float f = (fHM.at<float>(x, y+1) - fHM.at<float>(x, y-1)) / 2.f;

		solve2x2(a, b, c, d, e, f, xytheta[0], xytheta[1]);
		
		xytheta[0] = x - xytheta[0];
		xytheta[1] = y - xytheta[1];

		float cSGx = SGx.at<float>(x, y);
		float cSGy = SGy.at<float>(x, y);
		float SGlen = sqrt(cSGx * cSGx + cSGy * cSGy);
		
		xytheta[2] = atan2(cSGy / SGlen, cSGx / SGlen);

		// Descriptor
		float pixvs[64];
		bool bad_desc = 0;
		
		for(int dx = 0; dx <= 7; dx ++)
		{
			for(int dy = 0; dy <= 7; dy ++)
			{
				float cost = cos(xytheta[2]), sint = sin(xytheta[2]);
				float delx =  5 * (dx - 3.5), dely =  5 * (dy - 3.5);

				float sx = xytheta[0] + delx * cost + dely * sint;
				float sy = xytheta[1] + delx * (-sint) + dely * cost;

				float pixv = subpix(Blur_gray, sx, sy);
				if(pixv < 0)
				{
					bad_desc = 1;
					break;
				}
				pixvs[dx * 8 + dy] = pixv;

			}
			if(bad_desc) break;
		}

		// Store the point
		for(int l = 0; l < level; l++)
		{
			xytheta[0] *= 2;
			xytheta[1] *= 2;
		}
		//printf("%d %d -> %f %f %f\n", x, y, xytheta[0], xytheta[1], xytheta[2]);

		if(!bad_desc)
		{
			descriptor dpix;

			dpix.x = xytheta[0];
			dpix.y = xytheta[1];
			dpix.theta = xytheta[2];

			// Normalize pixvs
			float mu = 0, sigma = 0;

			for(int k = 0; k < 64; k++)
				mu += pixvs[k] / 64.f;

			for(int k = 0; k < 64; k++)
				sigma += (pixvs[k] - mu) * (pixvs[k] - mu);
			sigma = sqrt(sigma);

			for(int k = 0; k < 64; k++)
				dpix.inten[k] = (pixvs[k] - mu) / sigma;
			
			vec_desc.push_back(dpix);
			//printf("%f %f %f (mu %f sigma %f)\n", dpix.x, dpix.y, dpix.theta, mu, sigma);
		}

		intrst.push_back(xytheta); // for visualization
	
		// Color the original picture w/ detected corners using ANMS
		for(int di = -1; di <= 1; di++)
			for(int dj = -1; dj <= 1; dj++)
				clone_pic2.at<Vec3b>(x+di, y+dj) = Vec3b(0, 0, 255);
	}

	imwrite("anms_feat" + lvl + ".jpg", clone_pic2);

	return intrst;
}

void visualize_intrst_pts(Mat pic, vector<Vec3f> intrst_pt[], int level)
{
	Mat interest_pic = pic.clone();

	int len = 20;
	float clr = 0;
	for(int l = 0; l < level; l++)
	{
		for(int i = 0; i < (int)intrst_pt[l].size(); i++)
		{
			Vec3f xytheta = intrst_pt[l][i];
			float x = xytheta[0];
			float y = xytheta[1];
			float theta = xytheta[2];
			arrowedLine(interest_pic, Point(y, x), Point(y + len * cos(theta), x + len * sin(theta)),
						Scalar(255 * clr, 0, 255 * (1 - clr)));
		}

		clr += 1.f / level;
		len *= 2;
	}

	imwrite("MSOP.jpg", interest_pic);
}

void find_feats(Mat pic, image_feats &img_f)
{
	img_f.feats.clear();

	vector<Vec3f> intrst_pt[4];
	intrst_pt[0] = interest_point(pic, 500, 0, img_f.feats);
	intrst_pt[1] = interest_point(pic,  50, 1, img_f.feats);
	intrst_pt[2] = interest_point(pic, 	25, 2, img_f.feats);
	intrst_pt[3] = interest_point(pic,  10, 3, img_f.feats);

	visualize_intrst_pts(pic, intrst_pt, 4);
	printf("descriptor_num: %lu\n", img_f.feats.size());	
}

void Cylinder_Warp(Mat &ori_pic, Mat &cyl_pic, float focal)
{
	Mat pic;
	ori_pic.convertTo(pic, CV_32FC3);

	float width = pic.cols - 1;
	float height = pic.rows - 1;
	float mx_t = focal * atan(width / 2.f / focal) * 2.f;

	cyl_pic = Mat(pic.rows, ceil(mx_t), CV_32FC3);
	
	for(int i = 0; i < cyl_pic.rows; i++)
	{
		for(int j = 0; j < cyl_pic.cols; j++)
		{
			float theta = (j - (cyl_pic.cols - 1) / 2.f);
			float hval = (i - (cyl_pic.rows - 1) / 2.f);
			float y = tan(theta / focal) * focal;
			float x = hval / focal * sqrt(y * y + focal * focal);
			float xshift = x + height / 2.f;
			float yshift = y + width / 2.f;

			cyl_pic.at<Vec3f>(i, j) = bilinear(pic, xshift, yshift);
		}
	}
}

void Crop_Cylinder(Mat &cyl_pic, Mat &crp_pic)
{
	int st = -1, end = cyl_pic.rows;
	for(int j = 0; j < cyl_pic.cols; j++)
	{
		int mnpos = cyl_pic.rows, mxpos = -1;
		for(int i = 0; i < cyl_pic.rows; i++)
		{
			if(cyl_pic.at<Vec3f>(i, j)[0] != -1)
			{
				mnpos = min(mnpos, i);
				mxpos = max(mxpos, i);
			}
		}
		st = max(st, mnpos);
		end = min(end, mxpos);
	}
	
	crp_pic = Mat(end - st + 1, cyl_pic.cols, CV_32FC3);
	for(int j = 0; j < cyl_pic.cols; j++)
	{
		for(int i = st; i <= end; i++)
			crp_pic.at<Vec3f>(i - st, j) = cyl_pic.at<Vec3f>(i, j);
	}
}

void Convert2RGB(Mat &fpic, Mat &pic)
{
	pic = Mat(fpic.rows, fpic.cols, CV_8UC3);
	
	bool badpic = 0;
	for(int i = 0; i < fpic.rows; i++)
	{
		for(int j = 0; j < fpic.cols; j++)
		{
			for(int c = 0; c < 3; c++)
			{
				float intens = fpic.at<Vec3f>(i, j)[c];
				if(intens < 0) badpic = 1;
				pic.at<Vec3b>(i, j)[c] = (int)max(intens, 0.f);
			}
		}
	}
	printf("Picture exists holes: %d\n", badpic);

	imwrite("cyl2rgb.jpg", pic);
}

void warpNcrop(Mat &pic, float focal)
{
	Mat cyl;
	Cylinder_Warp(pic, cyl, focal);

	Mat crp;
	Crop_Cylinder(cyl, crp);
	
	Convert2RGB(crp, pic);
}

void align2pics_w_feat(Mat pic1, Mat pic2, image_feats fts1, image_feats fts2, float &dx, float &dy)
{
	Mat pic1_mtch = pic1.clone();
	Mat pic2_mtch = pic2.clone();

	Mat feat_mat((int)fts2.feats.size(), 64, CV_32F);
	for(int i = 0; i < (int)fts2.feats.size(); i++)
		for(int j = 0; j < 64; j++)
			feat_mat.at<float>(i, j) = fts2.feats[i].inten[j];

	flann::AutotunedIndexParams kdtree_param(0.97);
	flann::Index kdtree(feat_mat, kdtree_param);
	
	vector<pair<int, int> > match;
	for(int i = 0; i < (int)fts1.feats.size(); i++)
	{
		descriptor des = fts1.feats[i];
		int j = find_match(kdtree, des);
		if(j < 0) continue;

		match.push_back(make_pair(i, j));
		
		int x1 = fts1.feats[i].x, y1 = fts1.feats[i].y;
		for(int di = -1; di <= 1; di++)
			for(int dj = -1; dj <= 1; dj++)
				pic1_mtch.at<Vec3b>(x1+di, y1+dj) = Vec3b(0, 0, 255);

		int x2 = fts2.feats[j].x, y2 = fts2.feats[j].y;
		for(int di = -1; di <= 1; di++)
			for(int dj = -1; dj <= 1; dj++)
				pic2_mtch.at<Vec3b>(x2+di, y2+dj) = Vec3b(0, 0, 255);
	}

	imwrite("match1.jpg", pic1_mtch);
	imwrite("match2.jpg", pic2_mtch);
	
	printf("# of matches: %d\n", (int)match.size());

	srand(1023);

	int mx_inlier = -1;
	for(int k = 0; k < 300; k++)
	{
		float tdx = 0, tdy = 0;
		for(int n = 0; n < 5; n++)
		{
			int s = rand() % match.size();
			int i = match[s].first;
			int j = match[s].second;

			int x1 = fts1.feats[i].x, y1 = fts1.feats[i].y;
			int x2 = fts2.feats[j].x, y2 = fts2.feats[j].y;

			tdx += (x2 - x1) / 5.f;
			tdy += (y2 - y1) / 5.f;
		}

		int inlier = 0;
		for(int n = 0; n < (int)match.size(); n++)
		{
			int i = match[n].first;
			int j = match[n].second;

			int x1 = fts1.feats[i].x, y1 = fts1.feats[i].y;
			int x2 = fts2.feats[j].x, y2 = fts2.feats[j].y;

			float dist = (x2 - x1 - tdx) * (x2 - x1 - tdx) + (y2 - y1 - tdy) * (y2 - y1 - tdy);
			if(dist < 5) inlier ++;
		}
	
		if(inlier > mx_inlier){
			dx = tdx;
			dy = tdy;
			mx_inlier = inlier;
		}
	}

	printf("Inlier number: %d\n", mx_inlier);

	Mat combined = Mat::zeros(pic1.rows + abs(dx), pic1.cols + abs(dy), CV_8UC3);

	int idx = (int)dx, idy = (int)dy;
	for(int i = 0; i < pic1.rows; i++)
		for(int j = 0; j < pic1.cols; j++)
			combined.at<Vec3b>(idx < 0? i: i + idx, idy < 0? j: j + idy) = pic1.at<Vec3b>(i, j);

	for(int i = 0; i < pic2.rows; i++)
		for(int j = 0; j < pic2.cols; j++)
			combined.at<Vec3b>(idx < 0? i - idx: i, idy < 0? j - idy: j) = pic2.at<Vec3b>(i, j);

	imwrite("combined.jpg", combined);
}

int main(int argc, char *argv[])
{
	if(argc != 3)
	{
		fprintf(stderr, "Usage: ./MSOP input_directory/ output_directory/\n");
		exit(-1);
	}

	string dir = string(argv[1]);
	if(dir.back() != '/') dir.push_back('/');

	string metadata_nme = dir + "img_list.txt";
	FILE *metadata = fopen(metadata_nme.c_str(), "r");
	
	float focal = 1000;
	fscanf(metadata, "%f", &focal);

	int imgcnt = 0;
	Mat pic[100];

	char img_name[100];
	while(~fscanf(metadata, "%s", img_name))
	{
		string path = dir + string(img_name);
		pic[imgcnt] = imread(path.c_str());

		if(!pic[imgcnt].data){
			fprintf(stderr, "No image data for %s\n", img_name);
			exit(-1);
		}
		warpNcrop(pic[imgcnt], focal);

		printf("Read in %s\n", img_name);
		imgcnt ++;
	}

	image_feats img_fts[100];
	for(int i = 0; i < imgcnt; i++)
		find_feats(pic[i], img_fts[i]);

	float dx[100], dy[100];
	for(int i = 0; i < imgcnt - 1; i++)
	{
		align2pics_w_feat(pic[i], pic[i+1], img_fts[i], img_fts[i+1], dx[i], dy[i]);
		printf("delta x = %f, delta y = %f\n", dx[i], dy[i]);
	}

	float cdx[100], cdy[100];
	cdx[0] = cdy[0] = 0;

	float mxcdx = 0, mncdx = 0;
	for(int i = 0; i < imgcnt - 1; i++)
	{
		dx[i] *= -1;
		dy[i] *= -1;
		printf("final delta x = %f, delta y = %f\n", dx[i], dy[i]);

		cdx[i+1] = cdx[i] + dx[i];
		cdy[i+1] = cdy[i] + dy[i];

		printf("corres cum dx = %f, cum dy = %f\n", cdx[i+1], cdy[i+1]);

		mxcdx = max(mxcdx, cdx[i+1]);
		mncdx = min(mncdx, cdx[i+1]);
	}

	int st = ceil(mxcdx) + 1, end = pic[0].rows - ceil(-mncdx) - 1;
	printf("%f %f -> %d~%d\n", mncdx, mxcdx, st, end);

	string out_dir = string(argv[2]);
	if(out_dir.back() != '/') out_dir.push_back('/');

	mkdir(out_dir.c_str(), 0777);
	string out_metadata_nme = out_dir + "metadata.txt";
	FILE *out_metadata = fopen(out_metadata_nme.c_str(), "w");

	for(int i = 0; i < imgcnt; i++)
	{
		Mat fpic; pic[i].convertTo(fpic, CV_32F);
		Mat fin_img = Mat(end - st, pic[0].cols - 2, CV_8UC3);
		
		for(int x = st; x < end; x++)
		{
			for(int y = 0; y < fin_img.cols; y++)
			{
				Vec3f v = bilinear(fpic, x - cdx[i], y);
				fin_img.at<Vec3b>(x - st, y) = Vec3b((int)v[0], (int)v[1], (int)v[2]);
			}
		}

		string fin_img_nme = out_dir + "fin_img" + to_string(i) + ".jpg";
		imwrite(fin_img_nme.c_str(), fin_img);

		fprintf(out_metadata, "%s %d\n", fin_img_nme.c_str(), (int)(round(cdy[i]) + 1e-6));
	}
}
