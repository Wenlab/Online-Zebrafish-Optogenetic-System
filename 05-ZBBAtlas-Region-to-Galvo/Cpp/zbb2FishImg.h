#pragma once
class zbb2FishImg
{

public:
	zbb2FishImg();
	~zbb2FishImg();

	void getZBB2FixAffineMatrix();
	void getFix2MovingAffineMatrix();
	void getCropPoint();
	void getRotationAngle();

	void ZBB2FixTransform();
	void Fix2MovingTransform();
	void Moving2FishImgTransform();


};

