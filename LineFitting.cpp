
//3D best line fitting  
template<class Vector3>
std::pair < Vector3, Vector3 > best_line_from_points(const std::vector<Vector3>& c)
{
	// copy coordinates to  matrix in Eigen format
	size_t num_atoms = c.size();
	Eigen::Matrix< Vector3::Scalar, Eigen::Dynamic, Eigen::Dynamic > centers(num_atoms, 3);
	for (size_t i = 0; i < num_atoms; ++i) centers.row(i) = c[i];

	Vector3 origin = centers.colwise().mean();
	Eigen::MatrixXd centered = centers.rowwise() - origin.transpose();
	Eigen::MatrixXd cov = centered.adjoint() * centered;
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(cov);
	Vector3 axis = eig.eigenvectors().col(2).normalized();

	return std::make_pair(origin, axis);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//3d points to 3d line distance calculation
//D = ||(P2-P1) * (P1-P0)|| / ||P2-P1|| = norm (cross (p2-p1, p1-p0)) / norm(p2-p1)
double suqareDis = (line_dir.cross(line_pt - pt)).squaredNorm() / line_dir.squaredNorm();

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//#include <Eigen/QR>
//#include <stdio.h>
//#include <vector>
//2D line fitting
void polyfit(const std::vector<double>& xv, const std::vector<double>& yv, std::vector<double>& coeff, int order)
{
	Eigen::MatrixXd A(xv.size(), order + 1);
	Eigen::VectorXd yv_mapped = Eigen::VectorXd::Map(&yv.front(), yv.size());
	Eigen::VectorXd result;

	assert(xv.size() == yv.size());
	assert(xv.size() >= order + 1);

	// create matrix
	for (size_t i = 0; i < xv.size(); i++)
		for (size_t j = 0; j < order + 1; j++)
			A(i, j) = pow(xv.at(i), j);

	// solve for linear least squares fit
	result = A.householderQr().solve(yv_mapped);

	coeff.resize(order + 1);
	for (size_t i = 0; i < order + 1; i++)
		coeff[i] = result[i];
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//2D line fit
bool linearFit(double& A, double& B, double& C, vector<float> fitVx, vector<float> fitVy) {

	int size = fitVx.size();

	if (size < 2)
	{
		//		cout << "the number of points are too less for linear fit" <<endl;
		A = 0;
		B = 0;
		C = 0;
		return true;
	}

	double X = 0;
	double Y = 0;
	double XY = 0;
	double X2 = 0;
	double Y2 = 0;

	for (int i = 0; i < size; i++) {
		double x = fitVx[i];
		double y = fitVy[i];

		X += x;
		Y += y;
		XY += x * y;
		X2 += x * x;
		Y2 += y * y;
	}


	X /= size;
	Y /= size;
	XY /= size;
	X2 /= size;
	Y2 /= size;

	A = -(XY - X * Y); //!< Common for both solution

	double Bx = X2 - X * X;
	double By = Y2 - Y * Y;

	if (fabs(Bx) < fabs(By)) //!< Test verticality/horizontality
	{ // Line is more Vertical.
		B = By;
		std::swap(A, B);
	}
	else
	{   // Line is more Horizontal.
		// Classical solution, when we expect more horizontal-like line
		B = Bx;
	}
	C = -(A * X + B * Y);

	//	cout << "A= " << A << "   B=" << B << "   C=" << C << endl;
	return true;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

