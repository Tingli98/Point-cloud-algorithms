
// 3D plane 
////////////////////////////////////////////////////////////////////////////////////////////////////////
//transformation Matrix, project points to plane
ip = filteredNeighboursCloud[n];
op.x = static_cast<float> (TransformPlaneInvered(0, 0) * ip.x + TransformPlaneInvered(0, 1) * ip.y + TransformPlaneInvered(0, 2) * ip.z + TransformPlaneInvered(0, 3));
op.y = static_cast<float> (TransformPlaneInvered(1, 0) * ip.x + TransformPlaneInvered(1, 1) * ip.y + TransformPlaneInvered(1, 2) * ip.z + TransformPlaneInvered(1, 3));
op.z = static_cast<float> (TransformPlaneInvered(2, 0) * ip.x + TransformPlaneInvered(2, 1) * ip.y + TransformPlaneInvered(2, 2) * ip.z + TransformPlaneInvered(2, 3));

////////////////////////////////////////////////////////////////////////////////////////////////////////
//3D bet plane fitting
template<class Vector3>
std::pair<Vector3, Vector3> best_plane_from_points(const std::vector<Vector3>& c)
{
	// copy coordinates to  matrix in Eigen format
	size_t num_atoms = c.size();
	Eigen::Matrix< Vector3::Scalar, Eigen::Dynamic, Eigen::Dynamic > coord(3, num_atoms);
	for (size_t i = 0; i < num_atoms; ++i) coord.col(i) = c[i];

	// calculate centroid
	Vector3 centroid(coord.row(0).mean(), coord.row(1).mean(), coord.row(2).mean());

	// subtract centroid
	coord.row(0).array() -= centroid(0); coord.row(1).array() -= centroid(1); coord.row(2).array() -= centroid(2);

	// we only need the left-singular matrix here
	//  http://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points
	auto svd = coord.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
	Vector3 plane_normal = svd.matrixU().rightCols<1>();
	return std::make_pair(centroid, plane_normal);
}

//RANSAC plane fitting
/////////////////////////////////////////////////////////////////////////////////////////////////////////
void getSample(std::vector<CCVector3>& sample) {

	//find 3 random points
	size_t sample_size = 3;
	size_t index_size = all_points.size();
	for (unsigned int i = 0; i < sample_size; ++i) {
		std::swap(all_points[i], all_points[(rand() % (index_size - i)) + i]);
		sample.push_back(all_points[i]);
	}
}

bool computeModelCoefficients(const std::vector<CCVector3>& samples, Eigen::VectorXf& model_coefficients)
{
	// Need 3 samples
	if (samples.size() != 3)
	{
		std::cout << "[pcl::SampleConsensusModelPlane::computeModelCoefficients] Invalid set of samples given " << samples.size() << std::endl;
		return (false);
	}
	CCVector3 p0 = samples[0];
	CCVector3 p1 = samples[1];
	CCVector3 p2 = samples[2];

	Eigen::Array3f p1p0 = Eigen::Array3f(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z);

	// Compute the segment values (in 3d) between p2 and p0
	Eigen::Array3f p2p0 = Eigen::Array3f(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z);

	// Avoid some crashes by checking for collinearity here
	Eigen::Array3f dy1dy2 = p1p0 / p2p0;

	if ((dy1dy2[0] == dy1dy2[1]) && (dy1dy2[2] == dy1dy2[1]))          // Check for collinearity
		return (false);
	// Compute the plane coefficients from the 3 given points in a straightforward manner
	// calculate the plane normal n = (p2-p1) x (p3-p1) = cross (p2-p1, p3-p1)
	model_coefficients.resize(4);
	model_coefficients[0] = p1p0[1] * p2p0[2] - p1p0[2] * p2p0[1];
	model_coefficients[1] = p1p0[2] * p2p0[0] - p1p0[0] * p2p0[2];
	model_coefficients[2] = p1p0[0] * p2p0[1] - p1p0[1] * p2p0[0];
	model_coefficients[3] = 0;

	// Normalize
	model_coefficients.normalize();
	// ... + d = 0
  //  model_coefficients[3] = -1 * (model_coefficients.template head<4>().dot(p0.matrix ()));
	model_coefficients[3] = -1 * (model_coefficients[0] * p0.x + model_coefficients[1] * p0.y + model_coefficients[2] * p0.z);
	return (true);
}

std::size_t countWithinDistance(const Eigen::VectorXf& model_coefficients, const double threshold)
{
	// Needs a valid set of model coefficients
	if (model_coefficients.size() != 4)
	{
		std::cout << "[pcl::SampleConsensusModelPlane::countWithinDistance] Invalid number of model coefficients given (%lu)!\n";
		return (0);
	}

	std::size_t nr_p = 0;

	// Iterate through the 3d points and calculate the distances from them to the plane
	for (std::size_t i = 0; i < all_points.size(); ++i)
	{
		// Calculate the distance from the point to the plane normal as the dot product
		// D = (P-A).N/|N|
		Eigen::Vector4f pt(all_points[i].x, all_points[i].y, all_points[i].z, 1);
		if (std::abs(model_coefficients.dot(pt)) < threshold)
			nr_p++;
	}
	return (nr_p);
}

void selectWithinDistance(const Eigen::VectorXf& model_coefficients, const double threshold, std::vector<CCVector3>& inliers)
{
	// Needs a valid set of model coefficients
	if (model_coefficients.size() != 4)
	{
		std::cout << "[pcl::SampleConsensusModelPlane::selectWithinDistance] Invalid number of model coefficients given (%lu)!\n";
		return;
	}

	int nr_p = 0;
	inliers.resize(all_points.size());

	// Iterate through the 3d points and calculate the distances from them to the plane
	for (std::size_t i = 0; i < all_points.size(); ++i)
	{
		// Calculate the distance from the point to the plane normal as the dot product
		// D = (P-A).N/|N|
		Eigen::Vector4f pt(all_points[i].x, all_points[i].y, all_points[i].z, 1);
		float distance = std::abs(model_coefficients.dot(pt));
		if (distance < threshold)
		{
			// Returns the indices of the points whose distances are smaller than the threshold
			inliers[nr_p] = all_points[i];

			++nr_p;
		}
	}
	inliers.resize(nr_p);
}

bool run_RANSAC(std::vector<CCVector3>& input, std::vector<CCVector3>& inlier_points, double threshold) {

	all_points = input;
	if (threshold == std::numeric_limits<double>::max())
	{
		std::cout << "[pcl::RandomSampleConsensus::computeModel] No threshold set!\n";
		return (false);
	}

	int iterations_ = 0;
	int n_best_inliers_count = -INT_MAX;
	double k = 1.0;

	std::vector<CCVector3> selection;
	Eigen::VectorXf model_coefficients;

	// double log_probability  = log (1.0 - probability_);
	double log_probability = log(1.0 - 0.99);
	double one_over_indices = 1.0 / static_cast<double> (all_points.size());

	int n_inliers_count = 0;
	unsigned skipped_count = 0;
	// supress infinite loops by just allowing 10 x maximum allowed iterations for invalid model parameters!
	int max_iterations_ = 1000;
	const unsigned max_skip = max_iterations_ * 10;

	// Iterate
	while (iterations_ < k && skipped_count < max_skip)
	{
		selection.clear();
		// Get X samples which satisfy the model criteria
		getSample(selection);
		if (selection.empty())
		{
			std::cout << "[pcl::RandomSampleConsensus::computeModel] No samples could be selected!\n";
			continue;
		}
		Eigen::VectorXf model_coefficients;
		// Search for inliers in the point cloud for the current plane model M
		if (!computeModelCoefficients(selection, model_coefficients))
		{
			++skipped_count;
			continue;
		}
		n_inliers_count = countWithinDistance(model_coefficients, threshold);
		// Better match ?
		if (n_inliers_count > n_best_inliers_count)
		{
			n_best_inliers_count = n_inliers_count;

			// Save the current model/inlier/coefficients selection as being the best so far
			model_ = selection;
			model_coefficients_ = model_coefficients;

			// Compute the k parameter (k=log(z)/log(1-w^n))
			double w = static_cast<double> (n_best_inliers_count) * one_over_indices;
			double p_no_outliers = 1.0 - pow(w, static_cast<double> (selection.size()));
			p_no_outliers = (std::max) (std::numeric_limits<double>::epsilon(), p_no_outliers);       // Avoid division by -Inf
			p_no_outliers = (std::min) (1.0 - std::numeric_limits<double>::epsilon(), p_no_outliers);   // Avoid division by 0.
			k = log_probability / log(p_no_outliers);
		}
		++iterations_;
		//std:cout <<"[pcl::RandomSampleConsensus::computeModel] Trial %d out of %f: %d inliers (best is: %d so far).\n", iterations_, k, n_inliers_count, n_best_inliers_count);
		if (iterations_ > max_iterations_)
		{
			std::cout << "[pcl::RandomSampleConsensus::computeModel] RANSAC reached the maximum number of trials.\n";
			break;
		}
	}
	//  PCL_DEBUG ("[pcl::RandomSampleConsensus::computeModel] Model: %lu size, %d inliers.\n", model_.size (), n_best_inliers_count);
	if (model_.empty())
	{
		inlier_points.clear();
		return (false);
	}
	// Get the set of inliers that correspond to the best model found so far
	selectWithinDistance(model_coefficients_, threshold, inlier_points);
	return (true);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


