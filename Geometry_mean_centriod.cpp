template<typename Vector, typename Container>
Vector geometric_median(const Container& data, int iterations = 200)
{
	size_t N = data.size();
	if (N < 3) return data.front();
	size_t dim = data.front().size();
	std::vector<Vector> A(2, (data[0] + data[1]) / Scalar(2));

	for (int it = 0; it < iterations; it++) {
		Vector numerator; for (size_t i = 0; i < dim; i++) numerator[i] = 0;
		double denominator = 0;
		int t = it % 2;

		for (int n = 0; n < N; n++) {
			double dist = (data[n], A[t]).lpNorm<2>();
			if (dist != 0) {
				numerator += data[n] / dist;
				denominator += 1.0 / dist;
			}
		}

		A[1 - t] = numerator / denominator;
	}

	return A[iterations % 2];
}

template<typename Vector, typename Container>
Vector geometric_centroid(const Container& data)
{
	if (data.size() < 3) return data.front();
	size_t dim = data.front().size();
	Vector sum; for (size_t i = 0; i < dim; i++) sum[i] = 0;
	for (auto& d : data) sum += d;
	return sum / data.size();
}