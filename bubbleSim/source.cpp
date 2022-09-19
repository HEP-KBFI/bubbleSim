#include <vector>
#include <iostream>
#include <random>
#include <filesystem>

int main() {
	std::mt19937_64 generator(2);
	std::uniform_real_distribution<double> distribution(0, 1);

	std::cout << distribution(generator) << " , " << distribution(generator) << " , " << distribution(generator) << std::endl;

	std::mt19937_64 generator2(0);

	std::cout << distribution(generator2) << " , " << distribution(generator2) << " , " << distribution(generator2) << std::endl;

	std::filesystem::path a = std::filesystem::current_path();
	std::filesystem::path b = std::filesystem::relative(a);
	std::filesystem::path c = std::filesystem::absolute(a);

	std::cout << a << std::endl;
	std::cout << b << std::endl;
	std::cout << c << std::endl;

}