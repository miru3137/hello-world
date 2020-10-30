#include <iostream>
#include <opencv2/opencv.hpp>

/*
 * square process on device(GPU)
 */
__global__ void square(float* a, unsigned int n) {
    // get index
    unsigned int i = threadIdx.x;

    // compute square
    a[i] = a[i] * a[i];
}

/*
 * Main function
 */
int main() {
    /******************** Test Standard Library ********************/

    // print out message
    std::cout << "Hello, World!" << std::endl;

    /******************** Test CUDA Library ********************/

    // declare variables
    const unsigned int n = 20; // number of data
    float *h_a, *d_a; // host(CPU) & device(GPU) memory address

    // allocate host(CPU) memory
    h_a = new float[n];

    // allocate device(GPU) memory
    cudaMalloc(&d_a, n * sizeof(float));

    // set input data
    for (unsigned int i = 0; i < n; i++)
        h_a[i] = i;

    // copy input data from host(CPU) to device(GPU)
    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);

    // run square(^2) process
    square<<<1, n>>>(d_a, n);

    // copy output data from device(GPU) to host(CPU)
    cudaMemcpy(h_a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost);

    // print out result
    for (unsigned int i = 0; i < n; i++)
        std::cout << h_a[i] << " ";
    std::cout << std::endl;

    /******************** Test OpenCV Library ********************/

    // read image
    const auto image = cv::imread("../lenna.png");

    // show image
    cv::imshow("Lenna", image);

    // wait key input (with update window)
    cv::waitKey();

    /******************** Exit Program ********************/

    return 0;
}
