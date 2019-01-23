#ifndef BATCHGRADIENTDESCENT_BGD_H
#define BATCHGRADIENTDESCENT_BGD_H
#include <iostream>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <vector>

const int batch = 7;
const int paranum1 = 5;
const int paranum2 = 4;
const int paranum3 = 1;

std::vector<double> vInput = {
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
    0.2, 0.1, 0.2, 0.3, 0.4, 0.7, 0.6,
    0.3, 0.2, 0.1, 0.2, 0.3, 0.5, 0.4,
    -0.1, -0.2, -0.3, -0.4, -0.5, -0.1, 0.3,
    -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8
};

std::vector<double> vM1 = {
    0.1, 0.2, 0.3, 0.4, 0.5,
    0.2, 0.3, 0.4, 0.5, 0.6,
    0.3, 0.4, 0.5, 0.6, 0.7,
    -0.1, -0.2, -0.3, -0.4, -0.5
};

std::vector<double> vM2 = {
    0.5, 0.5, 0.5, 0.5
};

std::vector<double> vTarget = {
    0.694639, 0.688256, 0.68442, 0.683872, 0.683324, 0.697002, 0.700624
};

template<typename T>
boost::numeric::ublas::matrix<T> sigmoid(const boost::numeric::ublas::matrix<T> &A);

template<typename T>
boost::numeric::ublas::matrix<T> ele_prod(const boost::numeric::ublas::matrix<T> &A,
                                          const boost::numeric::ublas::matrix<T> &B);

template<typename T>
boost::numeric::ublas::matrix<T> getM(unsigned long x,
                                      unsigned long y,
                                      T value);

#endif //BATCHGRADIENTDESCENT_BGD_H
