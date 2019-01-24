#include "BGD.h"

int main() {

  // Init Matrix Input
  boost::numeric::ublas::matrix<double> mInput(paranum1, batch);
  std::copy(std::begin(vInput), std::end(vInput), mInput.data().begin());
  // Init Matrix m1
  boost::numeric::ublas::matrix<double> m1(paranum2, paranum1);
  std::copy(std::begin(vM1), std::end(vM1), m1.data().begin());
  // Init Matrix m2
  boost::numeric::ublas::matrix<double> m2(paranum3, paranum2);
  std::copy(std::begin(vM2), std::end(vM2), m2.data().begin());
  // Init Matrix target
  boost::numeric::ublas::matrix<double> mTarget(paranum3, batch);
  std::copy(std::begin(vTarget), std::end(vTarget), mTarget.data().begin());

  for (int i = 0; i != 1000000; i++) {
    auto mZ1 = prod(m1, mInput);
    auto mA1 = sigmoid<double>(mZ1);
    auto mZ2 = prod(m2, mA1);
    auto mOut = sigmoid<double>(mZ2);
    // -----back propagation-------------
    auto dz2 = ele_prod<double>(mOut - mTarget, ele_prod<double>(mOut, getM(mOut.size1(), mOut.size2(), 1) - mOut));
    auto dw2 = prod(dz2, trans(mA1));
    auto dz1 = ele_prod<double>(prod(trans(m2), dz2), ele_prod<double>(mA1, getM(mA1.size1(), mA1.size2(), 1) - mA1));
    auto dw1 = prod(dz1, trans(mInput));
    m1 -= ele_prod<double>(getM(dw1.size1(), dw1.size2(), 0.05 / batch), dw1);
    m2 -= ele_prod<double>(getM(dw2.size1(), dw2.size2(), 0.05 / batch), dw2);
    if (i % 10000 == 0)
      print(mOut - mTarget);
  }
  return 0;
}

template<typename T>
boost::numeric::ublas::matrix<T> sigmoid(const boost::numeric::ublas::matrix<T> &A) {
  boost::numeric::ublas::matrix<T> out(A.size1(), A.size2());
  for (auto iter1 = out.begin1(); iter1 != out.end1(); iter1++)
    for (auto iter2 = iter1.begin(); iter2 != iter1.end(); iter2++) {
      T tmp = A(iter2.index1(), iter2.index2());
      *iter2 = -0.004 * pow(tmp, 3) + 0.197 * tmp + 0.5;
    }
  return out;
}

template<typename T>
boost::numeric::ublas::matrix<T> ele_prod(const boost::numeric::ublas::matrix<T> &A,
                                          const boost::numeric::ublas::matrix<T> &B) {
  boost::numeric::ublas::matrix<T> out(A.size1(), A.size2());
  for (auto iter1 = out.begin1(); iter1 != out.end1(); iter1++)
    for (auto iter2 = iter1.begin(); iter2 != iter1.end(); iter2++) {
      *iter2 = A(iter2.index1(), iter2.index2()) * B(iter2.index1(), iter2.index2());
    }
  return out;
}

template<typename T>
boost::numeric::ublas::matrix<T> getM(unsigned long x,
                                      unsigned long y,
                                      T value) {
  boost::numeric::ublas::matrix<T> out(x, y);
  for (auto iter1 = out.begin1(); iter1 != out.end1(); iter1++)
    for (auto iter2 = iter1.begin(); iter2 != iter1.end(); iter2++) {
      *iter2 = value;
    }
  return out;
}

void print(const boost::numeric::ublas::matrix<double> m) {
  for (auto iter1 = m.begin1(); iter1 != m.end1(); iter1++) {
    for (auto iter2 = iter1.begin(); iter2 != iter1.end(); iter2++) {
      std::cout << *iter2 << ",";
    }
    std::cout << std::endl;
  }
}
