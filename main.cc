#include <iostream>
#include <string>
#include <Eigen/Dense>

using namespace std;
using Eigen::MatrixXd;
int main()
{
  MatrixXd m(2,2);
  MatrixXd a(2,2);
  m(0,0) = 1;
  m(0,1) = 2;
  m(1,0) = 4;
  m(1,1) = 2;
  a(0,0) = .5;
  a(0,1) = .25;
  a(1,0) = .75;
  a(1,1) = 1;
  cout << a << endl;
  a = a*m;
  cout << "Hello World" << endl;
  string name;
  cout << m << endl;
  cout << a << endl;
}
