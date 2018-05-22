/*
Copyright 2011. All rights reserved.
Institute of Measurement and Control Systems
Karlsruhe Institute of Technology, Germany

Authors: Andreas Geiger

libicp is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or any later version.

libicp is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
libicp; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA 
*/

#include "icp.h"
using namespace std;

Icp::Icp (double *M,const int32_t M_num,const int32_t dim) 
:m_dim(dim), m_max_iter(500), m_min_delta(1e-4)
{
  
	// check for correct dimensionality
	if (dim!=2 && dim!=3) {
		cout << "ERROR: LIBICP works only for data of dimensionality 2 or 3" << endl;
		m_kd_tree = 0;
		return;
	}
	// check for minimum number of points
	if (M_num<5) {
		cout << "ERROR: LIBICP works only with at least 5 model points" << endl;
		m_kd_tree = 0;
		return;
	}

	// copy model points to M_data
	m_kd_data.resize(boost::extents[M_num][dim]);
	for (int32_t m=0; m<M_num; m++)
		for (int32_t n=0; n<dim; n++)
		  m_kd_data[m][n] = (float)M[m*dim+n];

	// build a kd tree from the model point cloud
	m_kd_tree = new kdtree::KDTree(m_kd_data);
}

Icp::~Icp () {
  if (m_kd_tree)
    delete m_kd_tree;
}

double Icp::fit( double *T,const int32_t T_num,Matrix &R,Matrix &t,double indist/*=-1*/ )
{
	// make sure we have a model tree
	if (!m_kd_tree) {
		cout << "ERROR: No model available." << endl;
		return 0;
	}

	// check for minimum number of points
	if (T_num<5) {
		cout << "ERROR: Icp works only with at least 5 template points" << endl;
		return 0;
	}

	// set active points
	vector<int32_t> active;
	//if (indist<=0) {
	active.clear();
	for (int32_t i=0; i<T_num; i++)
	  active.push_back(i);
	//} else {
	//active = getInliers(T,T_num,R,t,indist);
	//assert(active.size() > 0);
	//}

	// run icp
	return fitIterate(T,T_num,R,t,indist);
}

double Icp::fitIterate( double *T,const int32_t T_num,Matrix &R,Matrix &t, double indist /*= -1*/ )
{
	//if(indist<=0){
		m_active.clear();m_active.resize(T_num);
		for(int32_t i=0;i<T_num;i++){
			m_active[i] = i;
		}
		m_inlier_ratio = 1;
	//}
	double delta = 1000;
	double min_indist = 0.1;
	int32_t iter;
	for(iter=0; iter<m_max_iter && delta>m_min_delta; iter++){
		//indist = std::max(indist*0.9, min_indist);

		//if(indist>min_indist){
		//	m_active = getInliers(T,T_num,R,t,indist);
	  //	m_inlier_ratio = (double)m_active.size()/T_num;
		//}

		if (m_active.size()<1){
		  return delta;
		}

		// delta=fitStep(T,T_num,R,t,m_active);
		fitStep(T,T_num,R,t,m_active);
		delta = getResidual(T,T_num,R,t,m_active);

		if(iter%10==0){
		  std::cout << "[" << iter << "] indist = " << indist << " m_active.size() = " << m_active.size() << " delta = " << delta << std::endl;
		}
	}

  std::cout << "[" << iter << "] indist = " << indist << " m_active.size() = " << m_active.size() << " delta = " << delta << std::endl;
	return delta;
}
