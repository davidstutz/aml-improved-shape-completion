#include <cstdio>
#include <cmath>
#include <cfloat>
#include <cstdarg>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <boost/filesystem.hpp>
#include <omp.h>
#include <H5Cpp.h>

#include "timer/timer.h"

/** \brief Read all files in a directory matching the given extension.
 * \param[in] directory path to directory
 * \param[out] files read file paths
 * \param[in] extension extension to filter for
 */
void read_directory(const boost::filesystem::path directory, std::map<int, boost::filesystem::path>& files, const std::string extension = ".off") {
  files.clear();
  boost::filesystem::directory_iterator end;

  for (boost::filesystem::directory_iterator it(directory); it != end; ++it) {
    if (it->path().extension().string() == extension) {
      if (!boost::filesystem::is_empty(it->path()) && !it->path().empty() && it->path().filename().string() != "") {
        int number = std::stoi(it->path().filename().string());
        files.insert(std::pair<int, boost::filesystem::path>(number, it->path()));
      }
    }
  }
}

/** \brief Just encapsulating vertices and faces. */
class Mesh {
public:
  /** \brief Empty constructor. */
  Mesh() {

  }

  /** \brief Add a vertex.
   * \param[in] vertex vertex to add
   */
  void add_vertex(Eigen::Vector3f& vertex) {
    this->vertices.push_back(vertex);
  }

  /** \brief Get the number of vertices.
   * \return number of vertices
   */
  int num_vertices() const {
    return static_cast<int>(this->vertices.size());
  }

  Eigen::Vector3f vertex(int v) {
    assert(v >= 0 && v < this->vertices.size());
    return this->vertices[v];
  }

  /** \brief Add a face.
   * \param[in] face face to add
   */
  void add_face(Eigen::Vector3i& face) {
    this->faces.push_back(face);
  }

  /** \brief Get the number of faces.
   * \return number of faces
   */
  int num_faces() const {
    return static_cast<int>(this->faces.size());
  }

  Eigen::Vector3i face(int f) {
    assert(f >= 0 && f < this->num_faces());
    return this->faces[f];
  }

  /** \brief Rotate the point cloud around the origin.
   * \param[in] rotation rotation matrix
   */
  void rotate(const Eigen::Matrix3f &rotation) {
    for (int v = 0; v < this->num_vertices(); ++v) {
      this->vertices[v] = rotation*this->vertices[v];
    }
  }

  /** \brief Translate the mesh.
   * \param[in] translation translation vector
   */
  void translate(const Eigen::Vector3f& translation) {
    for (int v = 0; v < this->num_vertices(); ++v) {
      for (int i = 0; i < 3; ++i) {
        this->vertices[v](i) += translation(i);
      }
    }
  }

  /** \brief Scale the mesh.
   * \param[in] scale scale vector
   */
  void scale(const Eigen::Vector3f& scale) {
    for (int v = 0; v < this->num_vertices(); ++v) {
      for (int i = 0; i < 3; ++i) {
        this->vertices[v](i) *= scale(i);
      }
    }
  }

  /** \brief Sample points from the mesh
   * \param[in] mesh mesh to sample from
   * \param[in] n batch index in points
   * \param[in] points pre-initialized tensor holding points
   */
  bool sample(const int N, const int k, Eigen::Tensor<float, 3, Eigen::RowMajor> &points) {

    // Stores the areas of faces.
    std::vector<float> areas(this->num_faces());
    float sum = 0;

    // Build a probability distribution over faces.
    for (int f = 0; f < this->num_faces(); f++) {
      Eigen::Vector3f a = this->vertices[this->faces[f][0]];
      Eigen::Vector3f b = this->vertices[this->faces[f][1]];
      Eigen::Vector3f c = this->vertices[this->faces[f][2]];

      // Angle between a->b and a->c.
      Eigen::Vector3f ab = b - a;
      Eigen::Vector3f ac = c - a;
      float cos_angle = ab.dot(ac)/(ab.norm()*ac.norm());
      float angle = std::acos(cos_angle);

      // Compute triangle area.
      float area = std::max(0., 0.5*ab.norm()*ac.norm()*std::sin(angle));
      //std::cout << area << " " << std::pow(area, 1./4.) << " " << angle << " " << ab.norm() << " " << ac.norm() << " " << std::sin(angle) << std::endl;

      // Accumulate.
      //area = std::sqrt(area);
      areas[f] = area;
      sum += area;
      //areas.push_back(1);
      //sum += 1;
    }

    //std::cout << sum << std::endl;
    if (sum < 1e-6) {
      std::cout << "[Error] face area sum of " << sum << std::endl;
      return false;
    }

    for (int f = 0; f < this->num_faces(); f++) {
      //std::cout << areas[f] << " ";
      areas[f] /= sum;
      //std::cout << areas[f] << std::endl;
    }

    std::vector<float> cum_areas(areas.size());
    cum_areas[0] = areas[0];

    for (int f = 1; f < this->num_faces(); f++) {
      cum_areas[f] = areas[f] + cum_areas[f - 1];
    }

    int j = 0;
    for (int f = 0; f < this->num_faces(); f++) {
      int n = std::max(static_cast<int>(areas[f]*N), 1);

      for (int i = 0; i < n; i++) {
        float r1 = 0;
        float r2 = 0;
        do {
          r1 = static_cast<float>(std::rand())/static_cast<float>(RAND_MAX);
          r2 = static_cast<float>(std::rand())/static_cast<float>(RAND_MAX);
        }
        while (r1 + r2 > 1.f);

        int s = std::rand()%3;
        //std::cout << face << " " << areas[face] << std::endl;

        Eigen::Vector3f a = this->vertices[this->faces[f](s)];
        Eigen::Vector3f b = this->vertices[this->faces[f]((s + 1)%3)];
        Eigen::Vector3f c = this->vertices[this->faces[f]((s + 2)%3)];

        Eigen::Vector3f ab = b - a;
        Eigen::Vector3f ac = c - a;

        Eigen::Vector3f point = a + r1*ab + r2*ac;
        points(k, j, 0) = point(0);
        points(k, j, 1) = point(1);
        points(k, j, 2) = point(2);
        j++;
      }
    }

    return true;
  }

  /** \brief Reading an off file and returning the vertices x, y, z coordinates and the
   * face indices.
   * \param[in] filepath path to the OFF file
   * \param[out] mesh read mesh with vertices and faces
   * \return success
   */
  static bool from_off(const std::string filepath, Mesh& mesh) {

    std::ifstream* file = new std::ifstream(filepath.c_str());
    std::string line;
    std::stringstream ss;
    int line_nb = 0;

    std::getline(*file, line);
    ++line_nb;

    if (line != "off" && line != "OFF") {
      std::cout << "[Error] Invalid header: \"" << line << "\", " << filepath << std::endl;
      return false;
    }

    size_t n_edges;
    std::getline(*file, line);
    ++line_nb;

    int n_vertices;
    int n_faces;
    ss << line;
    ss >> n_vertices;
    ss >> n_faces;
    ss >> n_edges;

    for (size_t v = 0; v < n_vertices; ++v) {
      std::getline(*file, line);
      ++line_nb;

      ss.clear();
      ss.str("");

      Eigen::Vector3f vertex;
      ss << line;
      ss >> vertex(0);
      ss >> vertex(1);
      ss >> vertex(2);

      mesh.add_vertex(vertex);
    }

    size_t n;
    for (size_t f = 0; f < n_faces; ++f) {
      std::getline(*file, line);
      ++line_nb;

      ss.clear();
      ss.str("");

      size_t n;
      ss << line;
      ss >> n;

      if(n != 3) {
        std::cout << "[Error] Not a triangle (" << n << " points) at " << (line_nb - 1) << std::endl;
        return false;
      }

      Eigen::Vector3i face;
      ss >> face(0);
      ss >> face(1);
      ss >> face(2);

      mesh.add_face(face);
    }

    if (n_vertices != mesh.num_vertices()) {
      std::cout << "[Error] Number of vertices in header differs from actual number of vertices." << std::endl;
      return false;
    }

    if (n_faces != mesh.num_faces()) {
      std::cout << "[Error] Number of faces in header differs from actual number of faces." << std::endl;
      return false;
    }

    file->close();
    delete file;

    return true;
  }

  /** \brief Write mesh to OFF file.
   * \param[in] filepath path to OFF file to write
   * \return success
   */
  bool to_off(const std::string filepath) {
    std::ofstream* out = new std::ofstream(filepath, std::ofstream::out);
    if (!static_cast<bool>(*out)) {
      return false;
    }

    (*out) << "OFF" << std::endl;
    (*out) << this->vertices.size() << " " << this->num_faces() << " 0" << std::endl;

    for (unsigned int v = 0; v < this->vertices.size(); v++) {
      (*out) << this->vertices[v](0) << " " << this->vertices[v](1) << " " << this->vertices[v](2) << std::endl;
    }

    for (unsigned int f = 0; f < this->num_faces(); f++) {
      (*out) << "3 " << this->faces[f](0) << " " << this->faces[f](1) << " " << this->faces[f](2) << std::endl;
    }

    out->close();
    delete out;

    return true;
  }

  /** \brief Write mesh to obj file.
   * \param[in] filepath
   * \param[in] mtl_lib
   * \param[in] materials
   * \return success
   */
  bool to_obj(const std::string filepath) {

    std::ofstream* out = new std::ofstream(filepath, std::ofstream::out);
    if (!static_cast<bool>(out)) {
      return false;
    }

    for (unsigned int v = 0; v < this->vertices.size(); v++) {
      (*out) << "v " << this->vertices[v](0) << " " << this->vertices[v](1) << " " << this->vertices[v](2) << std::endl;
    }

    for (unsigned int f = 0; f < this->num_faces(); f++) {
      (*out) << "f " << this->faces[f](0) + 1 << " " << this->faces[f](1) + 1 << " " << this->faces[f](2) + 1 << std::endl;
    }

    out->close();
    delete out;

    return true;
  }

private:

  /** \brief Vertices as (x,y,z)-vectors. */
  std::vector<Eigen::Vector3f> vertices;

  /** \brief Faces as list of vertex indices. */
  std::vector<Eigen::Vector3i> faces;
};

/** \brief Write the given set of volumes to h5 file.
 * \param[in] filepath h5 file to write
 * \param[in] n number of volumes
 * \param[in] height height of volumes
 * \param[in] width width of volumes
 * \param[in] depth depth of volumes
 * \param[in] dense volume data
 */
bool write_hdf5(const std::string filepath, Eigen::Tensor<float, 3, Eigen::RowMajor>& dense) {

  try {

    /*
     * Turn off the auto-printing when failure occurs so that we can
     * handle the errors appropriately
     */
    H5::Exception::dontPrint();

    /*
     * Create a new file using H5F_ACC_TRUNC access,
     * default file creation properties, and default file
     * access properties.
     */
    H5::H5File file(filepath, H5F_ACC_TRUNC);

    /*
     * Define the size of the array and create the data space for fixed
     * size dataset.
     */
    hsize_t rank = 3;
    hsize_t dimsf[rank];
    dimsf[0] = dense.dimension(0);
    dimsf[1] = dense.dimension(1);
    dimsf[2] = dense.dimension(2);
    H5::DataSpace dataspace(rank, dimsf);

    /*
     * Define datatype for the data in the file.
     * We will store little endian INT numbers.
     */
    H5::IntType datatype(H5::PredType::NATIVE_FLOAT);
    datatype.setOrder(H5T_ORDER_LE);

    /*
     * Create a new dataset within the file using defined dataspace and
     * datatype and default dataset creation properties.
     */
    H5::DataSet dataset = file.createDataSet("tensor", datatype, dataspace);

    /*
     * Write the data to the dataset using default memory space, file
     * space, and transfer properties.
     */
    float* data = static_cast<float*>(dense.data());
    dataset.write(data, H5::PredType::NATIVE_FLOAT);
  }  // end of try block

  // catch failure caused by the H5File operations
  catch(H5::FileIException error) {
    error.printError();
    return false;
  }

  // catch failure caused by the DataSet operations
  catch(H5::DataSetIException error) {
    error.printError();
    return false;
  }

  // catch failure caused by the DataSpace operations
  catch(H5::DataSpaceIException error) {
    error.printError();
    return false;
  }

  // catch failure caused by the DataSpace operations
  catch(H5::DataTypeIException error) {
    error.printError();
    return false;
  }

  return true;
}

/** \brief Main entrance point of script.
 * Expects one argument, the path to the config file.
 */
int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "[Error] Usage: sample off_directory h5_file" << std::endl;
    exit(1);
  }

  boost::filesystem::path off_directory(argv[1]);
  boost::filesystem::path h5_file(argv[2]);

  if (!boost::filesystem::is_directory(off_directory)) {
    std::cout << "[Error] directory " << off_directory << " not found" << std::endl;
    exit(1);
  }

  std::map<int, boost::filesystem::path> off_files;
  read_directory(off_directory, off_files);
  std::cout << "[ICP] found " << off_files.size() << " files" << std::endl;

  std::vector<int> indices;
  for (std::map<int, boost::filesystem::path>::iterator it = off_files.begin(); it != off_files.end(); it++) {
    indices.push_back(it->first);
  }

  int N_points = 1000000;
  Eigen::Tensor<float, 3, Eigen::RowMajor> points(indices.size(), N_points, 3);
  points.setZero();

  float total = 0;

  omp_set_num_threads(16);
  #pragma omp parallel
  {
    #pragma omp for
    for (unsigned int i = 0; i < indices.size(); i++) {
      int n = indices[i];
      std::string off_file = off_files[n].string();

      Mesh mesh;
      Mesh::from_off(off_file, mesh);

      Timer timer;
      timer.start();
      mesh.sample(N_points, i, points);
      timer.stop();

      float elapsed = timer.getElapsedTimeInMilliSec();
      std::cout << "[Sample] sampled " << off_file << " (" << elapsed << "ms)" << std::endl;

      #pragma omp critical
      {
        total += elapsed;
      }
    }
  }

  write_hdf5(h5_file.string(), points);
  std::cout << "[Sample] wrote " << h5_file.string() << std::endl;
  std::cout << "[Sample] took  on average " << total/indices.size() << "ms" << std::endl;
  exit(0);
}
