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

#include "icp/icpPointToPoint.h"
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
  bool sample(const int N, std::vector<Eigen::Vector3f> &points) {

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
        points.push_back(point);
      }
    }

    return true;
  }

  /** \brief Sample points from the mesh
   * \param[in] mesh mesh to sample from
   * \param[in] n batch index in points
   * \param[out] points sampled points
   */
  bool sample2(const int N, std::vector<Eigen::Vector3f> &points) {

    std::vector<float> areas(this->num_faces());

    #pragma omp parallel
    {
      #pragma omp for
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

        // Accumulate.
        areas[f] = area;
      }
    }

    float sum = 0;
    for (int f = 0; f < this->num_faces(); f++) {
      sum += areas[f];
    }

    if (sum < 1e-6) {
      std::cout << "[Error] face area sum of " << sum << std::endl;
      return false;
    }

    for (int f = 0; f < this->num_faces(); f++) {
      areas[f] /= sum;
    }

    std::vector<int> cum_sum(this->num_faces() + 1);
    cum_sum[0] = 0;

    for (int f = 1; f < this->num_faces() + 1; f++) {
      cum_sum[f] = std::max(static_cast<int>(std::ceil(areas[f - 1]*N)), 1) + cum_sum[f - 1];
    }

    //std::cout << cum_sum[this->num_faces()] << " " << sum << std::endl;
    points.resize(cum_sum[this->num_faces()]);

    #pragma omp parallel
    {
      for (int f = 0; f < this->num_faces(); f++) {
        int n = cum_sum[f + 1] - cum_sum[f];

        #pragma omp for
        for (int i = 0; i < n; i++) {
          float r1 = 0;
          float r2 = 0;
          do {
            r1 = static_cast<float>(std::rand())/static_cast<float>(RAND_MAX);
            r2 = static_cast<float>(std::rand())/static_cast<float>(RAND_MAX);
          }
          while (r1 + r2 > 1.f);

          int s = std::rand()%3;

          Eigen::Vector3f a = this->vertices[this->faces[f](s)];
          Eigen::Vector3f b = this->vertices[this->faces[f]((s + 1)%3)];
          Eigen::Vector3f c = this->vertices[this->faces[f]((s + 2)%3)];

          Eigen::Vector3f ab = b - a;
          Eigen::Vector3f ac = c - a;

          Eigen::Vector3f point = a + r1*ab + r2*ac;
          //std::cout << points.size() << " " << cum_sum[f] + i << std::endl;
          //assert(cum_sum[f] + i < points.size());
          points[cum_sum[f] + i] = point;
        }
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

/** \brief Class representing a point cloud in 3D. */
class PointCloud {
public:
  /** \brief Constructor. */
  PointCloud() {

  }

  /** \brief Copy constructor.
   * \param[in] point_cloud point cloud to copy
   */
  PointCloud(const PointCloud &point_cloud) {
    this->points.clear();
    this->colors.clear();

    for (unsigned int i = 0; i < point_cloud.points.size(); i++) {
      this->points.push_back(point_cloud.points[i]);
      this->colors.push_back(point_cloud.colors[i]);
    }
  }

  /** \brief Constructor
   * \param[in] points
   */
  PointCloud(const std::vector<Eigen::Vector3f> &points) {
    this->points.clear();
    this->colors.clear();

    for (unsigned int i = 0; i < points.size(); i++) {
      this->add_point(points[i]);
    }
  }

  /** \brief Constructor
   * \param[in] points
   */
  PointCloud(const int k, const Eigen::Tensor<float, 3, Eigen::RowMajor> &points) {
    this->points.clear();
    this->colors.clear();

    for (unsigned int i = 0; i < points.dimension(1); i++) {
      if (points(k, i, 0) != 0 && points(k, i, 1) != 0 && points(k, i, 0) != 2) {
        this->add_point(Eigen::Vector3f(points(k, i, 0), points(k, i, 1), points(k, i, 2)));
      }
    }
  }


  /** \brief Destructor. */
  ~PointCloud() {

  }

  /** \brief Read point cloud from txt file.
   * \param[in] filepath path to txt file
   * \param[out] point_cloud read point cloud
   */
  static bool from_txt(const std::string &filepath, PointCloud &point_cloud) {
    std::ifstream file(filepath.c_str());
    std::string line;
    std::stringstream ss;

    std::getline(file, line);
    ss << line;

    int n_points;
    ss >> n_points;

    if (n_points <= 0) {
      return false;
    }

    for (int i = 0; i < n_points; i++) {
      std::getline(file, line);

      ss.clear();
      ss.str("");
      ss << line;

      Eigen::Vector3f point;

      ss >> point(0);
      ss >> point(1);
      ss >> point(2);

      point_cloud.add_point(point);
    }

    return true;
  }

  /** \brief Read point cloud from binary file.
   * \param[in] filepath path to binary file
   * \param[out] point_cloud read point cloud
   */
  static bool from_bin(const std::string &filepath, PointCloud &point_cloud) {

    // allocate 4 MB buffer (only ~130*4*4 KB are needed)
    int32_t num = 1000000;
    float *data = (float*)malloc(num*sizeof(float));

    // pointers
    float *px = data+0;
    float *py = data+1;
    float *pz = data+2;
    float *pr = data+3;

    // load point cloud
    FILE *stream;
    stream = fopen (filepath.c_str(), "rb");
    num = fread(data,sizeof(float),num,stream)/4;
    for (int32_t i=0; i<num; i++) {
      point_cloud.add_point(Eigen::Vector3f(-*py,*pz,*px));
      //std::cout << *px << " " << *px << " " << *pz << std::endl;
      px+=4; py+=4; pz+=4; pr+=4;
    }

    fclose(stream);

    //exit(1);
    return true;
  }

  /** \brief Given the angle in radians, construct a rotation matrix around the x-axis.
   * \param[in] radians angle in radians
   * \param[out] rotation rotation matrix
   */
  static void rotation_matrix_x(const float radians, Eigen::Matrix3f &rotation) {
    rotation = Eigen::Matrix3f::Zero();

    rotation(0, 0) = 1;
    rotation(1, 1) = std::cos(radians); rotation(1, 2) = -std::sin(radians);
    rotation(2, 1) = std::sin(radians); rotation(2, 2) = std::cos(radians);
  }

  /** \brief Given the angle in radians, construct a rotation matrix around the y-axis.
   * \param[in] radians angle in radians
   * \param[out] rotation rotation matrix
   */
  static void rotation_matrix_y(const float radians, Eigen::Matrix3f &rotation) {
    rotation = Eigen::Matrix3f::Zero();

    rotation(0, 0) = std::cos(radians); rotation(0, 2) = std::sin(radians);
    rotation(1, 1) = 1;
    rotation(2, 0) = -std::sin(radians); rotation(2, 2) = std::cos(radians);
  }

  /** \brief Given the angle in radians, construct a rotation matrix around the z-axis.
   * \param[in] radians angle in radians
   * \param[out] rotation rotation matrix
   */
  static void rotation_matrix_z(const float radians, Eigen::Matrix3f &rotation) {
    rotation = Eigen::Matrix3f::Zero();

    rotation(0, 0) = std::cos(radians); rotation(0, 1) = -std::sin(radians);
    rotation(1, 0) = std::sin(radians); rotation(1, 1) = std::cos(radians);
    rotation(2, 2) = 1;
  }

  /** \brief Computes the rotation matrix corresponding to the given ray.
   * \param[in] ray ray defining the direction to rotate to
   * \param[out] rotation final rotation
   */
  static void rotation_matrix(const Eigen::Vector3f ray, Eigen::Matrix3f &rotation) {
    Eigen::Matrix3f rotation_x;
    Eigen::Matrix3f rotation_y;
    Eigen::Matrix3f rotation_z;

    Eigen::Vector3f axis_x = Eigen::Vector3f(1, 0, 0);
    Eigen::Vector3f axis_y = Eigen::Vector3f(0, 1, 0);
    Eigen::Vector3f axis_z = Eigen::Vector3f(0, 0, 1);

    Eigen::Vector3f ray_x = ray; ray_x(0) = 0; ray_x /= ray_x.norm();
    Eigen::Vector3f ray_y = ray; ray_y(1) = 0; ray_y /= ray_y.norm();
    Eigen::Vector3f ray_z = ray; ray_z(2) = 0; ray_y /= ray_z.norm();

    float radians_x = std::acos(axis_x.dot(ray_x));
    PointCloud::rotation_matrix_x(radians_x, rotation_x);

    float radians_y = std::acos(axis_y.dot(ray_y));
    PointCloud::rotation_matrix_y(radians_y, rotation_y);

    float radians_z = std::acos(axis_z.dot(ray_z));
    PointCloud::rotation_matrix_z(radians_z, rotation_z);
    std::cout << "[Data] radians " << radians_x << " " << radians_y << " " << radians_z << std::endl;

    rotation = Eigen::Matrix3f::Zero();
    rotation = rotation_z*rotation_y*rotation_x;
  }

  /** \brief Add a point to the point cloud.
   * \param[in] point point to add
   */
  void add_point(const Eigen::Vector3f &point) {
    this->points.push_back(point);
    this->colors.push_back(Eigen::Vector3i::Zero());
  }

  /** \brief Add a colored point to the point cloud.
   * \param[in] point point to add
   * \param[in] color color of the point
   */
  void add_point(const Eigen::Vector3f &point, const Eigen::Vector3i &color) {
    this->points.push_back(point);
    this->colors.push_back(color);
  }

  /** \brief Add points from a point cloud.
   * \param[in] point_cloud point cloud whose points to add
   */
  void add_points(const PointCloud &point_cloud) {
    for (unsigned int i = 0; i < point_cloud.num_points(); i++) {
      this->add_point(point_cloud.points[i], point_cloud.colors[i]);
    }
  }

  /** \brief Merge/add points from another point cloud.
   * \param[in] point_cloud point_cloud to take points from
   */
  void merge(const PointCloud &point_cloud) {
    for (unsigned int i = 0; i < point_cloud.points.size(); i++) {
      this->add_point(point_cloud.points[i], point_cloud.colors[i]);
    }
  }

  /** \brief Get number of points.
   * \return number of points
   */
  unsigned int num_points() const {
    return this->points.size();
  }

  /** \brief Write point cloud to txt file.
   * \param[in] filepath path to file
   * \return success
   */
  bool to_txt(const std::string &filepath) {
    std::ofstream* out = new std::ofstream(filepath, std::ofstream::out);
    if (!static_cast<bool>(*out)) {
      return false;
    }

    (*out) << this->points.size() << std::endl;
    for (unsigned int i = 0; i < this->points.size(); i++) {
     (*out) << this->points[i](0) << " " << this->points[i](1) << " " << this->points[i](2) << std::endl;
    }

    out->close();
    delete out;

    return true;
  }

  /** \brief Write point cloud to txt file.
   * \param[in] filepath path to file
   * \return success
   */
  bool to_ply(const std::string &filepath) {
    std::ofstream* out = new std::ofstream(filepath, std::ofstream::out);
    if (!static_cast<bool>(*out)) {
      return false;
    }

    if (this->points.size() != this->colors.size()) {
      return false;
    }

    (*out) << "ply" << std::endl;
    (*out) << "format ascii 1.0" << std::endl;
    (*out) << "element vertex " << this->points.size() << std::endl;
    (*out) << "property float32 x" << std::endl;
    (*out) << "property float32 y" << std::endl;
    (*out) << "property float32 z" << std::endl;
    (*out) << "property uchar red" << std::endl;
    (*out) << "property uchar green" << std::endl;
    (*out) << "property uchar blue" << std::endl;
    (*out) << "end_header" << std::endl;

    for (unsigned int i = 0; i < this->points.size(); i++) {
      //(*out) << this->points[i](0) << " " << this->points[i](1) << " " << this->points[i](2) << std::endl;
      (*out) << this->points[i](0) << " " << this->points[i](1) << " " << this->points[i](2) << " "
        << this->colors[i](0) << " " << this->colors[i](1) << " " << this->colors[i](2) << std::endl;
    }

    out->close();
    delete out;

    return true;
  }

  /** \brief Get the extents of the point cloud in all three axes.
   * \param[out] min minimum coordinate values per axis
   * \param[out] max maximum coordinate values per axis
   */
  void extents(Eigen::Vector3f &min, Eigen::Vector3f &max) {
    for (int d = 0; d < 3; d++) {
      min(d) = FLT_MAX;
      max(d) = FLT_MIN;
    }

    for (unsigned int i = 0; i < this->points.size(); i++) {
      for (int d = 0; d < 3; d++) {
        if (this->points[i](d) < min(d)) {
          min(d) = this->points[i](d);
        }
        if (this->points[i](d) > max(d)) {
          max(d) = this->points[i](d);
        }
      }
    }
  }

  /** \brief Scale the point cloud.
   * \param[in] scale scales for each axis
   */
  void scale(const Eigen::Vector3f &scale) {
    for (unsigned int i = 0; i < this->points.size(); i++) {
      for (int d = 0; d < 3; d++) {
        this->points[i](d) *= scale(d);
      }
    }
  }

  /** \brief Scale the point cloud.
   * \param[in] scale overlal scale for all axes
   */
  void scale(float scale) {
    this->scale(Eigen::Vector3f(scale, scale, scale));
  }

  /** \brief Rotate the point cloud around the origin.
   * \param[in] rotation rotation matrix
   */
  void rotate(const Eigen::Matrix3f &rotation) {
    for (unsigned int i = 0; i < this->points.size(); i++) {
      this->points[i] = rotation*this->points[i];
    }
  }

  /** \brief Translate the point cloud.
   * \brief translation translation vector
   */
  void translate(const Eigen::Vector3f &translation) {
    for (unsigned int i = 0; i < this->points.size(); i++) {
      this->points[i] += translation;
    }
  }

  /** \brief Get a point.
   * \param[in] n
   * \return point n
   */
  Eigen::Vector3f get(int n) const {
    assert(n < this->num_points() && n >= 0);
    return this->points[n];
  }

private:
  /** \brief The points of the point cloud. */
  std::vector<Eigen::Vector3f> points;
  /** \brief Colors of the points. */
  std::vector<Eigen::Vector3i> colors;

};

/** \brief Read a Hdf5 file into an Eigen tensor.
 * \param[in] filepath path to file
 * \param[out] dense Eigen tensor
 * \return success
 */
template<int RANK>
bool read_hdf5(const std::string filepath, Eigen::Tensor<float, RANK, Eigen::RowMajor>& dense) {

  try {
    H5::H5File file(filepath, H5F_ACC_RDONLY);
    H5::DataSet dataset = file.openDataSet("tensor");

    /*
     * Get filespace for rank and dimension
     */
    H5::DataSpace filespace = dataset.getSpace();

    /*
     * Get number of dimensions in the file dataspace
     */
    size_t rank = filespace.getSimpleExtentNdims();

    if (rank != RANK) {
      std::cout << "[Error] invalid rank read: " << rank << std::endl;
      exit(1);
    }

    /*
     * Get and print the dimension sizes of the file dataspace
     */
    hsize_t dimsf[rank];
    filespace.getSimpleExtentDims(dimsf);

    std::cout << "[Data] HDF5 size: ";
    for (int i = 0; i < RANK; ++i) {
      std::cout << dimsf[i] << " ";
    }
    std::cout << std::endl;

    /*
     * Define the memory space to read dataset.
     */
    std::cout << "[Data] HDF5 allocating ..." << std::endl;
    H5::DataSpace mspace(rank, dimsf);

    //size_t buffer_size = 1;
    //for (int i = 0; i < RANK; ++i) {
    //  buffer_size *= dimsf[i];
    //}

    std::cout << "[Data] HDF5 casting ..." << std::endl;
    float* buffer = static_cast<float*>(dense.data());
    std::cout << "[Data] HDF5 reading ..." << std::endl;
    dataset.read(buffer, H5::PredType::NATIVE_FLOAT, mspace, filespace);

    //for (int i = 0; i < buffer_size; ++i) {
    //  std::cout << buffer[i] << std::endl;
    //}
  }

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
}

/** \brief Perform ICP using a point-to-point distance.
 * \param[in] point_cloud_from
 * \param[in] point_cloud_to
 * \param[out] rotation
 * \param[out] translation
 */
void point_to_point_icp(const PointCloud &point_cloud_from, const PointCloud &point_cloud_to,
    Eigen::Matrix3f &rotation, Eigen::Vector3f &translation, float &residual, int &inliers) {

  double* M = new double[3*point_cloud_to.num_points()];
  for (unsigned int i = 0; i < point_cloud_to.num_points(); i++) {
    for (int d = 0; d < 3; d++) {
      M[i*3 + d] = point_cloud_to.get(i)(d);
    }
  }

  double* T = new double[3*point_cloud_from.num_points()];
  for (unsigned int i = 0; i < point_cloud_from.num_points(); i++) {
    for (int d = 0; d < 3; d++) {
      T[i*3 + d] = point_cloud_from.get(i)(d);
    }
  }

  unsigned int I = std::min(point_cloud_from.num_points(), point_cloud_to.num_points());
  translation(0) = 0; translation(1) = 0; translation(2) = 0;

  for (unsigned int i = 0; i < I; i++) {
    translation += point_cloud_to.get(i) - point_cloud_from.get(i);
  }

  translation /= I;

  Matrix R = Matrix::eye(3);
  Matrix t(3, 1);
  t.val[0][0] = 0;
  t.val[1][0] = 0;
  t.val[2][0] = 0;

  IcpPointToPoint icp(M, point_cloud_to.num_points(), 3);
  icp.setMaxIterations(100);
  icp.setMinDeltaParam(0.0001);

  residual = icp.fit(T, point_cloud_from.num_points(), R, t);
  inliers = icp.getInlierCount();

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      rotation(i, j) = R.val[i][j];
    }

    translation(i) = t.val[i][0];
  }

  delete[] M;
  delete[] T;
}

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cout << "[Error] Usage: icp txt_file off_directory h5_file out_off out_log" << std::endl;
    exit(1);
  }

  boost::filesystem::path txt_file(argv[1]);
  boost::filesystem::path off_directory(argv[2]);
  boost::filesystem::path h5_file(argv[3]);
  boost::filesystem::path out_off_file(argv[4]);

  boost::filesystem::path out_log_file;
  if (argc > 5) {
    out_log_file = boost::filesystem::path(argv[5]);
  }

  if (!boost::filesystem::is_regular_file(txt_file)) {
    std::cout << "[Error] file " << txt_file.string() << " not found" << std::endl;
    exit(1);
  }

  if (!boost::filesystem::is_directory(off_directory)) {
    std::cout << "[Error] directory " << off_directory.string() << " not found" << std::endl;
    exit(1);
  }

  if (!boost::filesystem::is_regular_file(h5_file)) {
    std::cout << "[Error] file " << h5_file.string() << " not found" << std::endl;
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
  read_hdf5(h5_file.string(), points);

  PointCloud point_cloud;
  PointCloud::from_txt(txt_file.string(), point_cloud);
  std::cout << "[ICP] read " << point_cloud.num_points() << " points" << std::endl;

  std::string log = "n n_points residual inliers\n";
  float min_residual = 1e32;
  float min_index = 0;

  Timer timer;
  float total = 0;

  std::vector<Eigen::Matrix3f> rotations;
  std::vector<Eigen::Vector3f> translations;
  for (unsigned int i = 0; i < indices.size(); i++) {
    rotations.push_back(Eigen::Matrix3f::Identity());
    translations.push_back(Eigen::Vector3f::Zero());
  }

  omp_set_num_threads(16);
  #pragma omp parallel
  {
    #pragma omp for
    for (unsigned int i = 0; i < indices.size(); i++) {
      int n = indices[i];

      PointCloud mesh_point_cloud(i, points);
      //mesh_point_cloud.to_ply(std::to_string(i) + ".ply");
      float residual = 1e32;
      int inliers = 0;

      timer.start();
      point_to_point_icp(point_cloud, mesh_point_cloud, rotations[i], translations[i], residual, inliers);
      timer.stop();

      float elapsed = timer.getElapsedTimeInMilliSec();
      std::cout << "[ICP] residual " << residual << " (inliers " << inliers << ", " << elapsed << "ms)" << std::endl;
      total += elapsed;

      #pragma omp critical
      {
        log += std::to_string(n) + " " + std::to_string(point_cloud.num_points()) + " " + std::to_string(residual) + " " + std::to_string(inliers) + "\n";

        if (residual < min_residual) {
          min_residual = residual;
          min_index = i;
        }
      }
    }
  }

  Mesh mesh;
  std::string off_file = off_files[indices[min_index]].string();
  Mesh::from_off(off_file, mesh);
  mesh.rotate(rotations[min_index].transpose());
  mesh.translate(-translations[min_index]);
  mesh.to_off(out_off_file.string());

  if (!out_log_file.empty()) {
    std::ofstream out(out_log_file.string());
    out << log;
    out.close();
  }
  else {
    std::cout << log;
  }

  std::cout << "[ICP] wrote " << off_file << " to " << out_off_file.string() << std::endl;
  std::cout << "[ICP] needed on average " << total/indices.size() << "ms" << std::endl;


  exit(0);
}
