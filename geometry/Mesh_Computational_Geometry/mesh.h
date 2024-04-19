#ifndef MESH_H
#define MESH_H

//#include <QGLWidget> //UpgradeQt6: 
#include <QOpenGLWidget>
#include <fstream> //ifstream, ofstream
#include <vector>
#include <iostream>
#include <cmath>

class Point
{
public:
    // empty contrusctor to initialize the face array in read_off with empty faces
    Point(){
        x = 0.;
        y = 0.;
        z = 0.;
    }

    Point(float x_, float y_, float z_){
        x = x_;
        y = y_;
        z = z_;
    }

    float get_x() const {
        return x;
    }

    float get_y() const {
        return y;
    }

    float get_z() const{
        return z;
    }

    void set_x(float _x){
        x = _x;
    }

    void set_y(float _y){
        y = _y;
    }

    void set_z(float _z){
        z = _z;
    }

    Point operator-(const Point& other) const {
        return Point(x - other.x, y - other.y, z - other.z);
    }

    double norm2(){
        return sqrt(x*x + y*y + z*z);
    }

private:
    float x;
    float y;
    float z;
};

class Vertex
{
public:
    // empty contrusctor to initialize the face array in read_off with empty faces
    Vertex(){
        p = Point();
        index_face = -1;
    }

    Vertex(Point p_, int index_face_){
        p = p_;
        index_face = index_face_;
    }

    Point get_point() const {
        return p;
    }

    void set_point(Point _p){
        p = _p;
    }

    float get_x() const {
        return p.get_x();
    }

    float get_y() const {
        return p.get_y();
    }

    float get_z() const{
        return p.get_z();
    }

    void set_x(float x){
        p.set_x(x);
    }

    void set_y(float y){
        p.set_y(y);
    }

    void set_z(float z){
        p.set_z(z);
    }

    int get_index_face() const {
        return index_face;
    }

    void set_index_face(int index){
        index_face = index;
    }

private:
    Point p;
    int index_face;
};

class Face
{
public:
    // empty constructor to initialize the face array in read_off with empty faces
    Face(){
        index_point_1 = -1;
        index_point_2 = -1;
        index_point_3 = -1;
        index_face_1 = -1;
        index_face_2 = -1;
        index_face_3 = -1;
        vector_points = {-1,-1,-1};
    }

    Face(int ip1, int ip2, int ip3, int if1, int if2, int if3){
        index_point_1 = ip1;
        index_point_2 = ip2;
        index_point_3 = ip3;
        index_face_1 = if1;
        index_face_2 = if2;
        index_face_3 = if3;
        vector_points = {index_point_1, index_point_2, index_point_3};
    }

    bool index_in_face(int index){
        return index_point_1 == index || index_point_2 == index || index_point_3 == index;
    }

    void set_index_point_1(int index){
        index_point_1 = index;
        vector_points[0] = index;
    }

    void set_index_point_2(int index){
        index_point_2 = index;
        vector_points[1] = index;
    }

    void set_index_point_3(int index){
        index_point_3 = index;
        vector_points[2] = index;
    }

    void set_index_face_1(int index){
        index_face_1 = index;
    }

    void set_index_face_2(int index){
        index_face_2 = index;
    }

    void set_index_face_3(int index){
        index_face_3 = index;
    }

    int get_index_point_1() const {
        return index_point_1;
    }

    int get_index_point_2() const {
        return index_point_2;
    }

    int get_index_point_3() const {
        return index_point_3;
    }

    int get_index_face_1() const {
        return index_face_1;
    }

    int get_index_face_2() const {
        return index_face_2;
    }

    int get_index_face_3() const {
        return index_face_3;
    }

    void update_face_split(int index_point, int index_new_face){
        // get next index, when starting at point which has index_point as index
        // useful when splitting a triangle, to connect a new triangle
        if (index_point == index_point_1){
            index_face_2 = index_new_face;
        }
        else if (index_point == index_point_2){
            index_face_3 = index_new_face;
        }
        else if (index_point == index_point_3){
            index_face_1 = index_new_face;
        }
    }

    int get_next_point(int index_point, int index_initial_point){
        // get the index of the next point, when rotating around a point in a mesh
        int i0 = 0;
        if (index_point == index_point_1){
            i0 = 0;
        }
        else if (index_point == index_point_2){
            i0 = 1;
        }
        else if (index_point == index_point_3){
            i0 = 2;
        }
        if (vector_points[(i0+1)%3] == index_initial_point){return vector_points[(i0+2)%3];}
        else {return vector_points[(i0+1)%3];}
    }


    int get_opposite_face(int index_point){
        if (index_point == index_point_1){return get_index_face_1();}
        if (index_point == index_point_2){return get_index_face_2();}
        if (index_point == index_point_3){return get_index_face_3();}
    }

    void set_opposite_face(int index_point, int index_new_face){
        if (index_point == index_point_1){index_face_1 = index_new_face;}
        if (index_point == index_point_2){index_face_2 = index_new_face;}
        if (index_point == index_point_3){index_face_3 = index_new_face;}
    }

    int get_index_point(int index_point){
        if (index_point == index_point_1){return 0;}
        if (index_point == index_point_2){return 1;}
        if (index_point == index_point_3){return 2;}
    }

    int get_index(int local_index){
        // get index of point based on local index
        if (local_index == 0){return index_point_1;}
        if (local_index == 1){return index_point_2;}
        if (local_index == 2){return index_point_3;}
    }

    void set_index_point(int ip, int new_index){
        // set index of point based on local index ip
        if (get_index(ip) == 0){set_index_point_1(new_index);}
        if (get_index(ip) == 1){set_index_point_2(new_index);}
        if (get_index(ip) == 2){set_index_point_3(new_index);}
    }


private:
    int index_point_1;
    int index_point_2;
    int index_point_3;
    int index_face_1;
    int index_face_2;
    int index_face_3;
    std::vector<int> vector_points;
};

class Mesh
{
public:
    Mesh(std::vector<Vertex> list_vertices, std::vector<Face> list_faces){
        vector_vertices = list_vertices;
        vector_faces = list_faces;
    };

    Mesh(){};

    ~Mesh();

    // Insert a Point p in the mesh
    void insert_point(const Point& p);

    void init_from_off(const std::string& filename);

    // Q1 TP3: add point into triangle
    //void triangle_split(int index_triangle, Point& p);

    void set_vector_vertices(std::vector<Vertex> _vector_vertices){
        vector_vertices = _vector_vertices;
    }

    void set_vector_faces(std::vector<Face> _vector_faces){
        vector_faces = _vector_faces;
    }

    int get_number_vertices(){
        return vector_vertices.size();
    }

    std::vector<Vertex>  get_vector_vertices() const {
        return vector_vertices;
    }

    std::vector<Face> get_vector_faces() const {
        return vector_faces;
    }


    std::vector<int> get_neighbors(int index_vertex) const;

    double laplacian_vertex(const std::vector<double> &u, int index_vertex);

    std::vector<double> laplacian(const std::vector<double> &u);

    int get_index_face(int ip1, int ip2, int ip3);

    //static bool is_counter_clockwise(const Point& p1, const Point& p2, const Point& p3); // return true if p1, p2, p3 counter-clockwise oriented


private:
  std::vector<Vertex> vector_vertices;
  std::vector<Face> vector_faces;
  std::vector<int> vector_i_convexhull; // vector indices convex hull
  bool is_inside(int i, const Point& p); // check if point p is inside i-th triangle
  void triangle_split(int index_triangle, const Point& p); // split i-th face (triangle) into 3 by inserting point p
  void edge_split(int ip0, int if0, Point& p);
  void edge_flip(int ip0, int if0);
};
//void add_vertex(Point vertex){vector_vertices.push_back(vertex);}


class GeometricWorld //Here used to create a singleton instance
{
  QVector<Point> _bBox;  // Bounding box
public :
  GeometricWorld();
  void draw();
  void drawWireFrame();
  void drawPoint(const Point& point, int R, int G, int B, double size=10.0f);
  void drawMesh(const Mesh& mesh, bool edges = false);
  void drawTriangle(const Point& p1, const Point& p2, const Point& p3);
  void drawTriangleEdges(const Point& p1, const Point& p2, const Point& p3);
  void drawMeshes();
  void drawTetrahedron();
  void drawPyramid();
  void drawBoundingBox();
  void drawQueen();
  void drawNeighbors();
  void addPointTriangulation();
  void reset_face_to_draw();
  void set_face_to_draw(int i);
  void reset();
  std::vector<Vertex> get_neighbors(int index_vertex);
  void temperature();
  void drawLaplacian(const std::vector<double> &u);
  void drawTriangulation();

private:
  Mesh tetrahedron;
  Mesh pyramid;
  Mesh bounding_box;
  Mesh queen;
  Mesh triangulation;
  std::vector<Mesh> vector_meshes;
  std::vector<Vertex> vertices_tetrahedron;
  std::vector<Face> faces_tetrahedron;
  std::vector<bool> face_to_draw;
  std::vector<Point> vector_points_triangulation;
  bool computing_laplacian;
  bool draw_neighbors=false;
  std::vector<double> vector_temperature;
  int index_next_point_triangulation = 4;
};


#endif // MESH_H
