#ifndef MESH_H
#define MESH_H

//#include <QGLWidget> //UpgradeQt6: 
#include <QOpenGLWidget>
#include <fstream> //ifstream, ofstream
#include <vector>


// TO MODIFY
class Point
{
public:
    double _x;
    double _y;
    double _z;

    Point():_x(),_y(),_z() {}
    Point(float x_, float y_, float z_):_x(x_),_y(y_),_z(z_) {}
};


//** TP : TO MODIFY

class Mesh
{
private:
  // (Q ou STL)Vector of vertices
  std::vector<Point> vector_vertices;
  // (Q ou STL)Vector of faces

  // Faces are represented by the index of their vertices
  // Faces can be any geometrical form
  std::vector<std::vector<int>> vector_faces;

  // Vector of adjacent face
  // Tells to what face the i-th triangle is attached
  std::vector<std::vector<int>> adjacency_matrix;

public:
    Mesh(): vector_vertices(), vector_faces(), adjacency_matrix() {} // Constructors automatically called to initialize a Mesh (default strategy)

    Mesh(std::vector<Point> list_vertices, std::vector<std::vector<int>> list_faces, std::vector<std::vector<int>> adj_matrix){
        vector_vertices = list_vertices;
        vector_faces = list_faces;
        adjacency_matrix = adj_matrix;
    }

    ~Mesh(); // Destructor automatically called before a Mesh is destroyed (default strategy)
    //void drawMesh();
    //void drawMeshWireFrame();

    std::vector<Point> get_vector_vertices(){return vector_vertices;}
    std::vector<std::vector<int>> get_vector_faces(){return vector_faces;}
    //std::vector<std::vector<int>> get_adjacency_matrix(){return adjacency_matrix;}

    static void read_off(const std::string& filename){
        // Read the .off file and returns a mesh.
        // Open the file
          std::ifstream inputFile(filename);

          // Read the first line (header)
          std::string header;
          std::getline(inputFile, header);
          // Read the number of vertices, faces, and edges
          int numVertices, numFaces, numEdges;
          inputFile >> numVertices >> numFaces >> numEdges;

          // Faces array. Contains:
          // Indices of the points
          // Index of face
          std::vector<std::vector<int>> faces_vector(numFaces, std::vector<int>(6));

          std::vector<std::vector<float>> vertex_vector(numVertices, std::vector<float>(4));

          // Read and save the vertex coordinates
          // not the face coordinate yet
          for (int i = 0; i < numVertices; ++i) {
              double x, y, z;
              inputFile >> x >> y >> z;
              vertex_vector[i][0] = x;
              vertex_vector[i][1] = y;
              vertex_vector[i][2] = z;
          }

          // Go through all edges
          // For each

          std::map<std::pair<int, int>, int> seen_map;

          for (int i = 0; i < numFaces; ++i) {
              // k is the 3 at the start of each face line ; it is not needed.
              int k, a, b, c;
              inputFile >> k >> a >> b >> c;
              faces_vector[i][0] = a;
              faces_vector[i][1] = b;
              faces_vector[i][2] = c;

              // Add face to vertices, if vertex was not associated with any face
              if (vertex_vector[a][3] == 0){vertex_vector[a][3] = i;}
              if (vertex_vector[b][3] == 0){vertex_vector[b][3] = i;}
              if (vertex_vector[c][3] == 0){vertex_vector[c][3] = i;}

              std::pair<int, int> key_ab = {(a < b) ? a : b, (a < b) ? b : a};
              // check if key exists in the map
              if (seen_map.find(key_ab) != seen_map.end()){
                // map[key] not empty
                int index_face = seen_map[key_ab];
                faces_vector[i][5] = index_face;
              }else{
                faces_vector[i][5] = i;
              }

              std::pair<int, int> key_bc = {(b < c) ? b : c, (b < c) ? c : b};
              if (seen_map.find(key_bc) != seen_map.end()){
                int index_face = seen_map[key_bc];
                faces_vector[i][3] = index_face;
              }else{
                faces_vector[i][3] = i;
              }

              std::pair<int, int> key_ac = {(a < c) ? a : c, (a < c) ? c : a};
              if (seen_map.find(key_ac) != seen_map.end()){
                int index_face = seen_map[key_ac];
                faces_vector[i][4] = index_face;
              }else{
                faces_vector[i][4] = i;
              }
          }

          // Close the file
          inputFile.close();
    };
};
//void add_vertex(Point vertex){vector_vertices.push_back(vertex);}


class GeometricWorld //Here used to create a singleton instance
{
  QVector<Point> _bBox;  // Bounding box
public :
  GeometricWorld();
  void draw();
  void drawWireFrame();
  void drawMesh(Mesh mesh);
  // ** TP Can be extended with further elements;
  // Mesh _mesh;
  /*Mesh tetrahedron(std::vector<Point> list_vertices = {Point(0,0,0), Point(0,1,0), Point(1,0,0), Point(1,1,0)},
                   std::vector<std::vector<int>> list_faces = {{0,1,2}, {0,1,3}, {0,2,3}, {1,2,3}},
                   std::vector<std::vector<int>> adj_matrix = {{1,2,3}, {0,2,3}, {0,1,3}, {0,1,2}});*/
};


#endif // MESH_H
