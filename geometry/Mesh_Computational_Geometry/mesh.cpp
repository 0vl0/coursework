#include "mesh.h"
#include <thread>
#include <chrono>
#include <cstdlib>
#include <algorithm>

void sleep(double seconds) {
    std::chrono::duration<double> duration_seconds(seconds);
    std::this_thread::sleep_for(duration_seconds);
}

// Destructor definition
Mesh::~Mesh() {
    // Add any necessary cleanup code here
}

double dot(const Point& p1, const Point& p2){
    return p1.get_x()*p2.get_x() + p1.get_y()*p2.get_y() + p1.get_z()*p2.get_z();
}

double absolute_v(double x){
    if (x < 0){
        return -x;
    }
    else {return x;}
}

Point cross(const Point& p1, const Point& p2){
    return Point(p1.get_x()*p2.get_y() - p1.get_y()*p2.get_x(),
                 p1.get_y()*p2.get_z() - p1.get_z()*p2.get_y(),
                 p1.get_z()*p2.get_x() - p1.get_x()*p2.get_z());
}

void Mesh::triangle_split(int index_triangle, const Point& p){
    int N_points = vector_vertices.size();
    int N_faces = vector_faces.size();
    int index_point_1 = vector_faces[index_triangle].get_index_point_1();
    int index_point_2 = vector_faces[index_triangle].get_index_point_2();
    int index_point_3 = vector_faces[index_triangle].get_index_point_3();
    int index_face_1 = vector_faces[index_triangle].get_index_face_1();
    int index_face_2 = vector_faces[index_triangle].get_index_face_2();
    int index_face_3 = vector_faces[index_triangle].get_index_face_3();

    Face new_face_0 = Face(index_point_1, index_point_2, N_points, N_faces, N_faces+1, index_face_3);
    Face new_face_1 = Face(index_point_2, index_point_3, N_points, N_faces+1, index_triangle, index_face_1);
    Face new_face_2 = Face(index_point_1, N_points, index_point_3, N_faces, index_face_2, index_triangle);

    Vertex new_vertex = Vertex(p, index_triangle);
    vector_vertices.push_back(new_vertex);
    vector_faces[index_triangle] = new_face_0;
    vector_faces.push_back(new_face_1);
    vector_faces.push_back(new_face_2);

    vector_faces[index_face_1].update_face_split(index_point_2, N_faces);
    vector_faces[index_face_2].update_face_split(index_point_3, N_faces+1);
}

void Mesh::edge_split(int ip0, int if0, Point& p){
    int N_points = vector_vertices.size();
    int N_faces = vector_faces.size();

    vector_vertices.push_back(Vertex(p, if0));

    int il0 = vector_faces[if0].get_index_point(ip0);
    int ip1 = vector_faces[if0].get_index((il0+1)%3);
    int ip2 = vector_faces[if0].get_index((il0+2)%3);

    int if1 = vector_faces[if0].get_opposite_face(ip0);
    int if0p = vector_faces[if0].get_opposite_face((ip0+2)%3);
    //int if3p = vector_faces[if0].get_opposite_face((ip0+1)%3);
    //int if2p = vector_faces[if1].get_opposite_face(ip1);
    int if1p = vector_faces[if1].get_opposite_face(ip2);

    int il_ip1_if1 = vector_faces[if1].get_index_point(ip1);
    int ip3 = vector_faces[if1].get_index((il_ip1_if1+1)%3);

    Face new_face_1 = Face(ip0, ip1, N_points, N_faces+1, if0, if0p);
    Face new_face_2 = Face(N_points, ip1, ip3, if1p, if1, N_faces);
    vector_faces.push_back(new_face_1);
    vector_faces.push_back(new_face_2);

    vector_faces[if0].set_opposite_face(ip2, N_faces);
    vector_faces[if1].set_opposite_face(ip2, N_faces+1);

    vector_faces[if0].set_index_point(ip1, N_points);
    vector_faces[if1].set_index_point(ip1, N_points);
}

void Mesh::edge_flip(int ip0, int if0){
    int il0 = vector_faces[if0].get_index_point(ip0);
    int ip1 = vector_faces[if0].get_index((il0+1)%3);
    int ip2 = vector_faces[if0].get_index((il0+2)%3);

    int if1 = vector_faces[if0].get_opposite_face(ip0);
    int il_ip1_if1 = vector_faces[if1].get_index_point(ip1);
    int ip3 = vector_faces[if1].get_index((il_ip1_if1+1)%3);

    int if1p = vector_faces[if1].get_opposite_face(ip2);
    //int if3p = vector_faces[if0].get_opposite_face((ip0+1)%3); // faux
    int if3p = vector_faces[if0].get_opposite_face(ip1);

    //vector_faces[if0].set_index_point((ip0+2)%3, ip3);
    vector_faces[if0].set_index_point(ip2, ip3);
    vector_faces[if0].set_opposite_face(ip0, if1p);
    vector_faces[if0].set_opposite_face(ip1, if1);

    //vector_faces[if1].set_index_point((ip0+1)%3, ip0);
    vector_faces[if1].set_index_point(ip1, ip0);
    vector_faces[if1].set_opposite_face(ip3, if3p);
    vector_faces[if1].set_opposite_face(ip2, if0);
}

double Mesh::laplacian_vertex(const std::vector<double> &u, int index_vertex){
    // compute laplacian of function u at vertex of index index vertex
    // u: discrete function on which laplacian is computed
    // index_vertex: index of the vertex where laplacian is computed
    std::vector<int> index_neighbors = get_neighbors(index_vertex);
    Point p0 = vector_vertices[index_vertex].get_point();
    double area = 0;
    double res = 0;
    for (int i = 0; i<index_neighbors.size(); i++){
        Point p1 = vector_vertices[index_neighbors[i]].get_point();
        Point p2 = vector_vertices[index_neighbors[(i+1)%index_neighbors.size()]].get_point();
        Point p3 = vector_vertices[index_neighbors[(i+2)%index_neighbors.size()]].get_point();

        Point vector10 = p0-p1;
        Point vector12 = p2-p1;

        Point vector30 = p0-p3;
        Point vector32 = p2-p3;

        //Point vector02 = p2-p0;

        double area1 = cross(vector10, vector12).norm2();
        double area2 = cross(vector30, vector32).norm2();
        area += area1 + area2;

        double cotan_012 = sqrt(dot(vector10, vector12)/cross(vector10, vector12).norm2());
        double cotan_032 = sqrt(dot(vector30, vector32)/cross(vector30, vector32).norm2());
        //double u_center = u[index_vertex];
        //double u_nei_i = u[(i+1)%index_neighbors.size()];
        //double value_to_add = (cotan_012 + cotan_032)*(u[(i+1)%index_neighbors.size()]-u[index_vertex]);
        res += (cotan_012 + cotan_032)*(u[index_neighbors[(i+1)%index_neighbors.size()]]-u[index_vertex]);
    }
    // half the parallelogram area (/2), one third of the area (/3), half of the resulting area (/2) => /12
    res /= (area/12);
    return res;
}

std::vector<double> Mesh::laplacian(const std::vector<double> &u){
    // compute next laplacian based on current laplacian u
    int n_vertices = vector_vertices.size();
    std::vector<double> result(n_vertices, 0);
    for(int i = 0; i<n_vertices; i++){
//        result[i] = u[i]+0.001*laplacian_vertex(u, i);
        result[i] = u[i]+0.0000001*laplacian_vertex(u, i);
    }
    return result;
}

void Mesh::init_from_off(const std::string &filename){
    // Read the .off file and init the Mesh
    // Open the file
      std::ifstream inputFile(filename);
      std::cout << "Input file: " << std::endl;

      // Read the first line (header)
      std::string header;
      std::getline(inputFile, header);
      std::cout << "Header: " << header << std::endl;
      // Read the number of vertices, faces, and edges
      int numVertices, numFaces, numEdges;
      inputFile >> numVertices >> numFaces >> numEdges;
      std::cout << "Number of vertices: " << numVertices << std::endl;
      std::cout << "Number of faces: " << numFaces << std::endl;
      std::cout << "Number of edges: " << numEdges << std::endl;

      // Faces array. Contains:
      // Indices of the points
      // Index of face
      //std::vector<std::vector<int>> faces_vector(numFaces, std::vector<int>(6));
      std::vector<Face> face_vector(numFaces, Face());


      //std::vector<std::vector<float>> vertex_vector(numVertices, std::vector<float>(4));
      std::vector<Vertex> vertex_vector(numVertices, Vertex());

      // Read and save the vertex coordinates
      // not the face coordinate yet
      for (int i = 0; i < numVertices; ++i) {
          double x, y, z;
          inputFile >> x >> y >> z;
          vertex_vector[i].set_x(x);
          vertex_vector[i].set_y(y);
          vertex_vector[i].set_z(z);
      }

      // Go through all faces
      // For each

      std::map<std::pair<int, int>, int> seen_map;

      for (int i = 0; i < numFaces; ++i) {
          // k is the 3 at the start of each face line ; it is not needed.
          int k, a, b, c;
          inputFile >> k >> a >> b >> c;
          face_vector[i].set_index_point_1(a);
          face_vector[i].set_index_point_2(b);
          face_vector[i].set_index_point_3(c);


          // Add face to vertices, if vertex was not associated with any face
          if (vertex_vector[a].get_index_face() == -1){vertex_vector[a].set_index_face(i);}
          if (vertex_vector[b].get_index_face() == -1){vertex_vector[b].set_index_face(i);}
          if (vertex_vector[c].get_index_face() == -1){vertex_vector[c].set_index_face(i);}


          std::pair<int, int> key_ab = {(a < b) ? a : b, (a < b) ? b : a};
          // check if key exists in the map
          if (seen_map.find(key_ab) != seen_map.end()){
            // map[key] not empty
            int index_face = seen_map[key_ab];
            face_vector[i].set_index_face_3(index_face);
          }
          seen_map[key_ab] = i;

          std::pair<int, int> key_bc = {(b < c) ? b : c, (b < c) ? c : b};
          if (seen_map.find(key_bc) != seen_map.end()){
            int index_face = seen_map[key_bc];
            face_vector[i].set_index_face_1(index_face);
          }
          seen_map[key_bc] = i;

          std::pair<int, int> key_ac = {(a < c) ? a : c, (a < c) ? c : a};
          if (seen_map.find(key_ac) != seen_map.end()){
            int index_face = seen_map[key_ac];
            face_vector[i].set_index_face_2(index_face);
          }
          seen_map[key_ac] = i;
      }


      // Close the file
      inputFile.close();

      for (int i = 0; i < numFaces; ++i) {
        Face f = face_vector[i];
        if (f.get_index_face_3() == -1){
            int a = f.get_index_point_1();
            int b = f.get_index_point_2();
            std::pair<int, int> key_ab = {(a < b) ? a : b, (a < b) ? b : a};
            f.set_index_face_3(seen_map[key_ab]);
        }

        if (f.get_index_face_1() == -1){
            int b = f.get_index_point_2();
            int c = f.get_index_point_3();
            std::pair<int, int> key_bc = {(b < c) ? b : c, (b < c) ? c : b};
            f.set_index_face_1(seen_map[key_bc]);
        }

        if (f.get_index_face_2() == -1){
            int a = f.get_index_point_1();
            int c = f.get_index_point_3();
            std::pair<int, int> key_ac = {(a < c) ? a : c, (a < c) ? c : a};
            f.set_index_face_2(seen_map[key_ac]);
        }
        face_vector[i] = f;
      }

//      for (int i = 0; i<face_vector.size(); i++){
//        std::cout << "face: " << i;
//        std::cout << "; i_point 1: " << face_vector[i].get_index_point_1();
//        std::cout << "; i_point 2: " << face_vector[i].get_index_point_2();
//        std::cout << "; i_point 3: " << face_vector[i].get_index_point_3();
//        std::cout << "; i_face 1: " << face_vector[i].get_index_face_1();
//        std::cout << "; i_face 2: " << face_vector[i].get_index_face_2();
//        std::cout << "; i_face 3: " << face_vector[i].get_index_face_3() << std::endl;
//      }

      set_vector_faces(face_vector);
      set_vector_vertices(vertex_vector);
}

std::vector<int> Mesh::get_neighbors(int index_vertex) const{
    std::vector<int> neighbors = {};
    int index_initial_face = vector_vertices[index_vertex].get_index_face();
    Face f = vector_faces[index_initial_face];
    int index_next_point = index_vertex;
    int index_next_face = index_initial_face;
    int index_initial_point = index_vertex;

    index_next_point = f.get_next_point(index_initial_point, index_initial_point);

    do{
        neighbors.push_back(index_next_point);

        int temp = index_next_point;
        // vertex opposite to initial vertex
        index_next_point = f.get_next_point(index_next_point, index_initial_point);

        index_next_face = f.get_opposite_face(temp);
        f = vector_faces[index_next_face];

    }while(index_next_face != index_initial_face);

    return neighbors;
}

GeometricWorld::GeometricWorld()
{
    _bBox.push_back(Point(0,0,0));
    _bBox.push_back(Point(1,0,0));
    _bBox.push_back(Point(0,1,0));
    _bBox.push_back(Point(0,0,1));

    std::vector<Vertex> vertices_tetrahedron = {
            Vertex(Point(0.0, 0.0, 0.0), 0),  // Vertex 0
            Vertex(Point(1.0, 0.0, 0.0), 1),  // Vertex 1
            Vertex(Point(0.0, 0.0, 1.0), 2), // Vertex 2
            Vertex(Point(1.0, 1.0, 0.0), 3)   // Vertex 3
     };

    // Define faces of the tetrahedron
    std::vector<Face> faces_tetrahedron = {
        Face(0, 2, 1, 2, 1, 3), // Face 0 (connecting vertices 0, 1, and 2)
        Face(0, 1, 3, 2, 3, 0), // Face 1 (connecting vertices 0, 1, and 3)
        Face(2, 1, 3, 1, 3, 0), // Face 2 (connecting vertices 1, 2, and 3)
        Face(0, 2, 3, 2, 1, 0)  // Face 3 (connecting vertices 2, 0, and 3)
    };

    tetrahedron.set_vector_vertices(vertices_tetrahedron);
    tetrahedron.set_vector_faces(faces_tetrahedron);

    vector_meshes.push_back(tetrahedron);

    std::vector<Vertex> vertices_pyramid = {
        Vertex(Point(0.0, 0.0, 0.0), 0),
        Vertex(Point(1.0, 0.0, 0.0), 4),
        Vertex(Point(1.0, 0.0, 1.0), 1),
        Vertex(Point(0.0, 0.0, 1.0), 3),
        Vertex(Point(0.5, 1.0, 0.5), 2)
    };

    std::vector<Face> faces_pyramid = {
        Face(0, 3, 1, 1, 4, 3),
        Face(3, 2, 1, 5, 0, 2),
        Face(3, 2, 4, 5, 3, 1),
        Face(3, 4, 0, 4, 0, 2),
        Face(0, 1, 4, 5, 3, 0),
        Face(2, 1, 4, 4, 2, 1)
    };

    pyramid.set_vector_vertices(vertices_pyramid);
    pyramid.set_vector_faces(faces_pyramid);
    vector_meshes.push_back(pyramid);

    std::vector<Vertex> vertices_bounding_box = {
        Vertex(Point(0.0, 0.0, 0.0), 0),
        Vertex(Point(1.0, 0.0, 0.0), 0),
        Vertex(Point(1.0, 0.0, 1.0), 1),
        Vertex(Point(0.0, 0.0, 1.0), 1)
    };

    std::vector<Face> faces_bounding_box = {
            Face(0, 3, 1, 1, 1, 1),
            Face(3, 2, 1, 0, 0, 0)
        };

    bounding_box.set_vector_vertices(vertices_bounding_box);
    bounding_box.set_vector_faces(faces_bounding_box);

    vector_meshes.push_back(bounding_box);

    std::vector<Vertex> vertices_triangulation = {
        Vertex(Point(0.5, -5, 0.5), 0),
        Vertex(Point(0.0, 0.0, 0.0), 0),
        Vertex(Point(1.0, 0.0, 0.0), 0),
        Vertex(Point(0.5, 0.0, 1.0), 0)
    };

    std::vector<Face> faces_triangulation = {
        Face(0, 1, 2, 3, 1, 2),
        Face(0, 2, 3, 3, 2, 0),
        Face(1, 0, 3, 1, 3, 0),
        Face(1, 3, 2, 1, 0, 2)
    };

    vector_points_triangulation = {Point(0.5, -5, 0.5),
                                   Point(0.0, 0.0, 0.0),
                                   Point(1.0, 0.0, 0.0),
                                   Point(0.5, 0.0, 1.0),
                                   Point(0.5, 0.0, 0.5),
                                   Point(0.5, 0.0, 0.25),
                                   Point(-1.0, 0.0, 0.5),
                                   Point(0.75, 0.0, 1.5),
                                   };

    triangulation.set_vector_vertices(vertices_triangulation);
    triangulation.set_vector_faces(faces_triangulation);

    // à changer
    queen.init_from_off("/home/vl/Documents/4A/calcul_modelisation_geometrique/transfer_6728376_files_f11b4472/Mesh_Computational_Geometry/queen.off");
    vector_meshes.push_back(queen);

    vector_meshes.push_back(triangulation);

    face_to_draw = {false, false, false, true, false};

    int number_vertices = vector_meshes[3].get_number_vertices();
    this->vector_temperature = std::vector<double>(number_vertices, 0.0);
    computing_laplacian = false;
}

void GeometricWorld::set_face_to_draw(int i){
    reset_face_to_draw();
    face_to_draw[i] = true;
}

void GeometricWorld::reset_face_to_draw(){
    face_to_draw = {false, false, false, false, false};
}

void GeometricWorld::reset(){
    reset_face_to_draw();
}

bool is_counter_clockwise(const Point& p1, const Point& p2, const Point& p3){
    return cross(p2-p1, p3-p1).get_z() >= 0;
}

bool Mesh::is_inside(int i, const Point& p){
    // Check if point p is inside i-th face (triangle)
    // All points in the triangle are in trigonometric order, so all cross product should be > 0;
    // faire un schema pour verifier les formules
    Face f = vector_faces[i];
    Point p1 = vector_vertices[f.get_index_point_1()].get_point();
    Point p2 = vector_vertices[f.get_index_point_2()].get_point();
    Point p3 = vector_vertices[f.get_index_point_3()].get_point();
    if (is_counter_clockwise(p, p1, p2) == false){return false;}
    if (is_counter_clockwise(p, p2, p3) == false){return false;}
    if (is_counter_clockwise(p, p3, p1) == false){return false;}

    return true;
}

void Mesh::insert_point(const Point& p){
    // Insert a point p into the Mesh.
    // The point 0 of the Mesh is the infinite vertex, which has a z coordinate, contrary to all points.
    // Firt we check for all triangles if the point is inside one of the faces (triangles).
    bool intersection_face = false;
    int a = get_vector_faces().size();
    for (int i = 0; i<a; i++){
        if (is_inside(i, p)){
            intersection_face = true;
            triangle_split(i, p);
// insertion into i-th triangle
        }
    }
    if (intersection_face == false){
    // the point is outside convex hull, so it is appended to the convex hull.
    // loop through all edges of the convex hull until neighboring edge if found
    std::vector<int> indices_convex_hull = get_neighbors(0);
    for(int i = 0; i<indices_convex_hull.size(); i++){
        if(is_counter_clockwise(vector_vertices[indices_convex_hull[i]].get_point(), vector_vertices[indices_convex_hull[(i+1)%indices_convex_hull.size()]].get_point(), p)){
            int j = get_index_face(indices_convex_hull[i], indices_convex_hull[(i+1)%indices_convex_hull.size()], 0);
            triangle_split(j, p);
            i+=1;
            while (i<indices_convex_hull.size() && is_counter_clockwise(vector_vertices[indices_convex_hull[i]].get_point(), vector_vertices[indices_convex_hull[(i+1)%indices_convex_hull.size()]].get_point(), p)){
                int index_face = get_index_face(indices_convex_hull[i], 0, get_number_vertices()-1);
                edge_flip(get_number_vertices()-1, index_face);
                i += 1;
            }
            break; // Exit the loop once the condition is fulfilled
        }
    }

    }
}

int Mesh::get_index_face(int ip1, int ip2, int ip3) {
    for (int i = 0; i < vector_faces.size(); ++i) {
        const Face& face = vector_faces[i];
        if ((face.get_index_point_1() == ip1 || face.get_index_point_1() == ip2 || face.get_index_point_1() == ip3) &&
            (face.get_index_point_2() == ip1 || face.get_index_point_2() == ip2 || face.get_index_point_2() == ip3) &&
            (face.get_index_point_3() == ip1 || face.get_index_point_3() == ip2 || face.get_index_point_3() == ip3)) {
            return i; // Return index of the face
        }
    }
    // Return -1 if no matching face is found
    return -1;
}


void GeometricWorld::temperature(){
    // update temperature vector at regular intervals
    // temperature equation is solved on the queen.
//    drawQueen();
    if (computing_laplacian == false){
    //    std::vector<double> laplacian_vector(number_vertices, 1.0);
//        for (int i=0; i<this->vector_temperature.size(); i++){
//            this->vector_temperature[i] = rand() % 10 + 1;
//        }
        //std::cout << "computing laplacian false!" << std::endl;
        //std::vector<int> nei = vector_meshes[3].get_neighbors(1000);
        //std::vector<int> nei1 = vector_meshes[3].get_neighbors(13880);
        //std::vector<int> nei2 = vector_meshes[3].get_neighbors(13699);
        //std::vector<int> nei3 = vector_meshes[3].get_neighbors(18560);
        //std::vector<int> nei4 = vector_meshes[3].get_neighbors(17315);
        //std::vector<int> nei5 = vector_meshes[3].get_neighbors(22718);
        //std::vector<int> nei6 = vector_meshes[3].get_neighbors(17317);


        this->vector_temperature[1000] = 255;
    }
    computing_laplacian = true;
    // get number of vertices and create temperature array size of number vertices
    // value 1000 for one of the values
    // then compute temperature at each step, and pause the computation
//    for (int step=0; step<100; step++){
        this->vector_temperature = queen.laplacian(this->vector_temperature);
//        drawLaplacian(laplacian_vector);
//        sleep(0.01);
//    }
}

void glPointDraw(const Point& p) {
    glVertex3f(p.get_x(), p.get_y(), p.get_z());
}

void GeometricWorld::drawMeshes(){
    if (computing_laplacian){
        drawLaplacian(this->vector_temperature);
    }
    else{
        for (int i = 0; i<vector_meshes.size(); i++){
            if (face_to_draw[i]==true){
                drawMesh(vector_meshes[i], i==4);
                if (i==4){
                    for(int j=0; j<vector_points_triangulation.size(); j++){
                        drawPoint(vector_points_triangulation[j], 0.0, 1.0, 0.0);
                    }
                }
            }
        }
    }

}

void GeometricWorld::drawLaplacian(const std::vector<double> &u){
    // Laplacian is drawn on queen only
    // (not very interesting on the other small meshes)
    // u: vector of temperatures, or another function on which laplacian is computed
    const std::vector<Vertex> vector_vertices = vector_meshes[3].get_vector_vertices();
    for (int i =0; i<vector_vertices.size(); i++){
//        drawPoint(vector_meshes[3].get_vector_vertices()[i].get_point(), u[i], 0, 0, 5.0f);
        drawPoint(vector_vertices[i].get_point(), u[i], 0, 0, 5.0f);
    }
}

void GeometricWorld::drawTetrahedron(){
//    drawMesh(vector_meshes[0]);
    set_face_to_draw(0);
//    drawMesh(vector_meshes[0]);
}

void GeometricWorld::drawPyramid(){
//    drawMesh(vector_meshes[1]);
    set_face_to_draw(1);
//    drawMesh(vector_meshes[1]);
}

void GeometricWorld::drawBoundingBox(){
//    drawMesh(vector_meshes[2]);
    set_face_to_draw(2);
//    drawMesh(vector_meshes[2]);
}

void GeometricWorld::drawQueen(){
//    drawMesh(vector_meshes[3]);
    set_face_to_draw(3);
//    drawMesh(vector_meshes[3]);
}

void GeometricWorld::drawTriangulation(){
    set_face_to_draw(4);
}

void GeometricWorld::drawNeighbors(){
    if(draw_neighbors == false){
        draw_neighbors = true;
    }
    else{draw_neighbors = false;}
}

void GeometricWorld::addPointTriangulation(){
    vector_meshes[4].insert_point(vector_points_triangulation[index_next_point_triangulation]);
    index_next_point_triangulation += 1;
}

void GeometricWorld::drawMesh(const Mesh& mesh, bool edges){
    std::vector<Vertex> vertices_vector = mesh.get_vector_vertices();
    std::vector<Face> vector_faces = mesh.get_vector_faces();
    for (int i = 0; i<vector_faces.size(); i++){
        Point p1 = vertices_vector[vector_faces[i].get_index_point_1()].get_point();
        Point p2 = vertices_vector[vector_faces[i].get_index_point_2()].get_point();
        Point p3 = vertices_vector[vector_faces[i].get_index_point_3()].get_point();
        if (edges==true){
            drawTriangleEdges(p1, p2, p3);
        }
        else{drawTriangle(p1, p2, p3);}
    }
    if (draw_neighbors == true){
        drawPoint(vertices_vector[0].get_point(), 0.0, 0.0, 1.0);
        std::vector<int> index_neighbors = mesh.get_neighbors(0);
        for(int i = 0; i<index_neighbors.size(); i++){
            drawPoint(vertices_vector[index_neighbors[i]].get_point(), 1.0, 0.0, 0.0);
        }
    }
}

void GeometricWorld::drawPoint(const Point& point, int R, int G, int B, double size){
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glPointSize(size);  // Set the point size to 10
    glBegin(GL_POINTS);  // Begin drawing points
    glColor3d(R, G, B);
    glPointDraw(point);
    glEnd();
    glDisable(GL_BLEND);
}

void GeometricWorld::drawTriangle(const Point& p1, const Point& p2, const Point& p3){
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glColor4d(237, 252, 252, 0.5); // Set color with alpha (0.5 for semi-transparency)
    glBegin(GL_TRIANGLES);
    glPointDraw(p1);
    glPointDraw(p2);
    glPointDraw(p3);
    glEnd();
    glDisable(GL_BLEND); // Remember to disable blending when you're done
}

void GeometricWorld::drawTriangleEdges(const Point& p1, const Point& p2, const Point& p3) {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glColor4d(237, 252, 252, 0.5); // Set color with alpha (0.5 for semi-transparency)

    // Draw edges of the triangle
    glBegin(GL_LINE_LOOP);
    glPointDraw(p1);
    glPointDraw(p2);
    glPointDraw(p3);
    glEnd();

    glDisable(GL_BLEND); // Remember to disable blending when you're done
}

//Example with a bBox
void GeometricWorld::draw() {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glColor4d(237, 252, 252, 0.5); // Set color with alpha (0.5 for semi-transparency)
    glBegin(GL_TRIANGLES);
    glPointDraw(_bBox[0]);
    glPointDraw(_bBox[1]);
    glPointDraw(_bBox[2]);
    glEnd();
    glDisable(GL_BLEND); // Remember to disable blending when you're done
}

//Example with a wireframe bBox
void GeometricWorld::drawWireFrame() {
    glColor3d(1,0,0);
    glBegin(GL_LINE_STRIP);
    glPointDraw(_bBox[0]);
    glPointDraw(_bBox[1]);
    glEnd();
    glColor3d(0,1,0);
    glBegin(GL_LINE_STRIP);
    glPointDraw(_bBox[0]);
    glPointDraw(_bBox[2]);
    glEnd();
    glColor3d(0,0,1);
    glBegin(GL_LINE_STRIP);
    glPointDraw(_bBox[0]);
    glPointDraw(_bBox[3]);
    glEnd();
}
