#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/point_generators_2.h>
#include <CGAL/Random.h>
#include <array>
#include "./simulator.h"

typedef CGAL::Exact_predicates_inexact_constructions_kernel  Kernel;
typedef CGAL::Creator_uniform_2<double, Kernel::Point_2>     Creator;
typedef CGAL::Delaunay_triangulation_2<Kernel>               Delaunay;

// Simulator to work on Delaunay triangulation, with siulations returning
// 2 double values in a std::array<double, 6>.
typedef Simulator<Delaunay, double, 2> My_simulator;

using namespace std;

int main()
{    
    // A simulator that uses 4 threads.
    My_simulator simulator{4};

    // A simulation to run.
    // We compute the average degree, and return it (along with the number
    // of points in the triangulation).
    auto sim = [](Delaunay const& tr)
    {
        unsigned sum   = 0;
        unsigned count = 0;
        
        for (auto v = tr.finite_vertices_begin(); 
                  v!= tr.finite_vertices_end();   ++v )
        {
            ++count;
            sum += v->degree();
        }

        // Return average degree, and the number of vertices (as double...)
        return array<double, 2>{{ sum/double(count), double(count) }};
    };
    
    // Run (at least) 100 repeats of 'sim', on randomly generated triangulations
    // with 2^20 points in a box of side 0.5, output to './out.txt'. Returns a
    // pair of two arrays, one containing values, one containing confidence
    // intervals (if there were sufficiently many repeats.)
    auto result = simulator.run(sim, 1<<20, 0.5, 100, "out.txt");

    // Show output with confidence intervals.
    for (int i=0; i!=My_simulator::tuple_size; ++i)
    {
        cout 
            << "Result "
            << i
            << " is "
            << result.first[i]                     
            << " "                                 
            << result.first[i] - result.second[i]  
            << " "                                 
            << result.first[i] + result.second[i]
            << endl;
    }
}