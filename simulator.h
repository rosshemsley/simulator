/******************************************************************************/
#ifndef SIMULATOR_H
#define SIMULATOR_H
/******************************************************************************/

#include <fstream>
#include <string>
#include <random>
#include <vector>
#include <string>
#include <cmath>
#include <array>

#include <boost/thread.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/poisson_distribution.hpp>
#include <CGAL/point_generators_2.h>
#include <CGAL/Random.h>

using namespace std;

// A class to do multi-thread simulations on geometric objects taking points.
// N is the number of values returned by the function.
template <typename Structure, typename result_type, int N>
class Simulator
{
    typedef boost::lock_guard<boost::recursive_mutex> Lock;

    typedef unsigned long long                              Integer;
    typedef typename Structure::Geom_traits::Point_2        Point;
    typedef CGAL::Creator_uniform_2<double, Point>          Creator;
    typedef vector<Point>                                   Points;

    unsigned int thread_count;
    unsigned int seed;


    // This class synchronises the results.
    class Accumulator
    {
    private:
        mutable boost::recursive_mutex   mutex;

        // We store all intermediate values at each step in order
        // to estimate the confidence values.
        vector<array<result_type, N>> values;

        Integer        iterations_target;
        string         filename;
    public:

        /**********************************************************************/

        Accumulator(Integer iterations_target, string filename) : 
            iterations_target(iterations_target),
            filename(filename)
        {
            cout << "Writing output to: " << filename << endl;
        }

        /**********************************************************************/

        bool finished() 
        const
        {
            Lock lock(mutex);

            return (values.size() >= iterations_target);
        }

        /**********************************************************************/

        std::array<result_type,N> estimate() 
        const
        {
            // Avoid race conditions when values are updated during loop.
            Lock lock(mutex);

            // Value-initialise the return array to zero.
            // We probably don't have to copy this to the return because of 
            // RVO.
            std::array<result_type,N> result{};

            // Use reference to avoid unnecesary copying of array.
            // For each computed array of values.
            for (auto& v : values)
            {
                for (int i=0; i!=N; ++i)
                    result[i] += v[i];
            }

            for (auto& r : result)
                r = r/double(N);

            return result;
        }

        /**********************************************************************/

        // First values are estimates, second values are confidence regions.
        std::pair< std::array<result_type,N>, std::array<double,N> > 
        estimate_with_confidence(double confidence = 0.95) const
        {
            Lock lock(mutex);

            using boost::math::quantile;
            using boost::math::students_t;
            using boost::math::complement;

            Integer n = values.size();

            auto inf = std::numeric_limits<double>::infinity();

            // The intermediate results for each of the values.
            std::array<result_type, N> sum     {};
            std::array<result_type, N> sd      {};
            std::array<result_type, N> mean    {};
            std::array<double,      N> interval;

            // Set confidence intervals to be infinity by default.
            for (int i=0; i!=N; ++i)
                interval[i] = inf;

            // Compute estimate for mean.
            for (auto const& v : values) 
                for (int i=0; i!=N; ++i)
                    sum[i] += v[i];
            
            for (int i=0; i!=N; ++i)
                mean[i] = sum[i]/double(n);

            if (n>=30)
            {
                // Zero the sum array.
                for (int i=0; i!=N; ++i)
                    sum[i] = 0;

                // Compute sample estimate for standard deviation.
                for (auto const& v : values) 
                    for (int i=0; i!=N; ++i)
                        sum[i] += std::pow((v[i]-mean[i]),2);

                for (int i=0; i!=N; ++i)
                    sd[i]   = std::sqrt( sum[i]/double(n-1.5) );

                // Students-t distribution with paramter n-1.
                students_t dist(n - 1);

                // Compute confidence.
                double t = quantile(complement(dist,(1-confidence)/double(2)));
                
                for (int i=0; i!=N; ++i)
                    interval[i] = sd[i]*t/std::sqrt(double(n));
            }

            return std::make_pair(mean, interval);
        }

        /**********************************************************************/

        Integer completed() 
        const
        {
            Lock lock(mutex);

            return values.size();
        }

        /**********************************************************************/

        bool insert( array<result_type, N> const& v)
        {
            Lock lock(mutex);
            values.push_back(v);
            return !finished();
        }

        /**********************************************************************/

        // Return false if the process should stop.
        template <typename InputIterator>
        bool insert(InputIterator begin, InputIterator end)
        {
            Lock lock(mutex); 
            values.insert(begin, end);
            return !finished();
        }
        
    };

    /**************************************************************************/

    template<typename Functor>
    class Worker
    {
    private:
        boost::thread  worker_thread;
        Functor        f;
        unsigned int   seed;
        Accumulator&   accumulator;
        Structure      structure;    
        Points         points;

    public:
        
        Worker( Functor f, unsigned int seed, Accumulator & accumulator )
            : f(f), seed(seed), accumulator(accumulator) {}

        /**********************************************************************/

        void process(unsigned int lambda, double side)
        {

            // Seed from hardware random number generator if available.
            // If not available, this is probably just a mersenne twister.
            // We xor in the time just in case something strange happens.
            std::random_device rd;
            seed ^= rd();
            seed ^= std::time(0);

            std::uniform_int_distribution<int> dist(0, 9);

            // Generate Poisson r.v.
            // This could be more efficiently if we could access the CGAL
            // random generator. Do this later.
            boost::mt19937 boost_rng;
            boost_rng.seed(static_cast<unsigned int>(seed) );
            boost::poisson_distribution<> gen(lambda);

            // Create random generator.
            CGAL::Random rng( seed );
            CGAL::Random_points_in_square_2<Point,Creator> g( side/2, rng );
    
            while (true)
            {                
                // Create a local copy of the functor to ensure that 
                // the state is not affected between runs.
                Functor temp_f(f);
                points.clear();
                structure.clear();

                auto n = gen(boost_rng);

                // Generate a Poisson(lambda) number of points.
                CGAL::cpp11::copy_n(g, n, std::back_inserter(points));
                structure.insert(points.begin(), points.end());    
                        
                // Stop if we have finished enough.
                if (!  accumulator.insert( temp_f(structure) )  )
                    return;
            }
        }

        /**********************************************************************/

        void start(unsigned int lambda, double side)
        {
            // Create a thread to do the jobn.
            worker_thread = boost::thread(&Worker::process, this, lambda, side);  
        }

        /**********************************************************************/

        void join()
        {
            worker_thread.join();
        }

    };

    /**************************************************************************/

    class Supervisor
    {
        // The supervisor doesn't change the accumulator.
        Accumulator const &  accumulator;
        boost::thread        supervisor_thread;
    public:

        /**********************************************************************/

        Supervisor( Accumulator const & accumulator) 
            : accumulator(accumulator) {}        

        /**********************************************************************/    

        void display() 
        const
        {
            while(true)
            {
                std::cout << "Completed: " << accumulator.completed();
                if (accumulator.completed() > 0)
                { 
                    auto estimate = accumulator.estimate_with_confidence();
                    std::cout << " Current estimate: ";

                    for (int i=0; i!=N; ++i)
                    {
                        std::cout << "{" 
                                  << estimate.first[i]
                                  << " +-"
                                  << estimate.second[i]
                                  << "}";
                        if (i!=N-1) 
                            std::cout << ", ";
                    }
                }
                cout << endl;

                if ( accumulator.finished() )
                    return;
                sleep(1);
            }
        }

        /**********************************************************************/

        void start()
        {
            supervisor_thread = boost::thread(&Supervisor::display, this);
        }

        /**********************************************************************/

        void join()
        {
            supervisor_thread.join();
        }

    };    

    /**************************************************************************/

public:

    // Make the tuple size public.
    static const int tuple_size  = N;

    Simulator( int thread_count )
        : thread_count(thread_count) {}

    /**************************************************************************/
    template<typename Functor>
    std::pair<std::array<result_type,N>, std::array<double,N>> 
    run( 
            Functor&&       f, 
            Integer         lambda,
            double          side,
            Integer         repeats,
            string          filename
        )
    {
        // Store the simulation results here.
        Accumulator accumulator(repeats, filename);        

        // Something to output what's going on.
        Supervisor  supervisor(accumulator);

        // Start the supervisor thread.
        supervisor.start();

        // The workers.
        vector<Worker<Functor>*> workers;
        for (unsigned int i = 0; i < thread_count; ++i)
            workers.push_back( new Worker<Functor>( f, i , accumulator ) );        

        // Run all of the jobs.
        for ( auto worker : workers )    
            worker->start( lambda, side );

        // Wait until they have all finished.
        for ( auto worker : workers )
            worker->join();

        // End the supervisor thread.
        supervisor.join();

        // Free all memory allocated.
        for ( auto worker : workers )
            delete worker;
        
        workers.clear();

        // Don't allow seed to be the same.
        ++seed;

        // Return the final approximation.
        return accumulator.estimate_with_confidence();
    }

    /**************************************************************************/

};

/******************************************************************************/
#endif
/******************************************************************************/