/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 1000;  // TODO: Set the number of particles
  std::normal_distribution<double> x_dist(x, std[0]);
  std::normal_distribution<double> y_dist(y, std[1]);
  std::normal_distribution<double> theta_dist(theta, std[2]);
  
  std::default_random_engine gen;
  
  //add noise
  
  for (auto i=0U ; i<num_particles ; ++i)
  {
    particles[i].x = x_dist(gen);
    particles[i].y = y_dist(gen);
    particles[i].theta = theta_dist(gen);
    particles[i].weight = 1.0;
    particles[i].id = i;
    weights.emplace_back(1.0);
    
  }
  
  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  
  // add noise , account for yaw_rate =0
  
  const double meu{0.0};
  
  std::normal_distribution<double> x_dist(meu,std_pos[0]);
  std::normal_distribution<double> y_dist(meu,std_pos[1]);
  std::normal_distribution<double> theta_dist(meu,std_pos[2]);
  
  std::default_random_engine gen;
  double epsilon{std::numeric_limits<double>::epsilon()};
  double velocity_div_yaw_rate{(velocity/(yaw_rate))};
  double yaw_rate_times_delta_t{yaw_rate * delta_t};
  
  for (auto i = 0U; i< particles.size(); ++i)
  {
    if(fabs(yaw_rate) < epsilon)
    {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }
    else
    {
      particles[i].x += velocity_div_yaw_rate * (sin(particles[i].theta + yaw_rate_times_delta_t) - sin(particles[i].theta));
      particles[i].y += velocity_div_yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate_times_delta_t));
      particles[i].theta+= yaw_rate * delta_t;
    }
                                              
   //adding gaussian noise
    particles[i].x += x_dist(gen);
    particles[i].y += y_dist(gen);
    particles[i].theta += theta_dist(gen);
                
  }
  
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  
  double min_distance{std::numeric_limits<double>::max()};
  
  for (auto &observation : observations)
  {
    
    for (auto &prediction : predicted)
    {
      double distance{dist(prediction.x,prediction.y,observation.x,observation.y)};
      
      if(distance < min_distance)
      {
        observation.id = prediction.id;
        min_distance = distance;
      }
    }
  }
  
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  
  vector<LandmarkObs> transformed_observations{};
  vector<LandmarkObs> landmarks_with_sensor_range{};
  
  for (auto &particle : particles)
  {
    int j{0};
    
    for(auto i = 0 ; i < observations.size() ; ++i)
    {
      transformed_observations[i].x = (observations[i].x * cos(particle.theta)) - (observations[i].y * sin(particle.theta)) + particle.x;
      transformed_observations[i].y = (observations[i].x * sin(particle.theta)) + (observations[i].y * sin(particle.theta)) + particle.y;
    }
    
    for (auto &landmark : map_landmarks.landmark_list)
    {
      if(dist(landmark.x_f, landmark.y_f, particle.x, particle.y) <= sensor_range)
      {
        landmarks_with_sensor_range.emplace_back(LandmarkObs{landmark.id_i, static_cast<double>(particle.x), static_cast<double>(particle.y)});
      }
    }
    
    dataAssociation(landmarks_with_sensor_range, transformed_observations);
    
    particle.weight = 1.0;
    
    const double std_x_2{pow(std_landmark[0], 2.0)};
    const double std_y_2{pow(std_landmark[1], 2.0)};
    const double norm_factor{(1 / (2 * M_PI * std_landmark[0] * std_landmark[1]))};
    
    for (auto i = 0 ; i < transformed_observations.size(); ++i)
    {
      int nearest_landmark_id = transformed_observations[i].id - 1;
      
      const double diff_x_2{pow(transformed_observations[i].x - landmarks_with_sensor_range[i].x, 2.0)};
      const double diff_y_2{pow(transformed_observations[i].y - landmarks_with_sensor_range[i].y, 2.0)};
      
      
      particle.weight *= norm_factor * exp(-diff_x_2 / (2 * std_x_2) + diff_y_2 / (2 * std_y_2));
    }
    
    weights[j] = particle.weight;
    
    j+=1;
    
 }
  

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
  std::default_random_engine gen;
  std::discrete_distribution<int> importance_dist(weights.begin(), weights.end());
  std::vector<Particle> resampled_particles{};
  
  for (auto i = 0; i < num_particles; ++i)
  {
    resampled_particles.emplace_back(particles[importance_dist(gen)]);
  }
  
  particles = resampled_particles;

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}