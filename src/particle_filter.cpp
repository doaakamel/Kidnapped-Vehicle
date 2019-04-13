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
#define EPS 0.00001
using std::string;
using std::vector;
using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
 if (is_initialized)
    return;
  random_device rd;
   default_random_engine gen(rd());
  num_particles =90;
  weights.resize(num_particles);
  particles.resize(num_particles);

normal_distribution<double> dist_x(x, std[0]);
normal_distribution<double> dist_y(y, std[1]);
normal_distribution<double> dist_theta(theta, std[2]);


for (int i = 0; i < num_particles; i++) {
    particles[i].id = i;
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
    particles[i].weight = 1.0;
    }
is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  double std_x=std_pos[0];
  double std_y=std_pos[1];
  double std_th=std_pos[2];
  
  default_random_engine gen;
  normal_distribution<double> xNoise(0,std_x);
  normal_distribution<double> yNoise(0,std_y);
  normal_distribution<double> thetaNoise(0,std_th);
  for (int i = 0; i < num_particles; i++) 
  {
    if (fabs(yaw_rate) == 0)
    {  
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    } 
    else
    {
      particles[i].x += (velocity / yaw_rate) * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      particles[i].y += (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }
  
    particles[i].x += xNoise(gen);
    particles[i].y += yNoise(gen);
    particles[i].theta += thetaNoise(gen);
  }

}

//void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted//                                      vector<LandmarkObs>& observations)  
//   for(int i_tobs=0; i_tobs<observations.size(); i_s++){
//     double tobs_x = observationstobs].x;
//     double tobs_y = observati[i_tobs].y;
//     double min_distVal = 100000.0;  //very big value
//     int min_distIdx = 0;
//     for(int i_ldmk=0; i_ldmk<predid.size(); i_ldmk++){
//       double distance = dist(tobs_x, tobs_y, predicted[i_ldmk, predicted[i_ldmk].y);
//     f(distance < min_distVal){
//      min_distVal = distanc//      _distIdx = i_ldmk;
//       }
//     }
// observations[i_tobs].id = min_distIdx;
// }}



void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks)

  {
   double param1 = 1 / ( M_PI * std_landmark[0] * std_landmark[1]*2);
   double den_x =  std_landmark[0] * std_landmark[0]*2;
   double den_y =  std_landmark[1] * std_landmark[1]*2;

  for (int i = 0; i < num_particles; ++i) 
  {
    double Gauss_dis = 1.0;
    for (int j = 0; j < observations.size(); ++j)
    {
      double trans_obs_x, trans_obs_y;
      trans_obs_x = observations[j].x * cos(particles[i].theta) - observations[j].y * sin(particles[i].theta) + particles[i].x;
      trans_obs_y = observations[j].x * sin(particles[i].theta) + observations[j].y * cos(particles[i].theta) + particles[i].y;
      
      vector<Map::single_landmark_s> landmarks = map_landmarks.landmark_list;
      vector<double> landmark_obs_dist (landmarks.size());
      for (int k = 0; k < landmarks.size(); k++) {
       
        double landmark_part_dist = sqrt(pow(particles[i].x - landmarks[k].x_f, 2) + pow(particles[i].y - landmarks[k].y_f, 2));
        if (landmark_part_dist < sensor_range) {
          landmark_obs_dist[k] = sqrt(pow(trans_obs_x - landmarks[k].x_f, 2) + pow(trans_obs_y - landmarks[k].y_f, 2));

        } else {
      
          landmark_obs_dist[k] = 100000.0;
          
        }
        
      }
   
      int minPosition = distance(landmark_obs_dist.begin(),min_element(landmark_obs_dist.begin(),landmark_obs_dist.end()));
      double Curx = landmarks[minPosition].x_f;
      double Cury = landmarks[minPosition].y_f;

      double xObsMinCur = trans_obs_x - Curx;
      double yobsMinCur = trans_obs_y - Cury;
      double param2 = ((pow(xObsMinCur,2)) / den_x) + ((pow(yobsMinCur,2)) / den_y);
      Gauss_dis =Gauss_dis* param1 * exp(-param2);
      
    }
    
    particles[i].weight = Gauss_dis;
    weights[i] = particles[i].weight;

  }
                                     }

 


void ParticleFilter::resample() {

 vector<double> newweights;
 random_device rd;
 default_random_engine gen(rd());
  
  double max = numeric_limits<double>::min();
  for(int i = 0; i < num_particles; ++i) {
    newweights.push_back(particles[i].weight);
    if ( particles[i].weight > max ) {
      max = particles[i].weight;
    }
  }

  uniform_int_distribution<int> distInt(0, num_particles - 1);
  uniform_real_distribution<double> distDouble(0.0, max);
  double var = 0.0;
  int index = distInt(gen);
  
  vector<Particle> newParticles;
  for(int i = 0; i < num_particles; ++i)
  {
    var =var+ (distDouble(gen) * 2.0);
    while( var > newweights[index])
    {
       index = (index + 1) % num_particles;
       var -= newweights[index];
    }
    newParticles.push_back(particles[index]);
  }

  particles = newParticles;
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