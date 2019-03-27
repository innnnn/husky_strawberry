#include <vector> // std::vector
#include <utility> // std::pair
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/passthrough.h>
#include <std_msgs/ColorRGBA.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseArray.h>
#include <image_prediction/bboxList.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

typedef pcl::PointCloud<pcl::PointXYZ> PointXYZ; // Not sure why xyzrgb cannot work
typedef pcl::PointCloud<pcl::PointXYZRGB> PointXYZRGB;
typedef std::pair<int, int> Pixel; // Image coordinate
typedef std::pair<Pixel, int> PixelWithR; // Image coordinate with its corresponding R-ch value
typedef std::vector<PixelWithR> PixelWithRVector;

const double PCNUM_THRES = 100; // Threshold number of points to cossidered as target
const double IMG_WIDTH = 640; // Image width
const std::string FRAME = "camera_color_optical_frame";
const std::string PC_STR = "/camera/depth_registered/points";

class PCPerception{
 private:
  ros::NodeHandle nh_, pnh_;
  ros::Subscriber sub_bbox;
  ros::Publisher pub_marker, pub_pc, pub_aff;
  PointXYZ pc_;
  bool verbose; // If true, also publish pc of last target
  double std_mul;
  /* Initial marker with given id number
     @param
       visualization_msgs::Marker&: target marker
       int: input id
  */
  void initMarker(visualization_msgs::Marker&, int);
  /*
      Callback function for subscriber
      Extract the pixel in bounding box to point cloud and visualize the center
  */
  void callback(const image_prediction::bboxList);
  /*
      Statistical processing for calculating mean and standard derivation to
      find the inlier pixel
      @param
        const PixelWithRVector: given target array
        double&: mean reference
        double&: std reference
  */
  void statistic_processing(const PixelWithRVector, double&, double&);
 public:
  /*
      Constructor
      @param
        ros::NodeHandle: public node handler
        ros::NodeHandle: private one
  */ 
  PCPerception(ros::NodeHandle, ros::NodeHandle);
};

PCPerception::PCPerception(ros::NodeHandle nh, ros::NodeHandle pnh): \
	nh_(nh), pnh_(pnh){
  sub_bbox = nh_.subscribe("bounding_box", 1, &PCPerception::callback, this);
  pub_marker = pnh_.advertise<visualization_msgs::MarkerArray>("affordance_marker", 1);
  pub_aff = pnh_.advertise<geometry_msgs::PoseArray>("affordance", 1);
  if(!pnh_.getParam("std_mul", std_mul)){
    std_mul = 0.5;
    ROS_WARN("std_mul use default value: %f", std_mul);
  }
  if(!pnh_.getParam("verbose", verbose)){
    verbose = false;
    ROS_WARN("std_mul use default value: %s", (verbose==true?"true":"false"));
  } if(verbose) pub_pc = pnh_.advertise<sensor_msgs::PointCloud2>("points", 1);
}

void PCPerception::initMarker(visualization_msgs::Marker& marker, int id){
  marker.header.frame_id = FRAME;
  marker.id = id;
  // Sphere
  marker.type = visualization_msgs::Marker::SPHERE;
  marker.action = visualization_msgs::Marker::ADD;
  // 5 cm
  marker.scale.x = 0.05;
  marker.scale.y = 0.05;
  marker.scale.z = 0.05;
  // Black
  marker.color.r = 0.0f;
  marker.color.g = 0.0f;
  marker.color.b = 1.0f;
  marker.color.a = 1.0f;
}

void PCPerception::callback(const image_prediction::bboxList msg){
  int numOfTarget = msg.num;
  if(numOfTarget == 0) return; // No target
  // Convert Image to cv image
  cv_bridge::CvImagePtr cv_ptr;
  cv_ptr = cv_bridge::toCvCopy(msg.img, sensor_msgs::image_encodings::BGR8);
  sensor_msgs::PointCloud2ConstPtr pc = ros::topic::waitForMessage<sensor_msgs::PointCloud2>\
   										 (PC_STR, ros::Duration(3.0));
  if(pc == NULL) {ROS_ERROR("Not point cloud received."); return;}
  pcl::fromROSMsg(*pc, pc_);
  geometry_msgs::PoseArray poseArray;
  visualization_msgs::MarkerArray markerArray;
  visualization_msgs::Marker marker;
  PointXYZ temp;
  for(int i=0; i<numOfTarget; ++i){ 
    initMarker(marker, i);
    PixelWithRVector pwrv;
    int range[4] = {msg.bbox[i].bb[0], msg.bbox[i].bb[1], msg.bbox[i].bb[2], msg.bbox[i].bb[3]};
    // Encode: xmin, ymin, xmax, ymax
    for(int x=range[0]; x<=range[2]; ++x){
      for(int y=range[1]; y<=range[3]; ++y){
        Pixel p = std::make_pair(x, y);
        PixelWithR pwr = std::make_pair(p, cv_ptr->image.at<cv::Vec3b>(y, x)[2]);
        pwrv.push_back(pwr);
      } // end y
    } // end x
    double mean, std;
    statistic_processing(pwrv, mean, std);
    // Inlier
    PixelWithRVector inlier;
    for(int num=0; num<pwrv.size(); ++num){ 
      if(pwrv[num].second > std_mul*std+mean){ // Standardized data greater than std_mul 
        inlier.push_back(pwrv[num]);
      } // end if
    } // end num
    // Extract inlier pixels to pc
    for(int num=0; num<inlier.size(); ++num){
      //temp.push_back(pc_.at(inlier[num].first.first, inlier[num].first.second)); //Not Organized!!
      int idx = inlier[num].first.second * IMG_WIDTH + inlier[num].first.first;
      if(idx>=pc_.size()){
        ROS_WARN("Index out of range, ignore...");
      } else temp.push_back(pc_.at(idx)); 
    } // end num
    if(temp.size()<PCNUM_THRES){
      ROS_WARN("Points too few, ignore...");
      return;
    }
    // Get centroid
    Eigen::Vector4f centroid; 
    pcl::compute3DCentroid(temp,centroid);
    // Check if nan
    if(std::isnan(centroid[0])){
      ROS_WARN("Get nan, ignore...");
      return;
    } 
    geometry_msgs::Pose pos; 
    pos.position.x = centroid[0]; pos.position.y = centroid[1]; pos.position.z = centroid[2];
    ROS_INFO("%f %f %f", centroid[0], centroid[1], centroid[2]);
    pos.orientation.w = 1.0;
    marker.pose = pos;
    markerArray.markers.push_back(marker);
    poseArray.poses.push_back(pos);
  } // end i
  // Publish pose array
  pub_aff.publish(poseArray);
  // Publish marker
  pub_marker.publish(markerArray);
  // Publish pc
  if(verbose){
    sensor_msgs::PointCloud2 pcout;
    pcl::PCLPointCloud2 pc2;
    pcl::toPCLPointCloud2(temp, pc2); // First convert to pclpc2
    pcl_conversions::fromPCL(pc2, pcout);
    pcout.header.frame_id = FRAME; // Add frame id
    pub_pc.publish(pcout); // Then publish
  }
}

void PCPerception::statistic_processing(const PixelWithRVector pwrv, double& mean, double& std){
  double sum = 0, square_sum = 0;
  for(int i=0; i<pwrv.size(); ++i){
    sum += pwrv[i].second;
    square_sum += pwrv[i].second*pwrv[i].second;
  }
  mean = sum/(float)pwrv.size();
  std = std::sqrt(square_sum/(float)pwrv.size() - mean*mean);
}
