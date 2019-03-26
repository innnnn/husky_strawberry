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
#include <image_prediction/bboxList.h>
#include <visualization_msgs/Marker.h>

typedef pcl::PointCloud<pcl::PointXYZ> PointXYZ; // Not sure why xyzrgb cannot work
typedef std::pair<int, int> Pixel;
typedef std::pair<Pixel, int> PixelWithR; 
typedef std::vector<PixelWithR> PixelWithRVector;

const std::string FRAME = "camera_color_optical_frame";
const std::string PC_STR = "/camera/depth_registered/points";

class PCPerception{
 private:
  ros::NodeHandle nh_, pnh_;
  ros::Subscriber sub_bbox;
  ros::Publisher pub_marker, pub_pc;
  PointXYZ pc_;
  double std_mul;
  void initMarker(visualization_msgs::Marker&);
  void callback(const image_prediction::bboxList);
  void statistic_processing(const PixelWithRVector, double&, double&);
 public:
  PCPerception(ros::NodeHandle nh, ros::NodeHandle pnh);
};

PCPerception::PCPerception(ros::NodeHandle nh, ros::NodeHandle pnh): \
	nh_(nh), pnh_(pnh){
  sub_bbox = nh_.subscribe("/bounding_box", 1, &PCPerception::callback, this);
  pub_marker = pnh_.advertise<visualization_msgs::Marker>("affordance", 1);
  pub_pc = pnh_.advertise<sensor_msgs::PointCloud2>("points", 1);
  if(!pnh_.getParam("std_mul", std_mul)){
    std_mul = 0.5;
    ROS_WARN("std_mul use default value");
  }
}

void PCPerception::initMarker(visualization_msgs::Marker& marker){
  marker.header.frame_id = FRAME;
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
  visualization_msgs::Marker marker;
  initMarker(marker);
  PointXYZ temp;
  for(int i=0; i<numOfTarget; ++i){ 
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
      if(pwrv[num].second > std_mul*std+mean){
        inlier.push_back(pwrv[num]);
      } // end if
    } // end num
    // -------------*****************-------------*****************-------------*****************
    // Extract inlier pixels to pc
    for(int num=0; num<inlier.size(); ++num){
      //temp.push_back(pc_.at(inlier[num].first.first, inlier[num].first.second)); //Not Organized!!
      int idx = inlier[num].first.second * 640 + inlier[num].first.first;
      if(pc_.at(idx).x!=0. and pc_.at(idx).y!=0. and pc_.at(idx).z!=0.){
        temp.push_back(pc_.at(idx));
      } 
    } // end num
    if(temp.size()<100){
      ROS_WARN("Points too few, ignore...");
      return;
    }

    // Get centroid
    Eigen::Vector4f centroid; 
    pcl::compute3DCentroid(temp,centroid); 
    geometry_msgs::Pose pos; 
    pos.position.x = centroid[0]; pos.position.y = centroid[1]; pos.position.z = centroid[2];
    ROS_INFO("%f %f %f", centroid[0], centroid[1], centroid[2]);
    pos.orientation.w = 1.0;
    marker.pose = pos;
  } // end i
  if(!std::isnan(marker.pose.position.x))
    pub_marker.publish(marker);
  else{ROS_WARN("Get nan. ignore...");}
  sensor_msgs::PointCloud2 pcout;
  pcl::PCLPointCloud2 pc2;
  pcl::toPCLPointCloud2(temp, pc2); // First convert to pclpc2
  pcl_conversions::fromPCL(pc2, pcout);
  pcout.header.frame_id = FRAME;
  pub_pc.publish(pcout); // Then publish
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
