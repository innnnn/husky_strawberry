#include "find_affordance.hpp"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "find_affordance_node");
  ros::NodeHandle nh, pnh("~");
  PCPerception pcp(nh, pnh);
  while(ros::ok()) ros::spinOnce();
  return 0;
}
