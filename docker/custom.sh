# install 'autonomy_datasets_msgs' package only
cd /docker-ros/ws
git clone --branch v1.5.0 https://github.com/thinking-cars/autonomy_datasets.git src/autonomy_datasets
rosdep update && rosdep install -y -i --from-paths src/autonomy_datasets/autonomy_datasets_msgs
source /opt/ros/${ROS_DISTRO}/setup.bash
colcon build --packages-up-to autonomy_datasets_msgs
rm -r src/autonomy_datasets log build
