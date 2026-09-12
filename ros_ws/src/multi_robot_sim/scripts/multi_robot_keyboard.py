#!/usr/bin/env python3

import rospy
import sys
import select
import termios
import tty

from geometry_msgs.msg import Twist


class MultiRobotKeyboard:

    def __init__(self):

        rospy.init_node(
            "multi_robot_keyboard"
        )

        # ============================================================
        # Publishers
        # ============================================================

        self.robot1_pub = rospy.Publisher(
            "/robot1/cmd_vel",
            Twist,
            queue_size=1
        )

        self.robot2_pub = rospy.Publisher(
            "/robot2/cmd_vel",
            Twist,
            queue_size=1
        )

        # ============================================================
        # Parameters
        # ============================================================

        self.linear_speed = rospy.get_param(
            "~linear_speed",
            4.0
        )

        self.angular_speed = rospy.get_param(
            "~angular_speed",
            10.0
        )

        # ============================================================
        # Current selected robot
        # ============================================================

        self.current_robot = 1

        # ============================================================
        # Terminal
        # ============================================================

        self.settings = termios.tcgetattr(
            sys.stdin
        )

        rospy.loginfo(
            "=========================================="
        )

        rospy.loginfo(
            "Multi Robot Keyboard Teleop"
        )

        rospy.loginfo(
            "=========================================="
        )

        rospy.loginfo(
            "1 : Control robot1"
        )

        rospy.loginfo(
            "2 : Control robot2"
        )

        rospy.loginfo(
            "W : Forward"
        )

        rospy.loginfo(
            "S : Backward"
        )

        rospy.loginfo(
            "A : Turn left"
        )

        rospy.loginfo(
            "D : Turn right"
        )

        rospy.loginfo(
            "SPACE : Stop"
        )

        rospy.loginfo(
            "Q : Quit"
        )

        rospy.loginfo(
            "------------------------------------------"
        )

        rospy.loginfo(
            "Current robot: robot1"
        )

    # ================================================================
    # Get keyboard input
    # ================================================================

    def get_key(self):

        tty.setraw(
            sys.stdin.fileno()
        )

        key = sys.stdin.read(1)

        termios.tcsetattr(
            sys.stdin,
            termios.TCSADRAIN,
            self.settings
        )

        return key

    # ================================================================
    # Create Twist
    # ================================================================

    def create_twist(
        self,
        linear_x=0.0,
        angular_z=0.0
    ):

        cmd = Twist()

        cmd.linear.x = linear_x
        cmd.linear.y = 0.0
        cmd.linear.z = 0.0

        cmd.angular.x = 0.0
        cmd.angular.y = 0.0
        cmd.angular.z = angular_z

        return cmd

    # ================================================================
    # Publish command
    # ================================================================

    def publish_command(
        self,
        cmd
    ):

        if self.current_robot == 1:

            self.robot1_pub.publish(
                cmd
            )

        elif self.current_robot == 2:

            self.robot2_pub.publish(
                cmd
            )

    # ================================================================
    # Stop both robots
    # ================================================================

    def stop_all(self):

        stop_cmd = self.create_twist()

        self.robot1_pub.publish(
            stop_cmd
        )

        self.robot2_pub.publish(
            stop_cmd
        )

    # ================================================================
    # Process keyboard
    # ================================================================

    def process_key(
        self,
        key
    ):

        # ------------------------------------------------------------
        # Select robot1
        # ------------------------------------------------------------

        if key == "1":

            self.current_robot = 1

            self.stop_all()

            rospy.loginfo(
                "Current robot: robot1"
            )

        # ------------------------------------------------------------
        # Select robot2
        # ------------------------------------------------------------

        elif key == "2":

            self.current_robot = 2

            self.stop_all()

            rospy.loginfo(
                "Current robot: robot2"
            )

        # ------------------------------------------------------------
        # Forward
        # ------------------------------------------------------------

        elif key.lower() == "w":

            cmd = self.create_twist(
                linear_x=self.linear_speed
            )

            self.publish_command(
                cmd
            )

        # ------------------------------------------------------------
        # Backward
        # ------------------------------------------------------------

        elif key.lower() == "s":

            cmd = self.create_twist(
                linear_x=-self.linear_speed
            )

            self.publish_command(
                cmd
            )

        # ------------------------------------------------------------
        # Turn left
        # ------------------------------------------------------------

        elif key.lower() == "a":

            cmd = self.create_twist(
                angular_z=self.angular_speed
            )

            self.publish_command(
                cmd
            )

        # ------------------------------------------------------------
        # Turn right
        # ------------------------------------------------------------

        elif key.lower() == "d":

            cmd = self.create_twist(
                angular_z=-self.angular_speed
            )

            self.publish_command(
                cmd
            )

        # ------------------------------------------------------------
        # Stop
        # ------------------------------------------------------------

        elif key == " ":

            cmd = self.create_twist()

            self.publish_command(
                cmd
            )

        # ------------------------------------------------------------
        # Quit
        # ------------------------------------------------------------

        elif key.lower() == "q":

            self.stop_all()

            rospy.loginfo(
                "Keyboard teleop stopped."
            )

            return False

        return True

    # ================================================================
    # Run
    # ================================================================

    def run(self):

        rate = rospy.Rate(20)

        try:

            while not rospy.is_shutdown():

                key = self.get_key()

                if not self.process_key(
                    key
                ):
                    break

                rate.sleep()

        except Exception as e:

            rospy.logerr(
                "Keyboard teleop error: %s",
                str(e)
            )

        finally:

            self.stop_all()

            termios.tcsetattr(
                sys.stdin,
                termios.TCSADRAIN,
                self.settings
            )


if __name__ == "__main__":

    try:

        node = MultiRobotKeyboard()

        node.run()

    except rospy.ROSInterruptException:

        pass