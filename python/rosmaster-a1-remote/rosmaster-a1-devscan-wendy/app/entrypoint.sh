#!/usr/bin/env bash
# Report which serial nodes the host actually has, and what each by-id symlink
# points at. Deliberately claims no serial entitlement, so it deploys whatever
# the hardware is doing; an entitlement naming an absent device hard-fails
# container creation, which is the very failure this exists to diagnose.
echo "DEVSCAN begin"
echo "DEVSCAN by-id:"
ls -la /dev/serial/by-id/ 2>&1 | sed 's/^/DEVSCAN   /'
echo "DEVSCAN tty nodes:"
ls -la /dev/ttyUSB* /dev/ttyACM* 2>&1 | sed 's/^/DEVSCAN   /'
echo "DEVSCAN sysfs tty:"
ls -la /sys/class/tty/ 2>/dev/null | grep -E 'USB|ACM' | sed 's/^/DEVSCAN   /'
echo "DEVSCAN end"
sleep 3600
