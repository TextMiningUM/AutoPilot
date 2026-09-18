#!/bin/bash
#-------------------------------------------------
# Launch script for scenario: s08_overtaking_os_stands_on
# Usage: ./launch.sh [time_warp]
#-------------------------------------------------

TIME_WARP=${1:-1}

# NOTE: for TIME_WARP != 1, also set MOOSTimeWarp in each .moos file
#       before launching, or use the --warp flag if your MOOS-IvP
#       build's pAntler/mission tooling supports it directly.

echo "Launching shoreside..."
pAntler shoreside.moos >> log_shoreside.txt 2>&1 &
sleep 1

echo "Launching opship..."
pAntler opship.moos >> log_opship.txt 2>&1 &
sleep 1

echo "Launching ts1..."
pAntler ts1.moos >> log_ts1.txt 2>&1 &
sleep 1

echo "All communities launched. Use pMarineViewer (shoreside) to observe."
echo "To deploy vehicles: click DEPLOY in pMarineViewer, or:"
echo "  uPokeDB opship.moos DEPLOY=true"

#-------------------------------------------------
# To kill all processes for this scenario:
#   kill -9 $(ps -ef | awk \'/MOOSDB|pAntler|pHelmIvP|uSimMarine|pMarinePID|pMarineViewer|pLogger|pShare|pHostInfo|uFldNodeBroker|uFldShoreBroker|pNodeReporter/{print $2}\')
#-------------------------------------------------
