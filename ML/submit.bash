#!/bin/bash

# Usage check
if [ $# -ne 1 ]; then
    echo "Usage: $0 HH:MM"
    exit 1
fi

# Parse time from argument
IFS=: read TARGET_HOUR TARGET_MIN <<< "$1"

# Validate input
if ! [[ "$TARGET_HOUR" =~ ^[0-9]{1,2}$ && "$TARGET_MIN" =~ ^[0-9]{1,2}$ ]]; then
    echo "Invalid time format. Use HH:MM (e.g. 23:30)"
    exit 1
fi

# Compute target time in total minutes
TARGET_TOTAL_MIN=$((10#$TARGET_HOUR * 60 + 10#$TARGET_MIN))

# Coordinates to click
X=1103
Y=642

while true; do
    CUR_HOUR=$(date +"%H")
    CUR_MIN=$(date +"%M")
    CUR_TOTAL_MIN=$((10#$CUR_HOUR * 60 + 10#$CUR_MIN))
    
    echo "Current time: $CUR_HOUR:$CUR_MIN"

    if [ "$CUR_TOTAL_MIN" -ge "$TARGET_TOTAL_MIN" ]; then
        #cliclick c:$X,$Y
        osascript -e 'tell application "System Events" to click at {1103, 642}'
        echo "Clicked at $X,$Y at $CUR_HOUR:$CUR_MIN"
        break
    else
        sleep 5
    fi
done

