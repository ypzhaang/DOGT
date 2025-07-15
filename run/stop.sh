#!/bin/bash

logdir=$(ls -td logs_* 2>/dev/null | head -n 1)

if [ -z "$logdir" ]; then
    echo "No logs_* directory found. Cannot locate pid file."
    exit 1
fi

pidfile="$logdir/pids.txt"

if [ ! -f "$pidfile" ]; then
    echo "No PID file found in $logdir! Cannot stop processes."
    exit 1
fi

echo "Stopping processes listed in $pidfile..."

read -a pids < "$pidfile"

for pid in "${pids[@]}"; do
    if kill "$pid" > /dev/null 2>&1; then
        echo "Stopped process with PID $pid"
    else
        echo "Failed to stop PID $pid (might already be stopped)"
    fi
done

echo "All listed processes handled."
