#!/usr/bin/env bash

set -euo pipefail

stop_services() {
    hbase-daemon.sh stop thrift || true
    stop-hbase.sh || true
}

trap stop_services EXIT INT TERM

start-hbase.sh 
hbase-daemon.sh start thrift -p 9090 --infoport 9095

until curl --fail --silent http://127.0.0.1:16010/master-status >/dev/null; do
    sleep 5
done

while curl --fail --silent \
    http://127.0.0.1:16010/master-status >/dev/null; do
    sleep 5
done

exit 1