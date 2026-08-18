#!/usr/bin/env bash
set -Eeuo pipefail
printf 'status=pilot_execution_not_authorized_fail_closed\n'
printf 'future_pilot_execution_authorization_record=absent\n'
exit 65
