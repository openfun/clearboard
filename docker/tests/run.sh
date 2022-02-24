#!/usr/bin/env bash
#
# Test that the container can be running with the default command
#
# If the command fails this script will return a non-zero exit code (it should
# be the gunicorn process exit code). Else, if it must send a kill signal after
# 5 seconds, it will return a zero exit code, because it means that gunicorn has
# booted with no errors and was running.

CLEARBOARD_IMAGE_NAME="${1:-clearboard}"
CLEARBOARD_IMAGE_TAG="${2:-latest}"

echo "Running image ${CLEARBOARD_IMAGE_NAME}:${CLEARBOARD_IMAGE_TAG}..."

timeout --preserve-status -k 5s 5s \
    docker run --rm --env-file=env.d/development.dist "${CLEARBOARD_IMAGE_NAME}:${CLEARBOARD_IMAGE_TAG}"

exit_code=$?
expected=137

if [[ $exit_code -ne $expected ]]; then
    echo "Test run: FAILED (exit code: $exit_code)"
    exit $exit_code
else
    echo "Test run: SUCCEED"
    exit 0
fi
