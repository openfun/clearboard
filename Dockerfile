# Clearboard enhances blackboard images

# ---- base image to inherit from ----
FROM python:3.9-slim as base

# Upgrade pip to its latest release to speed up dependencies installation
RUN python -m pip install --upgrade pip

# Upgrade system packages to install security updates
RUN apt-get update && \
  apt-get -y upgrade && \
  rm -rf /var/lib/apt/lists/*

# ---- Back-end builder image ----
FROM base as back-builder

WORKDIR /builder

# Copy required python dependencies
COPY ./src/backend /builder

RUN mkdir /install && \
  pip install --prefix=/install .

# ---- Core application image ----
FROM base as core

ENV PYTHONUNBUFFERED=1

# Copy installed python dependencies
COPY --from=back-builder /install /usr/local

# Copy application
COPY ./src/backend /app/

# Copy entrypoint
COPY ./docker/files/usr/local/bin/entrypoint /usr/local/bin/entrypoint

# Give the "root" group the same permissions as the "root" user on /etc/passwd
# to allow a user belonging to the root group to add new users; typically the
# docker user (see entrypoint).
RUN chmod g=u /etc/passwd

WORKDIR /app

# We wrap commands run in this container by the following entrypoint that
# creates a user on-the-fly with the container user ID (see USER) and root group
# ID.
ENTRYPOINT [ "/usr/local/bin/entrypoint" ]

# ---- Development image ----
FROM core as development

# Switch back to the root user to install development dependencies
USER root:root

# Uninstall clearboard and re-install it in editable mode along with development
# dependencies
RUN pip uninstall -y clearboard
RUN pip install -e .[dev]

# Restore the un-privileged user running the application
ARG DOCKER_USER
USER ${DOCKER_USER}

# Run uvicorn as development server
CMD ["uvicorn", "clearboard.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--reload", "--log-level", "debug"]

# ---- Production image ----
FROM core as production

# Un-privileged user running the application
ARG DOCKER_USER
USER ${DOCKER_USER}

# The default command runs gunicorn WSGI server in clearboard's main module
CMD ["gunicorn", "-c", "/usr/local/etc/gunicorn/clearboard.py", "app.main:app"]
